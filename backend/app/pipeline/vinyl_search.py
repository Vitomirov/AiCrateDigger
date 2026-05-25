"""Consolidated vinyl-search pipeline (Redis cache + single Tavily call).

This module is the production hot path for the ``/search`` endpoint. It is a
deliberately small, linear function so latency / cost behaviour stays obvious
in production logs:

1. Parse the user query (one ``gpt-4o-mini`` call → :class:`ParsedQuery`).
2. Resolve the album anchor (literal title or Discogs ordinal).
3. Build the deterministic Redis key and short-circuit on a cache **hit**:
   zero Tavily credits, zero further LLM tokens for 7 days.
4. **Local-shop top-up** (city queries only, **non-blocking in production**):
   schedules :func:`ensure_local_coverage` via FastAPI ``BackgroundTasks`` so
   Tavily + LLM discovery upserts ``whitelist_stores`` after the HTTP response.
   Tests and callers without ``BackgroundTasks`` still ``await`` it inline so
   behaviour stays deterministic.
5. Load the active store hosts (curated + discovered) and pass them to the
   pre-filter as a **positive whitelist signal**.
6. Issue ONE consolidated Tavily call (``max_results`` from
   ``Settings.tavily_single_call_max_results``, default ``10``;
   ``search_depth="advanced"``) returning a European candidate pool for one
   Tavily credit.
7. Python pre-filter — blacklist noise (YouTube, news portals, …), require
   PDP-shaped URLs from unknown hosts, dedupe per host. Cap to ~10 candidates.
8. LLM extract + verification via :func:`extract_listings`.
9. Project to :class:`ListingResult`, **deduplicate by host** so one shop never
   surfaces twice in the visible result list.
10. Persist the response to BOTH Redis (7-day TTL, fast hot-path read) AND
    the Postgres ``search_response_cache`` table (operator visibility via
    DBeaver / SQL).

Production note: Because store discovery runs in a background task, the *current*
HTTP response uses the whitelist as it existed when :func:`load_active_stores`
ran; newly discovered domains apply on subsequent searches.

Fall-back behaviour
-------------------
* No ``REDIS_URL`` / unreachable Redis → Redis layer is a no-op; pipeline runs
  live and writes Postgres only.
* Tavily circuit breaker tripped → empty result list, fast return, no cache
  write (so a transient outage does not poison 7 days of cache).
* No OpenAI key / extractor failure → empty result list, no cache write.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks

from app.agents.extractor import extract_listings
from app.agents.parser.parse_user_query import parse_user_query
from app.config import get_settings
from app.db.cache import (
    build_search_cache_key,
    set_cached_search_payload,
)
from app.db.redis_cache import (
    build_redis_search_key,
    get_cached_search,
    set_cached_search,
)
from app.db.store_loader import ensure_local_coverage, load_active_stores
from app.domain.listing_schema import Listing
from app.models.result import ListingResult
from app.pipeline_context import stage_timer
from app.policies.format_detect import detect_format_token
from app.policies.store_domain import canonical_store_domain
from app.services.discogs_service import resolve_album_by_index
from app.services.tavily import tavily_circuit_breaker_scope
from app.services.tavily.prefilter import prefilter_tavily_results
from app.services.tavily.single_call import run_consolidated_tavily_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


async def _stage_parse(query: str) -> Any:
    """Run the parser inside the ``parse`` stage timer."""
    with stage_timer("parse", input={"query": query}) as rec:
        parsed = await parse_user_query(query)
        rec.output = parsed.model_dump()
    return parsed


async def _stage_resolve_album_title(parsed: Any) -> str | None:
    """Resolve a Discogs release (ordinal queries) or take the literal album."""
    with stage_timer("discogs"):
        if parsed.album_index is not None:
            resolution = await resolve_album_by_index(
                artist=parsed.artist,
                album_index=parsed.album_index,
            )
            raw_title = resolution.album.title if resolution.album else None
            return (raw_title or "").strip() or None
        if parsed.album:
            return (parsed.album or "").strip() or None
        return None


async def _stage_ensure_local_coverage(parsed: Any) -> None:
    """Top up indie ``local_shop`` rows in the DB for the resolved city.

    Runs only when the parser resolved a specific city + country. Uses
    Tavily + LLM under the hood (:mod:`app.services.store_discovery`) and
    upserts new rows into ``whitelist_stores`` — so the next request (and the
    Python prefilter on this request) automatically trusts them.
    """
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip()
    if not city or not cc:
        return
    with stage_timer(
        "store_discovery",
        input={"city": city, "country_code": cc},
    ) as rec:
        coverage = await ensure_local_coverage(city=city, country_code=cc)
        rec.output = coverage
        disc = coverage.get("discovery") or {}
        if not coverage.get("triggered"):
            rec.status = "empty"
        elif not (disc.get("inserted") or disc.get("updated")):
            rec.status = "empty"


async def _background_ensure_local_coverage(parsed: Any) -> None:
    """Persist discovered indies **after** the HTTP response finishes.

    FastAPI queues this coroutine via :class:`~fastapi.BackgroundTasks`, so UI
    latency is no longer gated on Tavily + OpenAI discovery. Each call obtains
    its own SQLAlchemy session(s) inside :func:`~app.db.store_loader.ensure_local_coverage`
    — no request-scoped connection is captured or reopened here.

    We intentionally skip :func:`stage_timer` / pipeline context tracing: route
    contextvars reset once the handler returns and would otherwise synthesize a
    throwaway `PipelineContext` for every background run.
    """
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip()
    if not city or not cc:
        return

    try:
        coverage = await ensure_local_coverage(city=city, country_code=cc)
    except Exception:
        logger.exception(
            "background_store_discovery_failed",
            extra={"stage": "store_discovery", "city": city, "country_code": cc},
        )
        return

    disc = coverage.get("discovery") or {}
    logger.info(
        "background_store_discovery_finished",
        extra={
            "stage": "store_discovery",
            "city": city,
            "country_code": cc,
            "triggered": coverage.get("triggered"),
            "inserted": disc.get("inserted"),
            "updated": disc.get("updated"),
        },
    )


async def _kick_off_local_shop_discovery(
    *,
    parsed: Any,
    background_tasks: BackgroundTasks | None,
) -> None:
    """Inline ``store_discovery`` for tests/callers **or** background for HTTP."""
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip()
    if not city or not cc:
        return
    if background_tasks is not None:
        background_tasks.add_task(_background_ensure_local_coverage, parsed)
        logger.info(
            "store_discovery_background_scheduled",
            extra={"stage": "store_discovery", "city": city, "country_code": cc},
        )
        return

    await _stage_ensure_local_coverage(parsed)


async def _load_known_shop_hosts() -> frozenset[str]:
    """Active host set from ``whitelist_stores`` (curated + discovered indies).

    Returns an empty set if the DB is unavailable. The prefilter treats this as
    a positive signal — every host listed here passes the noise gate.
    """
    try:
        stores = await load_active_stores()
    except Exception:
        logger.exception(
            "load_active_stores_failed",
            extra={"stage": "stores"},
        )
        return frozenset()
    hosts: set[str] = set()
    for s in stores:
        dom = canonical_store_domain(getattr(s, "domain", "") or "")
        if dom:
            hosts.add(dom)
    return frozenset(hosts)


def _host_of(url: str) -> str | None:
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except Exception:
        return None
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":", 1)[0] or None


def _listing_to_api_row(listing: Listing) -> ListingResult:
    """Project a verified :class:`Listing` to the API-facing :class:`ListingResult`.

    A minimal, deterministic projection — the consolidated pipeline does NOT
    use the legacy composite scorer (which depends on the multi-tier loop),
    so the API ``score`` reflects pure extractor confidence:
    1.0 when in stock, 0.85 otherwise.
    """
    title = (listing.title or "").strip()
    if len(title) < 5:
        title = f"{title} · shop"

    price_str: str | None = None
    try:
        price_val = float(listing.price or 0.0)
        if price_val > 0:
            currency = (listing.currency or "EUR").strip().upper() or "EUR"
            price_str = f"{price_val:.2f} {currency}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        price_str = None

    host = _host_of(listing.url) or (listing.store or None)
    score = 1.0 if listing.in_stock else 0.85

    return ListingResult(
        url=listing.url,
        title=title,
        score=score,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=host,
        artist_match=1.0,
        album_match=1.0,
        match_reason="consolidated_pipeline",
    )


def _dedupe_listings_by_host(listings: list[Listing]) -> list[Listing]:
    """One listing per registrable host. In-stock rows win; among ties the first
    occurrence (preserving extractor order) wins.

    The Python prefilter intentionally lets up to 2 deep links per host through
    so the LLM has redundancy. After extraction we hard-collapse to 1 per host
    so the user-facing result list always shows variety across shops.
    """
    best: dict[str, Listing] = {}
    for lst in listings:
        host = _host_of(lst.url)
        if not host:
            continue
        prev = best.get(host)
        if prev is None:
            best[host] = lst
            continue
        if lst.in_stock and not prev.in_stock:
            best[host] = lst
    return list(best.values())


def _empty_response(query: str, parsed: Any | None, reason: str | None) -> dict[str, Any]:
    return {
        "query": query,
        "results": [],
        "parsed": parsed,
        "reason": reason,
    }


async def _persist_cache_payload(
    *,
    redis_key: str,
    pg_key: str,
    payload: dict[str, Any],
    redis_ttl_seconds: int,
    pg_ttl_seconds: int,
) -> None:
    """Write the response to BOTH Redis (hot read) and Postgres (operator audit).

    Failures on either tier are logged but never raised — the user already has
    the live response in hand by the time this runs.
    """
    try:
        await set_cached_search(redis_key, payload, ttl_seconds=redis_ttl_seconds)
    except Exception:
        logger.exception(
            "redis_cache_write_failed",
            extra={"stage": "redis_cache", "cache_key_head": redis_key[:64]},
        )
    try:
        await set_cached_search_payload(pg_key, payload, ttl_seconds=pg_ttl_seconds)
    except Exception:
        logger.exception(
            "postgres_cache_write_failed",
            extra={"stage": "search_cache", "cache_key_head": pg_key[:16]},
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_vinyl_search(
    query: str,
    *,
    background_tasks: BackgroundTasks | None = None,
) -> dict[str, Any]:
    """Entry point used by the FastAPI router.

    Wraps the inner implementation in the Tavily circuit-breaker scope so any
    burst of 432/433 throttling fails fast for the rest of the request.

    Pass ``BackgroundTasks`` from the route handler so Indie store discovery runs
    after the response returns; omit it for tests or CLI callers (discovery
    awaits inline inside the caller's event loop instead).
    """
    settings = get_settings()
    with tavily_circuit_breaker_scope(
        failure_threshold=settings.tavily_circuit_breaker_failure_threshold,
    ):
        return await _run_vinyl_search_inner(query, background_tasks=background_tasks)


async def _run_vinyl_search_inner(
    query: str,
    *,
    background_tasks: BackgroundTasks | None = None,
) -> dict[str, Any]:
    settings = get_settings()

    # ---- Stage 1: parse -------------------------------------------------
    parsed = await _stage_parse(query)

    # ---- Stage 2: resolve album anchor ---------------------------------
    album_title = await _stage_resolve_album_title(parsed)
    if album_title is None:
        return _empty_response(query, parsed, reason="album_unresolved")

    # ---- Stage 3: Redis cache lookup (7-day TTL short-circuit) ---------
    format_token = detect_format_token(parsed.original_query)
    redis_key = build_redis_search_key(
        format_token=format_token,
        artist=parsed.artist,
        album=album_title,
        country_code=parsed.country_code,
    )
    pg_key = build_search_cache_key(
        user_query=query,
        artist=parsed.artist,
        album_title=album_title,
        debug=settings.debug,
    )

    with stage_timer(
        "redis_cache_lookup",
        input={"cache_key": redis_key},
    ) as rec:
        cached = await get_cached_search(redis_key)
        rec.output = {"hit": cached is not None}
        rec.status = "success" if cached is not None else "empty"

    if cached is not None:
        try:
            cached_rows = cached.get("results") or []
            hydrated_rows = [ListingResult.model_validate(row) for row in cached_rows]
        except Exception:
            logger.warning(
                "redis_cache_payload_invalid_falling_back_to_live",
                extra={"stage": "redis_cache", "cache_key_head": redis_key[:64]},
            )
        else:
            logger.info(
                "redis_cache_hit",
                extra={
                    "stage": "redis_cache",
                    "cache_key_head": redis_key[:64],
                    "result_count": len(hydrated_rows),
                },
            )
            return {
                "query": query,
                "results": hydrated_rows,
                "parsed": parsed,
                "reason": None,
            }

    # ---- Stage 4: indie-shop top-up — background in HTTP, inline in tests -----
    await _kick_off_local_shop_discovery(parsed=parsed, background_tasks=background_tasks)

    # ---- Stage 5: load active shop hosts (positive prefilter signal) ---
    known_shop_hosts = await _load_known_shop_hosts()
    logger.info(
        "known_shop_hosts_loaded",
        extra={"stage": "stores", "host_count": len(known_shop_hosts)},
    )

    # ---- Stage 6: ONE consolidated Tavily call (bounded by Settings, advanced) -----
    raw_results = await run_consolidated_tavily_search(
        artist=parsed.artist,
        album=album_title,
        format_token=format_token,
        country_code=parsed.country_code,
        max_results=settings.tavily_single_call_max_results,
    )
    if not raw_results:
        return _empty_response(query, parsed, reason=None)

    # ---- Stage 7: Python pre-filter (blacklist + dedupe per host) ------
    with stage_timer(
        "prefilter",
        input={"raw_count": len(raw_results)},
    ) as rec:
        candidates, prefilter_diag = prefilter_tavily_results(
            raw_results,
            max_candidates=settings.pipeline_prefilter_max_candidates,
            max_per_host=settings.pipeline_prefilter_max_per_host,
            known_shop_hosts=known_shop_hosts,
        )
        rec.output = prefilter_diag
        rec.status = "success" if candidates else "empty"

    if not candidates:
        return _empty_response(query, parsed, reason=None)

    # ---- Stage 8: LLM extract + verify (gpt-4o-mini) -------------------
    # ``allowed_domains`` for ``extract_listings`` = the hosts that survived the
    # Python prefilter. We derive it from the candidates themselves so every
    # candidate (curated, discovered indie, or PDP-shaped unknown) passes the
    # extractor's host gate. The dynamic whitelist is loaded earlier and acts
    # as a *positive* signal inside the prefilter — extract_listings doesn't
    # need to re-check it.
    allowed_domains = {c["host"] for c in candidates if c.get("host")}
    with stage_timer("extractor", input={"candidate_count": len(candidates)}) as rec:
        report = await extract_listings(
            raw_results=[
                {
                    "url": c["url"],
                    "title": c["title"],
                    "content": c["content"],
                    "score": c.get("score", 0.0),
                }
                for c in candidates
            ],
            artist=parsed.artist,
            album=album_title,
            allowed_domains=allowed_domains,
        )
        rec.output = {
            "final_count": len(report.listings),
            "diagnostic": report.diagnostic,
        }
        rec.status = "success" if report.listings else "empty"

    if not report.listings:
        return _empty_response(query, parsed, reason=None)

    # ---- Stage 9: per-host dedupe + API projection ---------------------
    unique_listings = _dedupe_listings_by_host(report.listings)
    api_rows = [_listing_to_api_row(lst) for lst in unique_listings]
    api_rows.sort(key=lambda r: r.score, reverse=True)
    api_rows = api_rows[: settings.pipeline_max_results]
    logger.info(
        "results_dedupe_summary",
        extra={
            "stage": "pipeline",
            "extractor_listings": len(report.listings),
            "deduped_listings": len(unique_listings),
            "api_rows": len(api_rows),
        },
    )

    # ---- Stage 10: persist to BOTH Redis (hot) AND Postgres (audit) ----
    cache_payload = {
        "query": query,
        "results": [row.model_dump(mode="json") for row in api_rows],
        "cache_meta": {
            "format": format_token,
            "artist": (parsed.artist or "").strip() or None,
            "album": album_title,
            "country_code": parsed.country_code,
            "resolved_city": getattr(parsed, "resolved_city", None),
            "known_shop_host_count": len(known_shop_hosts),
        },
    }
    await _persist_cache_payload(
        redis_key=redis_key,
        pg_key=pg_key,
        payload=cache_payload,
        redis_ttl_seconds=settings.redis_search_cache_ttl_seconds,
        pg_ttl_seconds=settings.redis_search_cache_ttl_seconds,
    )

    return {
        "query": query,
        "results": api_rows,
        "parsed": parsed,
        "reason": None,
    }
