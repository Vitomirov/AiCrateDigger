"""Consolidated vinyl-search pipeline (Redis cache + single Tavily call).

This module is the production hot path for the ``/search`` endpoint. It is a
deliberately small, linear function so latency / cost behaviour stays obvious
in production logs:

1. Parse the user query (one ``gpt-4o-mini`` call → :class:`ParsedQuery`,
   including ordinal ``resolved_album`` when applicable).
2. Resolve the album anchor (``resolved_album`` or literal ``album``).
3. Build the deterministic Redis key and short-circuit on a cache **hit**:
   zero Tavily credits, zero further LLM tokens for 7 days.
4. **Local-shop top-up** (city queries only, **non-blocking**):
   :func:`ensure_local_coverage` counts ``local_shop`` rows for the resolved
   city; when below ``LOCAL_COVERAGE_THRESHOLD`` it marks discovery as
   **scheduled** and the pipeline schedules :func:`discover_new_stores` via
   ``BackgroundTasks`` (or ``asyncio.create_task`` when tasks are omitted) so
   Tavily + LLM discovery does not block the user's search. Newly upserted
   indie domains benefit the **next** request.
5. Load the active store hosts (curated + discovered) and pass them to the
   pre-filter as a **positive whitelist signal**.
6. Issue ONE consolidated Tavily call (``max_results`` from
   ``Settings.tavily_single_call_max_results``, default ``10``;
   ``search_depth="advanced"``) returning a European candidate pool for one
   Tavily credit.
6.5. **Opportunistic store discovery** (gated by
   ``Settings.pipeline_opportunistic_store_discovery_enabled``, **scheduled**):
   when Tavily surfaces unknown-host shop-shaped snippets for the resolved
   city/country, we schedule :func:`store_discovery.discover_stores_from_snippets`
   in the background so LLM verification + DB upsert do not add latency.
   Verified hosts enter ``whitelist_stores`` for **future** searches.
7. Python pre-filter — blacklist noise (YouTube, news portals, …), require
   PDP-shaped URLs from unknown hosts, dedupe per host. Cap to ~10 candidates.
8. LLM extract + verification via :func:`extract_listings`.
9. Project to :class:`ListingResult`, **deduplicate by host** so one shop never
   surfaces twice in the visible result list.
10. Persist the response to BOTH Redis (7-day TTL, fast hot-path read) AND
    the Postgres ``search_response_cache`` table (operator visibility via
    DBeaver / SQL).

Production note: Store discovery is deferred so the consolidated search path
stays fast. Pass ``BackgroundTasks`` from the route handler so discovery runs
after the HTTP response; tests/CLI callers without tasks still schedule work
via ``asyncio.create_task`` (same process, non-blocking for the pipeline).

Fall-back behaviour
-------------------
* No ``REDIS_URL`` / unreachable Redis → Redis layer is a no-op; pipeline runs
  live and writes Postgres only.
* Tavily circuit breaker tripped → empty result list, fast return, no cache
  write (so a transient outage does not poison 7 days of cache).
* No OpenAI key / extractor failure → empty result list, no cache write.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks

from app.domains.engine.extraction import extract_listings
from app.domains.query_parser.parse_user_query import parse_user_query
from app.core.config import get_settings
from app.core.db.cache import get_cached_search_payload, set_cached_search_payload
from app.core.db.redis_cache import get_cached_search, set_cached_search
from app.core.db.search_cache_key import build_pipeline_search_cache_keys
from app.core.db.store_loader import ensure_local_coverage, load_active_stores
from app.domains.engine.listing_schema import Listing
from app.domains.search_pipeline.models.result import ListingResult
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.policies.format_detect import detect_format_token
from app.domains.engine.policies.store_domain import canonical_store_domain
from app.domains.engine.search import tavily_circuit_breaker_scope
from app.domains.engine.search.prefilter import (
    _host_in_whitelist,
    _is_blacklisted,
    _registrable_host,
    prefilter_tavily_results,
)
from app.domains.engine.search.single_call import run_consolidated_tavily_search

logger = logging.getLogger(__name__)


async def _run_background_store_discovery(
    work: Callable[[], Awaitable[object]],
    *,
    label: str,
) -> None:
    """Execute deferred discovery; log failures without affecting the HTTP response."""
    try:
        await work()
    except Exception:  # noqa: BLE001 — background work must never surface to the client.
        logger.exception(
            "background_store_discovery_failed",
            extra={"stage": "store_discovery", "discovery_kind": label},
        )


def _schedule_background_store_discovery(
    background_tasks: BackgroundTasks | None,
    work: Callable[[], Awaitable[object]],
    *,
    label: str,
) -> None:
    """Run discovery after the response (BackgroundTasks) or concurrently (create_task)."""

    async def _runner() -> None:
        await _run_background_store_discovery(work, label=label)

    if background_tasks is not None:
        background_tasks.add_task(_runner)
        return
    asyncio.create_task(_runner())


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
    """Pick the album search anchor from parser output (fail closed when missing)."""
    with stage_timer("album_resolve") as rec:
        title = (getattr(parsed, "effective_album", None) or "").strip() or None
        rec.output = {"album_title": title}
        rec.status = "success" if title else "empty"
    return title


async def _stage_ensure_local_coverage(
    parsed: Any,
    *,
    background_tasks: BackgroundTasks | None,
) -> dict[str, object]:
    """Evaluate indie coverage and schedule discovery when the city is thin.

    Runs only when the parser resolved a specific city + country. Internally
    idempotent: when ``count_local_shops_in_city(city, cc) >= LOCAL_COVERAGE_THRESHOLD``
    no Tavily / OpenAI traffic is generated — it is just a single ``COUNT(*)``
    on ``whitelist_stores``. When coverage is insufficient, discovery is
    scheduled in the background so the search hot path is not blocked.

    Returns the coverage summary dict (empty when inputs are missing).
    """
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip()
    if not city or not cc:
        return {}
    with stage_timer(
        "store_discovery",
        input={"city": city, "country_code": cc},
    ) as rec:
        try:
            coverage = await ensure_local_coverage(city=city, country_code=cc)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never fail the user's search.
            logger.exception(
                "store_discovery_failed",
                extra={
                    "stage": "store_discovery",
                    "city": city,
                    "country_code": cc,
                },
            )
            rec.status = "fail"
            rec.error = str(exc)[:240]
            return {"triggered": False, "error": str(exc)[:240]}
        if coverage.get("scheduled"):
            from app.domains.engine.search.store_discovery import discover_new_stores

            _schedule_background_store_discovery(
                background_tasks,
                lambda: discover_new_stores(city=city, country_code=cc),
                label="primary",
            )
        rec.output = coverage
        if not coverage.get("triggered"):
            rec.status = "empty"
        elif coverage.get("scheduled"):
            rec.status = "success"
        return coverage


def _primary_discovery_should_skip_opportunistic(
    primary_discovery_summary: dict[str, object] | None,
) -> bool:
    """True when inline store discovery already ran for this request."""
    if not primary_discovery_summary:
        return False
    if primary_discovery_summary.get("triggered"):
        return True
    disc = primary_discovery_summary.get("discovery")
    if isinstance(disc, dict) and int(disc.get("inserted") or 0) > 0:
        return True
    return False


def _tavily_city_token(parsed: Any) -> str | None:
    """Resolved city for Tavily query injection when geo is city-level."""
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    if city:
        return city
    if getattr(parsed, "geo_granularity", None) == "city":
        return (getattr(parsed, "location", None) or "").strip() or None
    return None


def _select_unknown_host_snippets_for_discovery(
    raw_results: list[dict[str, Any]],
    *,
    known_shop_hosts: frozenset[str],
) -> list[dict[str, str]]:
    """Filter main-Tavily results down to *plausible-shop* unknown-host snippets.

    Excludes hosts that are:
      * blacklisted (YouTube, Discogs, news, social, …),
      * already in the active whitelist (curated + previously discovered).

    Keeps one snippet per unique host — the discovery LLM needs a clear shop
    name signal, not 5 deep links from the same domain.
    """
    chosen: dict[str, dict[str, str]] = {}
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        host = _registrable_host(url)
        if host is None or _is_blacklisted(host):
            continue
        if _host_in_whitelist(host, known_shop_hosts):
            continue
        if host in chosen:
            continue
        chosen[host] = {
            "title": str(row.get("title") or "").strip()[:240],
            "url": url,
            "content": str(row.get("content") or "").strip()[:1500],
        }
    return list(chosen.values())


def _schedule_opportunistic_store_discovery(
    *,
    background_tasks: BackgroundTasks | None,
    parsed: Any,
    raw_results: list[dict[str, Any]],
    known_shop_hosts: frozenset[str],
    primary_discovery_summary: dict[str, object] | None = None,
) -> None:
    """Schedule LLM verification of unknown-host snippets from the main Tavily call.

    Upserts run in the background so the prefilter / extractor stages are not
    blocked. Verified domains benefit subsequent searches via ``whitelist_stores``.
    """
    settings = get_settings()
    if not settings.pipeline_opportunistic_store_discovery_enabled:
        return

    if _primary_discovery_should_skip_opportunistic(primary_discovery_summary):
        city = (getattr(parsed, "resolved_city", None) or "").strip()
        cc = (getattr(parsed, "country_code", None) or "").strip()
        logger.info(
            "opportunistic_store_discovery_skipped_primary_discovery_triggered",
            extra={
                "stage": "opportunistic_store_discovery",
                "city": city,
                "country_code": cc,
                "skipped_reason": "skipped_primary_discovery_triggered",
                "primary_triggered": bool(
                    (primary_discovery_summary or {}).get("triggered")
                ),
            },
        )
        with stage_timer(
            "opportunistic_store_discovery",
            input={"skipped_reason": "skipped_primary_discovery_triggered"},
        ) as rec:
            rec.status = "empty"
            rec.output = {"skipped_reason": "skipped_primary_discovery_triggered"}
        return

    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip()
    if not city or not cc:
        return

    snippets = _select_unknown_host_snippets_for_discovery(
        raw_results,
        known_shop_hosts=known_shop_hosts,
    )
    min_required = settings.pipeline_opportunistic_discovery_min_unknown_hosts
    if len(snippets) < min_required:
        logger.info(
            "opportunistic_store_discovery_skipped_thin_signal",
            extra={
                "stage": "store_discovery",
                "city": city,
                "country_code": cc,
                "unknown_host_count": len(snippets),
                "min_required": min_required,
            },
        )
        return

    from app.domains.engine.search.store_discovery import discover_stores_from_snippets

    with stage_timer(
        "opportunistic_store_discovery",
        input={
            "city": city,
            "country_code": cc,
            "unknown_host_count": len(snippets),
            "scheduled": True,
        },
    ) as rec:
        _schedule_background_store_discovery(
            background_tasks,
            lambda: discover_stores_from_snippets(
                city=city,
                country_code=cc,
                snippets=snippets,
            ),
            label="opportunistic",
        )
        rec.status = "success"
        rec.output = {"scheduled": True, "unknown_host_count": len(snippets)}


async def _load_active_stores_catalogue() -> tuple[StoreEntry, ...]:
    """Single DB read of the active whitelist for one pipeline request.

    Robustness contract: an unexpected failure must NEVER leave the pipeline
    without a usable catalogue — we degrade to the in-code seed
    (:func:`get_active_stores`) so curated indies are not demoted to unknown hosts.
    """
    try:
        return await load_active_stores()
    except Exception:
        logger.exception(
            "load_active_stores_failed_falling_back_to_code_seed",
            extra={"stage": "stores"},
        )
        from app.domains.engine.policies.eu_stores import get_active_stores

        return get_active_stores()


def _known_shop_hosts_from_stores(stores: tuple[StoreEntry, ...]) -> frozenset[str]:
    """All active hosts — positive prefilter signal for curated + discovered shops."""
    hosts: set[str] = set()
    for s in stores:
        dom = canonical_store_domain(getattr(s, "domain", "") or "")
        if dom:
            hosts.add(dom)
    return frozenset(hosts)


def _trusted_local_shop_hosts_from_stores(
    stores: tuple[StoreEntry, ...],
    parsed: Any,
) -> frozenset[str]:
    """City-matched ``local_shop`` domains from an already-loaded catalogue."""
    city = (getattr(parsed, "resolved_city", None) or "").strip()
    cc = (getattr(parsed, "country_code", None) or "").strip().upper()
    if not cc:
        return frozenset()
    from app.domains.engine.policies.geo_proximity import cities_match

    hosts: set[str] = set()
    for s in stores:
        if getattr(s, "store_type", None) != "local_shop" or not s.domain:
            continue
        if (s.country_code or "").strip().upper() != cc:
            continue
        if city:
            store_city = (getattr(s, "city", None) or "").strip()
            if store_city and not cities_match(city, store_city):
                continue
        dom = canonical_store_domain(s.domain)
        if dom:
            hosts.add(dom)
    return frozenset(hosts)


async def _load_pipeline_shop_hosts(parsed: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(known_shop_hosts, trusted_local_shop_hosts)`` from one catalogue load."""
    stores = await _load_active_stores_catalogue()
    return (
        _known_shop_hosts_from_stores(stores),
        _trusted_local_shop_hosts_from_stores(stores, parsed),
    )


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

    Pass ``BackgroundTasks`` from the route handler so store discovery runs after
    the HTTP response; omit it for tests or CLI callers (discovery is still
    scheduled via ``asyncio.create_task`` and does not block the pipeline).
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
    redis_key, pg_key = build_pipeline_search_cache_keys(
        format_token=format_token,
        artist=parsed.artist,
        album=album_title,
        country_code=parsed.country_code,
        resolved_city=getattr(parsed, "resolved_city", None),
        geo_granularity=getattr(parsed, "geo_granularity", None),
    )

    with stage_timer(
        "redis_cache_lookup",
        input={"cache_key": redis_key, "pg_cache_key_head": pg_key[:16]},
    ) as rec:
        cached = await get_cached_search(redis_key)
        cache_source = "redis" if cached is not None else None
        if cached is None:
            cached = await get_cached_search_payload(pg_key)
            if cached is not None:
                cache_source = "postgres"
        rec.output = {"hit": cached is not None, "source": cache_source}
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
                "search_cache_hit",
                extra={
                    "stage": "search_cache",
                    "cache_key_head": redis_key[:64],
                    "result_count": len(hydrated_rows),
                    "source": cache_source or "redis",
                },
            )
            return {
                "query": query,
                "results": hydrated_rows,
                "parsed": parsed,
                "reason": None,
            }

    # ---- Stage 4: indie-shop coverage check + deferred discovery --------
    primary_discovery_summary = await _stage_ensure_local_coverage(
        parsed,
        background_tasks=background_tasks,
    )

    # ---- Stage 5: load active shop hosts (positive prefilter signal) ---
    known_shop_hosts, trusted_local_shop_hosts = await _load_pipeline_shop_hosts(parsed)
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
        resolved_city=_tavily_city_token(parsed),
        max_results=settings.tavily_single_call_max_results,
    )
    if not raw_results:
        return _empty_response(query, parsed, reason=None)

    # ---- Stage 6.5: opportunistic store discovery (scheduled) ------------
    _schedule_opportunistic_store_discovery(
        background_tasks=background_tasks,
        parsed=parsed,
        raw_results=raw_results,
        known_shop_hosts=known_shop_hosts,
        primary_discovery_summary=primary_discovery_summary,
    )

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
            trusted_local_shop_hosts=trusted_local_shop_hosts,
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
