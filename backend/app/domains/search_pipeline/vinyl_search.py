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

Stage implementations live under :mod:`app.domains.search_pipeline.stages`.

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

import logging
from typing import Any

from fastapi import BackgroundTasks

from app.core.config import get_settings
from app.core.db.search_cache_key import build_pipeline_search_cache_keys
from app.domains.engine.extraction import extract_listings
from app.domains.engine.policies.format_detect import detect_format_token
from app.domains.engine.search import tavily_circuit_breaker_scope
from app.domains.engine.search.prefilter import prefilter_tavily_results
from app.domains.engine.search.single_call import run_consolidated_tavily_search
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.search_pipeline.search_intent import (
    cache_album_segment,
    empty_reason_for_unresolved,
    resolve_search_intent,
)
from app.domains.search_pipeline.stages import cache as cache_stage
from app.domains.search_pipeline.stages import discovery as discovery_stage
from app.domains.search_pipeline.stages import parse as parse_stage
from app.domains.search_pipeline.stages import results as results_stage
from app.domains.search_pipeline.stages import stores as stores_stage

logger = logging.getLogger(__name__)


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
    parsed = await parse_stage.stage_parse(query)

    # ---- Stage 2: resolve search intent + album anchor -----------------
    search_intent = resolve_search_intent(parsed)
    album_title = await parse_stage.stage_resolve_album_title(parsed, search_intent)
    if search_intent == "unresolved":
        return cache_stage.empty_response(
            query,
            parsed,
            reason=empty_reason_for_unresolved(parsed),
        )

    # ---- Stage 3: Redis cache lookup (7-day TTL short-circuit) ---------
    format_token = detect_format_token(parsed.original_query)
    redis_key, pg_key = build_pipeline_search_cache_keys(
        format_token=format_token,
        artist=parsed.artist,
        album=cache_album_segment(intent=search_intent, album=album_title),
        country_code=parsed.country_code,
        resolved_city=parsed.resolved_city,
        geo_granularity=parsed.geo_granularity,
    )

    cache_hit = await cache_stage.stage_cache_lookup(
        query=query,
        parsed=parsed,
        redis_key=redis_key,
        pg_key=pg_key,
    )
    if cache_hit is not None:
        return cache_hit

    # ---- Stage 4: indie-shop coverage check + deferred discovery --------
    primary_discovery_summary = await discovery_stage.stage_ensure_local_coverage(
        parsed,
        background_tasks=background_tasks,
    )

    # ---- Stage 5: load active shop hosts (positive prefilter signal) ---
    known_shop_hosts, trusted_local_shop_hosts = await stores_stage.load_pipeline_shop_hosts(
        parsed,
    )
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
        resolved_city=discovery_stage.tavily_city_token(parsed),
        max_results=settings.tavily_single_call_max_results,
    )
    if not raw_results:
        return cache_stage.empty_response(query, parsed, reason=None)

    # ---- Stage 6.5: opportunistic store discovery (scheduled) ------------
    discovery_stage.schedule_opportunistic_store_discovery(
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
        return cache_stage.empty_response(query, parsed, reason=None)

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
            search_intent=search_intent,
            allowed_domains=allowed_domains,
        )
        rec.output = {
            "final_count": len(report.listings),
            "diagnostic": report.diagnostic,
        }
        rec.status = "success" if report.listings else "empty"

    if not report.listings:
        return cache_stage.empty_response(query, parsed, reason=None)

    # ---- Stage 9: per-host dedupe + API projection ---------------------
    api_rows = results_stage.finalize_api_rows(
        report.listings,
        settings=settings,
        extractor_listing_count=len(report.listings),
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
            "resolved_city": parsed.resolved_city,
            "known_shop_host_count": len(known_shop_hosts),
        },
    }
    await cache_stage.persist_cache_payload(
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
