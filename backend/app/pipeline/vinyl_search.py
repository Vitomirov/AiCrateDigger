"""Deterministic vinyl-search pipeline orchestrator.

Owns the full request lifecycle as a small, readable sequence of stages.
The bulk of the per-stage logic lives in sibling modules of this package:

* :mod:`app.pipeline.tier_runner` — per-tier (city → … → global) iteration
* :mod:`app.pipeline.finalize`     — post-loop sort / dedupe / cap / scoring
* :mod:`app.pipeline.listing_dedupe`, :mod:`app.pipeline.local_first`,
  :mod:`app.pipeline.store_selection`, :mod:`app.pipeline.stop_floors`,
  :mod:`app.pipeline.constants`, :mod:`app.pipeline.api_mapper`

Each stage is wrapped in ``stage_timer`` for structured tracing. NO business
logic lives in this module — every transformation lives in the called helper.
No fallbacks, no error swallowing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.db.cache import (
    build_search_cache_key,
    get_cached_search_payload,
    hydrate_cached_pipeline_dict,
    set_cached_search_payload,
)
from app.db.store_loader import ensure_local_coverage, load_active_stores
from app.agents.parser.parse_user_query import parse_user_query
from app.pipeline.api_mapper import listing_to_api_row
from app.pipeline.finalize import (
    compute_local_shop_signals,
    emit_ranking_trace,
    emit_widening_summary_trace,
    score_final_listings,
    sort_dedupe_and_cap_listings,
)
from app.pipeline.tier_runner import (
    TierContext,
    TierLoopState,
    process_tier,
)
from app.pipeline_context import stage_timer
from app.policies.geo_scope import (
    Tier,
    geo_intent_from_parsed,
    normalized_geo_from_parsed,
    tier_fallback_order,
)
from app.policies.physical_local import (
    curated_city_local_shop_domains,
    should_prioritize_physical_local_shops,
)
from app.policies.search_dsl import plan_tavily_query_strings
from app.services.discogs_service import resolve_album_by_index
from app.services.tavily import tavily_circuit_breaker_scope
from app.validators.listings import normalize_whitelist_domain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — parse + Discogs resolution
# ---------------------------------------------------------------------------


async def _stage_parse(query: str) -> Any:
    """Parse the user query inside the ``parse`` stage timer."""
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


# ---------------------------------------------------------------------------
# Stage 2 — geo + local-coverage discovery + active store load
# ---------------------------------------------------------------------------


def _stage_geo_norm(parsed: Any) -> tuple[Any, Any]:
    """Emit the ``geo_norm`` trace and return ``(geo, norm)``."""
    geo = geo_intent_from_parsed(parsed)
    norm = normalized_geo_from_parsed(parsed)
    with stage_timer("geo_norm") as rec:
        rec.output = {
            "raw_location": norm.raw_location,
            "resolved_country": norm.resolved_country,
            "resolved_city": norm.resolved_city,
            "granularity": norm.granularity,
            "confidence": round(float(norm.confidence), 3),
            "search_scope": geo.search_scope,
            "region": geo.region,
        }
    return geo, norm


async def _stage_ensure_local_coverage(norm: Any) -> None:
    """LOCAL-FIRST STRIKE phase 2: top up indie ``local_shop`` rows on demand.

    If the resolved city has fewer than the configured number of indie
    ``local_shop`` rows, discover more via Tavily + LLM before we load the
    active store catalogue.
    """
    if not (norm.resolved_city and norm.resolved_country):
        return
    with stage_timer(
        "store_discovery",
        input={
            "city": norm.resolved_city,
            "country_code": norm.resolved_country,
        },
    ) as rec:
        coverage = await ensure_local_coverage(
            city=norm.resolved_city,
            country_code=norm.resolved_country,
        )
        rec.output = coverage
        disc = coverage.get("discovery") or {}
        if not coverage.get("triggered"):
            rec.status = "empty"
        elif not (disc.get("inserted") or disc.get("updated")):
            rec.status = "empty"


async def _stage_load_stores() -> tuple[Any, ...]:
    with stage_timer("stores"):
        return await load_active_stores()


# ---------------------------------------------------------------------------
# Stage 3 — Tavily query planning
# ---------------------------------------------------------------------------


def _plan_queries(
    parsed: Any, album_title: str, norm: Any
) -> tuple[str, Any]:
    return plan_tavily_query_strings(
        parsed.artist,
        album_title,
        country_code_for_variants=norm.resolved_country,
        resolved_city=norm.resolved_city,
    )


# ---------------------------------------------------------------------------
# Stage 4 — geo widening loop
# ---------------------------------------------------------------------------


async def _run_tier_loop(
    ctx: TierContext,
    state: TierLoopState,
    tier_queue: list[Tier],
) -> bool:
    """Drive the geo widening loop, returning whether the global fallback ran."""
    tier_ix = 0
    executed_global_fallback = False
    while True:
        if tier_ix < len(tier_queue):
            tier = tier_queue[tier_ix]
            tier_ix += 1
            widening_reason = "planned_tier"
        elif (
            not state.aggregated
            and not executed_global_fallback
            and "global" not in tier_queue
        ):
            tier = "global"
            executed_global_fallback = True
            widening_reason = "fallback_no_results"
        else:
            break

        early_stop = await process_tier(
            tier=tier,
            widening_reason=widening_reason,
            ctx=ctx,
            state=state,
        )
        if early_stop:
            break
    return executed_global_fallback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_vinyl_search(query: str) -> dict[str, Any]:
    settings = get_settings()
    with tavily_circuit_breaker_scope(
        failure_threshold=settings.tavily_circuit_breaker_failure_threshold,
    ):
        return await _run_vinyl_search_inner(query, settings=settings)


async def _run_vinyl_search_inner(query: str, *, settings: Any) -> dict[str, Any]:
    debug_enabled = settings.debug

    parsed = await _stage_parse(query)
    album_title = await _stage_resolve_album_title(parsed)

    # No searchable release anchor — skip Tavily entirely and surface a
    # machine-readable reason for UI / API clients.
    if album_title is None:
        return {
            "query": query,
            "results": [],
            "parsed": parsed,
            "reason": "album_unresolved",
        }

    cache_key_value = build_search_cache_key(
        user_query=query,
        artist=parsed.artist,
        album_title=album_title,
        debug=debug_enabled,
    )

    cached_payload = await get_cached_search_payload(cache_key_value)
    if cached_payload is not None:
        hydrated = hydrate_cached_pipeline_dict(cached_payload)
        # `parsed` is intentionally re-attached from the live parse rather than
        # served from the cache: the cache key already incorporates the
        # parser-derived identity (artist + album_title), so the freshly-parsed
        # object is always equivalent AND insulates us from schema drift if
        # `ParsedQuery` ever gains/loses fields.
        hydrated["parsed"] = parsed
        hydrated["reason"] = None
        return hydrated

    geo, norm = _stage_geo_norm(parsed)
    await _stage_ensure_local_coverage(norm)
    stores = await _stage_load_stores()

    prioritize_physical_locals = should_prioritize_physical_local_shops(geo, norm)
    curated_city_local_domains = (
        curated_city_local_shop_domains(stores, norm)
        if prioritize_physical_locals
        else frozenset()
    )

    core_query, tavily_relaxation_queries = _plan_queries(parsed, album_title, norm)

    all_allowed = frozenset(
        normalize_whitelist_domain(s.domain) for s in stores if getattr(s, "domain", None)
    )

    tier_queue = list(tier_fallback_order(geo, norm))
    ctx = TierContext(
        parsed=parsed,
        album_title=album_title,
        geo=geo,
        norm=norm,
        stores=stores,
        settings=settings,
        core_query=core_query,
        tavily_relaxation_queries=tavily_relaxation_queries,
        curated_city_local_domains=curated_city_local_domains,
        prioritize_physical_locals=prioritize_physical_locals,
        all_allowed=all_allowed,
    )
    state = TierLoopState()

    executed_global_fallback = await _run_tier_loop(ctx, state, tier_queue)

    # ---- Post-loop finalisation -----------------------------------------
    list_out = list(state.aggregated.values())

    (
        generic_any_local_shop_present,
        primary_target_local_shop_present,
        pool_for_giant_penalty,
    ) = compute_local_shop_signals(
        list_out,
        store_lookup=state.store_lookup,
        norm=norm,
        prioritize_physical_locals=prioritize_physical_locals,
    )

    deduped_listings, head, domain_drops = sort_dedupe_and_cap_listings(
        list_out,
        store_lookup=state.store_lookup,
        norm=norm,
        listing_tier_map=state.listing_tier_map,
        album_title=album_title or "",
        artist=parsed.artist,
        album_match_by_url=state.album_match_by_url,
        pool_for_giant_penalty=pool_for_giant_penalty,
        generic_any_local_shop_present=generic_any_local_shop_present,
        primary_target_local_shop_present=primary_target_local_shop_present,
        prioritize_physical_locals=prioritize_physical_locals,
        max_results=settings.pipeline_max_results,
    )

    breakdowns = score_final_listings(
        head,
        store_lookup=state.store_lookup,
        listing_tier_map=state.listing_tier_map,
        last_tier=state.last_tier,
        album_match_by_url=state.album_match_by_url,
        norm=norm,
        album_title=album_title or "",
        artist=parsed.artist,
        pool_for_giant_penalty=pool_for_giant_penalty,
    )

    emit_ranking_trace(head, breakdowns)
    emit_widening_summary_trace(
        tier_queue=tier_queue,
        tiers_attempted=state.tiers_attempted,
        executed_global_fallback=executed_global_fallback,
        aggregated=state.aggregated,
        deduped_listings=deduped_listings,
        domain_drops=domain_drops,
        head=head,
        generic_any_local_shop_present=generic_any_local_shop_present,
        primary_target_local_shop_present=primary_target_local_shop_present,
        prioritize_physical_locals=prioritize_physical_locals,
        pool_for_giant_penalty=pool_for_giant_penalty,
        verifier_summary=state.verifier_summary,
        tier_traces=state.tier_traces,
    )

    api_rows = [
        listing_to_api_row(listing, breakdown=breakdown)
        for listing, breakdown in zip(head, breakdowns, strict=True)
    ]

    # Cache payload deliberately excludes `parsed` — see the cache-hit branch
    # above for the rationale (live re-attach beats cached re-hydration).
    await set_cached_search_payload(
        cache_key_value,
        {"query": query, "results": api_rows},
        ttl_seconds=settings.search_cache_ttl_seconds,
    )

    return {
        "query": query,
        "results": api_rows,
        "parsed": parsed,
        "reason": None,
    }
