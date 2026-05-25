"""Single-tier pipeline orchestration (one widening iteration)."""

from __future__ import annotations

import logging

from app.core.db.cache import set_cached_tavily_tier_payload
from app.domains.search_pipeline.stop_floors import effective_stop_floor
from app.domains.search_pipeline.tier_runner.aggregate import merge_into_aggregate
from app.domains.search_pipeline.tier_runner.context import TierContext, TierLoopState
from app.domains.search_pipeline.tier_runner.evaluator import (
    compute_early_stop,
    log_localized_early_exit_if_applicable,
)
from app.domains.search_pipeline.tier_runner.extract_validate import (
    run_extract_stage,
    stage_dedupe_listings,
    validate_listings_for_tier,
)
from app.domains.search_pipeline.tier_runner.selection import (
    record_skipped_tier_trace,
    select_tier_stores,
)
from app.domains.search_pipeline.tier_runner.tavily_fetch import run_tavily_and_fanout_merged
from app.domains.search_pipeline.tier_runner.trace import build_tier_trace
from app.domains.search_pipeline.tier_runner.verify_city import run_city_verify_stage
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.policies.geo_scope import Tier
from app.domains.engine.search import editorial_discovery_blocked_hosts_from_raw_results

logger = logging.getLogger(__name__)


async def process_tier(
    *,
    tier: Tier,
    widening_reason: str,
    ctx: TierContext,
    state: TierLoopState,
) -> bool:
    """Run one full tier iteration; return True iff widening should stop."""
    state.tiers_attempted.append(tier)

    selection = select_tier_stores(
        ctx, tier, editorial_discovery_blocked=state.editorial_discovery_blocked
    )
    if selection.is_below_floor:
        record_skipped_tier_trace(
            tier=tier,
            selection=selection,
            widening_reason=widening_reason,
            confidence=ctx.norm.confidence,
        )
        return False

    for dom, row in selection.tier_lookup.items():
        state.store_lookup[dom] = row

    raw_results, cache_hit, tkey, local_site_count, local_site_kept = (
        await run_tavily_and_fanout_merged(ctx, tier, selection)
    )

    if not cache_hit and raw_results and tkey:
        try:
            await set_cached_tavily_tier_payload(
                tkey,
                [r.model_dump() for r in raw_results],
                ttl_seconds=max(60, int(ctx.settings.tavily_intermediate_cache_ttl_seconds)),
            )
        except Exception:
            logger.warning(
                "tavily_tier_cache_write_failed",
                extra={"stage": "search_cache"},
                exc_info=True,
            )

    extract_report = await run_extract_stage(
        ctx, tier, raw_results, selection.domains_for_tavily
    )
    listings = extract_report.listings

    if extract_report.diagnostic.get("deterministic_miss"):
        state.editorial_discovery_blocked.update(
            editorial_discovery_blocked_hosts_from_raw_results(
                raw_results,
                deterministic_failed=True,
            ),
        )

    batch = stage_dedupe_listings(listings)
    accepted, rejected_reasons = validate_listings_for_tier(
        ctx=ctx,
        state=state,
        tier=tier,
        batch=batch,
        tier_lookup=selection.tier_lookup,
    )

    if tier == "city" and accepted:
        accepted = await run_city_verify_stage(
            ctx=ctx,
            state=state,
            tier=tier,
            accepted=accepted,
            tier_lookup=selection.tier_lookup,
            rejected_reasons=rejected_reasons,
        )

    accepted_this_tier_urls = merge_into_aggregate(state, tier, accepted)
    state.last_tier = tier

    need = effective_stop_floor(
        tier, settings=ctx.settings, confidence=ctx.norm.confidence
    )
    aggregated_count = len(state.aggregated)
    early_stop, early_reason = compute_early_stop(
        tier=tier,
        aggregated_validated_count=aggregated_count,
        stop_floor_need=need,
    )
    log_localized_early_exit_if_applicable(
        tier=tier,
        aggregated_validated_count=aggregated_count,
        reason=early_reason,
    )

    tier_trace = build_tier_trace(
        tier=tier,
        widening_reason=widening_reason,
        selection=selection,
        raw_results=raw_results,
        cache_hit=cache_hit,
        local_site_count=local_site_count,
        local_site_kept=local_site_kept,
        listings_out=len(listings),
        accepted=accepted,
        rejected_reasons=rejected_reasons,
        accepted_this_tier_urls=accepted_this_tier_urls,
        aggregated=state.aggregated,
        stop_floor=need,
        confidence=ctx.norm.confidence,
        early_stop=early_stop,
        early_stop_reason=early_reason,
    )
    state.tier_traces.append(tier_trace)

    with stage_timer("geo_tier", input={"tier": tier}) as rec:
        rec.output = tier_trace

    return early_stop
