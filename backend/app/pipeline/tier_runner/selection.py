"""Store pool selection and skipped-tier tracing."""

from __future__ import annotations

import logging

from app.pipeline.constants import MIN_STORE_DOMAINS_CITY, MIN_STORE_DOMAINS_DEFAULT
from app.pipeline.store_selection import dedupe_store_entries_by_domain, store_type_distribution
from app.pipeline.tier_runner.context import TierContext, TierStoreSelection
from app.pipeline_context import stage_timer
from app.policies.geo_scope import Tier, cap_stores, filter_stores_for_tier, max_domains_for_tier, sort_stores_for_tier
from app.validators.listings import normalize_whitelist_domain

logger = logging.getLogger(__name__)


def select_tier_stores(
    ctx: TierContext,
    tier: Tier,
    *,
    editorial_discovery_blocked: set[str],
) -> TierStoreSelection:
    """Filter / sort / cap / dedupe the tier pool and derive Tavily inputs."""
    pool = filter_stores_for_tier(ctx.stores, ctx.geo, tier, norm=ctx.norm)
    sorted_pool = sort_stores_for_tier(pool, ctx.geo, tier, norm=ctx.norm)
    max_d = max_domains_for_tier(
        tier,
        local_max=ctx.settings.pipeline_geo_local_max_domains,
        regional_max=ctx.settings.pipeline_geo_regional_max_domains,
        global_max=ctx.settings.pipeline_geo_global_max_domains,
    )

    capped = cap_stores(sorted_pool, max_domains=max_d)
    capped = dedupe_store_entries_by_domain(capped)
    min_dom = MIN_STORE_DOMAINS_CITY if tier == "city" else MIN_STORE_DOMAINS_DEFAULT

    tier_lookup = {normalize_whitelist_domain(s.domain): s for s in capped}
    store_domains = [normalize_whitelist_domain(s.domain) for s in capped if s.domain]
    domains_for_tavily = [
        d
        for d in store_domains
        if d and normalize_whitelist_domain(d) not in editorial_discovery_blocked
    ]
    if not domains_for_tavily:
        domains_for_tavily = store_domains

    return TierStoreSelection(
        capped=capped,
        pool_size=len(pool),
        min_dom=min_dom,
        store_domains=store_domains,
        domains_for_tavily=domains_for_tavily,
        tier_lookup=tier_lookup,
    )


def record_skipped_tier_trace(
    *,
    tier: Tier,
    selection: TierStoreSelection,
    widening_reason: str,
    confidence: float,
) -> None:
    """Emit the structured ``geo_tier`` trace used when a tier is skipped."""
    logger.info(
        "tier_skipped_low_signal",
        extra={"tier": tier, "domains": len(selection.capped)},
    )
    with stage_timer("geo_tier", input={"tier": tier}) as rec:
        rec.status = "empty"
        rec.output = {
            "tier": tier,
            "selected_domains": [s.domain for s in selection.capped],
            "pool_size": selection.pool_size,
            "store_types": store_type_distribution(selection.capped),
            "min_domain_floor": selection.min_dom,
            "widening_reason": widening_reason,
            "early_stop": False,
            "skipped": True,
            "skip_reason": "below_min_domain_floor",
            "geo_confidence": round(float(confidence), 3),
        }
