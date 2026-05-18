"""Cross-tier aggregate merge (tier-narrowness aware)."""

from __future__ import annotations

from typing import Any

from app.pipeline.tier_runner.context import TierLoopState
from app.policies.geo_scope import TIER_NARROWNESS, Tier
from app.services.tavily_service import normalize_url


def merge_into_aggregate(
    state: TierLoopState,
    tier: Tier,
    accepted: list[Any],
) -> list[str]:
    """Merge accepted listings into the cross-tier aggregate.

    A previously-seen URL is replaced only when this tier is *narrower* than
    the tier that first introduced it (``TIER_NARROWNESS``).
    """
    accepted_this_tier_urls: list[str] = []
    for lst in accepted:
        k = normalize_url(str(lst.url))
        prev_tier = state.listing_tier_map.get(k)
        if prev_tier is None or TIER_NARROWNESS[tier] < TIER_NARROWNESS[prev_tier]:
            state.aggregated[k] = lst
            state.listing_tier_map[k] = tier
            accepted_this_tier_urls.append(k)
    return accepted_this_tier_urls
