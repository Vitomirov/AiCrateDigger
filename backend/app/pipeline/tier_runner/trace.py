"""Structured ``geo_tier`` trace payloads."""

from __future__ import annotations

from typing import Any

from app.models.search_query import SearchResult
from app.pipeline.store_selection import store_type_distribution
from app.pipeline.tier_runner.context import TierStoreSelection
from app.policies.geo_scope import Tier


def build_tier_trace(
    *,
    tier: Tier,
    widening_reason: str,
    selection: TierStoreSelection,
    raw_results: list[SearchResult],
    cache_hit: bool,
    local_site_count: int,
    local_site_kept: int,
    listings_out: int,
    accepted: list[Any],
    rejected_reasons: list[dict[str, str]],
    accepted_this_tier_urls: list[str],
    aggregated: dict[str, Any],
    stop_floor: int,
    confidence: float,
    early_stop: bool,
    early_stop_reason: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tier": tier,
        "widening_reason": widening_reason,
        "pool_size": selection.pool_size,
        "selected_domains": [s.domain for s in selection.capped],
        "store_types": store_type_distribution(selection.capped),
        "tavily_raw": len(raw_results),
        "tavily_cache_hit": cache_hit,
        "tavily_local_fanout_calls": local_site_count,
        "tavily_local_fanout_kept": local_site_kept,
        "listings_extracted": listings_out,
        "listings_accepted": len(accepted),
        "listings_rejected": len(rejected_reasons),
        "rejected_sample": rejected_reasons[:5],
        "accepted_urls": accepted_this_tier_urls[:10],
        "aggregated_running_total": len(aggregated),
        "stop_floor": stop_floor,
        "geo_confidence": round(float(confidence), 3),
        "early_stop": early_stop,
        "skipped": False,
    }
    if early_stop_reason is not None:
        payload["early_stop_reason"] = early_stop_reason
    return payload
