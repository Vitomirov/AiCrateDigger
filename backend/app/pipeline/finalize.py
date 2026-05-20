"""Post-loop finalisation: local-shop signals, sort/dedupe/cap, scoring + traces.

Once the geo widening loop has built an aggregated pool of validated listings,
this module turns that pool into the final ordered, scored response set and
emits the matching ``ranking`` / ``geo_widening_summary`` debug traces.
"""

from __future__ import annotations

from typing import Any

from app.pipeline.constants import (
    MAX_MARKETPLACE_WHEN_LOCAL_PRESENT,
    MAX_REGIONAL_WHEN_LOCAL_PRESENT,
    MAX_REGIONAL_WHEN_PRIORITIZED_PRIMARY_LOCAL_PRESENT,
)
from app.pipeline.listing_dedupe import dedupe_listings_by_domain
from app.pipeline.local_first import apply_local_first_caps
from app.pipeline_context import stage_timer
from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import (
    NormalizedGeoIntent,
    Tier,
    sort_validated_listings_geo,
)
from app.policies.listing_rank import (
    ListingRankBreakdown,
    composite_listing_score,
    resolve_store_for_url,
)
from app.policies.physical_local import pool_has_qualifying_physical_local_row
from app.services.tavily import normalize_url


# ---------------------------------------------------------------------------
# Local-shop signal computation
# ---------------------------------------------------------------------------


def compute_local_shop_signals(
    list_out: list[Any],
    *,
    store_lookup: dict[str, StoreEntry],
    norm: NormalizedGeoIntent,
    prioritize_physical_locals: bool,
) -> tuple[bool, bool, bool]:
    """Return ``(generic_any_local, primary_target_local, pool_for_giant_penalty)``.

    * ``generic_any_local_shop_present`` — at least one validated row is a
      ``local_shop`` somewhere in the world (drives the soft cap).
    * ``primary_target_local_shop_present`` — at least one validated row is a
      target-city physical-local shop (drives the stricter prioritised cap).
    * ``pool_for_giant_penalty`` — whether the giant-retailer penalty should
      fire during ranking, derived from whichever signal applies.
    """
    def _row_store(lst: Any) -> StoreEntry | None:
        return resolve_store_for_url(str(getattr(lst, "url", "") or ""), store_lookup)

    generic_any_local_shop_present = any(
        (s := _row_store(r)) is not None and s.store_type == "local_shop"
        for r in list_out
    )
    primary_target_local_shop_present = pool_has_qualifying_physical_local_row(
        list_out,
        store_lookup=store_lookup,
        norm=norm,
    )
    pool_for_giant_penalty = (
        primary_target_local_shop_present
        if prioritize_physical_locals
        else generic_any_local_shop_present
    )
    return (
        generic_any_local_shop_present,
        primary_target_local_shop_present,
        pool_for_giant_penalty,
    )


# ---------------------------------------------------------------------------
# Sort + dedupe + cap (HHV duplicate fix BEFORE Local-First cap)
# ---------------------------------------------------------------------------


def sort_dedupe_and_cap_listings(
    list_out: list[Any],
    *,
    store_lookup: dict[str, StoreEntry],
    norm: NormalizedGeoIntent,
    listing_tier_map: dict[str, Tier],
    album_title: str,
    artist: str | None,
    album_match_by_url: dict[str, bool],
    pool_for_giant_penalty: bool,
    generic_any_local_shop_present: bool,
    primary_target_local_shop_present: bool,
    prioritize_physical_locals: bool,
    max_results: int,
) -> tuple[list[Any], list[Any], dict[str, int]]:
    """Sort by geo, collapse same-domain rows, then apply Local-First caps.

    Returns ``(deduped_listings, head, domain_drops)``. ``head`` is the
    final ordered slice (truncated to ``max_results``).
    """
    sorted_listings = sort_validated_listings_geo(
        list_out,
        store_by_domain=store_lookup,
        norm=norm,
        listing_tier_map=listing_tier_map,
        album_title=album_title or "",
        artist=artist,
        local_present_in_pool=pool_for_giant_penalty,
        album_match_by_url=album_match_by_url,
        prioritize_physical_city_locals=prioritize_physical_locals,
    )

    # HHV duplicate fix: collapse same-domain rows down to the top-ranked one
    # BEFORE the local-first cap runs, so that "5x hhv.de" can never crowd out
    # five separate indie shops just because Tavily liked one giant catalogue.
    # Prefer the listing that best evidences the requested album (plus verifier flags),
    # instead of blindly keeping the first composite-ranked row on that host.
    deduped_listings, domain_drops = dedupe_listings_by_domain(
        sorted_listings,
        store_by_domain=store_lookup,
        album_title=album_title,
        artist=artist,
        album_match_by_url=(album_match_by_url if len(album_match_by_url) > 0 else None),
    )

    # Result capping: when locals exist, hold non-indie store types to the cap
    # BEFORE truncating to ``pipeline_max_results``.
    head = apply_local_first_caps(
        deduped_listings,
        store_by_domain=store_lookup,
        generic_local_shop_present_in_pool=generic_any_local_shop_present,
        prioritize_physical_locals=prioritize_physical_locals,
        primary_target_local_shop_present=primary_target_local_shop_present,
        max_results=max_results,
    )
    return deduped_listings, head, domain_drops


# ---------------------------------------------------------------------------
# Composite scoring for the final head
# ---------------------------------------------------------------------------


def score_final_listings(
    head: list[Any],
    *,
    store_lookup: dict[str, StoreEntry],
    listing_tier_map: dict[str, Tier],
    last_tier: Tier,
    album_match_by_url: dict[str, bool],
    norm: NormalizedGeoIntent,
    album_title: str,
    artist: str | None,
    pool_for_giant_penalty: bool,
) -> list[ListingRankBreakdown]:
    """Compute one :class:`ListingRankBreakdown` per row in ``head``."""
    breakdowns: list[ListingRankBreakdown] = []
    for lst in head:
        u = str(getattr(lst, "url", "") or "")
        nk = normalize_url(u)
        st = resolve_store_for_url(u, store_lookup)
        tier_for = listing_tier_map.get(nk, last_tier)
        confirmed = album_match_by_url.get(u, album_match_by_url.get(nk, True))
        breakdowns.append(
            composite_listing_score(
                lst,
                store=st,
                discovery_tier=tier_for,
                resolved_country=norm.resolved_country,
                resolved_city=norm.resolved_city,
                album_title=album_title or "",
                artist=artist,
                local_present_in_pool=pool_for_giant_penalty,
                album_match_confirmed=confirmed,
            )
        )
    return breakdowns


# ---------------------------------------------------------------------------
# Debug traces — ranking + widening summary
# ---------------------------------------------------------------------------


def emit_ranking_trace(
    head: list[Any], breakdowns: list[ListingRankBreakdown]
) -> None:
    with stage_timer("ranking") as rec:
        avg_total = (
            round(sum(b.total for b in breakdowns) / len(breakdowns), 3)
            if breakdowns
            else 0.0
        )
        rec.output = {
            "scored_count": len(breakdowns),
            "avg_total_points": avg_total,
            "breakdowns": [
                {
                    "url": str(getattr(lst, "url", ""))[:200],
                    "title": str(getattr(lst, "title", ""))[:160],
                    **b.as_dict(),
                }
                for lst, b in zip(head, breakdowns, strict=True)
            ],
        }


def emit_widening_summary_trace(
    *,
    tier_queue: list[Tier],
    tiers_attempted: list[Tier],
    executed_global_fallback: bool,
    aggregated: dict[str, Any],
    deduped_listings: list[Any],
    domain_drops: dict[str, int],
    head: list[Any],
    generic_any_local_shop_present: bool,
    primary_target_local_shop_present: bool,
    prioritize_physical_locals: bool,
    pool_for_giant_penalty: bool,
    verifier_summary: list[dict[str, Any]],
    tier_traces: list[dict[str, Any]],
) -> None:
    with stage_timer("geo_widening_summary") as rec:
        rec.output = {
            "tiers_planned": list(tier_queue),
            "tiers_attempted": tiers_attempted,
            "executed_global_fallback": executed_global_fallback,
            "total_validated": len(aggregated),
            "after_domain_dedupe": len(deduped_listings),
            "domain_duplicates_dropped": domain_drops,
            "final_returned": len(head),
            "local_present_in_pool": generic_any_local_shop_present,
            "primary_target_city_local_shop_present": primary_target_local_shop_present,
            "prioritize_physical_local_shops": prioritize_physical_locals,
            "giant_penalty_pool_active": pool_for_giant_penalty,
            "local_first_caps": {
                "regional_ecommerce_max": (
                    MAX_REGIONAL_WHEN_PRIORITIZED_PRIMARY_LOCAL_PRESENT
                    if prioritize_physical_locals and primary_target_local_shop_present
                    else MAX_REGIONAL_WHEN_LOCAL_PRESENT
                ),
                "marketplace_max": MAX_MARKETPLACE_WHEN_LOCAL_PRESENT,
            },
            "verifier_summary": verifier_summary[:25],
            "verifier_totals": {
                "confirmed": sum(1 for v in verifier_summary if v["verdict"] == "confirmed"),
                "reject": sum(1 for v in verifier_summary if v["verdict"] == "reject"),
                "unsure": sum(1 for v in verifier_summary if v["verdict"] == "unsure"),
            },
            "per_tier": tier_traces,
        }
