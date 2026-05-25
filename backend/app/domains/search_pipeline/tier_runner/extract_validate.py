"""Extractor + validation stages for one geo tier."""

from __future__ import annotations

from typing import Any

from app.domains.engine.extraction import ExtractListingsReport, extract_listings
from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.search_pipeline.listing_dedupe import dedupe_listings_by_normalized_url
from app.domains.search_pipeline.tier_runner.context import TierContext, TierLoopState
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.policies.geo_scope import Tier
from app.domains.engine.policies.listing_rank import resolve_store_for_url
from app.domains.engine.policies.physical_local import qualifies_as_target_city_local_shop
from app.domains.engine.validators.listings import (
    global_fallback_matches_parsed_entity,
    validate_listing,
)


async def run_extract_stage(
    ctx: TierContext,
    tier: Tier,
    raw_results: list[SearchResult],
    domains_for_tavily: list[str],
) -> ExtractListingsReport:
    """Run the snippet extractor and emit the ``extract`` stage trace."""
    with stage_timer("extract") as rec:
        report = await extract_listings(
            [r.model_dump() for r in raw_results],
            artist=ctx.parsed.artist,
            album=ctx.album_title,
            allowed_domains=set(domains_for_tavily),
            snippet_relax_hosts=ctx.curated_city_local_domains,
        )
        rec.output = {
            "tier": tier,
            "raw_results_in": len(raw_results),
            "listings_out": len(report.listings),
            "diagnostic": report.diagnostic,
        }
    return report


def stage_dedupe_listings(listings: list[Any]) -> list[Any]:
    """URL-level dedupe wrapped in the ``validate`` stage timer."""
    with stage_timer("validate"):
        return dedupe_listings_by_normalized_url(listings)


def validate_listings_for_tier(
    *,
    ctx: TierContext,
    state: TierLoopState,
    tier: Tier,
    batch: list[Any],
    tier_lookup: dict[str, StoreEntry],
) -> tuple[list[Any], list[dict[str, str]]]:
    """Per-listing validation loop with relaxed-local-indie + global gates."""
    accepted: list[Any] = []
    rejected_reasons: list[dict[str, str]] = []

    # Local-First Strike: a row from the city tier whose host resolves to a
    # ``local_shop`` whitelist row passes the validator with relaxed fuzz floors.
    for lst in batch:
        enriched = lst.model_copy(
            update={
                "validation_album": ctx.album_title,
                "validation_artist": ctx.parsed.artist,
            }
        )
        relaxed = False
        enriched_url = str(getattr(enriched, "url", "") or "")
        host_tier = resolve_store_for_url(enriched_url, tier_lookup)
        if tier == "city" and host_tier is not None and host_tier.store_type == "local_shop":
            relaxed = True
        elif ctx.prioritize_physical_locals and qualifies_as_target_city_local_shop(
            listing_url=enriched_url,
            store_lookup=state.store_lookup,
            norm=ctx.norm,
        ):
            relaxed = True
        if tier == "global":
            snip = getattr(enriched, "source_snippet", None)
            if not global_fallback_matches_parsed_entity(
                listing_title=str(getattr(enriched, "title", "") or ""),
                source_snippet=snip if isinstance(snip, str) else None,
                validation_artist=ctx.parsed.artist,
                validation_album=ctx.album_title or "",
            ):
                rejected_reasons.append(
                    {
                        "url": str(getattr(enriched, "url", ""))[:160],
                        "title": str(getattr(enriched, "title", ""))[:120],
                        "reason": "global_fallback_core_entity_mismatch",
                    }
                )
                continue
        if not validate_listing(
            enriched,
            allowed_domains=ctx.all_allowed,
            relaxed_local_indie=relaxed,
        ):
            rejected_reasons.append(
                {
                    "url": str(getattr(enriched, "url", ""))[:160],
                    "title": str(getattr(enriched, "title", ""))[:120],
                }
            )
            continue
        accepted.append(enriched)

    return accepted, rejected_reasons
