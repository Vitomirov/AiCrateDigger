"""Per-tier orchestration for the geo widening loop.

The deterministic vinyl-search pipeline keeps widening the geographic store
pool (city → country → region → continental → global) until enough validated
listings are gathered. This module owns one full iteration of that loop:

    1. Pick + cap stores for the tier (skip with a structured empty trace if
       there is not enough signal),
    2. Run batched Tavily for the selected domains (with per-tier cache),
    3. In the city tier only, do a per-store Tavily fanout and merge it,
    4. Run the extractor on the raw results,
    5. Run per-listing validation (with relaxed-local-indie rules),
    6. In the city tier only, run the LLM album-title verifier,
    7. Merge the surviving listings into the cross-tier aggregate,
    8. Emit the ``geo_tier`` trace and report early-stop.

The orchestrator in :mod:`app.pipeline.vinyl_search` only has to drive the
``tier_queue`` + global-fallback loop and hand each tier to :func:`process_tier`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.agents.extractor import ExtractListingsReport, extract_listings
from app.agents.extractor.verify_album_match import verify_album_match
from app.db.cache import (
    build_tavily_tier_cache_key,
    get_cached_tavily_tier_payload,
)
from app.domain.parse_schema import ParsedQuery
from app.models.search_query import SearchResult
from app.pipeline.constants import (
    MIN_STORE_DOMAINS_CITY,
    MIN_STORE_DOMAINS_DEFAULT,
)
from app.pipeline.listing_dedupe import dedupe_listings_by_normalized_url
from app.pipeline.stop_floors import effective_stop_floor
from app.pipeline.store_selection import (
    dedupe_store_entries_by_domain,
    store_type_distribution,
)
from app.pipeline_context import stage_timer
from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import (
    GeoIntent,
    NormalizedGeoIntent,
    TIER_NARROWNESS,
    Tier,
    cap_stores,
    filter_stores_for_tier,
    max_domains_for_tier,
    sort_stores_for_tier,
)
from app.policies.listing_rank import resolve_store_for_url
from app.policies.physical_local import qualifies_as_target_city_local_shop
from app.services.tavily_service import (
    editorial_discovery_blocked_hosts_from_raw_results,
    normalize_url,
    run_local_site_searches,
    run_tavily_for_store_domains,
)
from app.validators.listings import (
    global_fallback_matches_parsed_entity,
    normalize_whitelist_domain,
    validate_listing,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request immutables + cross-tier mutable state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TierContext:
    """Per-request immutables threaded through every tier iteration."""

    parsed: ParsedQuery
    album_title: str
    geo: GeoIntent
    norm: NormalizedGeoIntent
    stores: tuple[StoreEntry, ...]
    settings: Any
    core_query: str
    tavily_relaxation_queries: Any
    curated_city_local_domains: frozenset[str]
    prioritize_physical_locals: bool
    all_allowed: frozenset[str]


@dataclass(slots=True)
class TierLoopState:
    """Mutable state carried across iterations of the geo widening loop."""

    aggregated: dict[str, Any] = field(default_factory=dict)
    listing_tier_map: dict[str, Tier] = field(default_factory=dict)
    store_lookup: dict[str, StoreEntry] = field(default_factory=dict)
    editorial_discovery_blocked: set[str] = field(default_factory=set)
    album_match_by_url: dict[str, bool] = field(default_factory=dict)
    verifier_summary: list[dict[str, Any]] = field(default_factory=list)
    tier_traces: list[dict[str, Any]] = field(default_factory=list)
    tiers_attempted: list[Tier] = field(default_factory=list)
    last_tier: Tier = "continental"


# ---------------------------------------------------------------------------
# Internal per-tier selection structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TierStoreSelection:
    """Per-tier store pool after filtering, sorting, capping and deduping."""

    capped: tuple[StoreEntry, ...]
    pool_size: int
    min_dom: int
    store_domains: list[str]
    domains_for_tavily: list[str]
    tier_lookup: dict[str, StoreEntry]

    @property
    def is_below_floor(self) -> bool:
        return len(self.capped) < self.min_dom


# ---------------------------------------------------------------------------
# Step 1 — store selection
# ---------------------------------------------------------------------------


def _select_tier_stores(
    ctx: TierContext,
    tier: Tier,
    *,
    editorial_discovery_blocked: set[str],
) -> _TierStoreSelection:
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

    return _TierStoreSelection(
        capped=capped,
        pool_size=len(pool),
        min_dom=min_dom,
        store_domains=store_domains,
        domains_for_tavily=domains_for_tavily,
        tier_lookup=tier_lookup,
    )


def _record_skipped_tier_trace(
    *,
    tier: Tier,
    selection: _TierStoreSelection,
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


# ---------------------------------------------------------------------------
# Step 2 — Tavily (with per-tier cache) + step 3 — city local fanout
# ---------------------------------------------------------------------------


async def _run_tavily_stage(
    ctx: TierContext,
    tier: Tier,
    domains_for_tavily: list[str],
) -> tuple[list[SearchResult], bool]:
    """Batched Tavily call (with per-tier cache) — wraps the ``tavily`` stage."""
    tkey = build_tavily_tier_cache_key(
        artist=ctx.parsed.artist,
        album_title=ctx.album_title or "",
        tier=tier,
    )
    cached_raw = await get_cached_tavily_tier_payload(tkey)
    cache_hit = cached_raw is not None

    # Visibility hook for the local-first audit: dump the exact
    # ``include_domains`` payload we are about to send to Tavily. This is
    # the canonical place to verify that newly-discovered domains (e.g.
    # ``misbits.ro``, ``phono.cz``) are actually included in the request.
    logger.info(
        "tavily_include_domains",
        extra={
            "stage": "tavily",
            "tier": tier,
            "core_query": ctx.core_query,
            "include_domains_count": len(domains_for_tavily),
            "include_domains": domains_for_tavily,
            "cache_hit": cache_hit,
        },
    )

    with stage_timer("tavily") as rec:
        rec.input = {
            "tier": tier,
            "include_domains": domains_for_tavily,
            "cache_hit": cache_hit,
        }
        if cache_hit:
            raw_results = [SearchResult.model_validate(x) for x in (cached_raw or [])]
        else:
            raw_results, _ = await run_tavily_for_store_domains(
                ctx.core_query,
                domains_for_tavily,
                tier=tier,
                relaxation_queries=ctx.tavily_relaxation_queries,
            )

    return raw_results, cache_hit


async def _run_city_local_fanout(
    ctx: TierContext,
    tier: Tier,
    capped: tuple[StoreEntry, ...],
    raw_results: list[SearchResult],
) -> tuple[list[SearchResult], int, int]:
    """City-tier only: per-indie-store Tavily fanout, merged into ``raw_results``.

    The batched Tavily call gives every store one shared slate; the fanout
    gives EACH indie local domain its own ``include_domains=[domain]`` request
    so deep PDPs surface on small shops that get drowned out in batches.
    """
    if tier != "city":
        return raw_results, 0, 0

    local_domains_for_fanout = [
        normalize_whitelist_domain(s.domain)
        for s in capped
        if s.domain and s.store_type == "local_shop"
    ]
    local_domains_for_fanout = [d for d in local_domains_for_fanout if d]
    if not local_domains_for_fanout:
        return raw_results, 0, 0

    logger.info(
        "tavily_local_fanout_start",
        extra={
            "stage": "tavily",
            "tier": tier,
            "core_query": ctx.core_query,
            "local_domains_count": len(local_domains_for_fanout),
            "local_domains": local_domains_for_fanout,
        },
    )

    with stage_timer("tavily_local_fanout") as rec_fan:
        fanout_results, n_calls = await run_local_site_searches(
            ctx.core_query,
            local_domains_for_fanout,
            tier=tier,
            artist=ctx.parsed.artist,
            album_title=ctx.album_title or "",
        )
        rec_fan.input = {
            "tier": tier,
            "local_domains": local_domains_for_fanout,
        }
        rec_fan.output = {
            "http_calls": n_calls,
            "kept": len(fanout_results),
        }
        rec_fan.status = "success" if fanout_results else "empty"

    # Merge fanout results into raw_results (URL-dedup, keep top score).
    existing_by_url: dict[str, SearchResult] = {
        normalize_url(r.url): r for r in raw_results
    }
    for r in fanout_results:
        k = normalize_url(r.url)
        prev = existing_by_url.get(k)
        if prev is None or r.score > prev.score:
            existing_by_url[k] = r
    return list(existing_by_url.values()), n_calls, len(fanout_results)


# ---------------------------------------------------------------------------
# Step 4 — extractor stage
# ---------------------------------------------------------------------------


async def _run_extract_stage(
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


# ---------------------------------------------------------------------------
# Step 5 — validation
# ---------------------------------------------------------------------------


def _stage_dedupe_listings(listings: list[Any]) -> list[Any]:
    """URL-level dedupe wrapped in the ``validate`` stage timer."""
    with stage_timer("validate"):
        return dedupe_listings_by_normalized_url(listings)


def _validate_listings_for_tier(
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


# ---------------------------------------------------------------------------
# Step 6 — city-only album-title verification
# ---------------------------------------------------------------------------


async def _run_city_verify_stage(
    *,
    ctx: TierContext,
    state: TierLoopState,
    tier: Tier,
    accepted: list[Any],
    tier_lookup: dict[str, StoreEntry],
    rejected_reasons: list[dict[str, str]],
) -> list[Any]:
    """LLM album-title verification on city-tier ``local_shop`` rows.

    City-tier indie rows receive the +500 bonus downstream, so a wrong-album
    title would unfairly dominate. The verifier returns:

    * ``"reject"``    → drop the row entirely from this tier (unless soft-kept
      because it is a prioritised target-city physical-local shop).
    * ``"confirmed"`` → mark ``album_match_by_url[url] = True`` (bonus eligible).
    * ``"unsure"``    → mark ``False`` (no bonus, no drop).
    """
    to_verify: list[Any] = []
    for lst in accepted:
        url = str(getattr(lst, "url", "") or "")
        host_store = resolve_store_for_url(url, tier_lookup)
        if host_store is not None and host_store.store_type == "local_shop":
            to_verify.append(lst)

    if not to_verify:
        return accepted

    with stage_timer("verify_album_match") as rec_v:
        rec_v.input = {
            "tier": tier,
            "candidates": len(to_verify),
            "album_title": ctx.album_title or "",
            "artist": ctx.parsed.artist,
        }
        verdicts = await verify_album_match(
            to_verify,
            artist=ctx.parsed.artist,
            album_title=ctx.album_title or "",
        )
        rec_v.output = {
            "verdicts_returned": len(verdicts),
            "rejected": sum(
                1 for v in verdicts.values() if v.verdict == "reject"
            ),
            "confirmed": sum(
                1 for v in verdicts.values() if v.verdict == "confirmed"
            ),
            "unsure": sum(
                1 for v in verdicts.values() if v.verdict == "unsure"
            ),
        }
        rec_v.status = "success" if verdicts else "empty"

    # Drop rejects, mark confirmed/unsure for downstream scoring.
    surviving: list[Any] = []
    for lst in accepted:
        url = str(getattr(lst, "url", "") or "")
        host_store = resolve_store_for_url(url, tier_lookup)
        is_local = (
            host_store is not None
            and host_store.store_type == "local_shop"
        )
        if not is_local:
            surviving.append(lst)
            continue
        v = verdicts.get(url)
        if v is None:
            # Verifier silently skipped this row → treat as unsure
            # (keep, no bonus).
            state.album_match_by_url[url] = False
            state.album_match_by_url[normalize_url(url)] = False
            surviving.append(lst)
            continue
        if v.verdict == "reject":
            keep_physical_local_despite_strict_reject = (
                ctx.prioritize_physical_locals
                and qualifies_as_target_city_local_shop(
                    listing_url=url,
                    store_lookup=state.store_lookup,
                    norm=ctx.norm,
                )
            )
            if keep_physical_local_despite_strict_reject:
                state.album_match_by_url[url] = False
                state.album_match_by_url[normalize_url(url)] = False
                state.verifier_summary.append(
                    {
                        "tier": tier,
                        "url": url[:160],
                        "verdict": "reject_soft_kept",
                        "reason": v.reason[:120],
                    }
                )
                surviving.append(lst)
                continue
            rejected_reasons.append(
                {
                    "url": url[:160],
                    "title": str(getattr(lst, "title", ""))[:120],
                    "reason": f"verify_album_match:{v.reason[:80]}",
                }
            )
            state.verifier_summary.append(
                {
                    "tier": tier,
                    "url": url[:160],
                    "verdict": "reject",
                    "reason": v.reason[:120],
                }
            )
            continue
        state.album_match_by_url[url] = v.verdict == "confirmed"
        state.album_match_by_url[normalize_url(url)] = v.verdict == "confirmed"
        surviving.append(lst)
        state.verifier_summary.append(
            {
                "tier": tier,
                "url": url[:160],
                "verdict": v.verdict,
                "reason": v.reason[:120],
            }
        )
    return surviving


# ---------------------------------------------------------------------------
# Step 7 — aggregate merge (tier-narrowness aware)
# ---------------------------------------------------------------------------


def _merge_into_aggregate(
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


# ---------------------------------------------------------------------------
# Step 8 — per-tier trace
# ---------------------------------------------------------------------------


def _build_tier_trace(
    *,
    tier: Tier,
    widening_reason: str,
    selection: _TierStoreSelection,
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
) -> dict[str, Any]:
    return {
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


# ---------------------------------------------------------------------------
# Public entry point — process_tier
# ---------------------------------------------------------------------------


async def process_tier(
    *,
    tier: Tier,
    widening_reason: str,
    ctx: TierContext,
    state: TierLoopState,
) -> bool:
    """Run one full tier iteration; return True iff early-stop fires."""
    state.tiers_attempted.append(tier)

    selection = _select_tier_stores(
        ctx, tier, editorial_discovery_blocked=state.editorial_discovery_blocked
    )
    if selection.is_below_floor:
        _record_skipped_tier_trace(
            tier=tier,
            selection=selection,
            widening_reason=widening_reason,
            confidence=ctx.norm.confidence,
        )
        return False

    for dom, row in selection.tier_lookup.items():
        state.store_lookup[dom] = row

    raw_results, cache_hit = await _run_tavily_stage(
        ctx, tier, selection.domains_for_tavily
    )
    raw_results, local_site_count, local_site_kept = await _run_city_local_fanout(
        ctx, tier, selection.capped, raw_results
    )

    extract_report = await _run_extract_stage(
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

    batch = _stage_dedupe_listings(listings)
    accepted, rejected_reasons = _validate_listings_for_tier(
        ctx=ctx,
        state=state,
        tier=tier,
        batch=batch,
        tier_lookup=selection.tier_lookup,
    )

    if tier == "city" and accepted:
        accepted = await _run_city_verify_stage(
            ctx=ctx,
            state=state,
            tier=tier,
            accepted=accepted,
            tier_lookup=selection.tier_lookup,
            rejected_reasons=rejected_reasons,
        )

    accepted_this_tier_urls = _merge_into_aggregate(state, tier, accepted)
    state.last_tier = tier

    need = effective_stop_floor(
        tier, settings=ctx.settings, confidence=ctx.norm.confidence
    )
    early_stop = len(state.aggregated) >= need

    tier_trace = _build_tier_trace(
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
    )
    state.tier_traces.append(tier_trace)

    with stage_timer("geo_tier", input={"tier": tier}) as rec:
        rec.output = tier_trace

    return early_stop
