"""Deterministic vinyl-search pipeline orchestrator.

Owns the full request lifecycle. Each stage is wrapped in `stage_timer` for
structured tracing. NO business logic in this module — every transformation
lives in the called sub-module. No fallbacks, no error swallowing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.db.cache import (
    build_search_cache_key,
    build_tavily_tier_cache_key,
    get_cached_search_payload,
    get_cached_tavily_tier_payload,
    hydrate_cached_pipeline_dict,
    set_cached_search_payload,
)
from app.db.store_loader import ensure_local_coverage, load_active_stores
from app.llm.extract_listings import ExtractListingsReport, extract_listings
from app.llm.parse_user_query import parse_user_query
from app.models.search_query import SearchResult
from app.models.result import ListingResult
from app.pipeline_context import stage_timer
from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import (
    Tier,
    TIER_NARROWNESS,
    cap_stores,
    filter_stores_for_tier,
    geo_intent_from_parsed,
    max_domains_for_tier,
    normalized_geo_from_parsed,
    sort_stores_for_tier,
    sort_validated_listings_geo,
    tier_fallback_order,
)
from app.policies.listing_rank import (
    ListingRankBreakdown,
    composite_listing_score,
    resolve_store_for_url,
)
from app.policies.search_dsl import build_tavily_core_query
from app.policies.store_domain import canonical_store_domain
from app.llm.verify_album_match import verify_album_match
from app.services.discogs_service import resolve_album_by_index
from app.services.tavily_service import (
    normalize_url,
    run_local_site_searches,
    run_tavily_for_store_domains,
)
from app.validators.listings import normalize_whitelist_domain, validate_listing

logger = logging.getLogger(__name__)

_MIN_STORE_DOMAINS_DEFAULT = 2
_MIN_STORE_DOMAINS_CITY = 1
#: Result capping (Local-First Strike): when at least one ``local_shop`` validated,
#: hold non-indie store types to this cap in the final response.
_MAX_REGIONAL_WHEN_LOCAL_PRESENT = 1
_MAX_MARKETPLACE_WHEN_LOCAL_PRESENT = 0


def _dedupe_store_entries_by_domain(capped: tuple[StoreEntry, ...]) -> tuple[StoreEntry, ...]:
    seen: set[str] = set()
    out: list[StoreEntry] = []
    for s in capped:
        k = normalize_whitelist_domain(s.domain)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return tuple(out)


def _tier_validated_stop_floor(tier: Tier, *, settings: Any) -> int:
    if tier in ("city", "country"):
        return int(settings.pipeline_geo_stop_country)
    if tier == "region":
        return int(settings.pipeline_geo_stop_region)
    if tier == "continental":
        return int(settings.pipeline_geo_stop_continental)
    return 9999


def _effective_stop_floor(tier: Tier, *, settings: Any, confidence: float) -> int:
    """High geo confidence → stop widening sooner; low → require more hits first."""
    base = _tier_validated_stop_floor(tier, settings=settings)
    if confidence >= 0.88:
        return max(1, base)
    bump = int((0.88 - confidence) * 6)
    return min(20, max(1, base + max(0, bump)))


def _dedupe_listings_by_normalized_url(listings: list[Any]) -> list[Any]:
    order: list[str] = []
    by_key: dict[str, Any] = {}
    for lst in listings:
        u = getattr(lst, "url", None)
        if not u:
            continue
        key = normalize_url(str(u))
        if key not in by_key:
            by_key[key] = lst
            order.append(key)
    return [by_key[k] for k in order]


def _listing_to_api_row(
    listing: Any,
    *,
    breakdown: ListingRankBreakdown,
) -> ListingResult:
    title = str(listing.title or "")
    if len(title) < 5:
        title = f"{title} · shop"

    price_str = None
    cur = listing.currency or "EUR"
    try:
        if listing.price is not None and float(listing.price) > 0:
            price_str = f"{float(listing.price):.2f} {cur}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        price_str = f"{listing.price} {cur}" if listing.price else None

    match_reason = (
        f"tier={breakdown.discovery_tier}|store_type={breakdown.store_type}"
        f"|geo={breakdown.geo_proximity:.1f}|sem={breakdown.semantic_match:.1f}"
        f"|vinyl={breakdown.vinyl_confidence:.1f}"
    )

    return ListingResult(
        url=listing.url,
        title=title,
        score=breakdown.score_normalized,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=listing.store,
        artist_match=1.0,
        album_match=1.0,
        match_reason=match_reason,
    )


def _store_type_distribution(stores: tuple[StoreEntry, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in stores:
        k = s.store_type or "regional_ecommerce"
        out[k] = out.get(k, 0) + 1
    return out


def _dedupe_listings_by_domain(
    sorted_listings: list[Any],
    *,
    store_by_domain: dict[str, StoreEntry],
) -> tuple[list[Any], dict[str, int]]:
    """Collapse multiple listings from the same domain down to the top-ranked one.

    Input MUST already be sorted by composite rank (best first); the first
    occurrence of any domain wins and subsequent rows from that same host are
    dropped. This is the "HHV duplicate fix": when ``hhv.de`` returns five PDPs
    in one tier they collapse to a single row, freeing slots for distinct
    stores (especially indie locals).

    Returns ``(deduped_listings, dropped_by_domain)`` so we can surface the
    decision in the debug payload.
    """
    seen: set[str] = set()
    dropped: dict[str, int] = {}
    out: list[Any] = []
    for lst in sorted_listings:
        url = str(getattr(lst, "url", "") or "")
        store = resolve_store_for_url(url, store_by_domain)
        if store is not None and store.domain:
            key = normalize_whitelist_domain(store.domain)
        else:
            key = canonical_store_domain(url)
        if not key:
            continue
        if key in seen:
            dropped[key] = dropped.get(key, 0) + 1
            continue
        seen.add(key)
        out.append(lst)
    return out, dropped


def _apply_local_first_caps(
    sorted_listings: list[Any],
    *,
    store_by_domain: dict[str, StoreEntry],
    local_present_in_pool: bool,
    max_results: int,
) -> list[Any]:
    """Local-First result capping.

    If at least one ``local_shop`` validated:
      * keep every ``local_shop`` row (in their composite-score order),
      * limit ``regional_ecommerce`` to ``_MAX_REGIONAL_WHEN_LOCAL_PRESENT``,
      * limit ``marketplace`` to ``_MAX_MARKETPLACE_WHEN_LOCAL_PRESENT``.

    When no locals are present the helper is a pass-through (input order kept).
    Always returns at most ``max_results`` rows.
    """
    if max_results <= 0:
        return []
    if not local_present_in_pool:
        return sorted_listings[:max_results]

    out: list[Any] = []
    seen_regional = 0
    seen_market = 0
    for lst in sorted_listings:
        store = resolve_store_for_url(str(getattr(lst, "url", "") or ""), store_by_domain)
        st = (store.store_type if store is not None else "regional_ecommerce") or "regional_ecommerce"
        if st == "local_shop":
            out.append(lst)
        elif st == "regional_ecommerce":
            if seen_regional < _MAX_REGIONAL_WHEN_LOCAL_PRESENT:
                out.append(lst)
                seen_regional += 1
        elif st == "marketplace":
            if seen_market < _MAX_MARKETPLACE_WHEN_LOCAL_PRESENT:
                out.append(lst)
                seen_market += 1
        else:
            # Unknown / default: treat as regional.
            if seen_regional < _MAX_REGIONAL_WHEN_LOCAL_PRESENT:
                out.append(lst)
                seen_regional += 1
        if len(out) >= max_results:
            break
    return out


async def run_vinyl_search(query: str) -> dict[str, Any]:
    settings = get_settings()
    debug_enabled = settings.debug

    queries: list[str] = []
    core_query: str = ""
    store_domains: list[str] = []
    raw_results: list[Any] = []
    listings: list[Any] = []
    album_title: str | None = None
    stores: tuple[Any, ...] = ()

    with stage_timer("parse", input={"query": query}) as rec:
        parsed = await parse_user_query(query)
        rec.output = parsed.model_dump()

    with stage_timer("discogs") as rec:
        if parsed.album_index is not None:
            resolution = await resolve_album_by_index(
                artist=parsed.artist,
                album_index=parsed.album_index,
            )
            album_title = resolution.album.title if resolution.album else None
        elif parsed.album:
            album_title = parsed.album
        else:
            album_title = None

    if album_title is None:
        return {"query": query, "results": []}

    cache_key_value = build_search_cache_key(
        user_query=query,
        artist=parsed.artist,
        album_title=album_title,
        debug=debug_enabled,
    )

    cached_payload = await get_cached_search_payload(cache_key_value)
    if cached_payload is not None:
        return hydrate_cached_pipeline_dict(cached_payload)

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

    # LOCAL-FIRST STRIKE — phase 2: if the resolved city has fewer than the
    # configured number of indie ``local_shop`` rows, discover more on demand
    # via Tavily + LLM before we load the active store catalogue.
    if norm.resolved_city and norm.resolved_country:
        with stage_timer("store_discovery", input={
            "city": norm.resolved_city,
            "country_code": norm.resolved_country,
        }) as rec:
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

    with stage_timer("stores") as rec:
        stores = await load_active_stores()

    core_query = build_tavily_core_query(parsed.artist, album_title)
    queries = [core_query]

    tier_queue = list(tier_fallback_order(geo, norm))
    tier_ix = 0
    executed_global_fallback = False

    aggregated: dict[str, Any] = {}
    listing_tier_map: dict[str, Tier] = {}
    store_lookup: dict[str, StoreEntry] = {}
    # Per-URL LLM verdict on whether the listing's title actually names the
    # target release. Populated only for city-tier ``local_shop`` rows so the
    # +500 indie bonus doesn't fire on wrong-album local listings.
    album_match_by_url: dict[str, bool] = {}
    verifier_summary: list[dict[str, Any]] = []
    extract_report: ExtractListingsReport | None = None
    last_tier: Tier = "continental"
    tiers_attempted: list[Tier] = []
    tier_traces: list[dict[str, Any]] = []

    all_allowed = frozenset(
        normalize_whitelist_domain(s.domain) for s in stores if getattr(s, "domain", None)
    )

    while True:
        if tier_ix < len(tier_queue):
            tier = tier_queue[tier_ix]
            tier_ix += 1
            widening_reason = "planned_tier"
        elif (
            not aggregated
            and not executed_global_fallback
            and "global" not in tier_queue
        ):
            tier = "global"
            executed_global_fallback = True
            widening_reason = "fallback_no_results"
        else:
            break

        tiers_attempted.append(tier)

        pool = filter_stores_for_tier(stores, geo, tier, norm=norm)
        sorted_pool = sort_stores_for_tier(pool, geo, tier, norm=norm)
        max_d = max_domains_for_tier(
            tier,
            local_max=settings.pipeline_geo_local_max_domains,
            regional_max=settings.pipeline_geo_regional_max_domains,
            global_max=settings.pipeline_geo_global_max_domains,
        )

        capped = cap_stores(sorted_pool, max_domains=max_d)
        capped = _dedupe_store_entries_by_domain(capped)

        min_dom = _MIN_STORE_DOMAINS_CITY if tier == "city" else _MIN_STORE_DOMAINS_DEFAULT
        if len(capped) < min_dom:
            logger.info(
                "tier_skipped_low_signal",
                extra={"tier": tier, "domains": len(capped)},
            )
            with stage_timer("geo_tier", input={"tier": tier}) as rec:
                rec.status = "empty"
                rec.output = {
                    "tier": tier,
                    "selected_domains": [s.domain for s in capped],
                    "pool_size": len(pool),
                    "store_types": _store_type_distribution(capped),
                    "min_domain_floor": min_dom,
                    "widening_reason": widening_reason,
                    "early_stop": False,
                    "skipped": True,
                    "skip_reason": "below_min_domain_floor",
                    "geo_confidence": round(float(norm.confidence), 3),
                }
            continue

        tier_lookup = {normalize_whitelist_domain(s.domain): s for s in capped}

        for dom, row in tier_lookup.items():
            store_lookup[dom] = row

        store_domains = [normalize_whitelist_domain(s.domain) for s in capped if s.domain]

        tkey = build_tavily_tier_cache_key(
            artist=parsed.artist,
            album_title=album_title or "",
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
                "core_query": core_query,
                "include_domains_count": len(store_domains),
                "include_domains": store_domains,
                "cache_hit": cache_hit,
            },
        )

        with stage_timer("tavily") as rec:
            rec.input = {
                "tier": tier,
                "include_domains": store_domains,
                "cache_hit": cache_hit,
            }
            if cache_hit:
                raw_results = [SearchResult.model_validate(x) for x in (cached_raw or [])]
            else:
                raw_results, _ = await run_tavily_for_store_domains(
                    core_query,
                    store_domains,
                    tier=tier,
                )

        # Local site-search fanout — only in the city tier. The batched Tavily
        # call above gives every store one shared slate; the fanout below gives
        # EACH indie local domain its own ``include_domains=[domain]`` request
        # so deep PDPs surface on small shops that get drowned out in batches.
        local_site_count = 0
        local_site_kept = 0
        if tier == "city":
            local_domains_for_fanout = [
                normalize_whitelist_domain(s.domain)
                for s in capped
                if s.domain and s.store_type == "local_shop"
            ]
            local_domains_for_fanout = [d for d in local_domains_for_fanout if d]
            if local_domains_for_fanout:
                logger.info(
                    "tavily_local_fanout_start",
                    extra={
                        "stage": "tavily",
                        "tier": tier,
                        "core_query": core_query,
                        "local_domains_count": len(local_domains_for_fanout),
                        "local_domains": local_domains_for_fanout,
                    },
                )
                with stage_timer("tavily_local_fanout") as rec_fan:
                    fanout_results, n_calls = await run_local_site_searches(
                        core_query,
                        local_domains_for_fanout,
                        tier=tier,
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
                local_site_count = n_calls
                local_site_kept = len(fanout_results)

                # Merge fanout results into raw_results (URL-dedup, keep top score).
                existing_by_url: dict[str, SearchResult] = {
                    normalize_url(r.url): r for r in raw_results
                }
                for r in fanout_results:
                    k = normalize_url(r.url)
                    prev = existing_by_url.get(k)
                    if prev is None or r.score > prev.score:
                        existing_by_url[k] = r
                raw_results = list(existing_by_url.values())

        with stage_timer("extract") as rec:
            extract_report = await extract_listings(
                [r.model_dump() for r in raw_results],
                artist=parsed.artist,
                album=album_title,
                allowed_domains=set(store_domains),
            )
            listings = extract_report.listings
            rec.output = {
                "tier": tier,
                "raw_results_in": len(raw_results),
                "listings_out": len(listings),
                "diagnostic": extract_report.diagnostic,
            }

        with stage_timer("validate") as rec:
            batch = _dedupe_listings_by_normalized_url(listings)

        accepted: list[Any] = []
        rejected_reasons: list[dict[str, str]] = []
        # Local-First Strike: a row from the city tier whose host resolves to a
        # ``local_shop`` whitelist row passes the validator with relaxed fuzz floors.
        for lst in batch:
            enriched = lst.model_copy(
                update={
                    "validation_album": album_title,
                    "validation_artist": parsed.artist,
                }
            )
            relaxed = False
            if tier == "city":
                host_store = resolve_store_for_url(str(getattr(enriched, "url", "")), tier_lookup)
                if host_store is not None and host_store.store_type == "local_shop":
                    relaxed = True
            if not validate_listing(
                enriched,
                allowed_domains=all_allowed,
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

        # Title verification — only for city-tier ``local_shop`` rows. These
        # are the same rows that get the +500 indie bonus, so a wrong-album
        # title here would unfairly dominate. We LLM-confirm each title still
        # names the requested release. Verdicts:
        #   * "reject"   -> drop the row entirely from this tier.
        #   * "confirmed"-> mark ``album_match_by_url[url] = True`` (bonus eligible).
        #   * "unsure"   -> mark False (no bonus, no drop).
        if tier == "city" and accepted:
            to_verify: list[Any] = []
            for lst in accepted:
                url = str(getattr(lst, "url", "") or "")
                host_store = resolve_store_for_url(url, tier_lookup)
                if host_store is not None and host_store.store_type == "local_shop":
                    to_verify.append(lst)

            if to_verify:
                with stage_timer("verify_album_match") as rec_v:
                    rec_v.input = {
                        "tier": tier,
                        "candidates": len(to_verify),
                        "album_title": album_title or "",
                        "artist": parsed.artist,
                    }
                    verdicts = await verify_album_match(
                        to_verify,
                        artist=parsed.artist,
                        album_title=album_title or "",
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
                        album_match_by_url[url] = False
                        album_match_by_url[normalize_url(url)] = False
                        surviving.append(lst)
                        continue
                    if v.verdict == "reject":
                        rejected_reasons.append(
                            {
                                "url": url[:160],
                                "title": str(getattr(lst, "title", ""))[:120],
                                "reason": f"verify_album_match:{v.reason[:80]}",
                            }
                        )
                        verifier_summary.append(
                            {
                                "tier": tier,
                                "url": url[:160],
                                "verdict": "reject",
                                "reason": v.reason[:120],
                            }
                        )
                        continue
                    album_match_by_url[url] = v.verdict == "confirmed"
                    album_match_by_url[normalize_url(url)] = v.verdict == "confirmed"
                    surviving.append(lst)
                    verifier_summary.append(
                        {
                            "tier": tier,
                            "url": url[:160],
                            "verdict": v.verdict,
                            "reason": v.reason[:120],
                        }
                    )
                accepted = surviving

        accepted_this_tier_urls: list[str] = []
        for lst in accepted:
            k = normalize_url(str(lst.url))
            prev_tier = listing_tier_map.get(k)
            if prev_tier is None or TIER_NARROWNESS[tier] < TIER_NARROWNESS[prev_tier]:
                aggregated[k] = lst
                listing_tier_map[k] = tier
                accepted_this_tier_urls.append(k)

        last_tier = tier

        need = _effective_stop_floor(tier, settings=settings, confidence=norm.confidence)
        early_stop = len(aggregated) >= need

        tier_trace = {
            "tier": tier,
            "widening_reason": widening_reason,
            "pool_size": len(pool),
            "selected_domains": [s.domain for s in capped],
            "store_types": _store_type_distribution(capped),
            "tavily_raw": len(raw_results),
            "tavily_cache_hit": cache_hit,
            "tavily_local_fanout_calls": local_site_count,
            "tavily_local_fanout_kept": local_site_kept,
            "listings_extracted": len(listings),
            "listings_accepted": len(accepted),
            "listings_rejected": len(rejected_reasons),
            "rejected_sample": rejected_reasons[:5],
            "accepted_urls": accepted_this_tier_urls[:10],
            "aggregated_running_total": len(aggregated),
            "stop_floor": need,
            "geo_confidence": round(float(norm.confidence), 3),
            "early_stop": early_stop,
            "skipped": False,
        }
        tier_traces.append(tier_trace)

        with stage_timer("geo_tier", input={"tier": tier}) as rec:
            rec.output = tier_trace

        if early_stop:
            break

    list_out = list(aggregated.values())

    # Local-First Strike — gather pool flag once: are there any indie locals among
    # the validated rows? If yes, scorer + sorter activate the giant penalty.
    def _row_store(lst: Any) -> StoreEntry | None:
        return resolve_store_for_url(str(getattr(lst, "url", "") or ""), store_lookup)

    local_present_in_pool = any(
        (s := _row_store(r)) is not None and s.store_type == "local_shop"
        for r in list_out
    )

    sorted_listings = sort_validated_listings_geo(
        list_out,
        store_by_domain=store_lookup,
        norm=norm,
        listing_tier_map=listing_tier_map,
        album_title=album_title or "",
        artist=parsed.artist,
        local_present_in_pool=local_present_in_pool,
        album_match_by_url=album_match_by_url,
    )

    # HHV duplicate fix: collapse same-domain rows down to the top-ranked one
    # BEFORE the local-first cap runs, so that "5x hhv.de" can never crowd out
    # five separate indie shops just because Tavily liked one giant catalogue.
    deduped_listings, domain_drops = _dedupe_listings_by_domain(
        sorted_listings,
        store_by_domain=store_lookup,
    )

    # Result capping: when locals exist, hold non-indie store types to the cap
    # BEFORE truncating to ``pipeline_max_results``.
    head = _apply_local_first_caps(
        deduped_listings,
        store_by_domain=store_lookup,
        local_present_in_pool=local_present_in_pool,
        max_results=settings.pipeline_max_results,
    )

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
                artist=parsed.artist,
                local_present_in_pool=local_present_in_pool,
                album_match_confirmed=confirmed,
            )
        )

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

    with stage_timer("geo_widening_summary") as rec:
        rec.output = {
            "tiers_planned": list(tier_queue),
            "tiers_attempted": tiers_attempted,
            "executed_global_fallback": executed_global_fallback,
            "total_validated": len(aggregated),
            "after_domain_dedupe": len(deduped_listings),
            "domain_duplicates_dropped": domain_drops,
            "final_returned": len(head),
            "local_present_in_pool": local_present_in_pool,
            "local_first_caps": {
                "regional_ecommerce_max": _MAX_REGIONAL_WHEN_LOCAL_PRESENT,
                "marketplace_max": _MAX_MARKETPLACE_WHEN_LOCAL_PRESENT,
            },
            "verifier_summary": verifier_summary[:25],
            "verifier_totals": {
                "confirmed": sum(1 for v in verifier_summary if v["verdict"] == "confirmed"),
                "reject": sum(1 for v in verifier_summary if v["verdict"] == "reject"),
                "unsure": sum(1 for v in verifier_summary if v["verdict"] == "unsure"),
            },
            "per_tier": tier_traces,
        }

    out = {
        "query": query,
        "results": [
            _listing_to_api_row(listing, breakdown=breakdown)
            for listing, breakdown in zip(head, breakdowns, strict=True)
        ],
    }

    await set_cached_search_payload(
        cache_key_value,
        out,
        ttl_seconds=settings.search_cache_ttl_seconds,
    )

    return out
