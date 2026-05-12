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
from app.db.store_loader import load_active_stores
from app.llm.extract_listings import ExtractListingsReport, extract_listings
from app.llm.parse_user_query import parse_user_query
from app.models.search_query import SearchResult
from app.models.result import ListingResult
from app.pipeline_context import stage_timer
from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import (
    NormalizedGeoIntent,
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
from app.policies.listing_rank import composite_listing_rank, resolve_store_for_url
from app.policies.search_dsl import build_tavily_core_query
from app.services.discogs_service import resolve_album_by_index
from app.services.tavily_service import normalize_url, run_tavily_for_store_domains
from app.validators.listings import normalize_whitelist_domain, validate_listing

logger = logging.getLogger(__name__)

_MIN_STORE_DOMAINS_DEFAULT = 2
_MIN_STORE_DOMAINS_CITY = 1


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
    norm: NormalizedGeoIntent,
    store_by_domain: dict[str, StoreEntry],
    listing_tier_map: dict[str, Tier],
    default_tier: Tier,
    album_title: str,
    artist: str | None,
    score_scale: float,
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

    u = str(getattr(listing, "url", "") or "")
    nk = normalize_url(u)
    tier = listing_tier_map.get(nk, default_tier)
    st = resolve_store_for_url(u, store_by_domain)

    raw_rank = composite_listing_rank(
        listing,
        store=st,
        discovery_tier=tier,
        resolved_country=norm.resolved_country,
        resolved_city=norm.resolved_city,
        album_title=album_title,
        artist=artist,
    )
    denom = score_scale if score_scale > 0 else 1.0
    score_norm = min(max(raw_rank / denom, 0.0), 1.0)

    return ListingResult(
        url=listing.url,
        title=title,
        score=score_norm,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=listing.store,
        artist_match=1.0,
        album_match=1.0,
        match_reason="vinyl_pipeline",
    )


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

    with stage_timer("stores") as rec:
        stores = await load_active_stores()

    geo = geo_intent_from_parsed(parsed)
    norm = normalized_geo_from_parsed(parsed)

    core_query = build_tavily_core_query(parsed.artist, album_title)
    queries = [core_query]

    tier_queue = list(tier_fallback_order(geo, norm))
    tier_ix = 0
    executed_global_fallback = False

    aggregated: dict[str, Any] = {}
    listing_tier_map: dict[str, Tier] = {}
    store_lookup: dict[str, StoreEntry] = {}
    extract_report: ExtractListingsReport | None = None
    last_tier: Tier = "continental"
    tiers_attempted: list[Tier] = []

    all_allowed = frozenset(
        normalize_whitelist_domain(s.domain) for s in stores if getattr(s, "domain", None)
    )

    while True:
        if tier_ix < len(tier_queue):
            tier = tier_queue[tier_ix]
            tier_ix += 1
        elif (
            not aggregated
            and not executed_global_fallback
            and "global" not in tier_queue
        ):
            tier = "global"
            executed_global_fallback = True
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

        with stage_timer("tavily") as rec:
            if cache_hit:
                raw_results = [SearchResult.model_validate(x) for x in (cached_raw or [])]
            else:
                raw_results, _ = await run_tavily_for_store_domains(
                    core_query,
                    store_domains,
                    tier=tier,
                )

        with stage_timer("extract") as rec:
            extract_report = await extract_listings(
                [r.model_dump() for r in raw_results],
                artist=parsed.artist,
                album=album_title,
                allowed_domains=set(store_domains),
            )
            listings = extract_report.listings

        with stage_timer("validate") as rec:
            batch = _dedupe_listings_by_normalized_url(listings)

        accepted: list[Any] = []
        for lst in batch:
            enriched = lst.model_copy(
                update={
                    "validation_album": album_title,
                    "validation_artist": parsed.artist,
                }
            )
            if not validate_listing(enriched, allowed_domains=all_allowed):
                continue
            accepted.append(enriched)

        for lst in accepted:
            k = normalize_url(str(lst.url))
            prev_tier = listing_tier_map.get(k)
            if prev_tier is None or TIER_NARROWNESS[tier] < TIER_NARROWNESS[prev_tier]:
                aggregated[k] = lst
                listing_tier_map[k] = tier

        last_tier = tier

        need = _effective_stop_floor(tier, settings=settings, confidence=norm.confidence)
        if len(aggregated) >= need:
            break

    list_out = list(aggregated.values())
    sorted_listings = sort_validated_listings_geo(
        list_out,
        store_by_domain=store_lookup,
        norm=norm,
        listing_tier_map=listing_tier_map,
        album_title=album_title or "",
        artist=parsed.artist,
    )

    head = sorted_listings[: settings.pipeline_max_results]
    scores = [
        composite_listing_rank(
            lst,
            store=resolve_store_for_url(str(lst.url), store_lookup),
            discovery_tier=listing_tier_map.get(normalize_url(str(lst.url)), last_tier),
            resolved_country=norm.resolved_country,
            resolved_city=norm.resolved_city,
            album_title=album_title or "",
            artist=parsed.artist,
        )
        for lst in head
    ]
    score_scale = max(scores) if scores else 1.0

    out = {
        "query": query,
        "results": [
            _listing_to_api_row(
                listing,
                norm=norm,
                store_by_domain=store_lookup,
                listing_tier_map=listing_tier_map,
                default_tier=last_tier,
                album_title=album_title or "",
                artist=parsed.artist,
                score_scale=score_scale,
            )
            for listing in head
        ],
    }

    await set_cached_search_payload(
        cache_key_value,
        out,
        ttl_seconds=settings.search_cache_ttl_seconds,
    )

    return out
