"""Composite ranking for pipeline listings (semantic + geo + store + PDP hints).

Keeps Tavily/LLM stages unchanged; only affects ordering of accepted rows.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from rapidfuzz import fuzz

from app.domain.listing_schema import Listing
from app.policies.eu_stores import StoreEntry
from app.policies.geo_proximity import geo_proximity_bonus
from app.policies.geo_scope import Tier, country_to_region, expand_ships_to
from app.validators.listings import url_suggests_product_detail_page

_TIER_WEIGHT: dict[Tier, float] = {
    "city": 48.0,
    "country": 38.0,
    "region": 26.0,
    "continental": 14.0,
    "global": 4.0,
}

_TYPE_WEIGHT: dict[str, float] = {
    "local_shop": 12.0,
    "regional_ecommerce": 4.0,
    "marketplace": 0.0,
}


def _album_fuzz(listing_title: str, album_title: str) -> float:
    if not album_title.strip():
        return 0.0
    t = (listing_title or "").strip().lower()
    a = album_title.strip().lower()
    if not t:
        return 0.0
    return float(max(fuzz.partial_ratio(a, t), fuzz.token_set_ratio(a, t)))


def composite_listing_rank(
    listing: Listing,
    *,
    store: StoreEntry | None,
    discovery_tier: Tier,
    resolved_country: str | None,
    resolved_city: str | None,
    album_title: str,
    artist: str | None,
) -> float:
    """Higher is better. Used for final global sort before truncation."""
    geo_b = 0.0
    ships: frozenset[str] = frozenset()
    st_cc = None
    st_city = None
    st_reg = None
    tgt_reg = country_to_region(resolved_country) if resolved_country else None
    if store is not None:
        ships = expand_ships_to(store.ships_to)
        st_cc = store.country_code
        st_city = store.city
        st_reg = country_to_region(store.country_code)
        geo_b = geo_proximity_bonus(
            store_country=st_cc,
            store_city=st_city,
            store_commerce_region=st_reg,
            target_country=resolved_country,
            target_city=resolved_city,
            target_commerce_region=tgt_reg,
            ships_expanded=ships,
        )

    tier_w = _TIER_WEIGHT.get(discovery_tier, 4.0)
    st_key = store.store_type if store is not None else "regional_ecommerce"
    type_w = _TYPE_WEIGHT.get(st_key, 0.0)

    pri = float(store.priority if store is not None else 0)
    lq = float(getattr(store, "listing_quality", 5) if store is not None else 0) * 0.35

    alb = _album_fuzz(listing.title or "", album_title) * 0.28
    art = 0.0
    if (artist or "").strip():
        art = float(
            max(
                fuzz.partial_ratio(artist.strip().lower(), (listing.title or "").lower()),
                fuzz.token_set_ratio(artist.strip().lower(), (listing.title or "").lower()),
            )
        ) * 0.12

    stock = 0.0
    if listing.in_stock is True:
        stock = 6.0
    elif listing.in_stock is None:
        stock = 2.0

    pdp = 8.0 if url_suggests_product_detail_page(listing.url) else 0.0

    return (
        geo_b * 1.35
        + tier_w
        + type_w
        + pri * 0.45
        + lq
        + alb
        + art
        + stock
        + pdp
    )


def resolve_store_for_url(url: str, store_by_domain: dict[str, StoreEntry]) -> StoreEntry | None:
    try:
        host = urlsplit(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        host = ""

    st = store_by_domain.get(host)
    if st is not None:
        return st
    for dom, row in store_by_domain.items():
        if host == dom or host.endswith("." + dom):
            return row
    return None
