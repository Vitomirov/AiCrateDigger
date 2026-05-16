"""Signals for prioritising bricks-and-mortar indie shops vs regional giants.

Used when ``search_scope == "local"`` or the parser anchored the query at city
granularity — relaxes brittle snippet gates and sorts output so curated
city-matched ``local_shop`` rows stay above mega-retail PDPs unless no such
hits validated at all.
"""

from __future__ import annotations

from typing import Any

from app.policies.eu_stores import StoreEntry
from app.policies.geo_proximity import cities_match
from app.policies.geo_scope import GeoIntent, NormalizedGeoIntent
from app.policies.listing_rank import resolve_store_for_url
from app.validators.listings import normalize_whitelist_domain


def should_prioritize_physical_local_shops(geo: GeoIntent, norm: NormalizedGeoIntent) -> bool:
    """True when UX should aggressively favour target-area ``local_shop`` rows."""
    if geo.search_scope == "local":
        return True
    return norm.granularity == "city" and bool(norm.resolved_city and norm.resolved_country)


def curated_city_local_shop_domains(
    stores: tuple[StoreEntry, ...],
    norm: NormalizedGeoIntent,
) -> frozenset[str]:
    """Normalized whitelist domains for city-matching ``local_shop`` catalogue rows.

    Passed into :func:`app.llm.extract_listings.pipeline.extract_listings` so fuzzy
    prefilter / merge gates do not silently drop thin indie snippets while stricter
    mega-retailer HTML still passes unchanged.
    """
    if not norm.resolved_city or not norm.resolved_country:
        return frozenset()
    cc = norm.resolved_country.strip().upper()
    out: set[str] = set()
    for s in stores:
        if getattr(s, "store_type", None) != "local_shop" or not s.domain:
            continue
        if (s.country_code or "").strip().upper() != cc:
            continue
        if not s.city or not cities_match(norm.resolved_city, s.city):
            continue
        out.add(normalize_whitelist_domain(s.domain))
    return frozenset(out)


def qualifies_as_target_city_local_shop(
    *,
    listing_url: str,
    store_lookup: dict[str, StoreEntry],
    norm: NormalizedGeoIntent,
) -> bool:
    """Does this validated URL map to a curated indie in the target city/country?"""
    if not listing_url.strip():
        return False
    st = resolve_store_for_url(listing_url, store_lookup)
    if st is None or getattr(st, "store_type", None) != "local_shop":
        return False
    hq = (st.country_code or "").strip().upper()
    tgt = (norm.resolved_country or "").strip().upper()
    if not hq or not tgt or hq != tgt:
        return False
    if norm.granularity == "city" and norm.resolved_city:
        return bool(st.city and cities_match(norm.resolved_city, st.city))
    # Country-only local-first: same-country curated locals still count.
    return True


def pool_has_qualifying_physical_local_row(
    listings: list[Any],
    *,
    store_lookup: dict[str, StoreEntry],
    norm: NormalizedGeoIntent,
) -> bool:
    return any(
        qualifies_as_target_city_local_shop(
            listing_url=str(getattr(lst, "url", "") or ""),
            store_lookup=store_lookup,
            norm=norm,
        )
        for lst in listings
    )
