"""Commerce geography for progressive store widening and geo-aware ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.domain.parse_schema import ParsedQuery

from app.policies.eu_stores import StoreEntry
from app.policies.geo_proximity import cities_match
from app.services.tavily_service import normalize_url

SearchScope = Literal["local", "regional", "global"]
Region = Literal[
    "balkans",
    "central_europe",
    "western_europe",
    "southern_europe",
    "nordics",
    "uk",
    "baltics",
]
Tier = Literal["city", "country", "region", "continental", "global"]

#: Ordering index: narrower phases sort first when merging duplicate URLs.
TIER_NARROWNESS: dict[Tier, int] = {
    "city": 0,
    "country": 1,
    "region": 2,
    "continental": 3,
    "global": 4,
}

#: Commerce regions used for store filtering and fallback. Country-level only.
COUNTRY_TO_REGION: dict[str, Region] = {
    "RS": "balkans",
    "HR": "balkans",
    "SI": "balkans",
    "BA": "balkans",
    "ME": "balkans",
    "MK": "balkans",
    "AL": "balkans",
    "XK": "balkans",
    "BG": "balkans",
    "DE": "central_europe",
    "AT": "central_europe",
    "CH": "central_europe",
    "CZ": "central_europe",
    "SK": "central_europe",
    "HU": "central_europe",
    "PL": "central_europe",
    "RO": "central_europe",
    "FR": "western_europe",
    "NL": "western_europe",
    "BE": "western_europe",
    "LU": "western_europe",
    "IE": "western_europe",
    "GB": "uk",
    "IT": "southern_europe",
    "ES": "southern_europe",
    "PT": "southern_europe",
    "GR": "southern_europe",
    "MT": "southern_europe",
    "CY": "southern_europe",
    "SE": "nordics",
    "NO": "nordics",
    "DK": "nordics",
    "FI": "nordics",
    "IS": "nordics",
    "EE": "baltics",
    "LV": "baltics",
    "LT": "baltics",
}

#: EU member states for ``ships_to`` expansion (``"EU"`` token).
EU_MEMBER_CODES: frozenset[str] = frozenset(
    {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "SE", "ES",
    }
)


def country_to_region(country_code: str | None) -> Region | None:
    """Country (ISO-2) → commerce region (or ``None`` when unknown)."""
    if not country_code:
        return None
    cc = country_code.strip().upper()
    if cc == "UK":
        cc = "GB"
    return COUNTRY_TO_REGION.get(cc)


@dataclass(frozen=True, slots=True)
class GeoIntent:
    """Resolved geography for store filtering (never mixed into Tavily query text)."""

    search_scope: SearchScope
    raw_location: str | None
    country_code: str | None
    region: Region | None


@dataclass(frozen=True, slots=True)
class NormalizedGeoIntent:
    """Parser-normalized geo with confidence — controls widening + ranking."""

    raw_location: str | None
    resolved_city: str | None
    resolved_country: str | None
    confidence: float
    granularity: Literal["city", "country", "region", "global", "none"]


def geo_intent_from_parsed(parsed: "ParsedQuery") -> GeoIntent:
    """Map :class:`ParsedQuery` to the legacy :class:`GeoIntent` struct."""
    cc = parsed.country_code if parsed.country_code else None
    region = country_to_region(cc) if cc else None
    return GeoIntent(
        search_scope=parsed.search_scope,
        raw_location=parsed.location,
        country_code=cc,
        region=region,
    )


def normalized_geo_from_parsed(parsed: "ParsedQuery") -> NormalizedGeoIntent:
    """Build strict locality signals from parser output and safe defaults."""
    cc = parsed.country_code if parsed.country_code else None
    city = (parsed.resolved_city or "").strip() or None
    conf_raw = parsed.geo_confidence
    if conf_raw is None:
        conf = 0.82 if (parsed.location or "").strip() else 1.0
    else:
        conf = float(conf_raw)
    conf = max(0.0, min(1.0, conf))

    gran: Literal["city", "country", "region", "global", "none"]
    g = parsed.geo_granularity
    if parsed.search_scope == "global":
        gran = "global"
    elif g == "region" or parsed.search_scope == "regional":
        gran = "region"
    elif g in ("city", "country"):
        gran = g
    elif cc and city:
        gran = "city"
    elif cc:
        gran = "country"
    elif g == "none":
        gran = "none"
    else:
        gran = "global"

    return NormalizedGeoIntent(
        raw_location=parsed.location,
        resolved_city=city,
        resolved_country=cc,
        confidence=conf,
        granularity=gran,
    )


def expand_ships_to(ships_to: tuple[str, ...]) -> frozenset[str]:
    """Expand magic ``EU`` token into member ISO codes; keep explicit ISO-2 codes."""
    out: set[str] = set()
    for s in ships_to:
        u = s.strip().upper()
        if u == "EU":
            out |= set(EU_MEMBER_CODES)
        elif len(u) == 2 and u.isalpha():
            out.add(u)
    return frozenset(out)


def store_serves_country(store: "StoreEntry", target_cc: str) -> bool:
    """True if the store HQ matches the country or its shipping covers it."""
    tc = target_cc.strip().upper()
    if not tc:
        return False
    hq = (store.country_code or "").strip().upper()
    if hq == tc:
        return True
    return tc in expand_ships_to(store.ships_to)


def store_in_region(store: "StoreEntry", region: Region) -> bool:
    if not region:
        return False
    if store.region and store.region.strip().lower() == region:
        return True
    hq = (store.country_code or "").strip().upper()
    if hq and country_to_region(hq) == region:
        return True
    ships = expand_ships_to(store.ships_to)
    for cc in ships:
        if country_to_region(cc) == region:
            return True
    return False


def tier_fallback_order(geo: GeoIntent, norm: NormalizedGeoIntent) -> tuple[Tier, ...]:
    """Progressive widen: city → country (HQ-only) → region → continental; global is pipeline-only."""
    if geo.search_scope == "global" or norm.granularity == "global":
        return ("continental",)
    if geo.search_scope == "regional" or norm.granularity == "region":
        if geo.region:
            return ("region", "continental")
        return ("continental",)
    if geo.country_code:
        use_city = (
            norm.granularity == "city"
            and bool(norm.resolved_city)
            and norm.confidence >= 0.4
        )
        if use_city:
            return ("city", "country", "region", "continental")
        return ("country", "region", "continental")
    return ("continental",)


def store_matches_rs_expanded_country_tier(store: "StoreEntry") -> bool:
    """RS local-first pool: home market, Balkan neighbours, EU shippers."""
    if store_serves_country(store, "RS"):
        return True
    if store_in_region(store, "balkans"):
        return True
    raw = tuple(x.strip().upper() for x in store.ships_to)
    if "RS" in raw or "EU" in raw:
        return True
    return False


def filter_stores_for_tier(
    stores: tuple["StoreEntry", ...],
    geo: GeoIntent,
    tier: Tier,
    *,
    norm: NormalizedGeoIntent,
) -> tuple["StoreEntry", ...]:
    """Restrict the active catalogue to the current widening phase."""
    active = tuple(s for s in stores if s.is_active)
    if tier == "global":
        return active
    if tier == "continental":
        return active
    if tier == "region":
        if not geo.region:
            return ()
        return tuple(s for s in active if store_in_region(s, geo.region))
    if tier == "city":
        # LOCAL-FIRST STRIKE: the city tier is strictly indie. A store enters this
        # pool only when it is a ``local_shop`` AND its `city` fuzzy-matches the
        # resolved query city. Regional ecommerce / marketplaces are deliberately
        # withheld for the country/region fallback, even when they happen to be
        # HQ'd in the target city (Juno London, HHV Berlin, …).
        if not norm.resolved_country or not norm.resolved_city:
            return ()
        cc = norm.resolved_country.upper()
        out: list["StoreEntry"] = []
        for s in active:
            if (s.country_code or "").upper() != cc:
                continue
            if s.store_type != "local_shop":
                continue
            if not s.city or not cities_match(norm.resolved_city, s.city):
                continue
            out.append(s)
        return tuple(out)
    # country — HQ match only (RS keeps expanded neighbourhood semantics).
    if not geo.country_code:
        return ()
    hq_cc = geo.country_code.upper()
    if hq_cc == "RS":
        return tuple(s for s in active if store_matches_rs_expanded_country_tier(s))
    return tuple(s for s in active if (s.country_code or "").upper() == hq_cc)


def _rs_country_sort_key(s: "StoreEntry") -> tuple[int, int, int, str]:
    cc = (s.country_code or "").upper()
    if cc == "RS":
        rk = 0
    elif s.region and s.region.lower() == "balkans":
        rk = 1
    elif "RS" in tuple(x.strip().upper() for x in s.ships_to):
        rk = 2
    elif "EU" in tuple(x.strip().upper() for x in s.ships_to):
        rk = 3
    else:
        rk = 4
    return (rk, -s.priority, -s.listing_quality, s.domain)


def sort_stores_for_tier(
    stores: tuple["StoreEntry", ...],
    geo: GeoIntent,
    tier: Tier,
    *,
    norm: NormalizedGeoIntent,
) -> tuple["StoreEntry", ...]:
    """Prefer geo-relevant rows inside a phase (for Tavily domain caps)."""
    if tier == "country" and (geo.country_code or "").upper() == "RS":
        return tuple(sorted(stores, key=_rs_country_sort_key))

    if tier == "city":

        def city_key(s: "StoreEntry") -> tuple[int, int, int, int, str]:
            cm = (
                0
                if (norm.resolved_city and s.city and cities_match(norm.resolved_city, s.city))
                else 1
            )
            st = 0 if s.store_type == "local_shop" else 1
            return (cm, st, -s.priority, -s.listing_quality, s.domain)

        return tuple(sorted(stores, key=city_key))

    def key(s: "StoreEntry") -> tuple[int | float, ...]:
        if tier == "country":
            return (-s.priority, -s.listing_quality, 0, s.domain)
        if tier == "region":
            in_country = (
                0 if (geo.country_code and (s.country_code or "").upper() == geo.country_code.upper()) else 1
            )
            return (in_country, -s.priority, -s.listing_quality, s.domain)
        boost_region = 0 if (geo.region and store_in_region(s, geo.region)) else 1
        boost_country = (
            0 if (geo.country_code and (s.country_code or "").upper() == geo.country_code.upper()) else 1
        )
        return (boost_country, boost_region, -s.priority, -s.listing_quality, s.domain)

    return tuple(sorted(stores, key=key))


def cap_stores(
    sorted_stores: tuple["StoreEntry", ...],
    *,
    max_domains: int,
) -> tuple["StoreEntry", ...]:
    if max_domains <= 0:
        return ()
    return sorted_stores[:max_domains]


def max_domains_for_tier(
    tier: Tier,
    *,
    local_max: int,
    regional_max: int,
    global_max: int,
) -> int:
    if tier in ("city", "country"):
        return local_max
    if tier in ("region", "continental"):
        return regional_max
    return global_max


def sort_validated_listings_geo(
    listings: list[object],
    *,
    store_by_domain: dict[str, "StoreEntry"],
    norm: NormalizedGeoIntent,
    listing_tier_map: dict[str, Tier],
    album_title: str,
    artist: str | None,
    local_present_in_pool: bool = False,
    album_match_by_url: dict[str, bool] | None = None,
    prioritize_physical_city_locals: bool = False,
) -> list[object]:
    """Global ordering by composite rank (geo-heavy).

    ``local_present_in_pool`` activates the "giant penalty" inside
    :func:`app.policies.listing_rank.composite_listing_rank` so non-local
    stores fall below any qualified indie local result.

    ``album_match_by_url`` (when provided) gates the city-indie +500 bonus:
    only rows whose URL maps to ``True`` are bonused. Unknown / missing keys
    default to ``True`` (preserve backward-compat for callers that skip the
    LLM verifier).

    ``prioritize_physical_city_locals`` — target city/country ``local_shop`` rows are
    ordered strictly ahead of giants when locality-first UX applies.
    """
    from app.policies.physical_local import qualifies_as_target_city_local_shop
    from app.policies.listing_rank import composite_listing_rank, resolve_store_for_url

    match_lookup = album_match_by_url or {}

    def sort_key(lst: object) -> tuple[int, float, str]:
        u = str(getattr(lst, "url", "") or "")
        nk = normalize_url(u)
        tier = listing_tier_map.get(nk, "continental")
        st = resolve_store_for_url(u, store_by_domain)
        if not hasattr(lst, "title"):
            return (1, 0.0, u)
        confirmed = match_lookup.get(u, match_lookup.get(nk, True))
        composite = composite_listing_rank(
            lst,  # type: ignore[arg-type]
            store=st,
            discovery_tier=tier,
            resolved_country=norm.resolved_country,
            resolved_city=norm.resolved_city,
            album_title=album_title,
            artist=artist,
            local_present_in_pool=local_present_in_pool,
            album_match_confirmed=confirmed,
        )
        phys_bucket = (
            0
            if (
                prioritize_physical_city_locals
                and qualifies_as_target_city_local_shop(
                    listing_url=u,
                    store_lookup=store_by_domain,
                    norm=norm,
                )
            )
            else 1
        )
        return (phys_bucket, -composite, u)

    return sorted(listings, key=sort_key)
