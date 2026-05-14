"""Composite ranking for pipeline listings.

Final score is a deterministic, non-degenerate composite of five signals:

    semantic_match  — fuzzy artist/album hits in the listing title
    geo_proximity   — store HQ vs target country/city + region/borders
    vinyl_confidence— URL/title leans towards a vinyl PDP (vs CD/Book/Merch)
    store_quality   — priority × listing_quality + store_type bonus
    pdp_confidence  — URL structure looks like a product page + in-stock

Each signal is bounded so the **sum** has a known ceiling, enabling stable
[0..1] normalization across requests without min-max collapse to ``1.0``.

`composite_listing_rank` is kept as a thin wrapper for callers that only need
a scalar (e.g. the geo sort comparator), but `composite_listing_score` is the
authoritative entry point and returns a `ListingRankBreakdown` for debug tracing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from rapidfuzz import fuzz

from app.domain.listing_schema import Listing
from app.policies.eu_stores import StoreEntry
from app.policies.geo_proximity import geo_proximity_bonus
from app.policies.geo_scope import Tier, country_to_region, expand_ships_to
from app.validators.listings import url_suggests_product_detail_page

# Per-signal ceilings (in raw points). Sum = RANK_CEILING_TOTAL.
_W_SEMANTIC_MAX = 30.0       # album fuzzy + artist fuzzy
_W_GEO_MAX = 220.0           # geo_proximity_bonus (0..100) × 2.2 — heavy locality bias
_W_VINYL_MAX = 16.0          # URL + title format signals (vinyl/LP/12")
_W_STORE_QUALITY_MAX = 22.0  # priority + listing_quality
_W_PDP_MAX = 12.0            # PDP-shaped URL + in_stock
_W_STORE_TYPE_MAX = 60.0     # local_shop bonus — must clearly outrank regional
_W_TIER_MAX = 48.0           # discovery tier (city tier highest)

#: Local-First Strike: applied to ``total`` (sort key) only — keeps the normalized
#: 0..1 ``score_normalized`` informative for the rest of the field while pinning
#: indie locals to the top of any mixed pool.
_INDIE_LOCAL_CITY_BONUS: float = 500.0
#: Subtracted from non-``local_shop`` rows whenever the pool also contains a
#: qualified ``local_shop`` listing. Affects both ``total`` and ``score_normalized``.
_GIANT_PENALTY_POINTS: float = 80.0

RANK_CEILING_TOTAL: float = (
    _W_SEMANTIC_MAX
    + _W_GEO_MAX
    + _W_VINYL_MAX
    + _W_STORE_QUALITY_MAX
    + _W_PDP_MAX
    + _W_STORE_TYPE_MAX
    + _W_TIER_MAX
)

_TIER_WEIGHT: dict[Tier, float] = {
    "city": _W_TIER_MAX,
    "country": _W_TIER_MAX * 0.79,
    "region": _W_TIER_MAX * 0.54,
    "continental": _W_TIER_MAX * 0.29,
    "global": _W_TIER_MAX * 0.08,
}

# Higher = stronger boost for the listing. local_shop > regional > marketplace.
_STORE_TYPE_WEIGHT: dict[str, float] = {
    "local_shop": _W_STORE_TYPE_MAX,            # 60
    "regional_ecommerce": _W_STORE_TYPE_MAX / 3.0,  # 20
    "marketplace": 0.0,
}

# Token signals in URL path / title that indicate a vinyl PDP.
_VINYL_URL_HINTS: tuple[str, ...] = (
    "/vinyl",
    "/vinyl-records",
    "/lp/",
    "/lps/",
    "/12-inch",
    "/12inch",
    "/12-",
    "/album/",
    "/albums/",
)
_VINYL_TITLE_HINTS: tuple[str, ...] = (
    "vinyl",
    "[lp]",
    "(lp)",
    " lp ",
    "12\"",
    '12"',
    "12''",
    "180g",
    "180 gram",
    "180-gram",
    "half-speed",
    "gatefold",
    "double lp",
    "2lp",
    "reissue",
)
_VINYL_TITLE_RE = re.compile(r"\b(lp|ep)\b", re.IGNORECASE)

# Non-vinyl format tokens — penalize when present without a positive vinyl hint.
_NON_VINYL_HINTS: tuple[str, ...] = (
    "cd box",
    " cd ",
    " cd,",
    " cd]",
    " cd)",
    "compact disc",
    "blu-ray",
    "bluray",
    " dvd",
    " mp3",
    "digital download",
    "cassette",
    " tape ",
    " book",
    "hardcover",
    "paperback",
    "poster",
    "lithograph",
    "t-shirt",
    "tshirt",
    "tote bag",
    "slipmat",
    "stylus",
    "turntable",
)


@dataclass(frozen=True, slots=True)
class ListingRankBreakdown:
    total: float
    score_normalized: float
    semantic_match: float
    geo_proximity: float
    vinyl_confidence: float
    store_quality: float
    pdp_confidence: float
    store_type_weight: float
    tier_weight: float
    discovery_tier: Tier
    store_type: str
    store_domain: str | None
    geo_country_match: bool
    geo_city_match: bool

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def _album_fuzz(listing_title: str, album_title: str) -> float:
    if not album_title.strip():
        return 0.0
    t = (listing_title or "").strip().lower()
    a = album_title.strip().lower()
    if not t:
        return 0.0
    return float(max(fuzz.partial_ratio(a, t), fuzz.token_set_ratio(a, t)))


def _artist_fuzz(listing_title: str, artist: str | None) -> float:
    if not (artist or "").strip():
        return 0.0
    t = (listing_title or "").strip().lower()
    a = artist.strip().lower()
    if not t:
        return 0.0
    return float(max(fuzz.partial_ratio(a, t), fuzz.token_set_ratio(a, t)))


#: Floor applied to whitelisted ``local_shop`` rows: an indie record store on
#: our DB allowlist IS a vinyl-selling business by definition, so the URL
#: heuristics should never zero its vinyl signal.
_LOCAL_SHOP_VINYL_FLOOR: float = 0.6  # × _W_VINYL_MAX


def _vinyl_confidence(url: str, title: str, *, is_local_shop: bool = False) -> float:
    """Soft 0..`_W_VINYL_MAX` score that listing is a vinyl PDP, not a sibling format.

    When ``is_local_shop`` is True (host resolved to a curated ``local_shop``
    entry in ``whitelist_stores``), the negative-keyword penalty is skipped and
    the score is floored at :data:`_LOCAL_SHOP_VINYL_FLOOR` × :data:`_W_VINYL_MAX`.
    Rationale: the store IS a record shop because it's in our whitelist; an
    unfortunate listing title (e.g. a long catalogue number) must not nuke the
    vinyl signal that the URL hint and store identity both already provide.
    """
    path = ""
    try:
        path = (urlsplit(url).path or "").lower()
    except Exception:
        path = ""
    t = (title or "").lower()
    score = 0.0

    if any(h in path for h in _VINYL_URL_HINTS):
        score += 6.0
    title_hits = sum(1 for h in _VINYL_TITLE_HINTS if h in t)
    if title_hits:
        score += min(6.0, 2.0 * title_hits)
    if _VINYL_TITLE_RE.search(t):
        score += 2.0

    has_pos = any(h in t for h in ("vinyl", "lp", '12"', "180g")) or any(
        h in path for h in _VINYL_URL_HINTS
    )
    if not has_pos and any(h in t for h in _NON_VINYL_HINTS):
        # Indie locals are vinyl-by-definition: skip the negative gate for them.
        if not is_local_shop:
            score -= 6.0

    if is_local_shop:
        score = max(score, _W_VINYL_MAX * _LOCAL_SHOP_VINYL_FLOOR)
    return max(0.0, min(_W_VINYL_MAX, score))


def composite_listing_score(
    listing: Listing,
    *,
    store: StoreEntry | None,
    discovery_tier: Tier,
    resolved_country: str | None,
    resolved_city: str | None,
    album_title: str,
    artist: str | None,
    local_present_in_pool: bool = False,
    album_match_confirmed: bool = True,
) -> ListingRankBreakdown:
    """Authoritative scorer. Returns the full breakdown for debug tracing.

    Local-First Strike behaviour:

    * When ``store.store_type == "local_shop"`` AND it city-matches the resolved
      city AND ``album_match_confirmed`` is True, :data:`_INDIE_LOCAL_CITY_BONUS`
      is added to ``total`` (sort key only). The confirmation gate prevents the
      bonus from firing on a wrong-album local listing (e.g. an indie page
      named "Spacekid" surfacing for a query about "Andrew Red Hand").
    * When ``local_present_in_pool`` is True and this row is NOT a local_shop,
      :data:`_GIANT_PENALTY_POINTS` is subtracted from both ``total`` and the
      normalized score so giants visibly fall below the indies.
    * ``_vinyl_confidence`` runs in local-shop mode for whitelisted indies: it
      skips the negative-keyword penalty and floors the signal.
    """
    # --- geo proximity (0..100) → weighted to ceiling ---
    geo_raw = 0.0
    geo_country_match = False
    geo_city_match = False
    if store is not None:
        ships = expand_ships_to(store.ships_to)
        tgt_reg = country_to_region(resolved_country) if resolved_country else None
        st_reg = country_to_region(store.country_code) if store.country_code else None
        geo_raw = geo_proximity_bonus(
            store_country=store.country_code,
            store_city=store.city,
            store_commerce_region=st_reg,
            target_country=resolved_country,
            target_city=resolved_city,
            target_commerce_region=tgt_reg,
            ships_expanded=ships,
            store_type=store.store_type,
        )
        sc = (store.country_code or "").strip().upper()
        tc = (resolved_country or "").strip().upper()
        geo_country_match = bool(sc and tc and sc == tc)
        geo_city_match = bool(
            geo_country_match
            and resolved_city
            and store.city
            and (store.city.strip().lower() == resolved_city.strip().lower())
        )
    geo_w = (geo_raw / 100.0) * _W_GEO_MAX

    # --- semantic match (album 0.65 + artist 0.35) → 0..30 ---
    alb = _album_fuzz(listing.title or "", album_title)
    art = _artist_fuzz(listing.title or "", artist)
    if (artist or "").strip():
        sem_raw = (alb * 0.65 + art * 0.35) / 100.0
    else:
        sem_raw = alb / 100.0
    semantic_w = sem_raw * _W_SEMANTIC_MAX

    # --- vinyl confidence (0..16) ---
    st_key_early = store.store_type if store is not None else "regional_ecommerce"
    vinyl_w = _vinyl_confidence(
        str(getattr(listing, "url", "") or ""),
        listing.title or "",
        is_local_shop=(st_key_early == "local_shop"),
    )

    # --- store quality (priority + listing_quality) ---
    pri = float(store.priority if store is not None else 0)
    lq = float(getattr(store, "listing_quality", 0) if store is not None else 0)
    sq_raw = (pri / 10.0) * (_W_STORE_QUALITY_MAX * 0.55) + (lq / 10.0) * (_W_STORE_QUALITY_MAX * 0.45)
    store_quality_w = max(0.0, min(_W_STORE_QUALITY_MAX, sq_raw))

    # --- PDP confidence (URL + in_stock) ---
    pdp_w = 0.0
    if url_suggests_product_detail_page(str(getattr(listing, "url", "") or "")):
        pdp_w += _W_PDP_MAX * 0.7
    if listing.in_stock is True:
        pdp_w += _W_PDP_MAX * 0.25
    elif listing.in_stock is None:
        pdp_w += _W_PDP_MAX * 0.08
    pdp_w = min(_W_PDP_MAX, pdp_w)

    # --- store type / tier ---
    st_key = store.store_type if store is not None else "regional_ecommerce"
    type_w = _STORE_TYPE_WEIGHT.get(st_key, 0.0)
    tier_w = _TIER_WEIGHT.get(discovery_tier, _W_TIER_MAX * 0.29)

    base_total = semantic_w + geo_w + vinyl_w + store_quality_w + pdp_w + type_w + tier_w

    # Giant penalty: applied when an indie local is present in the same pool —
    # affects BOTH the visible normalized score and the sort key.
    is_local = st_key == "local_shop"
    penalized_total = base_total
    if local_present_in_pool and not is_local:
        penalized_total = base_total - _GIANT_PENALTY_POINTS

    score_norm = max(0.0, min(1.0, penalized_total / RANK_CEILING_TOTAL))

    # Indie city bonus: applied to ``total`` only so the visible normalized
    # score stays informative across the indie cohort while the sort tier-bumps
    # any city-matched local_shop above giants. GATED on
    # ``album_match_confirmed`` so an indie row with a confusing / off-target
    # title cannot ride the +500 bonus to the top of the response.
    total = penalized_total
    if is_local and geo_city_match and album_match_confirmed:
        total = penalized_total + _INDIE_LOCAL_CITY_BONUS

    return ListingRankBreakdown(
        total=round(total, 3),
        score_normalized=round(score_norm, 4),
        semantic_match=round(semantic_w, 3),
        geo_proximity=round(geo_w, 3),
        vinyl_confidence=round(vinyl_w, 3),
        store_quality=round(store_quality_w, 3),
        pdp_confidence=round(pdp_w, 3),
        store_type_weight=round(type_w, 3),
        tier_weight=round(tier_w, 3),
        discovery_tier=discovery_tier,
        store_type=st_key,
        store_domain=(store.domain if store is not None else None),
        geo_country_match=geo_country_match,
        geo_city_match=geo_city_match,
    )


def composite_listing_rank(
    listing: Listing,
    *,
    store: StoreEntry | None,
    discovery_tier: Tier,
    resolved_country: str | None,
    resolved_city: str | None,
    album_title: str,
    artist: str | None,
    local_present_in_pool: bool = False,
    album_match_confirmed: bool = True,
) -> float:
    """Back-compat scalar (used by `sort_validated_listings_geo`)."""
    return composite_listing_score(
        listing,
        store=store,
        discovery_tier=discovery_tier,
        resolved_country=resolved_country,
        resolved_city=resolved_city,
        album_title=album_title,
        artist=artist,
        local_present_in_pool=local_present_in_pool,
        album_match_confirmed=album_match_confirmed,
    ).total


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
