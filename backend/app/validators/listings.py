"""Hard-rule validators for extracted Listing objects.

Production strict mode: whitelist domain, reject known non-product URLs/titles, then
RapidFuzz **token_set_ratio** / **partial_ratio** on album (and artist when set).

Debug mode: looser fuzz floor via ``listing_validation_debug_album_fuzz_min``.

Rejections log a deterministic ``reason``.
"""

from __future__ import annotations

import logging
import math
import re
from urllib.parse import urlparse, urlsplit

from rapidfuzz import fuzz

from app.config import get_settings
from app.domain.listing_schema import Listing
from app.policies.eu_stores import ALLOWED_STORES
from app.policies.store_domain import canonical_store_domain

logger = logging.getLogger(__name__)

# Title phrases typical of hubs, blogs, gift guides (not SKU rows).
_TITLE_NON_PRODUCT_SNIPPETS: tuple[str, ...] = (
    "staff pick",
    "staff picks",
    "editor's pick",
    "editors pick",
    "gift guide",
    "buyer's guide",
    "buyers guide",
    "music news",
    "interview with",
    "essential metal",
    "essential rock",
    "top metal albums",
    "top rock albums",
    "best metal albums",
    "best rock albums",
    "our favourite albums",
    "our favorite albums",
    "our vinyl collection",
    "complete vinyl collection",
    "recommendation",
    "recommendations",
    "vinyl playlist",
    "browse our",
    "just in:",
)

# Strict non-vinyl PDP rejection: physical objects that ARE listed on shops but
# are not the record itself. Checked with case-insensitive substring match.
_TITLE_NON_VINYL_PDP_TOKENS: tuple[str, ...] = (
    "anniversary book",
    "anniversary edition book",
    "coffee table book",
    "photo book",
    "photobook",
    "art book",
    "biography",
    "autobiography",
    "memoir",
    "hardcover",
    "softcover",
    "paperback",
    "art print",
    "art prints",
    "poster print",
    "poster set",
    "lithograph",
    "screen print",
    "screenprint",
    "tour poster",
    "tour programme",
    "tour program",
    "calendar",
    "puzzle",
    "jigsaw",
    "fridge magnet",
    "tote bag",
    "t-shirt",
    "tshirt",
    "hoodie",
    "sweatshirt",
    "longsleeve",
    "long sleeve",
    "cap",
    " mug",
    " mug,",
    "mug ",
    "enamel pin",
    "keychain",
    "keyring",
    "patch",
    "sticker pack",
    "slipmat",
    "stylus",
    "cartridge",
    "turntable",
    "record player",
    "cleaning kit",
    "vinyl sleeve",
    "vinyl sleeves",
    "outer sleeve",
    "inner sleeve",
    "official 50th anniversary",
    "50th anniversary book",
    "the official book",
    "official book of",
)

# Path fragments that usually indicate editorial / artist hubs (not PDPs).
_URL_HUB_SUBSTRINGS: tuple[str, ...] = (
    "/blog",
    "/blogs/news",
    "/blogs/post",
    "/blogs/article",
    "/news/",
    "/magazine/",
    "/mag/",
    "/editorial/",
    "/features/",
    "/feature/",
    "/story/",
    "/stories/",
    "/podcast/",
    "/playlists/",
    "/playlist/",
    "/tag/",
    "/tags/",
    "/articles/",
    "/article/",
    "/lists/",
    "/guides/",
    "/guide/",
    "/recommendations/",
    "/recommendation/",
    "/charts/",
    "/chart/",
    "/bestseller",
    "/best-of",
    "/best_of",
    "/search?",
    "/search/",
    "/category/",
    "/categories/",
    "/genre/",
    "/genres/",
    "/about",
    "/contact",
    "/help",
    "/faq",
    # Non-vinyl merch / book / poster catalogue branches:
    "/books/",
    "/book/",
    "/posters/",
    "/poster/",
    "/prints/",
    "/print/",
    "/merchandise/",
    "/merch/",
    "/clothing/",
    "/apparel/",
    "/accessories/",
    "/turntables/",
    "/equipment/",
    "/hardware/",
    "/dvd/",
    "/dvds/",
    "/blu-ray/",
    "/bluray/",
)

# Shallow artist/band/label index paths (no product slug).
_ARTIST_HUB_PATH_RE = re.compile(
    r"/(artist[s]?|band[s]?|label[s]?|genre[s]?)/[^/]+/?$",
    re.IGNORECASE,
)
# When present, these suggest a buyable product URL — skip hub rejection.
_URL_PRODUCT_HINTS: tuple[str, ...] = (
    "/product/",
    "/products/",
    "/p/",
    "/item/",
    "/items/",
    "/shop/",
    "/buy/",
    "add-to-cart",
    ".html",
)
_URL_PRODUCT_SUFFIX_RE = re.compile(r"-p-?\d+(?:[/?#]|$)", re.IGNORECASE)


def normalize_whitelist_domain(domain: str) -> str:
    """Host-only lowercase key for allowlists and ``include_domains`` (no path/scheme)."""
    return canonical_store_domain(domain)


def url_suggests_product_detail_page(url: str) -> bool:
    """Heuristic PDP URL — relaxes fuzzy gates and extract prefilter when True."""
    try:
        parsed = urlparse(url.strip())
        path = parsed.path or "/"
    except Exception:
        return False
    if _url_has_product_hint(path):
        return True
    pl = path.lower()
    if re.search(r"/[a-z0-9_-]{3,}-\d{3,}(?:/)?$", pl):
        return True
    if re.search(r"/\d{5,}(?:[/?#]|$)", pl):
        return True
    return False


def _normalize_listing_url_domain(url: str) -> str | None:
    """Lowercase host, strip leading ``www.``, URL trailing ``/`` ignored."""
    host = urlsplit(url.rstrip("/")).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def default_allowed_domains() -> frozenset[str]:
    return frozenset(normalize_whitelist_domain(s.domain) for s in ALLOWED_STORES if s.is_active)


def _host_matches_store_whitelist(host: str, allowed: frozenset[str]) -> bool:
    if not host:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    if h in allowed:
        return True
    return any(h.endswith("." + d) for d in allowed)


def _reject(reason: str, *, listing: Listing, extra: dict | None = None) -> bool:
    payload: dict = {
        "stage": "validate_listing",
        "status": "reject",
        "reason": reason,
        "url": listing.url[:500],
        "title": (listing.title or "")[:200],
    }
    if extra:
        payload["detail"] = extra
    logger.debug("listing_validation", extra=payload)
    return False


def _url_has_product_hint(path: str) -> bool:
    pl = path.lower()
    if any(h in pl for h in _URL_PRODUCT_HINTS):
        return True
    if _URL_PRODUCT_SUFFIX_RE.search(pl):
        return True
    return False


def _url_looks_non_product(url: str) -> bool:
    """True when the URL clearly points at a hub / blog / category page.

    A positive product hint short-circuits hub rejection (Shopify nests PDPs as
    ``/collections/<x>/products/<y>`` and product hints win).
    """
    try:
        parsed = urlparse(url.strip())
        path = parsed.path or "/"
    except Exception:
        return True

    pl = path.lower()
    has_product_hint = _url_has_product_hint(pl)

    if has_product_hint:
        return False
    if _ARTIST_HUB_PATH_RE.search(pl):
        return True
    if "/collections/" in pl and "/products/" not in pl:
        return True
    return any(fragment in pl for fragment in _URL_HUB_SUBSTRINGS)


def _title_looks_non_product(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    return any(s in t for s in _TITLE_NON_PRODUCT_SNIPPETS)


def _title_looks_non_vinyl_object(title: str) -> bool:
    """Reject titles that name a physical artifact other than the record itself.

    Examples: "Pink Floyd: The Dark Side of the Moon (50th Anniversary Book)",
    "Tour Poster", "Coffee Table Book", "Slipmat".
    """
    t = (title or "").strip().lower()
    if not t:
        return False
    return any(tok in t for tok in _TITLE_NON_VINYL_PDP_TOKENS)


def _fuzz_best_album_artist(needle: str, haystack: str) -> tuple[int, int, int]:
    """Returns (partial_ratio, token_set_ratio, best_of_two)."""
    if not needle.strip():
        return 100, 100, 100
    n = needle.lower()
    h = haystack.lower()
    pr = int(fuzz.partial_ratio(n, h))
    ts = int(fuzz.token_set_ratio(n, h))
    return pr, ts, max(pr, ts)


def _debug_minimal_pass(listing: Listing, *, allowed_domains: frozenset[str]) -> tuple[bool, str | None]:
    url = listing.url
    if not url.startswith("http") or len(url) < 10:
        return False, "url_short_or_not_http"
    host = _normalize_listing_url_domain(url)
    if host is None or not _host_matches_store_whitelist(host, allowed_domains):
        return False, "domain_not_allowed"
    if not (listing.title or "").strip():
        return False, "empty_title"
    return True, None


def validate_listing(
    listing: Listing,
    *,
    allowed_domains: frozenset[str] | None = None,
) -> bool:
    """Return True iff validation passes. Always logs on reject in strict mode."""

    settings = get_settings()
    allowed = allowed_domains if allowed_domains is not None else default_allowed_domains()
    album_min = int(settings.listing_validation_album_fuzz_min)
    artist_min = int(settings.listing_validation_artist_fuzz_min)
    relief = int(settings.listing_validation_pdp_fuzz_relief)
    debug_album_min = int(settings.listing_validation_debug_album_fuzz_min)

    url = listing.url
    title = (listing.title or "").strip()

    if settings.debug:
        ok, reason = _debug_minimal_pass(listing, allowed_domains=allowed)
        if not ok:
            return _reject(reason or "debug_minimal_fail", listing=listing)

        if _url_looks_non_product(url):
            return _reject("non_product_url", listing=listing)
        if _title_looks_non_product(title):
            return _reject("non_product_title", listing=listing)
        if _title_looks_non_vinyl_object(title):
            return _reject("non_vinyl_object_title", listing=listing, extra={"title": title[:160]})

        album_needle = (listing.validation_album or "").strip()
        artist_needle = (listing.validation_artist or "").strip()

        if album_needle:
            pr, ts, best = _fuzz_best_album_artist(album_needle, title)
            if best < debug_album_min:
                return _reject(
                    "debug_album_fuzzy_below_threshold",
                    listing=listing,
                    extra={
                        "needle": album_needle[:80],
                        "partial_ratio": pr,
                        "token_set_ratio": ts,
                        "best": best,
                        "min": debug_album_min,
                    },
                )
        if artist_needle:
            pr, ts, best = _fuzz_best_album_artist(artist_needle, title)
            if best < artist_min:
                return _reject(
                    "debug_artist_fuzzy_below_threshold",
                    listing=listing,
                    extra={
                        "needle": artist_needle[:80],
                        "partial_ratio": pr,
                        "token_set_ratio": ts,
                        "best": best,
                        "min": artist_min,
                    },
                )

        logger.debug(
            "listing_validation",
            extra={
                "stage": "validate_listing",
                "status": "accept_debug",
                "reason": "debug_fuzzy_gates",
                "url": listing.url[:500],
            },
        )
        return True

    host = _normalize_listing_url_domain(url)
    if host is None or not _host_matches_store_whitelist(host, allowed):
        return _reject("domain_not_allowed", listing=listing, extra={"host": host})

    if not url.startswith("http") or len(url) < 10:
        return _reject("url_short_or_not_http", listing=listing)

    if _url_looks_non_product(url):
        return _reject("non_product_url", listing=listing, extra={"url": url[:240]})

    if _title_looks_non_product(title):
        return _reject("non_product_title", listing=listing)

    if _title_looks_non_vinyl_object(title):
        return _reject("non_vinyl_object_title", listing=listing, extra={"title": title[:160]})

    album_needle = listing.validation_album
    if not album_needle or album_needle.strip() == "":
        return _reject("missing_validation_album", listing=listing)

    pdp = url_suggests_product_detail_page(url)
    eff_album_min = max(60, album_min - (relief if pdp else 0))
    eff_artist_min = max(55, artist_min - (relief if pdp else 0))

    pr_a, ts_a, best_album = _fuzz_best_album_artist(album_needle.strip(), title)
    if best_album < eff_album_min:
        return _reject(
            "album_fuzzy_below_threshold",
            listing=listing,
            extra={
                "needle": album_needle[:80],
                "partial_ratio": pr_a,
                "token_set_ratio": ts_a,
                "best": best_album,
                "min": eff_album_min,
                "pdp_relaxed": pdp,
            },
        )

    artist_needle = listing.validation_artist
    if artist_needle is not None and artist_needle.strip() != "":
        pr_ar, ts_ar, best_art = _fuzz_best_album_artist(artist_needle.strip(), title)
        if best_art < eff_artist_min:
            return _reject(
                "artist_fuzzy_below_threshold",
                listing=listing,
                extra={
                    "needle": artist_needle[:80],
                    "partial_ratio": pr_ar,
                    "token_set_ratio": ts_ar,
                    "best": best_art,
                    "min": eff_artist_min,
                    "pdp_relaxed": pdp,
                },
            )

    price = listing.price
    if price is None:
        pass
    elif isinstance(price, float):
        if math.isnan(price) or price < 0.0:
            return _reject("invalid_price", listing=listing, extra={"price": repr(price)})
    else:
        return _reject("invalid_price", listing=listing, extra={"price": repr(price)})

    cur = listing.currency or "EUR"
    if len(cur) != 3 or cur != cur.upper() or not cur.isascii() or not cur.isalpha():
        return _reject("invalid_currency", listing=listing, extra={"currency": cur})

    return True
