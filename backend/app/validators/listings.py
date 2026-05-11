"""Hard-rule validators for extracted Listing objects.

Strict checks when ``settings.debug`` is false. When ``debug`` is true, URL /
domain / non-empty title plus **rapidfuzz** partial_ratio ≥ 80 on artist & album
needles (when present).

Rejections log a deterministic ``reason``.
"""

from __future__ import annotations

import logging
import math
from urllib.parse import urlsplit

from rapidfuzz import fuzz

from app.config import get_settings
from app.domain.listing_schema import Listing
from app.policies.eu_stores import ALLOWED_STORES

logger = logging.getLogger(__name__)

_DEBUG_FUZZ_THRESHOLD = 80


def normalize_whitelist_domain(domain: str) -> str:
    s = domain.strip().lower()
    if s.startswith("www."):
        s = s[4:]
    return s


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
    logger.info("listing_validation", extra=payload)
    return False


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


def _fuzzy_needle_ok(needle: str, title: str) -> tuple[bool, int]:
    if not needle.strip():
        return True, 100
    score = int(fuzz.partial_ratio(needle.lower(), title.lower()))
    return score >= _DEBUG_FUZZ_THRESHOLD, score


def validate_listing(
    listing: Listing,
    *,
    allowed_domains: frozenset[str] | None = None,
) -> bool:
    """Return True iff validation passes. Always logs on reject in strict mode."""

    settings = get_settings()
    allowed = allowed_domains if allowed_domains is not None else default_allowed_domains()

    if settings.debug:
        ok, reason = _debug_minimal_pass(listing, allowed_domains=allowed)
        if not ok:
            return _reject(reason or "debug_minimal_fail", listing=listing)

        album_needle = (listing.validation_album or "").strip()
        artist_needle = (listing.validation_artist or "").strip()
        title = (listing.title or "").strip()

        if album_needle:
            ok_a, score_a = _fuzzy_needle_ok(album_needle, title)
            if not ok_a:
                return _reject(
                    "debug_album_fuzzy_below_threshold",
                    listing=listing,
                    extra={"needle": album_needle[:80], "partial_ratio": score_a, "min": _DEBUG_FUZZ_THRESHOLD},
                )
        if artist_needle:
            ok_ar, score_ar = _fuzzy_needle_ok(artist_needle, title)
            if not ok_ar:
                return _reject(
                    "debug_artist_fuzzy_below_threshold",
                    listing=listing,
                    extra={"needle": artist_needle[:80], "partial_ratio": score_ar, "min": _DEBUG_FUZZ_THRESHOLD},
                )

        logger.info(
            "listing_validation",
            extra={
                "stage": "validate_listing",
                "status": "accept_debug",
                "reason": "debug_fuzzy_minimal",
                "url": listing.url[:500],
            },
        )
        return True

    url = listing.url
    host = _normalize_listing_url_domain(url)
    if host is None or not _host_matches_store_whitelist(host, allowed):
        return _reject("domain_not_allowed", listing=listing, extra={"host": host})

    album_needle = listing.validation_album
    if not album_needle or album_needle.strip() == "":
        return _reject("missing_validation_album", listing=listing)
    title_lower = listing.title.lower()
    if album_needle.lower() not in title_lower:
        return _reject(
            "album_not_in_title",
            listing=listing,
            extra={"needle": album_needle[:80]},
        )
    artist_needle = listing.validation_artist
    if artist_needle is not None and artist_needle.lower() not in title_lower:
        return _reject(
            "artist_not_in_title",
            listing=listing,
            extra={"needle": artist_needle[:80]},
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

    if not url.startswith("http") or len(url) < 10:
        return _reject("url_short_or_not_http", listing=listing)

    return True
