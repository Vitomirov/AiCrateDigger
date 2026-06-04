"""Host parsing, blacklist checks, and whitelist matching."""

from __future__ import annotations

from urllib.parse import urlparse

from app.domains.engine.search.prefilter.constants import (
    BLACKLIST_HOST_SUBSTRINGS,
    NEWS_HOST_SUBSTRINGS,
    WWW_PREFIX_RE,
)


def registrable_host(url: str) -> str | None:
    """Lowercase host with leading ``www.`` stripped, or ``None`` if unparseable."""
    if not url:
        return None
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except Exception:
        return None
    if not netloc:
        return None
    return WWW_PREFIX_RE.sub("", netloc).split(":", 1)[0]


def is_blacklisted(host: str) -> bool:
    """``True`` when ``host`` matches any blacklist or news-portal substring."""
    if not host:
        return True
    h = host.lower()
    if any(token in h for token in BLACKLIST_HOST_SUBSTRINGS):
        return True
    if any(token in h for token in NEWS_HOST_SUBSTRINGS):
        return True
    return False


def host_in_whitelist(host: str, whitelist: frozenset[str] | None) -> bool:
    """``host`` is a known store from ``whitelist_stores`` (subdomain-safe).

    Matches registrable host equality, subdomains of a whitelist entry
    (``shop.example.com`` when ``example.com`` is listed), and parent domains
    when discovery persisted a shop subdomain but Tavily returns the apex host.
    """
    if not host or not whitelist:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    if h in whitelist:
        return True
    for d in whitelist:
        dl = d.lower()
        if h.endswith("." + dl):
            return True
        if dl.endswith("." + h):
            return True
    return False
