"""PDP URL heuristics for prefilter and extraction."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# When present, these suggest a buyable product URL.
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


def url_suggests_product_detail_page(url: str) -> bool:
    """Heuristic PDP URL — relaxes extract prefilter when True."""
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


def _url_has_product_hint(path: str) -> bool:
    pl = path.lower()
    if any(h in pl for h in _URL_PRODUCT_HINTS):
        return True
    if _URL_PRODUCT_SUFFIX_RE.search(pl):
        return True
    return False
