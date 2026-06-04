"""URL path signals and Tavily row scoring for pre-filter ranking."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.domains.engine.search.prefilter.constants import EDITORIAL_PATH_SUBSTRINGS
from app.domains.engine.search.scoring import product_signal_multiplier
from app.domains.engine.validators.listings import url_suggests_product_detail_page


def path_looks_editorial(url: str) -> bool:
    try:
        path = (urlparse(url).path or "").lower()
    except Exception:
        return False
    return any(token in path for token in EDITORIAL_PATH_SUBSTRINGS)


def looks_like_product_url(url: str) -> bool:
    """PDP-shaped URL heuristic for *unknown* hosts (not in the DB whitelist).

    Two independent signals — either is sufficient:
    * :func:`url_suggests_product_detail_page` already used by the validator,
      catches ``/product/``, ``/products/``, ``/p/``, ``/item/``, ``-p-1234``,
      slug-id suffixes, etc.
    * :func:`product_signal_multiplier` ≥ 1.0 → URL path hits a retail keyword.
    """
    if url_suggests_product_detail_page(url):
        return True
    try:
        path = (urlparse(url).path or "/")
    except Exception:
        return False
    return product_signal_multiplier(path) >= 1.0


def result_score(row: dict[str, Any], *, is_known_shop: bool) -> float:
    """Tavily ``score`` boosted by the PDP-vs-editorial path signal.

    Whitelisted shop hosts get a flat boost so a slightly-lower Tavily score
    on a known indie doesn't lose to a high-score editorial URL on an unknown
    host.
    """
    base = 0.0
    try:
        base = float(row.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    url = str(row.get("url") or "")
    try:
        path = (urlparse(url).path or "/")
    except Exception:
        path = "/"
    mult = product_signal_multiplier(path)
    if path_looks_editorial(url):
        mult = min(mult, 0.35)
    score = base * max(mult, 0.05)
    if is_known_shop:
        score = max(score * 1.25, score + 0.10)
    return score
