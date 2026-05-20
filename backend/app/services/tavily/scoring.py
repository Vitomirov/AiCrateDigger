"""Tavily result scoring heuristics (PDP vs editorial noise)."""

from __future__ import annotations

from urllib.parse import urlparse

from app.services.tavily.url_utils import normalize_url

_NON_RETAIL_PATH_SNIPPETS: tuple[str, ...] = (
    "/blog",
    "/blogs/",
    "/news/",
    "/nieuws/",
    "/magazine/",
    "/mag/",
    "/tag/",
    "/tags/",
    "/articles/",
    "/article/",
    "/editorial/",
    "/features/",
    "/story/",
    "/stories/",
    "/podcast/",
)

_RETAIL_PATH_BOOST_SNIPPETS: tuple[str, ...] = (
    "/product",
    "/products/",
    "/p/",
    "/item",
    "/items/",
    "/vinyl",
    "/lp",
    "/buy",
    "/shop/",
    "/catalog/",
    "/catalogue/",
    "-p-",
    ".html",
    "add-to-cart",
    "/cart",
    "/produse/",
    "/produs/",
)


def fanout_single_domain_threshold(min_score_base: float) -> float:
    """Score floor for single-domain local shop fanout (sparse PDP snippets)."""
    return max(0.022, min_score_base * 0.18)


def whitelist_include_domains_threshold(min_score_base: float) -> float:
    """Score floor when ``include_domains`` restricts hits to curated stores only."""
    return max(0.032, min_score_base * 0.30)


def product_signal_multiplier(path: str) -> float:
    """Down-rank obvious editorial/category noise; lightly boost PDP-like URLs."""
    p = (path or "").lower()
    if any(x in p for x in _NON_RETAIL_PATH_SNIPPETS):
        return 0.0
    seg = p.rstrip("/").count("/")
    if seg <= 2 and any(
        p.rstrip("/").endswith(s) for s in ("/artists", "/artist", "/bands", "/band", "/labels", "/label")
    ):
        return 0.4
    if any(x in p for x in _RETAIL_PATH_BOOST_SNIPPETS):
        return 1.0
    return 0.72


def buy_signal_multiplier_for_url(url: str) -> float:
    """Reuse Tavily PDP vs editorial heuristic for pipeline quality checks."""
    try:
        nu = normalize_url(url)
        path_m = urlparse(nu).path or "/"
    except Exception:
        path_m = "/"
    return float(product_signal_multiplier(path_m))
