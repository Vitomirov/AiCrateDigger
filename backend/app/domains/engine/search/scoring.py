"""Tavily result scoring heuristics (PDP vs editorial noise)."""

from __future__ import annotations

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
