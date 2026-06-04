"""Listing projection and per-host deduplication for API responses."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from app.core.config import Settings
from app.domains.engine.listing_schema import Listing
from app.domains.search_pipeline.models.result import ListingResult

logger = logging.getLogger(__name__)


def host_of(url: str) -> str | None:
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except Exception:
        return None
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":", 1)[0] or None


def listing_to_api_row(listing: Listing) -> ListingResult:
    """Project a verified :class:`Listing` to the API-facing :class:`ListingResult`.

    A minimal, deterministic projection — the consolidated pipeline does NOT
    use the legacy composite scorer (which depends on the multi-tier loop),
    so the API ``score`` reflects pure extractor confidence:
    1.0 when in stock, 0.85 otherwise.
    """
    title = (listing.title or "").strip()
    if len(title) < 5:
        title = f"{title} · shop"

    price_str: str | None = None
    try:
        price_val = float(listing.price or 0.0)
        if price_val > 0:
            currency = (listing.currency or "EUR").strip().upper() or "EUR"
            price_str = f"{price_val:.2f} {currency}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        price_str = None

    host = host_of(listing.url) or (listing.store or None)
    score = 1.0 if listing.in_stock else 0.85

    return ListingResult(
        url=listing.url,
        title=title,
        score=score,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=host,
        artist_match=1.0,
        album_match=1.0,
        match_reason="consolidated_pipeline",
    )


def dedupe_listings_by_host(listings: list[Listing]) -> list[Listing]:
    """One listing per registrable host. In-stock rows win; among ties the first
    occurrence (preserving extractor order) wins.

    The Python prefilter intentionally lets up to 2 deep links per host through
    so the LLM has redundancy. After extraction we hard-collapse to 1 per host
    so the user-facing result list always shows variety across shops.
    """
    best: dict[str, Listing] = {}
    for lst in listings:
        host = host_of(lst.url)
        if not host:
            continue
        prev = best.get(host)
        if prev is None:
            best[host] = lst
            continue
        if lst.in_stock and not prev.in_stock:
            best[host] = lst
    return list(best.values())


def finalize_api_rows(
    listings: list[Listing],
    *,
    settings: Settings,
    extractor_listing_count: int,
) -> list[ListingResult]:
    """Dedupe by host, project to API rows, sort by score, cap to pipeline max."""
    unique_listings = dedupe_listings_by_host(listings)
    api_rows = [listing_to_api_row(lst) for lst in unique_listings]
    api_rows.sort(key=lambda r: r.score, reverse=True)
    api_rows = api_rows[: settings.pipeline_max_results]
    logger.info(
        "results_dedupe_summary",
        extra={
            "stage": "pipeline",
            "extractor_listings": extractor_listing_count,
            "deduped_listings": len(unique_listings),
            "api_rows": len(api_rows),
        },
    )
    return api_rows
