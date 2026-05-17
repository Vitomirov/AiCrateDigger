"""Conversion from internal listing objects to :class:`ListingResult` API rows."""

from __future__ import annotations

from typing import Any

from app.models.result import ListingResult
from app.policies.listing_rank import ListingRankBreakdown


def listing_to_api_row(
    listing: Any,
    *,
    breakdown: ListingRankBreakdown,
) -> ListingResult:
    """Project an internal listing + score breakdown into the public API row."""
    title = str(listing.title or "")
    if len(title) < 5:
        title = f"{title} · shop"

    price_str = None
    cur = listing.currency or "EUR"
    try:
        if listing.price is not None and float(listing.price) > 0:
            price_str = f"{float(listing.price):.2f} {cur}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        price_str = f"{listing.price} {cur}" if listing.price else None

    match_reason = (
        f"tier={breakdown.discovery_tier}|store_type={breakdown.store_type}"
        f"|geo={breakdown.geo_proximity:.1f}|sem={breakdown.semantic_match:.1f}"
        f"|vinyl={breakdown.vinyl_confidence:.1f}"
    )

    return ListingResult(
        url=listing.url,
        title=title,
        score=breakdown.score_normalized,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=listing.store,
        artist_match=1.0,
        album_match=1.0,
        match_reason=match_reason,
    )
