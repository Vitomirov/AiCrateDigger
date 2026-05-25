"""Listing-level result schema. Canonical output of Agent 3 (Extractor).

Kept in its own module per the Cursor folder-structure rule. Downstream code
(e.g. router response models) may still use `SearchResult` from
`search_query.py` as the API-facing DTO; `ListingResult` is the internal
contract the extractor produces and the scorer consumes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Availability = Literal["available", "sold_out", "unknown"]
SellerType = Literal["store", "private", "unknown"]


class ListingResult(BaseModel):
    """Strict contract. Unknown fields MUST be explicitly set to null, never omitted."""

    url: str = Field(..., min_length=8, description="Normalized listing URL")
    title: str = Field(..., min_length=5, description="Listing title (>= 5 chars enforced)")
    # Extractor confidence. NOT the final sort score — scorer overrides with deterministic value.
    score: float = Field(..., ge=0.0, le=1.0)
    price: str | None = Field(default=None, description="Local-currency price string, or null")
    location: str | None = Field(default=None, description="City[, District] or null")
    availability: Availability = Field(default="unknown")
    seller_type: SellerType = Field(default="unknown")
    domain: str | None = Field(default=None, description="Base domain (e.g. 'metropolismusic.rs')")
    # Debugging / traceability fields:
    artist_match: float = Field(default=0.0, ge=0.0, le=1.0)
    album_match: float = Field(default=0.0, ge=0.0, le=1.0)
    match_reason: str | None = Field(
        default=None, description="Free-form reason (e.g. 'accepted', 'reject:merch')"
    )

    @model_validator(mode="after")
    def _url_required(self) -> "ListingResult":
        if not self.url.strip():
            raise ValueError("ListingResult.url cannot be blank")
        return self
