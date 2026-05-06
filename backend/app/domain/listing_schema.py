"""Listing schema for the deterministic vinyl-search pipeline.

Strict output shape returned by the extractor + validator and surfaced to the
API. Every field is required — the extraction layer must fill them or the
listing is rejected upstream.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Listing(BaseModel):
    """Buyable vinyl listing on a whitelisted EU store."""

    title: str = Field(..., min_length=1, description="Listing page title.")
    price: float = Field(..., gt=0.0, description="Numeric price; currency lives in `currency`.")
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO 4217 code (EUR, GBP, SEK, ...).",
    )
    in_stock: bool = Field(
        ...,
        description="True only when the page clearly states availability.",
    )
    url: str = Field(..., min_length=8, description="Canonical product URL on the source store.")
    store: str = Field(
        ...,
        min_length=1,
        description="Whitelisted store identifier (domain or short name).",
    )
    # Injected by the orchestrator before `validate_listing`; omit from LLM JSON.
    validation_artist: str | None = Field(
        default=None,
        description="If set, `title` must contain this substring (case-insensitive).",
    )
    validation_album: str | None = Field(
        default=None,
        description="If set, `title` must contain this substring (case-insensitive).",
    )
