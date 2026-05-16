"""Listing schema for the deterministic vinyl-search pipeline.

Extractor output: ``price`` / ``currency`` may be unknown — use defaults so
downstream validation can still accept slightly incomplete rows.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Listing(BaseModel):
    """Buyable vinyl listing on a whitelisted EU store."""

    title: str = Field(..., min_length=1, description="Listing page title.")
    price: Optional[float] = Field(
        default=0.0,
        description="Numeric price; ``0.0`` / ``None`` = unknown.",
    )
    currency: Optional[str] = Field(
        default="EUR",
        description="ISO 4217; unknown → EUR.",
    )
    in_stock: bool = Field(
        ...,
        description="``True`` when available; ``False`` only when explicitly sold out.",
    )
    url: str = Field(..., min_length=8, description="Canonical product URL on the source store.")
    store: str = Field(
        ...,
        min_length=1,
        description="Whitelisted store identifier (domain or short name).",
    )
    validation_artist: str | None = Field(default=None, description="Injected pre-validate.")
    validation_album: str | None = Field(default=None, description="Injected pre-validate.")
    source_snippet: str | None = Field(
        default=None,
        description="Raw SERP title+snippet excerpt used for grounding (verifier/evidence).",
    )

    @field_validator("price", mode="before")
    @classmethod
    def _price_before(cls, v: Any) -> Any:
        if v is None:
            return 0.0
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def _currency_before(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "EUR"
        return v

    @field_validator("currency", mode="after")
    @classmethod
    def _currency_after(cls, v: str | None) -> str:
        s = (v or "EUR").strip().upper()
        if len(s) != 3 or not s.isalpha():
            return "EUR"
        return s

    @field_validator("price", mode="after")
    @classmethod
    def _price_after(cls, v: float | None) -> float:
        if v is None or v < 0.0:
            return 0.0
        return float(v)
