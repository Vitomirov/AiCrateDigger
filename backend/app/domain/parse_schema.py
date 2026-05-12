"""ParsedQuery schema for the deterministic vinyl-search pipeline.

Parser output: core fields from one LLM call plus request echo (`original_query`)
and a fixed `language` placeholder until a dedicated detector exists.

``country_code`` and ``search_scope`` are **semantically inferred by the LLM** from
``location`` so the pipeline never needs a hardcoded city table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SearchScope = Literal["local", "regional", "global"]
GeoGranularity = Literal["city", "country", "region", "none"]


class ParsedQuery(BaseModel):
    """Parsed-intent contract returned by `parse_user_query` and the `/parse` API."""

    model_config = ConfigDict(extra="forbid")

    artist: str | None = Field(
        default=None,
        description="Artist / band name as extracted from the user query. Null when unknown.",
    )
    album: str | None = Field(
        default=None,
        description="Literal album title from the user query. Null for ordinal references.",
    )
    album_index: int | None = Field(
        default=None,
        description="1-based ordinal (1=debut, 2=second, ...). -1 means 'latest'. Null otherwise.",
    )
    location: str | None = Field(
        default=None,
        description="Verbatim city or country substring (free-form). Null when not stated.",
    )
    country_code: str | None = Field(
        default=None,
        description=(
            "ISO-3166-1 alpha-2 country code inferred from `location` (e.g. 'RS' for "
            "Kragujevac). Null when ambiguous, regional, or absent."
        ),
    )
    search_scope: SearchScope = Field(
        default="global",
        description=(
            "Commerce-routing scope: 'local' when a specific country/city is given, "
            "'regional' for multi-country phrases (Europe, Balkans, EU, Scandinavia), "
            "'global' when no location is stated."
        ),
    )
    resolved_city: str | None = Field(
        default=None,
        description=(
            "Canonical city name when `location` is city-level (e.g. 'Barcelona' for "
            "typos like 'barselona'). Null when unknown or country-only."
        ),
    )
    geo_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0–1 confidence for geo resolution; steers widening strictness.",
    )
    geo_granularity: GeoGranularity | None = Field(
        default=None,
        description="Explicit locality level from the parser; null = infer downstream.",
    )
    language: str = Field(
        default="unknown",
        min_length=1,
        description='Echo or placeholder; parser does not infer language (default "unknown").',
    )
    original_query: str = Field(
        ...,
        min_length=1,
        description="Exact user input string for this parse request.",
    )

    @field_validator("resolved_city", mode="before")
    @classmethod
    def _normalize_resolved_city(cls, v: object) -> object:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("country_code", mode="before")
    @classmethod
    def _normalize_country_code(cls, v: object) -> object:
        if v is None:
            return None
        s = str(v).strip().upper()
        if not s:
            return None
        if s == "UK":
            s = "GB"
        if len(s) == 2 and s.isalpha() and s.isascii():
            return s
        return None
