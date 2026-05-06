"""ParsedQuery schema for the deterministic vinyl-search pipeline.

Parser output: core fields from one LLM call plus request echo (`original_query`)
and a fixed `language` placeholder until a dedicated detector exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
        description="Country or city hint, free-form string. Null when not stated.",
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
