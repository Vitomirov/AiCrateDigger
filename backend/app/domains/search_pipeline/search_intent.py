"""Derive search intent from parser output (no extra LLM call)."""

from __future__ import annotations

from typing import Literal

from app.domains.query_parser.parse_schema import ParsedQuery

SearchIntent = Literal["release", "artist_catalog", "unresolved"]


def _has_geo_signal(parsed: ParsedQuery) -> bool:
    if parsed.country_code:
        return True
    if (parsed.resolved_city or "").strip():
        return True
    if (parsed.location or "").strip():
        return True
    return parsed.search_scope == "regional"


def resolve_search_intent(parsed: ParsedQuery) -> SearchIntent:
    """Classify how the pipeline should hunt listings for this parse."""
    if parsed.effective_album:
        return "release"

    artist = (parsed.artist or "").strip()
    if not artist:
        return "unresolved"

    if _has_geo_signal(parsed):
        return "artist_catalog"

    return "unresolved"


def empty_reason_for_unresolved(parsed: ParsedQuery) -> str:
    """Machine-readable reason when ``resolve_search_intent`` is ``unresolved``."""
    if parsed.album_index is not None and not parsed.effective_album:
        return "album_unresolved"
    return "intent_unresolved"


def cache_album_segment(*, intent: SearchIntent, album: str | None) -> str | None:
    """Stable cache token for the album slot (``catalog`` for artist+geo browse)."""
    if intent == "artist_catalog":
        return "catalog"
    return album


__all__ = [
    "SearchIntent",
    "cache_album_segment",
    "empty_reason_for_unresolved",
    "resolve_search_intent",
]
