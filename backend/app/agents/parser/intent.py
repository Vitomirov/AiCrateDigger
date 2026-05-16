"""Intent classification and hard fallback payloads."""

from __future__ import annotations

from app.agents.parser.coercion import guess_language_cheap
from app.models.search_query import IntentCompleteness, ParsedQuery


def classify_intent(
    *,
    artist: str | None,
    album: str | None,
    resolved_album: str | None,
) -> IntentCompleteness:
    has_artist = bool((artist or "").strip())
    has_album = bool((album or "").strip() or (resolved_album or "").strip())
    if has_artist and has_album:
        return "complete"
    if has_artist or has_album:
        return "partial"
    return "unknown"


def build_unknown_fallback(query: str) -> ParsedQuery:
    return ParsedQuery(
        language=guess_language_cheap(query),
        original_query=query,
        intent_completeness="unknown",
    )
