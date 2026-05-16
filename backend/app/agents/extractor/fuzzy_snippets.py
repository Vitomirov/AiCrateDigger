"""Deterministic fuzzy signals against Tavily snippets (pre-filter gates)."""

from __future__ import annotations

from rapidfuzz import fuzz

from app.agents.extractor.constants import PRE_FILTER_CONTENT_CHARS


def _haystack_lower(title: str, content: str) -> str:
    """Shared lowercased haystack for artist/album RapidFuzz partial_ratio."""
    return f"{title} {content[:PRE_FILTER_CONTENT_CHARS]}".lower()


def artist_fuzzy_score(artist: str, title: str, content: str) -> float:
    """RapidFuzz partial ratio of `artist` against (title + first N chars of content)."""
    return float(fuzz.partial_ratio(artist.lower(), _haystack_lower(title, content)))


def album_fuzzy_score(album: str, title: str, content: str) -> float:
    if not album:
        return 0.0
    return float(fuzz.partial_ratio(album.lower(), _haystack_lower(title, content)))
