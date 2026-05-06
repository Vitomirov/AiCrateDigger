"""Deterministic search-query DSL."""

from __future__ import annotations


def build_query(
    artist: str | None,
    album: str,
    domain: str,
    location: str | None = None,
) -> str:
    """``{artist?} {album} {location?} vinyl site:{domain}`` — omit empty parts."""
    chunks: list[str] = []
    if (artist or "").strip():
        chunks.append(str(artist).strip())
    chunks.append(str(album).strip())
    if (location or "").strip():
        chunks.append(str(location).strip())
    core = " ".join(chunks)
    return f"{core} vinyl site:{domain}"
