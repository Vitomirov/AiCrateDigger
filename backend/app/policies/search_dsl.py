"""Deterministic search-query DSL."""

from __future__ import annotations


def _intent_chunks(artist: str | None, album: str, location: str | None) -> list[str]:
    chunks: list[str] = []
    if (artist or "").strip():
        chunks.append(str(artist).strip())
    chunks.append(str(album).strip())
    if (location or "").strip():
        chunks.append(str(location).strip())
    return chunks


def build_tavily_core_query(artist: str | None, album: str) -> str:
    """Intent string for Tavily only.

    Omits **city/country tokens**: they are not store identifiers and (with
    ``include_domains`` already set) tend to skew results toward unrelated
    pages (e.g. live titles mentioning "Paris").

    Multi-word album titles are **double-quoted** to preserve phrase matching.
    """
    parts: list[str] = []
    if (artist or "").strip():
        parts.append(str(artist).strip())
    al = str(album).strip()
    if not al:
        return "vinyl"
    if " " in al:
        parts.append(f'"{al}"')
    else:
        parts.append(al)
    parts.append("vinyl")
    return " ".join(parts)


def build_query_core(
    artist: str | None,
    album: str,
    location: str | None = None,
) -> str:
    """Natural-language intent: ``{artist?} {album} {location?} vinyl`` (no ``site:``).

    Used for logging/debug parity with user wording. For live Tavily calls
    prefer :func:`build_tavily_core_query`.
    """
    core = " ".join(_intent_chunks(artist, album, location))
    return f"{core} vinyl"


def build_query(
    artist: str | None,
    album: str,
    domain: str,
    location: str | None = None,
) -> str:
    """``{core} site:{domain}`` for legacy per-domain probes."""
    return f"{build_query_core(artist, album, location)} site:{domain}"
