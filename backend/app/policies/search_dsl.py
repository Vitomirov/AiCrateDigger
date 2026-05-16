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

    Phrase quoting: **3+ word** album titles are double-quoted for phrase match.
    **1–2 word** titles stay unquoted so storefront indexes can match despite
    casing/styling differences (e.g. ``Strange Days`` vs ``"strange Days"``).
    """
    parts: list[str] = []
    if (artist or "").strip():
        parts.append(str(artist).strip())
    al = str(album).strip()
    if not al:
        return "vinyl"
    n_words = len(al.split())
    if n_words >= 2:
        parts.append(f'"{al}"' if n_words >= 3 else al)
    else:
        parts.append(al)
    parts.append("vinyl")
    return " ".join(parts)


def build_tavily_local_fanout_narrow_query(*, artist: str | None, album: str) -> str:
    """Artist + album for single-domain indie Tavily fanout (no ``vinyl`` / ``LP``).

    Quotes are **not** used even for long titles — small-shop indexes and Tavily's
    snippet matching often miss quoted 3–4 word phrases, yielding ``raw_count``>0
    but ``kept_count``=0 after scoring.
    """
    art = str(artist or "").strip()
    al = str(album or "").strip()

    if not al:
        return art if art else "vinyl"

    if art:
        return f"{art} {al}"

    return al


def build_tavily_local_fanout_plain_album_query(album: str) -> str:
    """Unquoted ``{album} vinyl LP`` variant — always safe for local fanout add-on."""
    al = str(album or "").strip()
    if not al:
        return ""
    return f"{al} vinyl LP"


def build_tavily_local_fanout_plain_artist_album_query(*, artist: str | None, album: str) -> str:
    """``{artist} {album} vinyl LP`` for storefronts that need both tokens with format hints."""
    art = str(artist or "").strip()
    al = str(album or "").strip()
    if not al:
        return ""
    if art:
        return f"{art} {al} vinyl LP"
    return f"{al} vinyl LP"


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
