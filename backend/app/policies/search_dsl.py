"""Deterministic search-query DSL."""

from __future__ import annotations

from typing import Iterable

from app.policies.locale_text_variants import (
    expand_album_glyph_variants,
    expand_artist_glyph_variants,
)


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


def build_tavily_geo_browse_queries(
    artist_variants: Iterable[str],
    album_title: str,
    *,
    resolved_city: str | None,
) -> list[str]:
    """Broad storefront-oriented queries — used when phrase-strict Tavily misses PDPs."""

    geo = (resolved_city or "").strip()
    if not geo:
        return []

    qs: list[str] = []
    al = album_title.strip()

    def _uniq_append(s: str) -> None:
        t = s.strip()
        if not t:
            return
        if t.casefold() not in {x.casefold() for x in qs}:
            qs.append(t)

    for art in artist_variants:
        a = (art or "").strip()
        if not a:
            continue
        _uniq_append(f"{a} vinyl {geo}")
        _uniq_append(f"{a} record shop {geo}")
        if al:
            _uniq_append(f"{a} {al} vinyl {geo}")

    return qs


def plan_tavily_query_strings(
    artist: str | None,
    album_title: str,
    *,
    country_code_for_variants: str | None,
    resolved_city: str | None,
) -> tuple[str, list[str]]:
    """Primary strict Tavily phrase + ordered relaxation ladder (glyph + geo browse).

    Deduped case-insensitively while preserving-first-wins ordering.
    """
    alb = album_title.strip()
    if not alb:
        return "", []

    art_variants = expand_artist_glyph_variants(artist, country_code=country_code_for_variants)
    alb_variants = expand_album_glyph_variants(alb, country_code=country_code_for_variants)

    strict_candidates: list[str] = []
    for av in art_variants[:5]:
        for bv in alb_variants[:5]:
            q = build_tavily_core_query(av or artist, bv)
            if q and q.casefold() not in {s.casefold() for s in strict_candidates}:
                strict_candidates.append(q)

    primary = strict_candidates[0] if strict_candidates else build_tavily_core_query(artist, alb)
    relax_geo = build_tavily_geo_browse_queries(
        art_variants[:3] or ([] if not (artist or "").strip() else [str(artist).strip()]),
        alb,
        resolved_city=resolved_city,
    )

    seen_cf: set[str] = set()
    relax: list[str] = []

    def _consume(q: str) -> None:
        k = q.strip().casefold()
        if k and k not in seen_cf and q.strip() != primary.strip():
            seen_cf.add(k)
            relax.append(q.strip())

    for q in strict_candidates[1:6]:
        _consume(q)

    for q in relax_geo[:6]:
        _consume(q)

    return primary, relax


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
