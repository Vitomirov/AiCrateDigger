"""Locale-aware Latin spelling variants for Tavily/query fan-out.

Uses ISO region → common native orthography swaps (facts-only), not catalog literals.
"""

from __future__ import annotations

import itertools
import unicodedata
from collections import OrderedDict
from typing import Iterable

_COUNTRY_PAIR_SUBSTITUTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "PL": (("l", "ł"), ("L", "Ł"), ("z", "ż"), ("Z", "Ż")),
    "CZ": (("r", "ř"), ("R", "Ř"), ("c", "č"), ("C", "Č")),
    "SK": (("r", "ŕ"), ("R", "Ŕ"), ("l", "ľ"), ("L", "Ľ")),
    "RO": (("s", "ș"), ("S", "Ș"), ("t", "ț"), ("T", "Ț")),
    "HU": (("o", "ő"), ("O", "Ő"), ("u", "ű"), ("U", "Ű")),
    "DE": (("a", "ä"), ("A", "Ä"), ("o", "ö"), ("O", "Ö"), ("u", "ü"), ("U", "Ü")),
    "AT": (("a", "ä"), ("A", "Ä"), ("o", "ö"), ("O", "Ö"), ("u", "ü"), ("U", "Ü")),
    "CH": (("a", "ä"), ("A", "Ä"), ("o", "ö"), ("O", "Ö"), ("u", "ü"), ("U", "Ü")),
    "ES": (("n", "ñ"), ("N", "Ñ")),
    "FR": (("e", "é"), ("E", "É"), ("a", "à"), ("A", "À"), ("o", "ô"), ("O", "Ô")),
    "SE": (("a", "å"), ("A", "Å"), ("o", "ö"), ("O", "Ö")),
    "NO": (("o", "ø"), ("O", "Ø"), ("a", "å"), ("A", "Å")),
    "DK": (("o", "ø"), ("O", "Ø"), ("a", "å"), ("A", "Å")),
    "IS": (("t", "þ"), ("T", "Þ"), ("d", "ð"), ("D", "Ð")),
    "LV": (("u", "ū"), ("U", "Ū")),
    "LT": (("u", "ų"), ("U", "Ų")),
}


def latin_ascii_fold(text: str) -> str:
    """Strip combining marks / map notable Latin-extension letters toward ASCII-ish forms."""
    if not text:
        return ""
    nk = unicodedata.normalize("NFKD", text)
    out: list[str] = []
    for ch in nk:
        cat = unicodedata.category(ch)
        if cat == "Mn":
            continue
        if ch in ("\u0142", "\u0141"):
            out.append("l" if ch == "\u0142" else "L")
        elif ch in ("\u00f8", "\u00d8"):
            out.append("o" if ch == "\u00f8" else "O")
        elif ch == "\u00e6":
            out.append("ae")
        elif ch == "\u0153":
            out.append("oe")
        else:
            out.append(ch)
    return "".join(out)


def _dedupe_ordered(items: Iterable[str]) -> list[str]:
    od: OrderedDict[str, None] = OrderedDict()
    for x in items:
        t = x.strip()
        if t:
            od.setdefault(t, None)
    return list(od.keys())


def combinatorial_pair_substitutions(word: str, pairs: tuple[tuple[str, str], ...], *, max_variants: int) -> list[str]:
    """At each ASCII position touched by ``pairs``, pick original or swapped letter."""
    if not pairs or len(word) > 58:
        return [word]

    choice_at: list[list[str]] = []
    for ch in word:
        opts = [ch]
        for src, dst in pairs:
            if len(src) != 1 or len(dst) != 1:
                continue
            if ch.lower() == src.lower():
                alt = dst.upper() if ch.isupper() else dst.lower()
                if alt != ch:
                    opts.append(alt)
        uniq: OrderedDict[str, None] = OrderedDict()
        for o in opts:
            uniq.setdefault(o, None)
        choice_at.append(list(uniq.keys()))

    out: OrderedDict[str, None] = OrderedDict()
    for combo in itertools.islice(itertools.product(*choice_at), max_variants):
        out["".join(combo)] = None
        if len(out) >= max_variants:
            break
    return list(out.keys()) or [word]


def expand_glyph_variants_phrase(phrase: str, *, country_code: str | None, max_variants: int) -> list[str]:
    """Permute glyphs inside ASCII whitespace-separated tokens."""
    raw = phrase.strip()
    if not raw:
        return []

    cc = (country_code or "").strip().upper()
    if cc == "UK":
        cc = "GB"
    pairs = _COUNTRY_PAIR_SUBSTITUTIONS.get(cc)
    merged: OrderedDict[str, None] = OrderedDict()
    merged.setdefault(raw, None)
    lf = latin_ascii_fold(raw)
    if lf != raw:
        merged.setdefault(lf, None)

    if not pairs:
        return list(merged.keys())

    tokens = raw.split()
    variant_lists: list[list[str]] = []
    for tok in tokens:
        alphas_only = "".join(ch for ch in tok if ch.isalpha())
        if alphas_only and alphas_only.isascii():
            variant_lists.append(combinatorial_pair_substitutions(tok, pairs, max_variants=min(10, max(4, max_variants))))
        else:
            variant_lists.append([tok])

    product_cap = max(1, max_variants)
    for combo in itertools.islice(itertools.product(*variant_lists), product_cap):
        merged[" ".join(combo)] = None

    for k in list(merged.keys()):
        fk = latin_ascii_fold(k)
        if fk:
            merged.setdefault(fk, None)
    return _dedupe_ordered(merged.keys())


def expand_artist_glyph_variants(artist: str | None, *, country_code: str | None, max_variants: int = 14) -> list[str]:
    if not (artist or "").strip():
        return []
    return expand_glyph_variants_phrase(artist.strip(), country_code=country_code, max_variants=max_variants)


def expand_album_glyph_variants(album: str, *, country_code: str | None, max_variants: int = 8) -> list[str]:
    """Album titles are usually English/other Latin; skip per-letter national swaps.

    Applying e.g. PL ``l→ł`` turns *Futility* into *Futiłity*, which breaks Tavily and Discogs-aligned queries.
    """
    _ = country_code  # Artist-only glyph fan-out uses locale; albums stay canon + ascii-fold.
    _ = max_variants
    if not album.strip():
        return []
    raw = album.strip()
    out: OrderedDict[str, None] = OrderedDict()
    out.setdefault(raw, None)
    lf = latin_ascii_fold(raw)
    if lf:
        out.setdefault(lf, None)
    return list(out.keys())
