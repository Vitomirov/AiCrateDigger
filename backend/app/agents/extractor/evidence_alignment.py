"""Evidence that SERP/snippet material actually mentions the hunted release."""

from __future__ import annotations

import unicodedata
from urllib.parse import unquote, urlsplit

from rapidfuzz import fuzz

# Latin extensions that ``unicodedata.NFKD`` does NOT decompose (it leaves the
# base codepoint unchanged). Without explicit mapping ``"mgła"`` and ``"mgla"``
# share zero characters at the ``ł`` position and every substring/fuzz score
# collapses — killing recall on Polish, Scandinavian and Balkan queries.
_LATIN_FOLD_OVERRIDES: dict[str, str] = {
    "ł": "l", "Ł": "L",
    "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D",
    "ß": "ss",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "þ": "th", "Þ": "Th",
    "ð": "d", "Ð": "D",
}


def ascii_fold(s: str) -> str:
    """Lowercase + strip combining marks + apply :data:`_LATIN_FOLD_OVERRIDES`.

    Purely additive vs the original string for ASCII input (identity); for
    diacritics it produces the canonical ASCII rendering most catalogues print
    in URLs and SEO titles. Symmetric: callers should fold both needle AND
    haystack so behaviour stays identical on plain ASCII queries.
    """
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    out: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        out.append(_LATIN_FOLD_OVERRIDES.get(ch, ch))
    return "".join(out).lower()


def url_path_evidence_text(url: str) -> str:
    """SERP-thin recall booster: turn the URL slug into searchable evidence text.

    Indie store PDPs frequently encode artist + album in the slug
    (``/produkty/mgla-exercises-in-futility-12-vinyl``) even when the Tavily
    snippet only echoes the shop name. Treating the decoded, separator-split
    path as additional blob text lets the deterministic gates ground on real
    evidence without an extra LLM call.

    Returns lowercase ASCII-folded text; empty string on malformed input.
    """
    if not url:
        return ""
    try:
        parsed = urlsplit(url.strip())
        path = unquote(parsed.path or "")
    except (ValueError, UnicodeDecodeError):
        return ""
    if not path:
        return ""
    folded = ascii_fold(path)
    for sep in ("-", "_", "/", ".", "+"):
        folded = folded.replace(sep, " ")
    return " ".join(folded.split())


def _artist_variants_lc(artist_lc: str) -> tuple[str, ...]:
    """Lowercased + ASCII-folded artist forms; also ``the …`` stripped when safe.

    Many SERP snippets say ``Doors`` / ``doors`` without the leading article even
    when the canonical artist is ``The Doors`` — substring checks must not miss that.
    """
    a = ascii_fold(artist_lc).strip()
    if not a:
        return ()
    out: list[str] = [a]
    if a.startswith("the ") and len(a) > 4:
        tail = a[4:].strip()
        if tail:
            out.append(tail)
    return tuple(dict.fromkeys(out))


def artist_substring_in_blob(artist_lc: str, blob_lc: str) -> bool:
    """True if any folded variant (including ``the``-stripped) appears in the folded blob."""
    folded_blob = ascii_fold(blob_lc)
    return any(v in folded_blob for v in _artist_variants_lc(artist_lc))


def artist_fuzzy_best_vs_blob(artist_lc: str, blob_lc: str) -> float:
    """Best RapidFuzz score across folded artist variants vs the folded blob."""
    variants = _artist_variants_lc(artist_lc)
    if not variants:
        return 0.0
    folded_blob = ascii_fold(blob_lc)
    return max(
        float(max(fuzz.partial_ratio(v, folded_blob), fuzz.token_set_ratio(v, folded_blob)))
        for v in variants
    )


def canonical_query_echo_title(artist: str | None, album: str) -> str:
    """Lowercase trimmed \"Artist Album\" fingerprint (avoid LLM copying the search)."""
    return f"{(artist or '').strip()} {album}".strip().lower()


def looks_like_pure_query_echo_title(
    title: str,
    *,
    artist: str | None,
    album: str,
    evidence_blob_lc: str,
) -> bool:
    """Title mirrors the user's search tokens but SERP excerpt does NOT name the album."""
    t = ascii_fold(title).strip()
    if not t:
        return False
    q = ascii_fold(canonical_query_echo_title(artist, album)).strip()
    if not q:
        return False
    alb = ascii_fold(album).strip()
    folded_blob = ascii_fold(evidence_blob_lc)
    blob_has_album = alb and alb in folded_blob
    if blob_has_album:
        return False
    token_hit = alb and any(tok and tok in folded_blob for tok in alb.split())
    if token_hit:
        return False
    if t == q or t.startswith(q + " ") or t.endswith(" " + q):
        return True
    if alb and alb in t and alb not in folded_blob and q in t:
        return True
    return False


def listing_title_grounded_in_evidence(llm_title: str, evidence_blob_lc: str, *, min_ratio: float = 52.0) -> bool:
    """LLM-returned title should overlap the raw snippet materially (diacritic-insensitive)."""
    lt = ascii_fold(llm_title).strip()
    if len(lt) < 4:
        return False
    folded_blob = ascii_fold(evidence_blob_lc)
    if lt in folded_blob and len(lt) >= 12:
        return True
    return float(max(fuzz.partial_ratio(lt, folded_blob), fuzz.token_set_ratio(lt, folded_blob))) >= min_ratio


def evidence_blob_matches_target_release(
    evidence_blob_lc: str,
    *,
    artist: str | None,
    album: str,
    album_partial_min: float = 62.0,
    artist_partial_min: float = 56.0,
) -> bool:
    """Deterministic gate: SERP blob plausibly describes the target release.

    Used in extract/merge and verifier pre-checks. Thin indie snippets are *not*
    auto-loosened here (that caused false positives vs regional SKUs); storefronts
    rely on intent relax for ``snippet_relax_hosts`` plus
    ``validate_listing(..., relaxed_local_indie=True)`` for title+snippet fuzz.

    Both sides are ASCII-folded so diacritic-bearing queries (Polish ``ł``,
    Scandinavian ``ø``, Serbian ``đ``, …) match plain-ASCII catalogue text and
    vice-versa.
    """
    b = ascii_fold(evidence_blob_lc)
    alb = ascii_fold(album).strip()
    if not alb or not b.strip():
        return False
    if alb in b:
        return True
    if float(max(fuzz.partial_ratio(alb, b), fuzz.token_set_ratio(alb, b))) >= album_partial_min:
        if not (artist or "").strip():
            return True
        ar = ascii_fold(artist).strip()
        if artist_substring_in_blob(ar, b):
            return True
        return artist_fuzzy_best_vs_blob(ar, b) >= artist_partial_min
    return False
