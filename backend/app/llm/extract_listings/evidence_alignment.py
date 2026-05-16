"""Evidence that SERP/snippet material actually mentions the hunted release."""

from __future__ import annotations

from rapidfuzz import fuzz


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
    t = (title or "").strip().lower()
    if not t:
        return False
    q = canonical_query_echo_title(artist, album).strip().lower()
    if not q:
        return False
    alb = album.strip().lower()
    blob_has_album = alb and alb in evidence_blob_lc
    if blob_has_album:
        return False
    token_hit = alb and any(tok and tok in evidence_blob_lc for tok in alb.split())
    if token_hit:
        return False
    if t == q or t.startswith(q + " ") or t.endswith(" " + q):
        return True
    if alb and alb in t and alb not in evidence_blob_lc and q in t:
        return True
    return False


def listing_title_grounded_in_evidence(llm_title: str, evidence_blob_lc: str, *, min_ratio: float = 52.0) -> bool:
    """LLM-returned title should overlap the raw snippet materially."""
    lt = (llm_title or "").strip().lower()
    if len(lt) < 4:
        return False
    if lt in evidence_blob_lc and len(lt) >= 12:
        return True
    return float(max(fuzz.partial_ratio(lt, evidence_blob_lc), fuzz.token_set_ratio(lt, evidence_blob_lc))) >= min_ratio


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
    """
    b = evidence_blob_lc
    alb = album.strip().lower()
    if not alb or not b.strip():
        return False
    if alb in b:
        return True
    if float(max(fuzz.partial_ratio(alb, b), fuzz.token_set_ratio(alb, b))) >= album_partial_min:
        if not (artist or "").strip():
            return True
        ar = artist.strip().lower()
        if ar in b:
            return True
        return float(max(fuzz.partial_ratio(ar, b), fuzz.token_set_ratio(ar, b))) >= artist_partial_min
    return False
