"""Artist/album intent gate on snippet text (fuzzy + URL heuristics)."""

from __future__ import annotations

from rapidfuzz import fuzz

from app.agents.extractor.evidence_alignment import (
    artist_fuzzy_best_vs_blob,
    artist_substring_in_blob,
    ascii_fold,
)
from app.validators.listings import url_suggests_product_detail_page

# Standard mega-retailer / rich-snippet blobs — album title often verbatim.
_ARTIST_DETAIL_PDP = 68
_ALBUM_PDP_ONLY = 72
_ARTIST_ALBUM_IN_BLOB_ARTIST_GAP = 65

# Curated indie shops: titles are often SKU-ish; avoid false negatives.
_RELAX_ARTIST_DETAIL_PDP = 52
_RELAX_ALBUM_PDP_ONLY = 56
_RELAX_ARTIST_ALBUM_IN_BLOB = 48


def intent_matches_snippet(
    *,
    url: str,
    blob: str,
    artist_l: str | None,
    album_l: str,
    relaxed_indie_candidate: bool = False,
) -> bool:
    """Gate whether snippet text references the hunted release.

    Album/artist tokens and the blob are ASCII-folded before comparison so
    diacritics (Polish ``ł``, Scandinavian ``ø``…) never cost a substring hit.
    """
    rd = (
        (_RELAX_ARTIST_DETAIL_PDP, _RELAX_ALBUM_PDP_ONLY, _RELAX_ARTIST_ALBUM_IN_BLOB)
        if relaxed_indie_candidate
        else (_ARTIST_DETAIL_PDP, _ALBUM_PDP_ONLY, _ARTIST_ALBUM_IN_BLOB_ARTIST_GAP)
    )
    td_pdp_art, td_pdp_album, td_blob_art = rd

    folded_blob = ascii_fold(blob)
    folded_album = ascii_fold(album_l)
    folded_artist = ascii_fold(artist_l) if artist_l else ""

    if folded_album and folded_album in folded_blob:
        if not folded_artist:
            return True
        if artist_substring_in_blob(folded_artist, folded_blob):
            return True
        if url_suggests_product_detail_page(url):
            if artist_fuzzy_best_vs_blob(folded_artist, folded_blob) >= td_pdp_art:
                return True
        if artist_fuzzy_best_vs_blob(folded_artist, folded_blob) >= 58.0:
            return True
        return False
    alb_fuzzy = float(max(fuzz.token_set_ratio(folded_album, folded_blob), fuzz.partial_ratio(folded_album, folded_blob)))
    if alb_fuzzy >= 62.0 and folded_artist:
        if artist_substring_in_blob(folded_artist, folded_blob):
            return True
        if artist_fuzzy_best_vs_blob(folded_artist, folded_blob) >= 54.0:
            return True
    if not url_suggests_product_detail_page(url):
        return False
    ab = max(fuzz.token_set_ratio(folded_album, folded_blob), fuzz.partial_ratio(folded_album, folded_blob))
    if ab < td_pdp_album:
        return False
    if not folded_artist:
        return True
    return artist_fuzzy_best_vs_blob(folded_artist, folded_blob) >= td_blob_art


def snippet_passes_release_intent(
    *,
    url: str,
    blob: str,
    artist_l: str | None,
    album_l: str,
    host: str | None,
    snippet_relax_hosts: frozenset[str] | None,
) -> bool:
    """Strict intent gate, then a second relaxed attempt for curated local_shop hosts."""
    if intent_matches_snippet(
        url=url,
        blob=blob,
        artist_l=artist_l,
        album_l=album_l,
        relaxed_indie_candidate=False,
    ):
        return True
    if not snippet_relax_hosts or not host or host not in snippet_relax_hosts:
        return False
    return intent_matches_snippet(
        url=url,
        blob=blob,
        artist_l=artist_l,
        album_l=album_l,
        relaxed_indie_candidate=True,
    )
