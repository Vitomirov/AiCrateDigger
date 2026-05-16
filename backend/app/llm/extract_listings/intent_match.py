"""Artist/album intent gate on snippet text (fuzzy + URL heuristics)."""

from __future__ import annotations

from rapidfuzz import fuzz

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
    """Gate whether snippet text references the hunted release."""
    rd = (
        (_RELAX_ARTIST_DETAIL_PDP, _RELAX_ALBUM_PDP_ONLY, _RELAX_ARTIST_ALBUM_IN_BLOB)
        if relaxed_indie_candidate
        else (_ARTIST_DETAIL_PDP, _ALBUM_PDP_ONLY, _ARTIST_ALBUM_IN_BLOB_ARTIST_GAP)
    )
    td_pdp_art, td_pdp_album, td_blob_art = rd

    if album_l in blob:
        if not artist_l or artist_l in blob:
            return True
        if url_suggests_product_detail_page(url):
            ar = max(fuzz.token_set_ratio(artist_l, blob), fuzz.partial_ratio(artist_l, blob))
            if ar >= td_pdp_art:
                return True
        return False
    if not url_suggests_product_detail_page(url):
        return False
    ab = max(fuzz.token_set_ratio(album_l, blob), fuzz.partial_ratio(album_l, blob))
    if ab < td_pdp_album:
        return False
    if not artist_l:
        return True
    ar = max(fuzz.token_set_ratio(artist_l, blob), fuzz.partial_ratio(artist_l, blob))
    return ar >= td_blob_art


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
