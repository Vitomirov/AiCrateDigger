"""Deterministic pre-filter: cheap gates before LLM extraction."""

from __future__ import annotations

from app.agents.extractor.constants import MIN_ARTIST_FUZZY, MIN_TITLE_LEN
from app.agents.extractor.merch_gate import listing_looks_like_merch
from app.agents.extractor.utils.fuzzy_snippets import album_fuzzy_score, artist_fuzzy_score
from app.agents.extractor.utils.rejection_logging import log_extractor_reject
from app.models.search_query import SearchResult


def run_pre_filter(
    *,
    batch: list[SearchResult],
    artist: str,
    album: str,
) -> tuple[list[SearchResult], list[float], list[float]]:
    survivors: list[SearchResult] = []
    artist_scores: list[float] = []
    album_scores: list[float] = []
    for cand in batch:
        if not cand.url.strip():
            log_extractor_reject(cand.url, "missing_url")
            continue
        title = (cand.title or "").strip()
        if len(title) < MIN_TITLE_LEN:
            log_extractor_reject(cand.url, "title_too_short", detail=title)
            continue
        if listing_looks_like_merch(title, cand.content):
            log_extractor_reject(cand.url, "merch_keyword", detail=title)
            continue

        # Partial-intent safety: without an artist anchor we cannot run the
        # fuzzy-match gate, so we let Tavily's own score be the only signal.
        # This path corresponds to `intent_completeness in {"partial","unknown"}`
        # where artist failed to extract. We still reject obvious garbage (no url,
        # short title, merch keywords) upstream.
        if artist.strip():
            a_score = artist_fuzzy_score(artist, title, cand.content)
            if a_score < MIN_ARTIST_FUZZY:
                log_extractor_reject(
                    cand.url,
                    "artist_mismatch",
                    artist_match=a_score,
                    detail=f"title={title!r}",
                )
                continue
        else:
            # No artist to match against — assign neutral mid-score so ranking
            # still works downstream but no quality assertion is made.
            a_score = 50.0

        b_score = album_fuzzy_score(album, title, cand.content)
        survivors.append(cand)
        artist_scores.append(a_score)
        album_scores.append(b_score)
    return survivors, artist_scores, album_scores
