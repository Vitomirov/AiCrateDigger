"""Orchestrator: pre-filter → LLM extract → ListingResult assembly."""

from __future__ import annotations

from pydantic import ValidationError

from app.agents.extractor.constants import MAX_AI_BATCH_SIZE, MAX_FINAL_RESULTS
from app.agents.extractor.hosts import normalize_domain
from app.agents.extractor.candidates.step_11_candidate_pre_filter import run_pre_filter
from app.agents.extractor.candidates.step_12_candidate_llm_runner import run_llm_extract
from app.agents.extractor.utils.field_normalizers import (
    clean_optional_string,
    normalize_availability,
    normalize_seller,
)
from app.agents.extractor.utils.rejection_logging import log_extractor_reject
from app.models.result import ListingResult
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer


async def extract_and_score_results(
    candidates: list[SearchResult],
    artist: str | None,
    album: str | None,
    music_format: str | None,
    country: str | None,
    city: str | None,
) -> list[ListingResult]:
    # Defensive normalization — upstream agents may now pass None for any of
    # these tokens in the partial-intent / bootstrap paths. The extractor is
    # the last gate and must never crash on a missing field.
    artist = (artist or "").strip()
    album = (album or "").strip()
    music_format = (music_format or "").strip()
    country = (country or "").strip()
    with stage_timer(
        "extractor",
        input={"artist": artist, "album": album, "candidates_in": len(candidates)},
    ) as rec:
        if not candidates:
            rec.status = "empty"
            return []

        batch = candidates[:MAX_AI_BATCH_SIZE]

        # -------- PASS 1: deterministic pre-filter --------
        survivors, pre_artist_scores, pre_album_scores = run_pre_filter(
            batch=batch, artist=artist, album=album
        )

        if not survivors:
            rec.status = "empty"
            rec.extra["rejected_all"] = True
            return []

        # -------- PASS 2: LLM structured extraction on survivors --------
        llm_output = await run_llm_extract(
            survivors=survivors,
            artist=artist,
            album=album,
            music_format=music_format,
            city=city,
            country=country,
        )

        # Map LLM output by URL for quick lookup; anything the LLM silently dropped is rejected.
        llm_by_url: dict[str, dict] = {}
        for item in llm_output:
            u = str(item.get("url") or "").strip()
            if u:
                llm_by_url[u] = item

        # -------- PASS 3: assemble validated ListingResult objects --------
        validated: list[ListingResult] = []
        rejections_post = 0

        for survivor, artist_match, album_match in zip(
            survivors, pre_artist_scores, pre_album_scores, strict=True
        ):
            item = llm_by_url.get(survivor.url)
            if item is None:
                log_extractor_reject(survivor.url, "llm_dropped_silently", artist_match=artist_match)
                rejections_post += 1
                continue

            try:
                llm_score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                llm_score = 0.0

            domain = normalize_domain(survivor.url)

            try:
                listing = ListingResult(
                    url=survivor.url,
                    title=(item.get("title") or survivor.title).strip(),
                    score=min(max(llm_score, 0.0), 1.0),
                    price=clean_optional_string(item.get("price")),
                    location=clean_optional_string(item.get("location")) or None,
                    availability=normalize_availability(item.get("availability")),
                    seller_type=normalize_seller(item.get("seller_type")),
                    domain=domain,
                    artist_match=round(artist_match / 100.0, 3),
                    album_match=round(album_match / 100.0, 3),
                    match_reason=str(item.get("match_reason") or "accepted").strip() or "accepted",
                )
            except ValidationError as exc:
                log_extractor_reject(
                    survivor.url,
                    "validation_failed",
                    artist_match=artist_match,
                    detail=str(exc.errors()[:2]),
                )
                rejections_post += 1
                continue

            validated.append(listing)

        final = sorted(validated, key=lambda x: x.score, reverse=True)[:MAX_FINAL_RESULTS]
        rec.output = {
            "kept": len(final),
            "rejected_pre": len(batch) - len(survivors),
            "rejected_post": rejections_post,
        }
        rec.status = "success" if final else "empty"
        return final
