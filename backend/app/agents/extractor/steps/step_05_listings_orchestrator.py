"""Orchestration: prefilter → LLM or deterministic path → merge and diagnostics."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.extractor.listing_constants import LLM_MAX_INPUT, SMALL_BATCH_NO_LLM
from app.agents.extractor.listing_domains import normalize_allowed_domains
from app.agents.extractor.report import ExtractListingsReport
from app.agents.extractor.steps.step_01_snippet_prefilter import collect_snippet_candidates
from app.agents.extractor.steps.step_02_listing_deterministic import deterministic_listings_from_candidates
from app.agents.extractor.steps.step_03_listing_llm_extract import llm_extract
from app.agents.extractor.steps.step_04_merge_llm_listings import merge_llm_rows_into_listings

logger = logging.getLogger("app.llm.extract_listings")


def _log_extraction_summary(diagnostic: dict[str, Any]) -> None:
    logger.info(
        "extraction_summary",
        extra={
            "stage": "extractor",
            "mode": diagnostic.get("extraction_mode"),
            "final_count": diagnostic.get("final_count", 0),
            "empty_reason": diagnostic.get("empty_reason"),
        },
    )


async def extract_listings(
    raw_results: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    allowed_domains: set[str],
    snippet_relax_hosts: frozenset[str] | None = None,
) -> ExtractListingsReport:
    diagnostic: dict[str, Any] = {
        "empty_reason": None,
        "prefilter_candidates": 0,
        "dropped_intent_mismatch": 0,
        "llm_rows_returned": 0,
        "json_parse_ok": True,
        "drop_url_not_in_candidates": 0,
        "drop_title_gate": 0,
        "drop_pydantic": 0,
        "extraction_mode": None,
        "snippet_relaxed_host_hint_count": (
            len(snippet_relax_hosts) if snippet_relax_hosts else 0
        ),
    }

    if not raw_results or not album:
        diagnostic["empty_reason"] = "no_raw_results_or_album"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    allowed = normalize_allowed_domains(allowed_domains)
    if not allowed:
        diagnostic["empty_reason"] = "empty_allowed_domains"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    artist_l = (artist or "").strip().lower() or None
    album_l = album.strip().lower()

    candidates, dropped_intent = collect_snippet_candidates(
        raw_results,
        artist_l=artist_l,
        album_l=album_l,
        allowed_hosts=allowed,
        snippet_relax_hosts=snippet_relax_hosts,
    )

    diagnostic["prefilter_candidates"] = len(candidates)
    diagnostic["dropped_intent_mismatch"] = dropped_intent

    logger.debug(
        "extract_listings_prefilter",
        extra={
            "stage": "extractor",
            "candidate_count": len(candidates),
            "dropped_intent_mismatch": dropped_intent,
            "snippet_relax_hosts_hint": diagnostic["snippet_relaxed_host_hint_count"],
        },
    )

    if not candidates:
        diagnostic["empty_reason"] = "prefilter_zero_candidates_intent_mismatch"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    if len(candidates) <= SMALL_BATCH_NO_LLM:
        diagnostic["extraction_mode"] = "deterministic_small_batch"
        det = deterministic_listings_from_candidates(
            candidates,
            artist=artist,
            album=album,
            snippet_relax_hosts=snippet_relax_hosts,
        )
        diagnostic["final_count"] = len(det)
        if not det:
            diagnostic["empty_reason"] = "deterministic_build_failed"
        else:
            diagnostic["empty_reason"] = None
        _log_extraction_summary(diagnostic)
        return ExtractListingsReport(listings=det, diagnostic=diagnostic)

    diagnostic["extraction_mode"] = "llm"
    extracted, raw_json = await llm_extract(candidates[:LLM_MAX_INPUT], diagnostic)
    diagnostic["llm_rows_returned"] = len(extracted)
    if not raw_json.strip() or raw_json.strip() == "{}":
        diagnostic["empty_reason"] = "llm_empty_response"
    if not extracted:
        diagnostic["empty_reason"] = diagnostic.get("empty_reason") or "llm_returned_empty_listings_array"

    out_list = merge_llm_rows_into_listings(
        extracted,
        candidates,
        artist=artist,
        album=album,
        artist_l=artist_l,
        album_l=album_l,
        diagnostic=diagnostic,
        snippet_relax_hosts=snippet_relax_hosts,
    )

    if out_list:
        diagnostic["empty_reason"] = None
    elif extracted:
        diagnostic["empty_reason"] = "post_llm_all_dropped"
    else:
        diagnostic["empty_reason"] = diagnostic.get("empty_reason") or (
            "llm_json_empty_or_failed" if not diagnostic.get("json_parse_ok", True) else "llm_returned_empty_listings_array"
        )

    diagnostic["final_count"] = len(out_list)
    _log_extraction_summary(diagnostic)
    return ExtractListingsReport(listings=out_list, diagnostic=diagnostic)
