"""Snippet / search-result extraction (LLM + deterministic helpers).

Lazy package attributes keep ``import app.domains.engine.extraction.constants`` (and
similar) from eagerly loading OpenAI / listing schema when tests only need constants.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EXTRACTOR_SYSTEM_PROMPT",
    "ExtractListingsReport",
    "LISTINGS_EXTRACTOR_SYSTEM_PROMPT",
    "MAX_AI_BATCH_SIZE",
    "MAX_FINAL_RESULTS",
    "MIN_ARTIST_FUZZY",
    "MIN_TITLE_LEN",
    "extract_and_score_results",
    "extract_listings",
    "normalize_domain",
]


def __getattr__(name: str) -> Any:
    if name == "EXTRACTOR_SYSTEM_PROMPT":
        from app.domains.engine.extraction.constants import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name in {"MAX_AI_BATCH_SIZE", "MAX_FINAL_RESULTS", "MIN_ARTIST_FUZZY", "MIN_TITLE_LEN"}:
        from app.domains.engine.extraction import constants as _c

        return getattr(_c, name)
    if name == "normalize_domain":
        from app.domains.engine.extraction.hosts import normalize_domain

        return normalize_domain
    if name == "LISTINGS_EXTRACTOR_SYSTEM_PROMPT":
        from app.domains.engine.extraction.listing_constants import EXTRACTOR_SYSTEM_PROMPT as lp

        return lp
    if name == "ExtractListingsReport":
        from app.domains.engine.extraction.report import ExtractListingsReport

        return ExtractListingsReport
    if name == "extract_and_score_results":
        from app.domains.engine.extraction.candidates.step_13_candidate_pipeline import extract_and_score_results

        return extract_and_score_results
    if name == "extract_listings":
        from app.domains.engine.extraction.steps.step_05_listings_orchestrator import extract_listings

        return extract_listings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
