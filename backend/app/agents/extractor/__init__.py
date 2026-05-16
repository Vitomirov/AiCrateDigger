"""Agent 3 — Extractor package.

Snippet/Tavily pipeline (``step_01``–``step_05`` + shared helpers): prefilter,
deterministic small-batch path, LLM extract, merge, orchestrator.

Search-result pipeline (``step_11``–``step_13``): deterministic pre-filter
(including merch gate), LLM structured scores, ``ListingResult`` assembly.

Public API mirrors the legacy flat ``agents.extractor`` module for README compatibility.
"""

from __future__ import annotations

from app.agents.extractor.constants import (
    MAX_AI_BATCH_SIZE,
    MAX_FINAL_RESULTS,
    MIN_ARTIST_FUZZY,
    MIN_TITLE_LEN,
    SYSTEM_PROMPT as EXTRACTOR_SYSTEM_PROMPT,
)
from app.agents.extractor.hosts import normalize_domain
from app.agents.extractor.listing_constants import EXTRACTOR_SYSTEM_PROMPT as LISTINGS_EXTRACTOR_SYSTEM_PROMPT
from app.agents.extractor.report import ExtractListingsReport
from app.agents.extractor.candidates.step_13_candidate_pipeline import extract_and_score_results
from app.agents.extractor.steps.step_05_listings_orchestrator import extract_listings

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
