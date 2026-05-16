"""Agent 3 — Extractor package.

Pipeline per candidate:
    1. Deterministic pre-filter  (gates: url / title-length / artist fuzzy / merch keywords)
    2. LLM structured extraction (price, location, availability, seller_type, reason)
    3. Pydantic validation       (ListingResult contract; bad shapes rejected)

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
from app.agents.extractor.pipeline import extract_and_score_results

__all__ = [
    "EXTRACTOR_SYSTEM_PROMPT",
    "MAX_AI_BATCH_SIZE",
    "MAX_FINAL_RESULTS",
    "MIN_ARTIST_FUZZY",
    "MIN_TITLE_LEN",
    "extract_and_score_results",
    "normalize_domain",
]
