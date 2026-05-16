"""LLM step 2 — map web-search snippets to ``Listing`` rows.

One LLM call with lenient extraction (unknown price/currency allowed), then
deterministic assembly for small batches. Exposes :class:`ExtractListingsReport`.
"""

from __future__ import annotations

from app.llm.extract_listings.constants import EXTRACTOR_SYSTEM_PROMPT
from app.llm.extract_listings.pipeline import extract_listings
from app.llm.extract_listings.report import ExtractListingsReport

__all__ = [
    "EXTRACTOR_SYSTEM_PROMPT",
    "ExtractListingsReport",
    "extract_listings",
]
