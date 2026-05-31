"""Daily quota bucket identifiers (account spend limits)."""

from __future__ import annotations

from enum import Enum


class QuotaKind(str, Enum):
    """Redis counter families — extend when adding new paid provider classes."""

    PARSE = "parse"
    TAVILY = "tavily"
    OPENAI_EXTRACT = "openai_extract"
