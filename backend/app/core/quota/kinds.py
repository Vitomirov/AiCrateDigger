"""Daily quota bucket identifiers (Phase 3 account spend fuse)."""

from __future__ import annotations

from enum import Enum


class QuotaKind(str, Enum):
    """Redis counter families — extend when adding new paid provider classes."""

    PARSE = "parse"
    TAVILY = "tavily"
    OPENAI_EXTRACT = "openai_extract"
