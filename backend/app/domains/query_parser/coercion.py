"""Coerce loosely-typed LLM JSON primitives into bounded forms."""

from __future__ import annotations

from typing import Any


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_format_literal(value: Any) -> str | None:
    s = safe_str(value)
    if not s:
        return None
    if s in {"Vinyl", "CD", "Cassette"}:
        return s
    lowered = s.lower()
    if "vinyl" in lowered or lowered in {"vinil", "lp"}:
        return "Vinyl"
    if lowered in {"cd"} or "compact disc" in lowered:
        return "CD"
    if "cassette" in lowered:
        return "Cassette"
    return None


def guess_language_cheap(query: str) -> str:
    return "English"
