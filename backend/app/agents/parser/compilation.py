"""Compilation / best-of detection (single keyword source of truth)."""

from __future__ import annotations

from app.agents.parser.constants import COMPILATION_REQUEST_KEYWORDS


def query_text_contains_compilation_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in COMPILATION_REQUEST_KEYWORDS)


def user_wants_compilation(query: str) -> bool:
    """Returns True when the user explicitly requests a compilation / best-of."""
    return query_text_contains_compilation_keyword(query)


def looks_like_compilation(title: str) -> bool:
    """Heuristic: does this title look like a compilation / best-of / greatest-hits?"""
    return query_text_contains_compilation_keyword(title)
