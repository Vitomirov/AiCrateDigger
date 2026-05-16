"""Agent 1 — Strict music + geography parser package.

Resolves NL queries into structured fields. Geography uses LLM-backed inference when
city is present without country (no deterministic city→country tables in code).

Exposes :func:`parse_user_input` (orchestrated Discogs-aware path) and
:func:`parse_user_query` (single-call contract used by the vinyl pipeline and ``/parse``).
"""

from __future__ import annotations

from app.agents.parser.constants import COMPILATION_REQUEST_KEYWORDS, PARSER_SYSTEM_PROMPT
from app.agents.parser.errors import ParserError
from app.agents.parser.parse_user_query import parse_user_query
from app.agents.parser.steps.step_05_discogs_finalize import validate_album_with_discogs
from app.agents.parser.steps.step_06_parser_pipeline import parse_user_input
from app.agents.parser.track_resolver import resolve_track_to_album

__all__ = [
    "COMPILATION_REQUEST_KEYWORDS",
    "PARSER_SYSTEM_PROMPT",
    "ParserError",
    "parse_user_input",
    "parse_user_query",
    "resolve_track_to_album",
    "validate_album_with_discogs",
]
