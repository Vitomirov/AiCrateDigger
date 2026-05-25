"""Query parsing: NL → structured music + geography, Discogs-assisted resolution."""

from __future__ import annotations

from typing import Any

__all__ = [
    "COMPILATION_REQUEST_KEYWORDS",
    "PARSER_SYSTEM_PROMPT",
    "ParserError",
    "parse_user_input",
    "parse_user_query",
    "resolve_track_to_album",
    "validate_album_with_discogs",
]


def __getattr__(name: str) -> Any:
    if name == "COMPILATION_REQUEST_KEYWORDS":
        from app.domains.query_parser.constants import COMPILATION_REQUEST_KEYWORDS

        return COMPILATION_REQUEST_KEYWORDS
    if name == "PARSER_SYSTEM_PROMPT":
        from app.domains.query_parser.constants import PARSER_SYSTEM_PROMPT

        return PARSER_SYSTEM_PROMPT
    if name == "ParserError":
        from app.domains.query_parser.errors import ParserError

        return ParserError
    if name == "parse_user_query":
        from app.domains.query_parser.parse_user_query import parse_user_query

        return parse_user_query
    if name == "parse_user_input":
        from app.domains.query_parser.steps.step_06_parser_pipeline import parse_user_input

        return parse_user_input
    if name == "validate_album_with_discogs":
        from app.domains.query_parser.steps.step_05_discogs_finalize import validate_album_with_discogs

        return validate_album_with_discogs
    if name == "resolve_track_to_album":
        from app.domains.query_parser.track_resolver import resolve_track_to_album

        return resolve_track_to_album
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
