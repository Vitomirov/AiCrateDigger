"""Multi-language physical-music format detector for the user query.

The parser (Agent 1) deliberately *ignores* format words so it can focus on
artist/album/location. The downstream pipeline still needs a format token for:

* the Redis cache key (``cratedigger:search:{format}:...``), so a ``vinyl``
  hit cannot collide with a future ``cd`` hit for the same artist/album, and
* the consolidated Tavily query string (``{format_type} shop ...``), which
  steers SERPs towards the right SKU type.

Detection is **regex-only**, no LLM call: the format vocabulary is small and
stable across European languages relevant to the catalogue.
"""

from __future__ import annotations

import re
from typing import Literal

FormatToken = Literal["vinyl", "cd", "cassette"]


_VINYL_TOKENS: tuple[str, ...] = (
    "vinyl",
    "vinyls",
    "vinil",
    "vinili",
    "vinile",
    "viniles",
    "vinilo",
    "vinilos",
    "vinyle",
    "vinyles",
    "schallplatte",
    "schallplatten",
    "ploca",
    "ploče",
    "ploca",
    "ploce",
    "lp",
    "lps",
    "12\"",
    "12in",
    "7\"",
    "7in",
    "record",
    "records",
    "gramofonska",
)


_CD_TOKENS: tuple[str, ...] = (
    "cd",
    "cds",
    "compact disc",
    "compact-disc",
)


_CASSETTE_TOKENS: tuple[str, ...] = (
    "cassette",
    "cassettes",
    "kaseta",
    "kasete",
    "mc",
    "tape",
    "tapes",
)


def _build_word_re(tokens: tuple[str, ...]) -> re.Pattern[str]:
    escaped = sorted({re.escape(t) for t in tokens}, key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


_VINYL_RE = _build_word_re(_VINYL_TOKENS)
_CD_RE = _build_word_re(_CD_TOKENS)
_CASSETTE_RE = _build_word_re(_CASSETTE_TOKENS)


def detect_format_token(original_query: str | None) -> FormatToken:
    """Return ``"vinyl"`` / ``"cd"`` / ``"cassette"`` based on user wording.

    Default is ``"vinyl"`` (the catalogue's primary intent) when no format
    token is found, so the cache key stays deterministic for ambiguous queries
    like ``"Pink Floyd The Wall Berlin"``.

    Ordering: cassette > CD > vinyl. CD is checked before vinyl so the literal
    ``"cd"`` token is never swallowed by the broader vinyl regex set.
    """
    text = (original_query or "").strip()
    if not text:
        return "vinyl"
    if _CASSETTE_RE.search(text):
        return "cassette"
    if _CD_RE.search(text):
        return "cd"
    if _VINYL_RE.search(text):
        return "vinyl"
    return "vinyl"


__all__ = ["FormatToken", "detect_format_token"]
