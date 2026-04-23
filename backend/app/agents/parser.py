"""Agent 1 — Permissive Parser (Deferred Resolution Architecture).

Contract:
- Music search queries are partial and intent-driven. The parser MUST NOT reject
  a query for missing album / format / country. It extracts what it can and
  DOWNGRADES confidence; downstream agents handle enrichment and fallbacks.
- `intent_completeness` is the primary signal for the query completion layer.
- The LLM is NEVER allowed to invent album titles. Relative references flow
  through `album_index` and get resolved deterministically via Discogs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import get_settings
from app.models.search_query import IntentCompleteness, ParsedQuery
from app.pipeline_context import stage_timer
from app.services.discogs_service import resolve_album_by_index

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """
You are Agent 1 (Parser) for AiCrateDigger. You extract facts; you do NOT guess titles.
Partial queries are EXPECTED. If a field is unknown, return null — do not fabricate.

### STRICT OUTPUT SCHEMA (JSON, no extra keys)
{
  "artist": "string|null",
  "album": "string|null",
  "album_index": number|null,
  "format": "Vinyl|CD|Cassette|null",
  "country": "string|null",
  "city": "string|null",
  "language": "string"
}

### FIELD RULES

- artist: verbatim from user query. Preserve diacritics (č, ć, š, đ, ž, ü, ß, é ...).
  If the query names NO artist at all, return null. Do NOT invent.

- album & album_index:
  * Specific title in the query ("Strange Days", "Kuku Lele") -> `album` verbatim, `album_index=null`.
  * Relative reference ("first", "2nd", "debut", "latest") -> `album=null`, `album_index=<ordinal>`.
      Map: first/debut -> 1, second -> 2, third -> 3, latest/newest/most recent -> -1.
  * No album mentioned at all -> both null. THIS IS ACCEPTABLE.
  * NEVER invent a title from a relative reference.

- format: normalize to one of Vinyl / CD / Cassette if clearly implied.
  Schallplatte/ploča/vinil/vinilo/vinyle/vinile -> Vinyl
  CD/sidi/cede -> CD
  Kassette/kazeta/casete/cassette -> Cassette
  If the query does not imply a physical format at all, return null.

- city: ONLY if the user explicitly names a city. Otherwise null. Normalize to English
  conventional spelling (Beograd -> Belgrade, München -> Munich, Niš stays Niš).

- country:
  * If `city` is set, derive country from the city (Belgrade -> Serbia, Munich -> Germany).
  * Else if the user explicitly names a country, use it.
  * Else if the query's language strongly implies one geography, you MAY return a best guess.
  * Else null.

- language: the language of the ORIGINAL QUERY, not the country.

Return only the JSON object. No explanations.
""".strip()


class ParserError(RuntimeError):
    """Structured parser failure (e.g. LLM dead, invalid JSON). Not raised for partial intents."""


async def parse_user_input(query: str) -> ParsedQuery:
    """End-to-end parse: LLM extraction -> optional Discogs resolution -> validated model.

    NEVER raises on incomplete user intent. Only raises `ParserError` on hard
    infrastructure failures (LLM unavailable, invalid JSON).
    """
    with stage_timer("parser", input={"query": query}) as rec:
        try:
            extracted = await _extract_with_llm(query)
        except ParserError:
            # Emit a structured partial ParsedQuery with intent_completeness=unknown so
            # the pipeline keeps flowing rather than returning a 5xx.
            fallback = _build_unknown_fallback(query)
            rec.status = "fail"
            rec.error = "parser_llm_failed_fallback_to_unknown"
            rec.output = fallback.model_dump()
            logger.exception(
                "parser_hard_fail_fallback",
                extra={
                    "stage": "parser",
                    "status": "fail",
                    "reason": "llm_failure",
                    "intent_completeness": "unknown",
                    "fallback_triggered": True,
                },
            )
            return fallback

        rec.extra["llm_extraction"] = extracted

        album_index = _safe_int(extracted.get("album_index"))
        album_literal = _safe_str(extracted.get("album"))
        artist = _safe_str(extracted.get("artist"))

        resolved_album: str | None = None
        confidence = "unknown"
        fallback_triggered = False

        if album_index is not None and album_index != 0 and artist:
            # Deterministic path: Discogs-first. Swallow all errors — this step
            # is ENRICHMENT, not gatekeeping.
            try:
                resolution = await resolve_album_by_index(artist=artist, album_index=album_index)
                if resolution.album is not None:
                    resolved_album = resolution.album.title
                    confidence = resolution.confidence
                else:
                    resolved_album = await _llm_fallback_resolve(extracted, album_index)
                    confidence = "low" if resolved_album else "unknown"
                    fallback_triggered = True
            except Exception:
                logger.exception(
                    "parser_discogs_enrichment_failed",
                    extra={
                        "stage": "parser",
                        "status": "fail",
                        "reason": "discogs_exception",
                        "fallback_triggered": True,
                    },
                )
                confidence = "unknown"
                fallback_triggered = True
        elif album_literal:
            confidence = "unknown"  # user gave a specific title; no resolution needed.

        intent = _classify_intent(
            artist=artist,
            album=album_literal,
            resolved_album=resolved_album,
        )

        payload: dict[str, Any] = {
            "artist": artist or None,
            "album": album_literal,
            "album_index": album_index,
            "resolved_album": resolved_album,
            "resolution_confidence": confidence,
            "format": _safe_format(extracted.get("format")),
            "country": _safe_str(extracted.get("country")),
            "city": _safe_str(extracted.get("city")),
            "language": _safe_str(extracted.get("language")) or "English",
            "original_query": query,
            "intent_completeness": intent,
        }

        try:
            parsed = ParsedQuery.model_validate(payload)
        except ValidationError as exc:
            # With the permissive schema this should be practically unreachable,
            # but if it DOES fire we still must not crash — emit a minimal shell.
            logger.exception(
                "parser_schema_validation_failed",
                extra={
                    "stage": "parser",
                    "status": "fail",
                    "reason": f"schema: {exc.errors()[:2]}",
                    "fallback_triggered": True,
                },
            )
            parsed = _build_unknown_fallback(query)

        rec.output = parsed.model_dump()
        rec.status = "success"

        logger.info(
            "parser_done",
            extra={
                "stage": "parser",
                "status": "success",
                "intent_completeness": parsed.intent_completeness,
                "missing_fields": parsed.missing_fields(),
                "fallback_triggered": fallback_triggered,
                "resolution_confidence": parsed.resolution_confidence,
            },
        )
        if confidence == "low":
            logger.warning(
                "parser_low_confidence",
                extra={
                    "stage": "parser",
                    "status": "success",
                    "reason": "discogs_miss_llm_fallback",
                    "artist": parsed.artist,
                    "album_index": parsed.album_index,
                    "resolved_album": parsed.resolved_album,
                },
            )
        return parsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_intent(
    *, artist: str | None, album: str | None, resolved_album: str | None
) -> IntentCompleteness:
    has_artist = bool((artist or "").strip())
    has_album = bool((album or "").strip() or (resolved_album or "").strip())
    if has_artist and has_album:
        return "complete"
    if has_artist or has_album:
        return "partial"
    return "unknown"


def _build_unknown_fallback(query: str) -> ParsedQuery:
    """Minimal ParsedQuery for the 'something broke / nothing extracted' path.

    Contains only the original query text + language guess. Everything else is null.
    The pipeline can still search on this — it will use `original_query` as a Tavily
    bootstrap probe."""
    return ParsedQuery(
        language=_guess_language_cheap(query),
        original_query=query,
        intent_completeness="unknown",
    )


def _guess_language_cheap(query: str) -> str:
    """Ultra-light heuristic — the parser LLM usually gives us this field, but if
    the LLM itself is dead we still need SOMETHING so the composer prompt works.
    Falls back to 'English' so the neutral LLM composer remains usable."""
    return "English"


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_format(value: Any) -> str | None:
    s = _safe_str(value)
    if not s:
        return None
    if s in {"Vinyl", "CD", "Cassette"}:
        return s
    return None


async def _extract_with_llm(query: str) -> dict[str, Any]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
    except Exception as exc:
        logger.exception("parser_llm_failed", extra={"stage": "parser", "status": "fail"})
        raise ParserError("Parser LLM call failed.") from exc

    content = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error(
            "parser_llm_invalid_json",
            extra={"stage": "parser", "status": "fail", "reason": "json_decode", "output": content[:500]},
        )
        raise ParserError("Parser returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ParserError("Parser returned non-object JSON.")
    return data


async def _llm_fallback_resolve(extracted: dict[str, Any], album_index: int) -> str | None:
    """Last-resort LLM-only album resolution — only used when Discogs misses.
    Returns None unless the LLM is highly confident (per prompt)."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    artist = str(extracted.get("artist", "")).strip()
    if not artist:
        return None

    ordinal_label = (
        "latest studio album"
        if album_index == -1
        else f"studio album number {album_index} (1-based, chronological)"
    )
    prompt = (
        f"Resolve the canonical studio album title for artist '{artist}', specifically "
        f"their {ordinal_label}. If you are not highly confident, return an empty string. "
        f'Return JSON of the form {{"title": "<title or empty>"}}.'
    )

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(completion.choices[0].message.content or "{}")
    except Exception:
        logger.exception("parser_llm_fallback_failed", extra={"stage": "parser", "status": "fail"})
        return None

    title = str(data.get("title") or "").strip()
    return title or None
