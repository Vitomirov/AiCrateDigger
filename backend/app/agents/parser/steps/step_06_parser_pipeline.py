"""Orchestrator: primary LLM → geo fill → ordinal / compilation handling → validate & finalize."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from app.agents.parser.coercion import normalize_format_literal, safe_int, safe_str
from app.agents.parser.compilation import looks_like_compilation, user_wants_compilation
from app.agents.parser.errors import ParserError
from app.agents.parser.steps.step_01_primary_llm import extract_json_with_primary_llm
from app.agents.parser.steps.step_02_geo_inference import ensure_country_when_city_given
from app.agents.parser.steps.step_03_ordinal_fallback import resolve_ordinal_via_llm_fallback
from app.agents.parser.steps.step_04_intent import build_unknown_fallback, classify_intent
from app.agents.parser.steps.step_05_discogs_finalize import finalize_album_with_discogs
from app.models.search_query import ParsedQuery
from app.pipeline_context import stage_timer
from app.services.discogs_service import resolve_album_by_index

logger = logging.getLogger("app.agents.parser")


async def parse_user_input(query: str) -> ParsedQuery:
    """Parse -> optional Discogs enrichment -> forced country-if-city inference -> ParsedQuery."""

    wants_compilation = user_wants_compilation(query)

    with stage_timer("parser", input={"query": query}) as rec:
        try:
            extracted = await extract_json_with_primary_llm(query)
        except ParserError:
            fallback = build_unknown_fallback(query)
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

        artist = safe_str(extracted.get("artist"))
        album_literal = safe_str(extracted.get("album"))
        album_index = safe_int(extracted.get("album_index"))
        release_is_track = bool(extracted.get("release_is_track_title"))
        city_before = safe_str(extracted.get("city"))
        country_llm = safe_str(extracted.get("country"))
        language_hint = safe_str(extracted.get("language")) or "English"

        resolved_album: str | None = None
        confidence = "unknown"
        fallback_triggered = False

        # --- Country: MUST NOT be missing when city is present (secondary LLM, no geo tables).
        city = city_before
        country = await ensure_country_when_city_given(
            city=city_before,
            country=country_llm,
            language_hint=language_hint,
        )

        album_work = album_literal

        # If the LLM resolved the album to a compilation but user did NOT ask for one,
        # treat the literal as a track title so downstream resolution can find the studio album.
        if album_work and not wants_compilation and looks_like_compilation(album_work):
            logger.warning(
                "parser_llm_returned_compilation_override",
                extra={
                    "stage": "parser",
                    "status": "warn",
                    "album_literal": album_work,
                    "artist": artist,
                },
            )
            release_is_track = True  # force track→album resolution downstream

        # --- Ordinal album resolution (existing Discogs path).
        if album_index is not None and album_index != 0 and artist:
            try:
                resolution = await resolve_album_by_index(artist=artist, album_index=album_index)
                if resolution.album is not None:
                    resolved_album = resolution.album.title
                    confidence = str(resolution.confidence)
                    album_work = None
                    release_is_track = False
                else:
                    resolved_album = await resolve_ordinal_via_llm_fallback(extracted, album_index)
                    confidence = "low" if resolved_album else "low"
                    fallback_triggered = True
                    if resolved_album:
                        album_work = None
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
                confidence = "low"
                fallback_triggered = True

        # TRACK → parent ALBUM happens in `finalize_album_with_discogs` after validation.

        intent = classify_intent(
            artist=artist,
            album=album_work,
            resolved_album=resolved_album,
        )

        payload: dict[str, Any] = {
            "artist": artist or None,
            "album": album_work if not resolved_album else None,
            "album_index": album_index if album_index is None or album_index != 0 else None,
            "resolved_album": resolved_album if resolved_album else None,
            "resolution_confidence": confidence,
            "format": normalize_format_literal(extracted.get("format")),
            "country": country,
            "city": city,
            "language": language_hint,
            "original_query": query,
            "intent_completeness": intent,
        }

        try:
            parsed = ParsedQuery.model_validate(payload)
            parsed = await finalize_album_with_discogs(
                parsed,
                from_ordinal_resolve=(
                    album_index not in (None, 0) and bool(resolved_album)
                ),
                wants_compilation=wants_compilation,
            )
        except ValidationError as exc:
            logger.exception(
                "parser_schema_validation_failed",
                extra={
                    "stage": "parser",
                    "status": "fail",
                    "reason": f"schema: {exc.errors()[:2]}",
                    "fallback_triggered": True,
                },
            )
            parsed = build_unknown_fallback(query)

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
                    "reason": "discogs_or_llm_fallback",
                    "artist": parsed.artist,
                    "album_index": parsed.album_index,
                    "resolved_album": parsed.resolved_album,
                },
            )
        return parsed
