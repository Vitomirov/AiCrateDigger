"""Agent 1 — Strict music + geography parser.

Resolves NL queries into structured fields. Track titles are folded into parent album
releases via Discogs when indicated. Geography: country MUST be inferred when a
city is present (LLM-backed inference — no deterministic city→country tables in code).
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

# Keywords that indicate the user explicitly wants a compilation/best-of.
_COMPILATION_REQUEST_KEYWORDS: tuple[str, ...] = (
    "best of",
    "greatest hits",
    "compilation",
    "anthology",
)

PARSER_SYSTEM_PROMPT = """
You are a strict music + geography parser for AiCrateDigger.

TASK
Read the user's natural-language query and return STRICT JSON ONLY (single object, no markdown).

OUTPUT SCHEMA (exact keys):

{
  "artist": string | null,
  "album": string | null,
  "album_index": number | null,
  "release_is_track_title": boolean,
  "format": "Vinyl" | "CD" | "Cassette" | null,
  "city": string | null,
  "country": string | null,
  "language": string
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUSIC RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Extract the primary MUSICIAN / BAND as `artist` (preserve diacritics).

2. `album` field:
   - If the user names a confirmed STUDIO RELEASE → put that exact canonical TITLE here.
     Only do this when you are CERTAIN it is a studio album title, not a track title.
   - If the user names a SINGLE SONG/TRACK → put THAT PHRASE in `album` AND set
     `release_is_track_title` = true. The downstream resolver will find the parent album.
   - Ordinal references ("2nd album", "debut") → set `album` null, track flag false,
     and set `album_index` (1=debut, 2=second, …, -1=latest). Never invent a title from ordinals.

3. ALBUM RESOLUTION — 2-STEP DETERMINISTIC:

   STEP 1 — Candidate generation:
   When a partial title, ambiguous phrase, or song title is detected:
   - Mentally list up to 3 STUDIO album candidates for (artist + phrase).
   - EXCLUDE every compilation / best-of / greatest-hits / anthology — unless the user
     explicitly uses the words "best of", "greatest hits", "compilation", or "anthology".
   - Prefer chronologically earlier releases when multiple studio albums match equally.

   STEP 2 — Candidate selection (score-based, pick highest):
     +2  album title contains the query phrase as a substring
     +2  confirmed studio album (not live, not compilation)
     +1  artist name strongly matches
     -3  compilation / best-of / greatest hits (DISQUALIFIED if user did not ask for it)
   Pick the SINGLE highest-scoring candidate. No randomness. No fallback guessing.

4. COMPILATION EXCLUSION RULE (ABSOLUTE):
   UNLESS the query contains "best of", "greatest hits", "compilation", or "anthology":
   - NEVER set `album` to a compilation or best-of title.
   - NEVER use a song's presence on a compilation to resolve the album field.
   - When a song maps to both a studio album and a compilation → ALWAYS pick the studio album.

   REGRESSION EXAMPLE (must be correct):
   Query: "High Hopes Pink Floyd CD Marseille"
   Correct: artist="Pink Floyd", album="High Hopes", release_is_track_title=true
   WRONG:   album="Echoes (The Best Of Pink Floyd)"   ← NEVER do this

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOGRAPHY RULES  (GEO IS MARKETPLACE-ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GEO ISOLATION (ABSOLUTE):
  city and country are used EXCLUSIVELY for downstream marketplace search.
  They MUST NOT influence album selection, artist resolution, album_index, or confidence.
  Completely ignore city / country when resolving any music metadata.

1. Extract `city` when the query names or clearly implies ONE city.
2. Extract `country` only when explicitly stated ("in France", "Serbia").
3. COUNTRY WHEN CITY IS KNOWN — CRITICAL:
   - If `city` is non-null AND user did NOT name a country → INFER the correct sovereign state
     using world geography knowledge (English name: France, Germany, Norway, Serbia, …).
   - Examples: Marseille → France; Berlin → Germany; Oslo → Norway; Belgrade → Serbia.
   - Always populate `country` when `city` is populated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Set `language` to the predominant language OF THE ORIGINAL QUERY text (ISO language name:
English, Serbian, French, German, …). If ambiguous, derive from inferred `country`/region.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Normalize physical format synonyms to exactly Vinyl | CD | Cassette or null when absent.
- Be conservative about invented titles except where country/city inference is required.
"""


class ParserError(RuntimeError):
    """Structured parser failure (e.g. LLM dead, invalid JSON). Not raised for partial intents."""


async def parse_user_input(query: str) -> ParsedQuery:
    """Parse -> optional Discogs enrichment -> forced country-if-city inference -> ParsedQuery."""

    wants_compilation = _user_wants_compilation(query)

    with stage_timer("parser", input={"query": query}) as rec:
        try:
            extracted = await _extract_with_llm(query)
        except ParserError:
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

        artist = _safe_str(extracted.get("artist"))
        album_literal = _safe_str(extracted.get("album"))
        album_index = _safe_int(extracted.get("album_index"))
        release_is_track = bool(extracted.get("release_is_track_title"))
        city_before = _safe_str(extracted.get("city"))
        country_llm = _safe_str(extracted.get("country"))
        language_hint = _safe_str(extracted.get("language")) or "English"

        resolved_album: str | None = None
        confidence = "unknown"
        fallback_triggered = False

        # --- Country: MUST NOT be missing when city is present (secondary LLM, no geo tables).
        city = city_before
        country = await _ensure_country_when_city_given(
            city=city_before,
            country=country_llm,
            language_hint=language_hint,
        )

        album_work = album_literal

        # If the LLM resolved the album to a compilation but user did NOT ask for one,
        # treat the literal as a track title so downstream resolution can find the studio album.
        if album_work and not wants_compilation and _looks_like_compilation(album_work):
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
                    resolved_album = await _llm_fallback_resolve(extracted, album_index)
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

        # TRACK → parent ALBUM happens in `_finalize_album_with_discogs` after validation.

        intent = _classify_intent(
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
            "format": _safe_format(extracted.get("format")),
            "country": country,
            "city": city,
            "language": language_hint,
            "original_query": query,
            "intent_completeness": intent,
        }

        try:
            parsed = ParsedQuery.model_validate(payload)
            parsed = await _finalize_album_with_discogs(
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
                    "reason": "discogs_or_llm_fallback",
                    "artist": parsed.artist,
                    "album_index": parsed.album_index,
                    "resolved_album": parsed.resolved_album,
                },
            )
        return parsed


async def validate_album_with_discogs(artist: str, album: str) -> bool:
    """Returns True if album exists in Discogs studio discography."""

    from app.services.discogs_service import get_artist_discography

    try:
        albums = await get_artist_discography(artist)
        normalized = album.lower()
        for a in albums:
            if normalized in a.title.lower():
                return True
        return False
    except Exception:
        return True  # fail-open (do not break pipeline)


async def resolve_track_to_album(
    artist: str, track: str, *, wants_compilation: bool = False
) -> str | None:
    """Resolve a track title to its parent studio album.

    Strategy:
    1. Search Discogs for (artist + track) and apply deterministic scoring.
       Compilations are hard-rejected unless `wants_compilation=True`.
    2. If Discogs yields nothing useful, fall back to a 2-step LLM resolution.
    """
    from app.services.discogs_service import DiscogsAlbum, search_release_by_track

    try:
        results = await search_release_by_track(artist, track)
    except Exception:
        results = []

    if results:
        def _score(r: DiscogsAlbum) -> int:
            title = r.title.lower()
            # Hard-reject compilations when user did not ask for one.
            if not wants_compilation and _looks_like_compilation(title):
                return 9999
            s = 0
            if any(kw in title for kw in ("live", "concert", "in concert")):
                s += 100  # deprioritise live releases
            return s

        ranked = sorted(results, key=_score)
        best_score = _score(ranked[0])
        if best_score < 9999:
            return ranked[0].title

    # Discogs had no usable studio result → ask LLM.
    return await _llm_track_to_album(artist, track, wants_compilation=wants_compilation)


async def _llm_track_to_album(
    artist: str, track: str, *, wants_compilation: bool = False
) -> str | None:
    """2-step LLM resolution: generate studio album candidates, then score and pick the best.

    Step 1 — candidate generation via LLM (temperature 0, anti-compilation bias).
    Step 2 — deterministic scoring applied in-process (no LLM randomness).
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    compilation_instruction = (
        ""
        if wants_compilation
        else (
            "EXCLUDE compilations, best-of, greatest hits, and anthologies. "
            "Only list STUDIO albums."
        )
    )

    step1_prompt = (
        f"Artist: {artist}\n"
        f"Track: \"{track}\"\n\n"
        f"List up to 3 albums by {artist} that contain the track \"{track}\". "
        f"{compilation_instruction}\n"
        f'Return JSON only: {{"candidates": ["<album1>", "<album2>", "<album3>"]}}\n'
        f"If you are not sure which album contains this track, return your best guesses. "
        f"Never return an empty list if the artist is well-known."
    )

    try:
        r1 = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": step1_prompt}],
        )
        data1 = json.loads(r1.choices[0].message.content or "{}")
        candidates: list[str] = [
            str(c).strip() for c in (data1.get("candidates") or []) if c
        ]
    except Exception:
        logger.exception("llm_track_to_album_step1_failed", extra={"stage": "parser"})
        return None

    if not candidates:
        return None

    # Step 2: deterministic scoring — no LLM involvement.
    track_lower = track.lower()

    def _score_candidate(title: str) -> int:
        t = title.lower()
        if not wants_compilation and _looks_like_compilation(t):
            return 9999  # hard-reject
        s = 0
        if any(kw in t for kw in ("live", "concert", "in concert")):
            s += 100  # deprioritise live
        if track_lower in t:
            s -= 5  # small boost for titled-track albums
        return s

    ranked = sorted(candidates, key=_score_candidate)
    best = ranked[0]

    if _score_candidate(best) >= 9999:
        logger.warning(
            "llm_track_to_album_all_compilations",
            extra={"stage": "parser", "artist": artist, "track": track},
        )
        return None

    logger.info(
        "llm_track_to_album_resolved",
        extra={"stage": "parser", "artist": artist, "track": track, "resolved": best},
    )
    return best


async def _finalize_album_with_discogs(
    parsed: ParsedQuery,
    *,
    from_ordinal_resolve: bool,
    wants_compilation: bool = False,
) -> ParsedQuery:
    """After LLM parsing: validate album-shaped strings against Discogs; remap tracks."""

    artist = (parsed.artist or "").strip()
    effective = ((parsed.resolved_album or parsed.album) or "").strip()
    if not artist or not effective:
        return parsed

    # If effective looks like a compilation and user did not ask for one, skip straight to
    # track resolution — treat the string as a (bad) track query rather than an album title.
    if not wants_compilation and _looks_like_compilation(effective):
        logger.warning(
            "parser_finalize_compilation_bypassed",
            extra={"stage": "parser", "effective": effective, "artist": artist},
        )
        resolved = await resolve_track_to_album(
            artist, effective, wants_compilation=False
        )
        if resolved:
            return parsed.model_copy(
                update={
                    "album": resolved,
                    "resolved_album": resolved,
                    "resolution_confidence": "medium",
                    "intent_completeness": _classify_intent(
                        artist=parsed.artist,
                        album=resolved,
                        resolved_album=resolved,
                    ),
                }
            )
        # Could not salvage — clear the compilation artifact.
        return parsed.model_copy(
            update={
                "album": None,
                "resolved_album": None,
                "resolution_confidence": "low",
                "intent_completeness": _classify_intent(
                    artist=parsed.artist,
                    album=None,
                    resolved_album=None,
                ),
            }
        )

    is_valid = await validate_album_with_discogs(artist, effective)

    if is_valid:
        return parsed

    resolved = await resolve_track_to_album(
        artist, effective, wants_compilation=wants_compilation
    )

    if resolved:
        return parsed.model_copy(
            update={
                "album": resolved,
                "resolved_album": resolved,
                "resolution_confidence": "high",
                "intent_completeness": _classify_intent(
                    artist=parsed.artist,
                    album=resolved,
                    resolved_album=resolved,
                ),
            }
        )

    if from_ordinal_resolve:
        return parsed.model_copy(
            update={
                "resolution_confidence": "medium",
                "intent_completeness": _classify_intent(
                    artist=parsed.artist,
                    album=parsed.album,
                    resolved_album=parsed.resolved_album,
                ),
            }
        )

    # Track title was not resolvable to a canonical studio album.
    return parsed.model_copy(
        update={
            "album": None,
            "resolved_album": None,
            "resolution_confidence": "low",
            "intent_completeness": _classify_intent(
                artist=parsed.artist,
                album=None,
                resolved_album=None,
            ),
        }
    )


# ---------------------------------------------------------------------------
# Geography — LLM-only when city present but country absent (NO code tables).
# ---------------------------------------------------------------------------


async def _ensure_country_when_city_given(
    *,
    city: str | None,
    country: str | None,
    language_hint: str,
) -> str | None:
    """If LLM omitted country despite knowing city, ask a compact model call for sovereignty."""
    ct = (city or "").strip()
    if not ct:
        return country
    if (country or "").strip():
        return country.strip()

    inferred = await _infer_country_via_llm(ct, language_hint=language_hint)
    if inferred.strip():
        return inferred.strip()

    inferred2 = await _infer_country_via_llm(
        ct, language_hint=language_hint, retry=True
    )
    if inferred2.strip():
        return inferred2.strip()

    inferred3 = await _infer_country_via_llm(ct, language_hint=language_hint, insist=True)
    return inferred3.strip() if inferred3 else None


async def _infer_country_via_llm(
    city: str,
    *,
    language_hint: str,
    retry: bool = False,
    insist: bool = False,
) -> str:
    """Single-purpose JSON inference — no deterministic city lookup tables."""

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    base = (
        f'Infer the sovereign STATE (English short name preferred) whose territory contains '
        f'the locality "{city}". Return JSON only:'
        '{"country":"<name>"}'
    )
    if retry:
        base += (
            " Respond with ONLY a real UN member sovereign state commonly associated with "
            "this locality. If disputed, prefer the universally recognized administering state "
            "(e.g., Berlin -> Germany)."
        )
    if insist:
        base += (
            " You MUST produce a non-empty sovereign state name. Never return null or unknown."
        )

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"Detected query language hint: {language_hint}."},
                {"role": "user", "content": base},
            ],
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        return str(data.get("country") or "").strip()
    except Exception:
        logger.exception("parser_country_inference_failed")
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_wants_compilation(query: str) -> bool:
    """Returns True when the user explicitly requests a compilation / best-of."""
    lower = query.lower()
    return any(kw in lower for kw in _COMPILATION_REQUEST_KEYWORDS)


def _looks_like_compilation(title: str) -> bool:
    """Heuristic: does this title look like a compilation / best-of / greatest-hits?"""
    lower = title.lower()
    return any(kw in lower for kw in _COMPILATION_REQUEST_KEYWORDS)


def _classify_intent(
    *,
    artist: str | None,
    album: str | None,
    resolved_album: str | None,
) -> IntentCompleteness:
    has_artist = bool((artist or "").strip())
    has_album = bool((album or "").strip() or (resolved_album or "").strip())
    if has_artist and has_album:
        return "complete"
    if has_artist or has_album:
        return "partial"
    return "unknown"


def _build_unknown_fallback(query: str) -> ParsedQuery:
    return ParsedQuery(
        language=_guess_language_cheap(query),
        original_query=query,
        intent_completeness="unknown",
    )


def _guess_language_cheap(query: str) -> str:
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
    lowered = s.lower()
    if "vinyl" in lowered or lowered in {"vinil", "lp"}:
        return "Vinyl"
    if lowered in {"cd"} or "compact disc" in lowered:
        return "CD"
    if "cassette" in lowered:
        return "Cassette"
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
