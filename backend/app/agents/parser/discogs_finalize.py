"""Post-parse Discogs validation and track→album remapping."""

from __future__ import annotations

import logging

from app.agents.parser.compilation import looks_like_compilation
from app.agents.parser.intent import classify_intent
from app.agents.parser.track_resolver import resolve_track_to_album
from app.models.search_query import ParsedQuery

logger = logging.getLogger("app.agents.parser")


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


async def finalize_album_with_discogs(
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
    if not wants_compilation and looks_like_compilation(effective):
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
                    "intent_completeness": classify_intent(
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
                "intent_completeness": classify_intent(
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
                "intent_completeness": classify_intent(
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
                "intent_completeness": classify_intent(
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
            "intent_completeness": classify_intent(
                artist=parsed.artist,
                album=None,
                resolved_album=None,
            ),
        }
    )
