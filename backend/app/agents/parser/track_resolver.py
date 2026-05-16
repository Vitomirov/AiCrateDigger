"""Track title → parent studio album via Discogs with LLM fallback."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.agents.parser.compilation import looks_like_compilation
from app.config import get_settings

logger = logging.getLogger("app.agents.parser")


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
            if not wants_compilation and looks_like_compilation(title):
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
    return await llm_track_to_album(artist, track, wants_compilation=wants_compilation)


async def llm_track_to_album(
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
        if not wants_compilation and looks_like_compilation(t):
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
