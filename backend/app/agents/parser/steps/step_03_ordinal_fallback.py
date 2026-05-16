"""LLM ordinal resolution when Discogs ordinals yield no album."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger("app.agents.parser")


async def resolve_ordinal_via_llm_fallback(extracted: dict[str, Any], album_index: int) -> str | None:
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
