"""OpenAI structured extraction pass for survivor snippets."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.agents.extractor.constants import LLM_LISTING_CONTENT_CHARS, SYSTEM_PROMPT
from app.config import get_settings
from app.models.search_query import SearchResult

logger = logging.getLogger("app.agents.extractor")


async def run_llm_extract(
    *,
    survivors: list[SearchResult],
    artist: str,
    album: str,
    music_format: str,
    city: str | None,
    country: str,
) -> list[dict]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    payload = {
        "target_artist": artist,
        "target_album": album,
        "target_format": music_format,
        "target_city": city,
        "target_country": country,
        "listings": [
            {"url": r.url, "title": r.title, "content": r.content[:LLM_LISTING_CONTENT_CHARS]}
            for r in survivors
        ],
    }
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        logger.exception("extractor_llm_failed", extra={"stage": "extractor", "status": "fail"})
        return []

    scores_list = data.get("scores", []) or []
    return [item for item in scores_list if isinstance(item, dict)]
