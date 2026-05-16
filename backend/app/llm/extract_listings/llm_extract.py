"""OpenAI JSON extraction for search snippets."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.llm.extract_listings.constants import EXTRACTOR_SYSTEM_PROMPT

logger = logging.getLogger("app.llm.extract_listings")


async def llm_extract(
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Return parsed listing dicts and raw message content string."""
    if not candidates:
        return [], ""

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    user_payload = {
        "instructions": (
            "For each listings[] element, set `title` using ONLY substring material from "
            "that object's `title` + `content` fields. NEVER paste an artist/album that does "
            "not appear verbatim (or clearly as product naming) inside that same snippet blob."
        ),
        "listings": candidates,
    }

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "json\n" + json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    logger.debug(
        "extract_listings_llm_response_meta",
        extra={"stage": "extractor", "response_chars": len(raw)},
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        diagnostic["json_parse_ok"] = False
        logger.warning(
            "extract_listings_json_decode_error",
            extra={"stage": "extractor", "error": str(exc), "raw_head": raw[:500]},
        )
        return [], raw

    items = [x for x in data.get("listings", []) if isinstance(x, dict)]
    return items, raw
