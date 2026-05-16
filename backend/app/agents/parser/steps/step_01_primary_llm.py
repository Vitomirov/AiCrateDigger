"""Primary structured LLM parse (music + geography JSON)."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.agents.parser.constants import PARSER_SYSTEM_PROMPT
from app.agents.parser.errors import ParserError
from app.config import get_settings

logger = logging.getLogger("app.agents.parser")


async def extract_json_with_primary_llm(query: str) -> dict[str, Any]:
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
