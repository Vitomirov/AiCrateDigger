"""Geography — LLM-only when city present but country absent (no code tables)."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger("app.agents.parser")


async def infer_country_via_llm(
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


async def ensure_country_when_city_given(
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

    inferred = await infer_country_via_llm(ct, language_hint=language_hint)
    if inferred.strip():
        return inferred.strip()

    inferred2 = await infer_country_via_llm(ct, language_hint=language_hint, retry=True)
    if inferred2.strip():
        return inferred2.strip()

    inferred3 = await infer_country_via_llm(ct, language_hint=language_hint, insist=True)
    return inferred3.strip() if inferred3 else None
