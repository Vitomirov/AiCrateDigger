"""LLM step 1 — parse a free-form user query into ParsedQuery JSON.

Single OpenAI call, JSON-only output, temperature 0. Extraction only: no
inference, no guessing, no filling gaps. Discogs runs downstream.
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from app.config import get_settings
from app.domain.parse_schema import ParsedQuery

SYSTEM_PROMPT = """You are an extraction-only parser. Your job is to copy \
explicitly stated information from the user's text into JSON. You do NOT \
reason about intent, probability, or what the user "probably means".

Return STRICT JSON ONLY (one object, no markdown, no commentary).

OUTPUT SCHEMA — exact keys only, no extras:
{
  "artist":       string | null,
  "album":        string | null,
  "album_index":  number | null,
  "location":     string | null
}

CORE RULE (NON-NEGOTIABLE):
- Output a non-null value ONLY when that value is explicitly supported by \
the user text (verbatim substring, or album_index via the ordinal mapping \
below).
- If the text does not explicitly support a field → null.
- Never invent, infer, extrapolate, choose "most likely", or use world \
knowledge to fill a field.
- Uncertainty or ambiguity for a field → null for that field (never a guess).

1. ARTIST
   - Set artist only if a musician or band name appears as a contiguous \
substring in the user text (same spelling as written; do not correct typos \
or casing).
   - After ignoring format words (section 6), that substring must still be \
present in what remains of the text.
   - If no such substring → null.

2. ALBUM
   - Set album ONLY when the user states an explicit album title (a named \
release title) that appears in the text.
   - NEVER set album from ordinal phrases ("second album", "2nd album", \
"debut", "latest album", etc.). Those map to album_index only.
   - Ordinal-only discography references → album = null.
   - If it is unclear whether a phrase is a title vs a track, descriptor, \
or format → null.

3. ALBUM_INDEX
   - Set ONLY when an ordinal / position phrase appears in the text:
       "first" / "debut" / "1st" → 1
       "second" / "2nd" → 2
       "third" / "3rd" → 3
       (same pattern for higher ordinals when written out or as Nth)
       "latest" / "newest" / "most recent" (referring to an album) → -1
   - If no such phrase → null.
   - If both an explicit album title and an ordinal appear: set album to \
the verbatim title and set album_index = null (never output both).

4. LOCATION
   - Set location only when a city or country name appears literally in the \
user text (verbatim substring).
   - Do not translate, normalise, expand to a region, or infer location \
from language, context, or genre.
   - If both city and country appear explicitly, prefer the city substring as \
location (more specific literal mention).
   - If none → null.

5. NO HALLUCINATION
   - Any field not explicitly supported by the text → null for that field.
   - No extra keys. JSON only.

6. FORMAT WORDS — IGNORE (never assign to any field)
   - Tokens like: vinyl, lp, record, ploča, vinile, schallplatte, and similar \
physical-format words are not artist, album title, or location.
"""


async def parse_user_query(query: str) -> ParsedQuery:
    """Run a single OpenAI call and return the strict ParsedQuery contract.

    Fails fast: invalid JSON, schema mismatch, or transport errors propagate
    to the caller.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    if not isinstance(data, dict):
        msg = "Parser returned non-object JSON."
        raise ValueError(msg)
    llm_only = {k: data.get(k) for k in ("artist", "album", "album_index", "location")}
    return ParsedQuery.model_validate(
        {
            **llm_only,
            "original_query": query,
            "language": "unknown",
        }
    )
