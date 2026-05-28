"""Parse a free-form user query into ParsedQuery JSON (single OpenAI call).

JSON-only output, temperature 0. Extracts literal music fields, resolves ordinal
album references into ``resolved_album``, and semantically resolves geography
(city/country name → ISO-3166-1 alpha-2 + ``local|regional|global`` scope).
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.domains.query_parser.parse_schema import ParsedQuery, ResolutionConfidence

SYSTEM_PROMPT = """You are an extraction-and-geocoding parser for a vinyl \
ecommerce search. Copy explicitly stated music fields verbatim, resolve ordinal \
album references into canonical studio album titles, and resolve the user's \
geography to a country code using your world knowledge.

Return STRICT JSON ONLY (one object, no markdown, no commentary).

OUTPUT SCHEMA — exact keys only, no extras:
{
  "artist":        string | null,
  "album":         string | null,
  "album_index":   number | null,
  "resolved_album": string | null,
  "location":      string | null,
  "country_code":  string | null,
  "search_scope":  "local" | "regional" | "global",
  "resolved_city": string | null,
  "geo_confidence": number | null,
  "geo_granularity": "city" | "country" | "region" | "none" | null
}

CORE RULES:
- Music extraction fields (artist, album, album_index, location) must reflect \
the user text — never invent artist names or literal album titles.
- ``resolved_album`` is the ONE music field where world knowledge is allowed \
(ordinal → canonical studio album title). When unsure → null.
- Geography fields (country_code, search_scope) are SEMANTIC RESOLUTION of \
``location``.

1. ARTIST
   - Set when a musician or band name appears as a contiguous substring in the \
user text (preserve their spelling/casing).
   - Ignore format words (section 7) when checking.
   - Else null.

2. ALBUM
   - Set ONLY when a named album title or potential title placeholder appears in the text.
   - Copy the substring verbatim (e.g., user writes "Bossanova" -> album = "Bossanova").
   - Ordinal phrases ("second album", "debut", "latest album") → album = null.

3. ALBUM_INDEX
   - Set ONLY when an ordinal/position phrase appears:
       first / debut / 1st → 1
       second / 2nd → 2
       third / 3rd → 3   (etc.)
       latest / newest / most recent (album) → -1
   - If both an explicit title and an ordinal appear: set album to the title, \
album_index = null, resolved_album = null.

4. RESOLVED_ALBUM (Canonical resolution)
   - If ``album_index`` is non-null (ordinal phrase like "latest"): use your world knowledge to resolve it to the canonical studio album title.
   - If ``album`` is non-null (user provided a title like "Bossanova"): use your world knowledge to validate and normalize it to the official, full studio album title (e.g., "Bossanova").
   - Exclude live albums, compilations, EPs, unless specified. If unsure -> null.

5. LOCATION (verbatim only)
   - Set only when a city or country name appears literally in the user text.
   - Do NOT translate, normalise, expand, or invent it.
   - If both city and country appear, prefer the city substring (more specific).

6. COUNTRY_CODE (semantic, country-level)
   - When `location` is set to a single city or country, output its ISO-3166-1 \
alpha-2 code in uppercase. Use the United Kingdom code "GB" (never "UK").
   - Examples (use your knowledge, do NOT rely on this list being exhaustive):
       "kragujevac"        → "RS"
       "Novi Sad"          → "RS"
       "Belgrade"          → "RS"
       "Zagreb"            → "HR"
       "Ljubljana"         → "SI"
       "Sarajevo"          → "BA"
       "Berlin"            → "DE"
       "London"            → "GB"
       "Paris"             → "FR"
       "Amsterdam"         → "NL"
       "Oslo"              → "NO"
       "Stockholm"         → "SE"
       "Copenhagen"        → "DK"
       "Helsinki"          → "FI"
       "Warsaw"            → "PL"
       "Prague"            → "CZ"
       "Budapest"          → "HU"
       "Vienna"            → "AT"
       "Brussels"          → "BE"
       "Barcelona" / "barselona" → "ES"
   - When `location` is multi-country (continent / region / trade bloc), \
country_code = null.
       "Europe", "EU", "the Balkans", "Scandinavia" → country_code = null
   - When `location` is null or genuinely ambiguous (e.g. a name that could \
match many countries with no extra signal) → null.

7. SEARCH_SCOPE
   - "local"    → `location` resolves to ONE country (city or country name).
   - "regional" → `location` is a multi-country region/continent/trade bloc \
(Europe, EU, EEA, Balkans, Scandinavia, Benelux, DACH, Iberia, Baltics, etc.).
   - "global"   → `location` is null OR cannot be resolved at all.

8. FORMAT WORDS — IGNORE everywhere (never location/artist/album)
   - vinyl, lp, record, ploča, vinil, vinile, schallplatte, kaseta, cassette, \
cd, and similar physical-format tokens.

9. RESOLVED_CITY
   - When `location` is a city name (including common misspellings), set \
`resolved_city` to the standard English city name (e.g. user "barselona" → \
"Barcelona"; "belgrad" → "Belgrade").
   - When `location` is a country name only → `resolved_city` = null.
   - Never invent a city when the user did not express one.

10. GEO_CONFIDENCE (0.0–1.0)
   - How sure you are about country_code + city/country classification.
   - High (≥0.9) for well-known cities; lower for ambiguous names.
   - null → omit (downstream will default).

11. GEO_GRANULARITY
   - "city"    → user intent is a specific city.
   - "country" → user named a country or only a country-like location.
   - "region"  → regional / multi-country intent (mirror search_scope regional).
   - "none"    → no usable geo (location null).
"""


def _strip_optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _normalize_album_index(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    if idx == 0:
        return None
    return idx


def _derive_album_fields(
    *,
    album: str | None,
    album_index: int | None,
    resolved_album: str | None,
) -> tuple[str | None, int | None, str | None, ResolutionConfidence]:
    """Apply fail-closed rules after the LLM JSON is parsed."""
    if album:
        return album, None, None, "unknown"
    if album_index is not None and resolved_album:
        return None, album_index, resolved_album, "high"
    if album_index is not None:
        return None, album_index, None, "low"
    return None, None, None, "unknown"


def _build_parsed_payload(data: dict[str, Any], query: str) -> dict[str, Any]:
    scope_raw = str(data.get("search_scope") or "").strip().lower()
    if scope_raw not in ("local", "regional", "global"):
        scope_raw = "global"

    gran_raw = data.get("geo_granularity")
    gran: str | None
    if gran_raw is None or str(gran_raw).strip() == "":
        gran = None
    else:
        gran = str(gran_raw).strip().lower()
        if gran not in ("city", "country", "region", "none"):
            gran = None

    gconf = data.get("geo_confidence")
    geo_confidence: float | None
    if gconf is None or gconf == "":
        geo_confidence = None
    else:
        try:
            geo_confidence = float(gconf)
        except (TypeError, ValueError):
            geo_confidence = None

    album = _strip_optional_str(data.get("album"))
    album_index = _normalize_album_index(data.get("album_index"))
    resolved_album = _strip_optional_str(data.get("resolved_album"))
    album, album_index, resolved_album, resolution_confidence = _derive_album_fields(
        album=album,
        album_index=album_index,
        resolved_album=resolved_album,
    )

    return {
        "artist": _strip_optional_str(data.get("artist")),
        "album": album,
        "album_index": album_index,
        "resolved_album": resolved_album,
        "resolution_confidence": resolution_confidence,
        "location": _strip_optional_str(data.get("location")),
        "country_code": data.get("country_code"),
        "search_scope": scope_raw,
        "resolved_city": data.get("resolved_city"),
        "geo_confidence": geo_confidence,
        "geo_granularity": gran,
        "original_query": query,
        "language": "unknown",
    }


async def parse_user_query(query: str) -> ParsedQuery:
    """Run a single OpenAI call and return the strict ParsedQuery contract."""
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

    return ParsedQuery.model_validate(_build_parsed_payload(data, query))
