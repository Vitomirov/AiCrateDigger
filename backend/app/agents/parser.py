import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.search_query import ParsedQuery

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """
You are Agent 1 (Parser) for AiCrateDigg.

Your task:
1) Parse a user music search query into a strict JSON object that matches the ParsedQuery schema.
2) Resolve relative album references using internal music knowledge:
   - Examples: "first album", "debut", "latest release", "2nd album".
   - Convert these to an explicit album/release title whenever possible.
3) Detect the language of the user query and map it to a primary country:
   - Example: Serbian -> Serbia. Norvegian -> Norway.
   - If the language is ambiguous (English/Spanish), default to UK for English and Spain for Spanish.
4) Normalize format to exactly one of: Vinyl, CD, Cassette.
5) Detect the city of the user query and map it to a city name.
6) The language field must always match the language of the original_query. 
example: If the user asks in Serbian about Germany, language is 'Serbian' and country is 'Germany'
6) Preserve the original user text in original_query.

Output requirements:
- Return ONLY valid JSON.
- Follow this exact schema:
  {
    "artist": "string",
    "album": "string",
    "format": "Vinyl|CD|Cassette",
    "country": "string",
    "city": "string",
    "language": "string",
    "original_query": "string"
  }
- Do not include extra keys.
""".strip()


async def parse_user_input(query: str) -> ParsedQuery:
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
        logger.exception("Parser agent request failed.")
        raise RuntimeError("Failed to parse query with parser agent.") from exc

    content = completion.choices[0].message.content or "{}"

    try:
        parsed_payload = json.loads(content)
        parsed_payload["original_query"] = query
        return ParsedQuery.model_validate(parsed_payload)
    except Exception as exc:
        logger.exception("Parser agent produced invalid payload: %s", content)
        raise RuntimeError("Parser agent returned invalid JSON payload.") from exc
