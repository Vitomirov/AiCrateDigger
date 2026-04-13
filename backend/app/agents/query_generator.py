import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.search_query import ParsedQuery, SearchQueries

logger = logging.getLogger(__name__)

QUERY_GENERATOR_SYSTEM_PROMPT = """
You are Agent 2 (Query Generator) for AiCrateDigger.
Return the output ONLY in valid JSON format.

Your task is to generate 3-5 surgical search queries to find buyable music listings.

STRICT CONSTRAINTS:
1. **Geo-Fence:** ONLY use marketplaces and domains that are physically or operationally 
    present in the target country. Do not hallucinate international sites 
    (e.g., do not use Allegro unless the country is Poland).
2. **The 'Site' Rule:** Use the `site:` operator for at least two queries using the most 
    dominant local domains (e.g., site:wallapop.com for Spain, site:kupujemprodajem.com for Serbia).
3. **Local Lingo:** At least one query MUST be in the target country's official
     language using high-intent keywords like "kaufen", "comprar", "vendre", "prodaja".
4. **No Editorial:** Avoid queries that lead to reviews, Wikipedia, or news.
     Focus on "item for sale", "price", "stock", "in inventory".

QUERY STRUCTURE:
- [Artist] [Album] [Format] [City/Country]
- site:[local_marketplace_domain] [Artist] [Album] [Format]
- [Local Language Keyword] [Artist] [Album] [Format] [City]

Output Schema:
{
  "queries": ["string"]
}
""".strip()


async def generate_search_queries(parsed_data: ParsedQuery) -> SearchQueries:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    payload = parsed_data.model_dump()

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": QUERY_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
    except Exception as exc:
        logger.exception("Query generator request failed.")
        raise RuntimeError("Failed to generate search queries.") from exc

    content = completion.choices[0].message.content or "{}"

    try:
        parsed_payload = json.loads(content)
        return SearchQueries.model_validate(parsed_payload)
    except Exception as exc:
        logger.exception("Query generator produced invalid payload: %s", content)
        raise RuntimeError("Query generator returned invalid JSON payload.") from exc
