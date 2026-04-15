import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.search_query import ParsedQuery, SearchQueries

logger = logging.getLogger(__name__)

QUERY_GENERATOR_SYSTEM_PROMPT = """
You are Agent 2 (Query Generator) for AiCrateDigger. 
Your goal is to find music listings by thinking like a local collector in the target city.

STRICT RULES:
1. **No Technical Operators:** Do NOT use "-site:" or complex Google operators. Tavily works best with natural phrases.
2. **The "Local Shop" Mix:** - 1 Query for a major local marketplace (e.g., Gumtree for UK, KupujemProdajem for Serbia, Avito for Russia).
   - 2 Queries targeting physical record stores in the specific city (e.g., "Record shop in Manchester city center selling The Clash").
   - 1 Query targeting independent music dealers or specialized vinyl/CD forums.
   - 1 Query using a high-intent buying phrase in the local language.
3. **Marketplace Diversity:** Use eBay ONLY if it's the absolute last resort. Prefer local names.
4. **Specific Geography:** Use neighborhoods or city centers if a city is provided (e.g., "Northern Quarter Manchester" instead of just "Manchester").

QUERY PATTERNS:
- "Independent record stores in [City] having [Artist] [Album] [Format] in stock"
- "Buy [Artist] [Album] [Format] from local sellers in [City]"
- "[Local Marketplace Name] [Artist] [Album] [Format] [City]"
- "Used [Format] shops [City] [Artist] [Album]"

Output ONLY valid JSON:
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
