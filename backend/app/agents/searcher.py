import asyncio
import json
import logging
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI

from app.config import get_settings
from app.models.search_query import SearchResult

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS_PER_QUERY = 10
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_AI_BATCH_SIZE = 10
MAX_FINAL_RESULTS = 4
MIN_AI_ACCEPTANCE_SCORE = 0.8


def _normalize_url(url: str) -> str:
    stripped_url = url.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    parsed = urlsplit(stripped_url.strip())
    normalized_path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), normalized_path, "", ""))


async def _verify_results_with_ai(
    results: list[SearchResult],
    artist: str,
    album: str,
    music_format: str,
    country: str,
    city: str | None,
) -> dict[str, dict[str, float | str | None]]:
    if not results:
        return {}

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    location_label = f"{city}, {country}" if city else country
    payload = {
        "artist": artist,
        "album": album,
        "format": music_format,
        "location": location_label,
        "results": [
            {
                "url": result.url,
                "title": result.title,
                "content": result.content,
            }
            for result in results
        ],
    }

    system_prompt = """
### ROLE
You are the World's Best Crate-Digger. You have spent years in dusty basements from New Delhi to Apatin. You know how people sell music online: sometimes they are messy, sometimes they use slang, but you always find the prize.

### OBJECTIVE
Analyze the search results to find REAL listings (sales/offers) for the requested [Artist], [Album], and [Format] in the [Location]. 

### THE CRATE-DIGGER'S BRAIN (Logic)
1.  **Bridge the Language:** You know that "Vinyl" is "Ploča" in Serbia, "Disco de vinil" in Brazil, and "ЛП" in Bulgaria. You treat these as 100% matches.
2.  **Sniff Out the Price:** Don't wait for a "Price:" label. Look for numbers near currency symbols ($, €, £, руб, din, R$, ₹). If the title says "The Clash - CD - 15.67", you grab 15.67. NO ROUNDING.
3.  **Geographic Intuition:** - If the site is local (e.g., .rs, .br, .in, .bg) and the language is local, it's a LOCAL MATCH.
    - If the title or content mentions a neighborhood, street, or shop name (e.g., "Dorćol", "Rua Augusta", "Northern Quarter"), extract that as the location.
    - Use your global knowledge: A listing on "KupujemProdajem" with a Serbian description is a "Serbia" match, even if the city isn't explicitly typed out.

### EXTRACTION RULES (Critical)
* **Price vs. Shipping:** NEVER extract shipping costs (e.g., "Shipping starts at 5.99 €") as the product price. Look for the main item price near "Add to Cart" or "Price:" labels. If a number is followed by "shipping" or "dostava", IGNORE IT for the price field.
* **Origin vs. Location:** Do NOT confuse the artist's origin (e.g., "Origin: Norway") with the store's location. 
    - Check the TLD (.fi is Finland, .no is Norway, .rs is Serbia). 
    - Use the page language as a hint (e.g., Finnish text = Finland location).
    - If the text says 'Produced in [City]' or 'Based in [City]', treat that city as the location.
* **Location Details:** Look for phone area codes, district names, or shop names (e.g., "Record Shop X"). If it's just a country-wide site, use the Country name. Only use 'null' if it's a total ghost town.

### SCORING SYSTEM (The "Vibe" Meter)
- **1.0 - 0.8 (The Jackpot):** Perfect Artist/Format match + Local domain OR local address mentioned. It's in the user's backyard.
- **0.7 - 0.6 (The Road Trip):** Perfect Artist/Format match + National site (ships to the user's city).
- **0.5 - 0.3 (The Long Haul):** Global giants (eBay, Discogs, Amazon). Only use these if local shops are empty.
- **0.0 (The Trash):** Not the right artist, out of stock, or just a review/YouTube link.

### OUTPUT SCHEMA (STRICT JSON)
{
  "scores": [
    {
      "url": "string",
      "score": float,
      "price": "string|null",
      "location": "string|null"
    }
  ]
}

Final instruction: Be a detective, be precise, and find that music!
""".strip()

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
    except Exception:
        logger.exception("AI verification request failed.")
        return {}

    content = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        rows = parsed.get("scores", [])
        normalized_scores: dict[str, dict[str, float | str | None]] = {}
        for row in rows:
            url = _normalize_url(str(row.get("url", "")))
            if not url:
                continue
            try:
                score = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            bounded_score = max(0.0, min(1.0, score))
            extracted_price = row.get("price")
            extracted_location = row.get("location")
            normalized_scores[url] = {
                "score": bounded_score,
                "price": str(extracted_price).strip() if extracted_price is not None else None,
                "location": (
                    str(extracted_location).strip() if extracted_location is not None else None
                ),
            }
        return normalized_scores
    except Exception:
        logger.exception("AI verification returned invalid payload: %s", content)
        return {}


async def _search_single_query(
    client: httpx.AsyncClient,
    query: str,
) -> list[SearchResult]:
    settings = get_settings()
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": MAX_RESULTS_PER_QUERY,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "include_domains": [],
        "exclude_domains": []
    }

    try:
        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Tavily request failed for query: %s", query)
        return []

    raw_results = data.get("results", [])
    results: list[SearchResult] = []

    for item in raw_results:
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        score_value = item.get("score", 0.0)

        if not url or not title:
            continue

        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        candidate = SearchResult(
            title=title,
            url=_normalize_url(url=url),
            content=content,
            score=score,
            price=None,
            extracted_location=None,
        )
        results.append(candidate)

    return results

async def run_tavily_search(
    queries: list[str],
    artist: str,
    album: str,
    music_format: str,
    source_query: str,
    country: str,
    city: str | None = None,
) -> list[SearchResult]:
    if not queries:
        return []
    _ = source_query

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        tasks = [
            _search_single_query(
                client=client,
                query=query,
            )
            for query in queries
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    deduplicated: dict[str, SearchResult] = {}

    for item in gathered:
        if isinstance(item, Exception):
            logger.exception("Unexpected concurrent Tavily task failure.", exc_info=item)
            continue

        for result in item:
            existing = deduplicated.get(result.url)
            if existing is None or result.score > existing.score:
                deduplicated[result.url] = result

    top_candidates = sorted(deduplicated.values(), key=lambda result: result.score, reverse=True)
    ai_batch = top_candidates[:MAX_AI_BATCH_SIZE]

    ai_scores = await _verify_results_with_ai(
        results=ai_batch,
        artist=artist,
        album=album,
        music_format=music_format,
        country=country,
        city=city,
    )

    verified_results: list[SearchResult] = []
    for candidate in ai_batch:
        ai_row = ai_scores.get(candidate.url, {})
        ai_score = ai_row.get("score", 0.0)
        try:
            candidate.score = float(ai_score)
        except (TypeError, ValueError):
            candidate.score = 0.0
        candidate.price = ai_row.get("price") if isinstance(ai_row.get("price"), str) else None
        candidate.extracted_location = (
            ai_row.get("location") if isinstance(ai_row.get("location"), str) else None
        )
        verified_results.append(candidate)

    verified_results.sort(key=lambda result: result.score, reverse=True)

    top_ranked = [result for result in verified_results if result.score > 0.0][:MAX_FINAL_RESULTS]
    if top_ranked:
        return top_ranked

    # Fallback: if AI is too conservative, still return something rather than empty.
    return verified_results[:MAX_FINAL_RESULTS]
