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
MAX_RESULTS_PER_QUERY = 6
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
) -> dict[str, float]:
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
You are a professional crate-digger.
For each result, determine if it is a REAL listing (sale/offer) for the requested artist/release and format in or near the requested location.
You must recognize prices in ANY currency (symbols or local words) and formats in ANY language.
Return strict JSON only with this shape:
{
  "scores": [
    {"url": "string", "score": 0.0}
  ]
}
Rules:
- score must be a float from 0.0 to 1.0.
- evaluate each URL independently.
- use title and snippet content evidence.

RANKING-FIRST POLICY (do not be overly strict):
- Prefer scoring and ranking over discarding.
- Do NOT return an empty set of good candidates if there are any results that match the requested Artist and Format.

LOCATION + LOCALITY (high-intensity local ranking):
- TLD SUPREMACY: always rank target-country TLDs at the very top when location is specified (e.g., .ru for Russia, .rs for Serbia).
- Contextual inference is allowed: if the page language appears local (e.g., Russian) and the domain is the target TLD, treat it as a LOCAL MATCH even if the city name is not present.
- If the domain is a known local marketplace (e.g., Avito, Youla) and language appears local, treat it as LOCAL MATCH for a city within that country (e.g., Moscow).

LOCAL VS GLOBAL WEIGHTS:
- Local Match score band (1.0 - 0.8): correct Artist + correct Format + (target-country TLD OR strong local-language signals).
- Regional/nearby score band (0.8 - 0.6): correct Artist + correct Format + in-country marketplace/shop that ships nationally (even if city unclear).
- Global/Proxy score band (0.5 - 0.3): correct Artist + correct Format + global site (Discogs/eBay/etc.) — ONLY use this band if no meaningful local/regional matches exist.

LANGUAGE BRIDGE:
- Treat these as identical for scoring purposes: "кассета", "kaseti", "cassette".
- Apply the same equivalence logic for the requested format in any language (you must bridge synonyms/transliterations).

OUTPUT GUIDANCE:
- Use score 0.0 only when it is clearly NOT a relevant listing for the requested Artist and Format.
- Otherwise always assign a score, and enforce the strict local-first ordering via higher scores.
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
        normalized_scores: dict[str, float] = {}
        for row in rows:
            url = _normalize_url(str(row.get("url", "")))
            if not url:
                continue
            try:
                score = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            bounded_score = max(0.0, min(1.0, score))
            normalized_scores[url] = bounded_score
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
            raw_price=None,
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
        ai_score = ai_scores.get(candidate.url, 0.0)
        candidate.score = ai_score
        verified_results.append(candidate)

    verified_results.sort(key=lambda result: result.score, reverse=True)

    top_ranked = [result for result in verified_results if result.score > 0.0][:MAX_FINAL_RESULTS]
    if top_ranked:
        return top_ranked

    # Fallback: if AI is too conservative, still return something rather than empty.
    return verified_results[:MAX_FINAL_RESULTS]
