"""Tavily search-engine probes for indie record shops in a target city."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import get_settings
from app.domains.engine.search import fetch_tavily_results_body
from app.domains.engine.search.country_boost import tavily_country_from_iso3166_alpha2
from app.domains.engine.search.store_discovery.models import TAVILY_MAX_RESULTS, TAVILY_TIMEOUT_S

logger = logging.getLogger(__name__)


def build_probe_queries(city: str, country_code: str) -> list[str]:
    """Return the diverse probe queries fired against Tavily.

    Two complementary queries: a "best of" listicle hook (great for landing on
    listicle pages whose snippets enumerate concrete shop domains) and a
    direct shop-shaped query (great for landing on shop home / contact pages).
    The country code is appended so Tavily prefers TLD-matching results even
    before the structured ``country`` parameter kicks in.
    """
    cc = (country_code or "").strip().upper()
    city_clean = city.strip()
    base = [
        f"best independent record stores {city_clean} {cc} vinyl shop",
        f"vinyl record shop {city_clean} {cc} buy LP",
    ]
    return [q.strip() for q in base if q.strip()]


async def tavily_single_probe(
    *,
    client: httpx.AsyncClient,
    query: str,
    tavily_country: str | None,
) -> list[dict[str, str]]:
    """One Tavily HTTP call — returns the cleaned ``{title,url,content}`` list."""
    settings = get_settings()
    payload: dict[str, object] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        # Bug fix: Tavily's REST API parameter is ``max_results``, not
        # ``max_results_per_query``. The mis-named key was silently ignored,
        # so the probe was returning Tavily's default (5) instead of the
        # requested cap — starving the LLM verifier of evidence.
        "max_results": TAVILY_MAX_RESULTS,
    }
    if tavily_country:
        payload["topic"] = "general"
        payload["country"] = tavily_country
    data = await fetch_tavily_results_body(client, payload, query_for_log=query)
    if data is None:
        logger.warning(
            "store_discovery_tavily_http_error",
            extra={
                "stage": "store_discovery",
                "reason": "tavily_request_failed_after_retries",
                "query": query[:200],
            },
        )
        return []
    raw_items = data.get("results", []) or []
    cleaned: list[dict[str, str]] = []
    for item in raw_items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        cleaned.append(
            {
                "title": str(item.get("title", "")).strip()[:240],
                "url": url,
                "content": str(item.get("content", "")).strip()[:1500],
            }
        )
    return cleaned


async def tavily_probe(city: str, country_code: str) -> list[dict[str, str]]:
    """Multi-query search-engine probe for indie record shops in the target city.

    Returns a deduplicated list of raw ``{title, url, content}`` dicts across
    every probe query. Empty list on any HTTP / parse error.
    """
    tavily_country = tavily_country_from_iso3166_alpha2(country_code)
    queries = build_probe_queries(city, country_code)

    async with httpx.AsyncClient(timeout=TAVILY_TIMEOUT_S) as client:
        probes = await asyncio.gather(
            *(
                tavily_single_probe(
                    client=client,
                    query=q,
                    tavily_country=tavily_country,
                )
                for q in queries
            ),
            return_exceptions=True,
        )

    merged: dict[str, dict[str, str]] = {}
    for batch in probes:
        if isinstance(batch, BaseException):
            logger.warning(
                "store_discovery_probe_error",
                extra={"stage": "store_discovery", "reason": str(batch)[:200]},
            )
            continue
        for item in batch:
            url = item.get("url", "")
            if not url or url in merged:
                continue
            merged[url] = item
    return list(merged.values())
