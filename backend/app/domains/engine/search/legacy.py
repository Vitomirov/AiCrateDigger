"""Legacy per-query Tavily mode (one HTTP request per query string)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import get_settings
from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.engine.search.aggregation import cap_results_per_domain, top_domains
from app.domains.engine.search.constants import REQUEST_TIMEOUT_SECONDS
from app.domains.engine.search.filtering import is_valid_result
from app.domains.engine.search.search import search_single_query

logger = logging.getLogger(__name__)


async def run_tavily_search(queries: list[str]) -> list[SearchResult]:
    """Run legacy per-query Tavily calls.

    Prefer :func:`run_tavily_for_store_domains` when all queries share one intent.
    """
    if not queries:
        return []

    settings = get_settings()
    capped_queries = queries[: settings.tavily_max_http_calls]
    min_score = float(settings.tavily_min_result_score)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        tasks = [search_single_query(client, q) for q in capped_queries]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    unique_by_url: dict[str, SearchResult] = {}
    for task_result in gathered:
        if isinstance(task_result, list):
            for res in task_result:
                if res.score < min_score:
                    continue
                if not is_valid_result(res.url):
                    continue
                existing = unique_by_url.get(res.url)
                if existing is None or res.score > existing.score:
                    unique_by_url[res.url] = res

    sorted_candidates = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
    max_per_dom = settings.tavily_max_results_per_domain_aggregate
    final_results, dropped_by_domain_cap = cap_results_per_domain(
        sorted_candidates,
        max_per_dom=max_per_dom,
    )

    logger.info(
        "tavily_aggregate",
        extra={
            "stage": "tavily",
            "status": "success" if final_results else "empty",
            "count": len(final_results),
            "output": {
                "batched": False,
                "tavily_http_calls": len(capped_queries),
                "unique_urls": len(unique_by_url),
                "kept": len(final_results),
                "dropped_domain_cap": dropped_by_domain_cap,
                "top_domains": top_domains(final_results, limit=5),
            },
        },
    )

    return final_results
