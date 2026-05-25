"""Batched Tavily search constrained to whitelist store domains."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import httpx

from app.core.config import get_settings
from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.engine.search.aggregation import (
    apply_merge_floor_and_collect,
    cap_results_per_domain,
    retail_signal_satisfied,
    top_domains,
)
from app.domains.engine.search.constants import REQUEST_TIMEOUT_SECONDS
from app.domains.engine.search.domain_batches import chunk_include_domains
from app.domains.engine.search.filtering import enforce_include_domains_hosts
from app.domains.engine.search.scoring import whitelist_include_domains_threshold
from app.domains.engine.search.search import search_single_query
from app.domains.engine.search.url_utils import dedupe_domains

logger = logging.getLogger(__name__)


async def run_tavily_for_store_domains(
    core_query: str,
    store_domains: list[str],
    *,
    tier: str | None = None,
    relaxation_queries: Sequence[str] | None = None,
) -> tuple[list[SearchResult], int]:
    """Run Tavily with **hostname-only** ``include_domains``.

    Returns ``(results, http_call_count)``.
    """
    if not (core_query or "").strip() or not store_domains:
        return [], 0

    settings = get_settings()

    domains = dedupe_domains(list(store_domains))
    if not domains:
        return [], 0

    max_batch_results = settings.tavily_max_results_per_batch
    max_per_dom = settings.tavily_max_results_per_domain_aggregate
    max_domains_one_call = min(20, int(getattr(settings, "tavily_domain_chunk_threshold", 20)))

    qs_plan: list[str] = []
    primary = core_query.strip()
    if primary:
        qs_plan.append(primary)
    relax_raw = relaxation_queries if relaxation_queries else ()
    for rq in relax_raw:
        s = rq.strip()
        if not s or s.casefold() in {x.casefold() for x in qs_plan}:
            continue
        qs_plan.append(s)
        if len(qs_plan) >= 5:
            break

    chunks = chunk_include_domains(domains, max_domains_one_call)
    min_score = float(settings.tavily_min_result_score)
    merge_floor = whitelist_include_domains_threshold(min_score)
    aggregated: dict[str, SearchResult] = {}
    http_calls = 0

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for q_ix, q in enumerate(qs_plan):
            if q_ix > 0:
                await asyncio.sleep(0.12)
            logger.info(
                "tavily_request_parallel",
                extra={
                    "tier": tier,
                    "chunks": len(chunks),
                    "domains": sum(len(c) for c in chunks),
                    "query": q,
                },
            )
            chunk_tasks = [
                search_single_query(
                    client,
                    q,
                    include_domains=c,
                    max_results=max_batch_results,
                )
                for c in chunks
            ]
            batch_lists = list(await asyncio.gather(*chunk_tasks))
            http_calls += len(chunks)

            round_unique = apply_merge_floor_and_collect(batch_lists, merge_floor=merge_floor)
            for url_k, sr in round_unique.items():
                prev = aggregated.get(url_k)
                if prev is None or sr.score > prev.score:
                    aggregated[url_k] = sr
            merged_all = sorted(aggregated.values(), key=lambda x: x.score, reverse=True)
            if retail_signal_satisfied(merged_all):
                break

    unique_by_url = aggregated or {}

    sorted_candidates = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
    sorted_candidates = enforce_include_domains_hosts(sorted_candidates, domains)
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
                "batched": len(chunks) > 1,
                "tavily_http_calls": http_calls,
                "unique_domains_requested": len(domains),
                "unique_urls": len(unique_by_url),
                "kept": len(final_results),
                "dropped_domain_cap": dropped_by_domain_cap,
                "top_domains": top_domains(final_results, limit=5),
            },
        },
    )

    return final_results, http_calls
