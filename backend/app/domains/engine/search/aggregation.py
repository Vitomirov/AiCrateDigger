"""Merge, dedupe, and aggregate Tavily result batches."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.engine.search.filtering import is_valid_result
from app.domains.engine.search.scoring import buy_signal_multiplier_for_url
from app.domains.engine.search.url_utils import host_bucket


def top_domains(results: list[SearchResult], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for r in results:
        dom = urlparse(r.url).netloc.lower()
        if dom:
            counts[dom] = counts.get(dom, 0) + 1
    return [d for d, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


def apply_merge_floor_and_collect(
    batch_lists: list[list[SearchResult]],
    *,
    merge_floor: float,
) -> dict[str, SearchResult]:
    unique_by_url: dict[str, SearchResult] = {}
    for task_result in batch_lists:
        for res in task_result:
            if res.score < merge_floor:
                continue
            if not is_valid_result(res.url):
                continue
            existing = unique_by_url.get(res.url)
            if existing is None or res.score > existing.score:
                unique_by_url[res.url] = res
    return unique_by_url


def cap_results_per_domain(
    sorted_candidates: list[SearchResult],
    *,
    max_per_dom: int,
) -> tuple[list[SearchResult], int]:
    final_results: list[SearchResult] = []
    domain_counts: dict[str, int] = {}
    dropped_by_domain_cap = 0
    for res in sorted_candidates:
        domain = urlparse(res.url).netloc.lower()
        count = domain_counts.get(domain, 0)
        if count >= max_per_dom:
            dropped_by_domain_cap += 1
            continue
        final_results.append(res)
        domain_counts[domain] = count + 1
    return final_results, dropped_by_domain_cap


def retail_buy_signal_hits(rows: Iterable[SearchResult]) -> int:
    return sum(1 for r in rows if buy_signal_multiplier_for_url(r.url) >= 0.72)


def retail_signal_satisfied(rows: Iterable[SearchResult]) -> bool:
    """Enough PDP-like SERP artefacts to skip widening the query ladder."""
    collected = list(rows)
    hits = retail_buy_signal_hits(collected)
    if hits >= 2:
        return True
    if hits >= 1 and len({host_bucket(r.url) for r in collected if buy_signal_multiplier_for_url(r.url) >= 0.72}) >= 2:
        return True
    return hits >= 1 and len(collected) >= 5
