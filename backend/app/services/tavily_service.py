"""Tavily search client — one place for all outbound Tavily calls.

Traceability goals (STEP 4):
- Every query is logged with its raw response count.
- Every URL returned is visible in the trace (count, top domains).
- Per-domain cap and normalization happen here so the pipeline downstream sees
  a clean, deduplicated candidate list.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx

from app.config import get_settings
from app.db.marketplace_db import (
    TavilyObservation,
    get_marketplace_db,
    ingest_tavily_batch,
)
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer


@dataclass(slots=True)
class TavilyIntent:
    """Per-request intent tokens handed in by the caller. Used to score
    relevance (artist/album/format mention density) as Tavily results are
    fed back into the emergent `marketplaces` RAG."""

    artist: str
    album: str
    music_format: str
    location_hint: str | None = None

logger = logging.getLogger(__name__)

FORBIDDEN_DOMAINS = (
    "ebay.com",
    "amazon.",
    "acousticsounds.com",
)


def is_valid_result(url: str) -> bool:
    u = url.lower()
    return not any(domain in u for domain in FORBIDDEN_DOMAINS)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS_PER_QUERY = 10
MAX_RESULTS_PER_DOMAIN = 3
REQUEST_TIMEOUT_SECONDS = 15.0
# Tavily-reported relevance below this is usually noise (wrong artist/topic).
MIN_TAVILY_SCORE = 0.20


def normalize_url(url: str) -> str:
    """Strip query string, fragment, trailing slash. Lowercase host."""
    try:
        stripped = url.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
        parsed = urlsplit(stripped.strip())
        normalized_path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), normalized_path, "", ""))
    except Exception:
        return url


async def _search_single_query(client: httpx.AsyncClient, query: str) -> list[SearchResult]:
    settings = get_settings()
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
        "max_results_per_domain": MAX_RESULTS_PER_DOMAIN,
    }

    with stage_timer("tavily", input={"query": query}) as rec:
        try:
            response = await client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            rec.status = "fail"
            rec.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "tavily_http_error",
                extra={"stage": "tavily", "status": "fail", "reason": rec.error, "query": query},
            )
            return []
        except Exception as exc:
            rec.status = "fail"
            rec.error = str(exc)
            logger.exception("tavily_unexpected_error", extra={"stage": "tavily", "status": "fail"})
            return []

        raw_items = data.get("results", []) or []
        results: list[SearchResult] = []
        for item in raw_items:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=normalize_url(url),
                    content=str(item.get("content", "")).strip(),
                    score=float(item.get("score", 0.0) or 0.0),
                    price=None,
                    extracted_location=None,
                )
            )

        rec.output = {
            "raw_count": len(raw_items),
            "kept_count": len(results),
            "top_domains": _top_domains(results, limit=5),
        }
        rec.status = "success" if results else "empty"
        return results


def _top_domains(results: list[SearchResult], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for r in results:
        dom = urlparse(r.url).netloc.lower()
        if dom:
            counts[dom] = counts.get(dom, 0) + 1
    return [d for d, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


async def run_tavily_search(
    queries: list[str],
    intent: TavilyIntent | None = None,
) -> list[SearchResult]:
    """Run all queries in parallel, apply per-domain diversity, return unique candidates.

    If `intent` is supplied, every Tavily hit (both kept and dropped-for-domain-cap)
    is also forwarded to `marketplace_db` as a `TavilyObservation` — this is the
    system's self-improvement feedback loop. Every search teaches the next one.
    """
    if not queries:
        return []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        tasks = [_search_single_query(client, q) for q in queries]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten and dedup per-URL before the domain cap. We DO ingest all
    # candidates (including ones later dropped by the per-domain cap) so that
    # the emergent scorer observes full frequency signal, not just the top of
    # the funnel.
    unique_by_url: dict[str, SearchResult] = {}
    for task_result in gathered:
        if isinstance(task_result, list):
            for res in task_result:
                if res.score < MIN_TAVILY_SCORE:
                    continue
                if not is_valid_result(res.url):
                    continue
                existing = unique_by_url.get(res.url)
                if existing is None or res.score > existing.score:
                    unique_by_url[res.url] = res

    # Domain diversity cap — applies to what we return to the extractor.
    sorted_candidates = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
    final_results: list[SearchResult] = []
    domain_counts: dict[str, int] = {}
    dropped_by_domain_cap = 0

    for res in sorted_candidates:
        domain = urlparse(res.url).netloc.lower()
        count = domain_counts.get(domain, 0)
        if count >= MAX_RESULTS_PER_DOMAIN:
            dropped_by_domain_cap += 1
            continue
        final_results.append(res)
        domain_counts[domain] = count + 1

    logger.info(
        "tavily_aggregate",
        extra={
            "stage": "tavily",
            "status": "success" if final_results else "empty",
            "count": len(final_results),
            "output": {
                "unique_urls": len(unique_by_url),
                "kept": len(final_results),
                "dropped_domain_cap": dropped_by_domain_cap,
                "top_domains": _top_domains(final_results, limit=5),
            },
        },
    )

    # Feedback loop — non-blocking in spirit (we await, but it's fast and we
    # want the trace to reflect what landed in RAG for this request).
    if intent is not None and unique_by_url:
        await _ingest_tavily_feedback(list(unique_by_url.values()), intent)

    return final_results


async def _ingest_tavily_feedback(
    results: list[SearchResult],
    intent: TavilyIntent,
) -> None:
    with stage_timer(
        "rag_ingest_tavily",
        input={"candidate_count": len(results)},
    ) as rec:
        observations = [
            TavilyObservation(
                url=r.url,
                title=r.title,
                content=r.content,
                tavily_score=r.score,
                artist=intent.artist,
                album=intent.album,
                music_format=intent.music_format,
                location_hint=intent.location_hint,
            )
            for r in results
        ]
        try:
            service = get_marketplace_db()
            domains = await ingest_tavily_batch(service, observations)
        except Exception as exc:
            rec.status = "fail"
            rec.error = str(exc)
            logger.exception(
                "rag_ingest_tavily_failed",
                extra={"stage": "rag_ingest_tavily", "status": "fail"},
            )
            return
        rec.output = {"domains_written": domains, "count": len(domains)}
        rec.status = "success" if domains else "empty"
