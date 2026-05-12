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
import re
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx

from app.config import get_settings
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer
from app.policies.store_domain import canonical_store_domain
from app.services.tavily_domain_batches import chunk_include_domains as _chunk_include_domains

logger = logging.getLogger(__name__)

FORBIDDEN_DOMAINS = (
    "ebay.com",
    "amazon.",
    "acousticsounds.com",
    "kupujemprodajem.com",
    "kupindo.com",
)


def is_valid_result(url: str) -> bool:
    u = url.lower()
    return not any(domain in u for domain in FORBIDDEN_DOMAINS)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS_PER_QUERY = 8
REQUEST_TIMEOUT_SECONDS = 15.0
# Tavily-reported relevance below this is usually noise (wrong artist/topic).
# Default is overridden by ``Settings.tavily_min_result_score`` at runtime.
_NON_RETAIL_PATH_SNIPPETS: tuple[str, ...] = (
    "/blog",
    "/blogs/",
    "/news/",
    "/nieuws/",
    "/magazine/",
    "/mag/",
    "/tag/",
    "/tags/",
    "/articles/",
    "/article/",
    "/editorial/",
    "/features/",
    "/story/",
    "/stories/",
    "/podcast/",
)

_RETAIL_PATH_BOOST_SNIPPETS: tuple[str, ...] = (
    "/product",
    "/products/",
    "/p/",
    "/item",
    "/items/",
    "/vinyl",
    "/lp",
    "/buy",
    "/shop/",
    "/catalog/",
    "/catalogue/",
    "-p-",
    ".html",
    "add-to-cart",
    "/cart",
)


def _product_signal_multiplier(path: str) -> float:
    """Down-rank obvious editorial/category noise; lightly boost PDP-like URLs."""
    p = (path or "").lower()
    if any(x in p for x in _NON_RETAIL_PATH_SNIPPETS):
        return 0.0
    seg = p.rstrip("/").count("/")
    if seg <= 2 and any(
        p.rstrip("/").endswith(s) for s in ("/artists", "/artist", "/bands", "/band", "/labels", "/label")
    ):
        return 0.4
    if any(x in p for x in _RETAIL_PATH_BOOST_SNIPPETS):
        return 1.0
    return 0.72

_SITE_TAIL_RE = re.compile(r"\bsite:([^\s]+)\s*$", re.IGNORECASE)


def _include_domains_for_query(query: str) -> list[str] | None:
    m = _SITE_TAIL_RE.search(query.strip())
    if not m:
        return None
    dom = m.group(1).strip().lower()
    if dom.startswith("www."):
        dom = dom[4:]
    return [dom] if dom else None


def _normalize_store_domain(domain: str) -> str | None:
    d = canonical_store_domain(domain)
    return d or None


def _dedupe_domains(domains: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in domains:
        n = _normalize_store_domain(raw)
        if n is None or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def normalize_url(url: str) -> str:
    """Strip query string, fragment, trailing slash. Lowercase host."""
    try:
        stripped = url.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
        parsed = urlsplit(stripped.strip())
        normalized_path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), normalized_path, "", ""))
    except Exception:
        return url


async def _search_single_query(
    client: httpx.AsyncClient,
    query: str,
    *,
    include_domains: list[str] | None = None,
    max_results: int | None = None,
) -> list[SearchResult]:
    settings = get_settings()
    inc = include_domains if include_domains is not None else _include_domains_for_query(query)
    max_r = max_results if max_results is not None else MAX_RESULTS_PER_QUERY
    min_score = float(settings.tavily_min_result_score)
    depth = getattr(settings, "tavily_search_depth", None) or "basic"
    payload: dict[str, str | int | list[str]] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": depth,
        "max_results_per_query": max_r,
        "max_results_per_domain": settings.tavily_max_results_per_domain_aggregate,
    }
    if inc:
        payload["include_domains"] = inc

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
            nu = normalize_url(url)
            try:
                path_m = urlparse(nu).path or "/"
            except Exception:
                path_m = "/"
            prod = _product_signal_multiplier(path_m)
            base_score = float(item.get("score", 0.0) or 0.0)
            eff = base_score * prod
            if prod == 0.0 or eff < min_score:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=nu,
                    content=str(item.get("content", "")).strip(),
                    score=eff,
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


async def run_tavily_for_store_domains(
    core_query: str,
    store_domains: list[str],
    *,
    tier: str | None = None,
) -> tuple[list[SearchResult], int]:
    """Run Tavily with **hostname-only** ``include_domains``.

    * **≤ 20 domains:** exactly one HTTP POST (no chunking, no parallel calls).
    * **> 20 domains:** chunks of 20, **sequential** requests, merged and deduped.

    Returns ``(results, http_call_count)``.
    """
    if not (core_query or "").strip() or not store_domains:
        return [], 0

    settings = get_settings()
    q = core_query.strip()
    domains = _dedupe_domains(list(store_domains))
    if not domains:
        return [], 0

    max_batch_results = settings.tavily_max_results_per_batch
    max_per_dom = settings.tavily_max_results_per_domain_aggregate
    max_domains_one_call = min(20, int(getattr(settings, "tavily_domain_chunk_threshold", 20)))

    chunks = _chunk_include_domains(domains, max_domains_one_call)
    batch_lists: list[list[SearchResult]] = []
    http_calls = len(chunks)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for chunk in chunks:
            logger.info(
                "tavily_request",
                extra={
                    "tier": tier,
                    "domains": len(chunk),
                    "query": q,
                },
            )
            batch_lists.append(
                await _search_single_query(
                    client,
                    q,
                    include_domains=chunk,
                    max_results=max_batch_results,
                )
            )

    unique_by_url: dict[str, SearchResult] = {}
    min_score = float(settings.tavily_min_result_score)
    for task_result in batch_lists:
        for res in task_result:
            if res.score < min_score:
                continue
            if not is_valid_result(res.url):
                continue
            existing = unique_by_url.get(res.url)
            if existing is None or res.score > existing.score:
                unique_by_url[res.url] = res

    sorted_candidates = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
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
                "top_domains": _top_domains(final_results, limit=5),
            },
        },
    )

    return final_results, http_calls


def _top_domains(results: list[SearchResult], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for r in results:
        dom = urlparse(r.url).netloc.lower()
        if dom:
            counts[dom] = counts.get(dom, 0) + 1
    return [d for d, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


async def run_tavily_search(queries: list[str]) -> list[SearchResult]:
    """Run legacy per-query Tavily calls (one HTTP request per query string).

    Prefer :func:`run_tavily_for_store_domains` when all queries share one intent
    and differ only by trailing ``site:``.
    """
    if not queries:
        return []

    settings = get_settings()
    capped_queries = queries[: settings.tavily_max_http_calls]
    min_score = float(settings.tavily_min_result_score)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        tasks = [_search_single_query(client, q) for q in capped_queries]
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

    # Domain diversity cap — applies to what we return to the extractor.
    sorted_candidates = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
    final_results: list[SearchResult] = []
    domain_counts: dict[str, int] = {}
    dropped_by_domain_cap = 0

    max_per_dom = settings.tavily_max_results_per_domain_aggregate

    for res in sorted_candidates:
        domain = urlparse(res.url).netloc.lower()
        count = domain_counts.get(domain, 0)
        if count >= max_per_dom:
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
                "batched": False,
                "tavily_http_calls": len(capped_queries),
                "unique_urls": len(unique_by_url),
                "kept": len(final_results),
                "dropped_domain_cap": dropped_by_domain_cap,
                "top_domains": _top_domains(final_results, limit=5),
            },
        },
    )

    return final_results
