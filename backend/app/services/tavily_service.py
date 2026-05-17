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
from collections.abc import Iterable, Sequence
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx

from app.config import get_settings
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer
from app.policies.search_dsl import (
    build_tavily_local_fanout_narrow_query,
    build_tavily_local_fanout_plain_album_query,
    build_tavily_local_fanout_plain_artist_album_query,
)
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


def _fanout_single_domain_threshold(min_score_base: float) -> float:
    """Score floor for single-domain local shop fanout (sparse PDP snippets)."""
    return max(0.022, min_score_base * 0.18)


def _whitelist_include_domains_threshold(min_score_base: float) -> float:
    """Score floor when ``include_domains`` restricts hits to curated stores only.

    Batched city/country/regional tier calls use the same constraint: Tavily scores
    are miscalibrated low for indie domains, so the global ``tavily_min_result_score``
    would yield raw_count>0 but kept_count=0.
    """
    return max(0.032, min_score_base * 0.30)


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
    "/produse/",
    "/produs/",
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


def buy_signal_multiplier_for_url(url: str) -> float:
    """Reuse Tavily PDP vs editorial heuristic for pipeline quality checks."""
    try:
        nu = normalize_url(url)
        path_m = urlparse(nu).path or "/"
    except Exception:
        path_m = "/"
    return float(_product_signal_multiplier(path_m))


def _editorial_discovery_url(url: str) -> bool:
    """URLs that seldom yield deterministic listing rows (blogs, magazine hubs).

    Mirrors non-product rejection heuristics in ``validators.listings``.
    """
    try:
        pl = urlparse(url.strip()).path.lower() or "/"
    except Exception:
        pl = "/"
    if _product_signal_multiplier(pl) <= 0.0:
        return True
    from app.validators import listings as _lv

    if _lv._url_looks_non_product(url) and not _lv.url_suggests_product_detail_page(url):  # noqa: SLF001
        return True
    return False


def editorial_discovery_blocked_hosts_from_raw_results(rows: Sequence[SearchResult], *, deterministic_failed: bool) -> set[str]:
    """Identify hosts that look like SERP hubs when deterministic extraction stalled."""
    out: set[str] = set()
    if not deterministic_failed or not rows:
        return out
    for r in rows:
        if _editorial_discovery_url(r.url):
            host = urlparse(normalize_url(r.url)).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            host = canonical_store_domain(host)
            if host:
                out.add(host)
    return out

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


def _host_matches_include_domain(netloc: str, allowed_domain: str) -> bool:
    """True if URL host is ``allowed_domain`` or a subdomain of it (``www`` stripped)."""
    h = (netloc or "").lower().strip()
    if h.startswith("www."):
        h = h[4:]
    a = (allowed_domain or "").lower().strip()
    if a.startswith("www."):
        a = a[4:]
    if not h or not a:
        return False
    return h == a or h.endswith("." + a)


async def _search_single_query(
    client: httpx.AsyncClient,
    query: str,
    *,
    include_domains: list[str] | None = None,
    max_results: int | None = None,
    fanout_local_shop: bool = False,
) -> list[SearchResult]:
    settings = get_settings()
    inc = include_domains if include_domains is not None else _include_domains_for_query(query)
    max_r = max_results if max_results is not None else MAX_RESULTS_PER_QUERY
    min_score_base = float(settings.tavily_min_result_score)
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

    include_only = inc is not None and len(inc) > 0
    per_query_threshold = min_score_base
    if fanout_local_shop and include_only and len(inc) == 1:
        per_query_threshold = _fanout_single_domain_threshold(min_score_base)
    elif include_only:
        per_query_threshold = _whitelist_include_domains_threshold(min_score_base)

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
        except Exception:
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
            if include_only and prod > 0.0:
                p_floor = 0.91 if fanout_local_shop and len(inc) == 1 else 0.82
                eff = float(base_score) * max(float(prod), p_floor)
            if prod == 0.0 or eff < per_query_threshold:
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

        salvage_kept = 0
        if (
            fanout_local_shop
            and include_only
            and len(inc) == 1
            and not results
            and raw_items
        ):
            want_dom = inc[0]
            salvage_scored: list[tuple[float, str, dict[str, object]]] = []
            for item in raw_items:
                url = str(item.get("url", "")).strip()
                if not url or not is_valid_result(url):
                    continue
                nu = normalize_url(url)
                try:
                    netloc = urlparse(nu).netloc or ""
                    path_m = urlparse(nu).path or "/"
                except Exception:
                    continue
                if not _host_matches_include_domain(netloc, want_dom):
                    continue
                prod = _product_signal_multiplier(path_m)
                if prod <= 0.0:
                    continue
                base_score = float(item.get("score", 0.0) or 0.0)
                salvage_scored.append((base_score * prod, nu, item))
            salvage_scored.sort(key=lambda x: -x[0])
            for _, nu, item in salvage_scored[:max_r]:
                base_score = float(item.get("score", 0.0) or 0.0)
                try:
                    path_m = urlparse(nu).path or "/"
                except Exception:
                    path_m = "/"
                prod = _product_signal_multiplier(path_m)
                eff = max(float(base_score), 0.052) * 0.92 * max(float(prod), 0.72)
                results.append(
                    SearchResult(
                        title=str(item.get("title", "")).strip(),
                        url=nu,
                        content=str(item.get("content", "")).strip(),
                        score=min(eff, 0.49),
                        price=None,
                        extracted_location=None,
                    )
                )
                salvage_kept += 1

        rec.output = {
            "raw_count": len(raw_items),
            "kept_count": len(results),
            "salvage_kept": salvage_kept,
            "top_domains": _top_domains(results, limit=5),
        }
        rec.status = "success" if results else "empty"
        return results


def _retail_buy_signal_hits(rows: Iterable[SearchResult]) -> int:
    return sum(1 for r in rows if buy_signal_multiplier_for_url(r.url) >= 0.72)


def _retail_signal_satisfied(rows: Iterable[SearchResult]) -> bool:
    """Enough PDP-like SERP artefacts to skip widening the query ladder."""
    collected = list(rows)
    hits = _retail_buy_signal_hits(collected)
    if hits >= 2:
        return True
    if hits >= 1 and len({_host_bucket(r.url) for r in collected if buy_signal_multiplier_for_url(r.url) >= 0.72}) >= 2:
        return True
    return hits >= 1 and len(collected) >= 5


def _host_bucket(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if h.startswith("www."):
        h = h[4:]
    return h


def _apply_merge_floor_and_collect(
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


async def run_tavily_for_store_domains(
    core_query: str,
    store_domains: list[str],
    *,
    tier: str | None = None,
    relaxation_queries: Sequence[str] | None = None,
) -> tuple[list[SearchResult], int]:
    """Run Tavily with **hostname-only** ``include_domains``.

    * **≤ 20 domains:** exactly one HTTP POST (no chunking, no parallel calls).
    * **> 20 domains:** chunks of 20, **sequential** requests, merged and deduped.

    Optionally appends relaxation queries after the strict ``core_query`` when the first
    pass surfaces mostly editorial/hub artefacts.

    Returns ``(results, http_call_count)``.
    """
    if not (core_query or "").strip() or not store_domains:
        return [], 0

    settings = get_settings()

    domains = _dedupe_domains(list(store_domains))
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

    chunks = _chunk_include_domains(domains, max_domains_one_call)
    min_score = float(settings.tavily_min_result_score)
    merge_floor = _whitelist_include_domains_threshold(min_score)
    aggregated: dict[str, SearchResult] = {}
    http_calls = 0

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for _, q in enumerate(qs_plan):
            batch_lists: list[list[SearchResult]] = []
            batch_http = 0
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
                batch_http += 1
            http_calls += batch_http

            round_unique = _apply_merge_floor_and_collect(batch_lists, merge_floor=merge_floor)
            for url_k, sr in round_unique.items():
                prev = aggregated.get(url_k)
                if prev is None or sr.score > prev.score:
                    aggregated[url_k] = sr
            merged_all = sorted(aggregated.values(), key=lambda x: x.score, reverse=True)
            if _retail_signal_satisfied(merged_all):
                break

    unique_by_url = aggregated or {}

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


async def run_local_site_searches(
    core_query: str,
    local_domains: list[str],
    *,
    tier: str | None = None,
    artist: str | None = None,
    album_title: str | None = None,
) -> tuple[list[SearchResult], int]:
    """One-call-per-domain Tavily fanout for indie local shops.

    Optionally runs a **narrow album-phrase** query in addition to the full ``core``
    string—some EU storefront search indexes barely surface rows for long artist+album packs.

    Per-request score floors relax when ``fanout_local_shop`` is enabled so SEO-sparse PDP
    URLs are not clipped before merge.

    Returns ``(deduplicated_results, http_call_count)``.
    """
    if not (core_query or "").strip() or not local_domains:
        return [], 0

    settings = get_settings()
    q = core_query.strip()
    domains = _dedupe_domains(list(local_domains))
    if not domains:
        return [], 0

    narrow = build_tavily_local_fanout_narrow_query(
        artist=artist,
        album=album_title or "",
    ).strip()
    sub_queries: list[str] = [q]
    seen_q = {q.lower()}
    for extra in (
        narrow,
        build_tavily_local_fanout_plain_album_query(album_title or ""),
        build_tavily_local_fanout_plain_artist_album_query(
            artist=artist,
            album=album_title or "",
        ),
    ):
        x = (extra or "").strip()
        if x and x.lower() not in seen_q:
            seen_q.add(x.lower())
            sub_queries.append(x)
    queries_per_domain = len(sub_queries)

    max_results = settings.tavily_max_results_per_batch
    aggregate_floor = _fanout_single_domain_threshold(float(settings.tavily_min_result_score))

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        async def _one(domain: str) -> list[SearchResult]:
            logger.info(
                "tavily_local_site_search",
                extra={
                    "stage": "tavily",
                    "tier": tier,
                    "domain": domain,
                    "queries": sub_queries,
                },
            )
            merged: dict[str, SearchResult] = {}
            for subq in sub_queries:
                try:
                    batch = await _search_single_query(
                        client,
                        subq,
                        include_domains=[domain],
                        max_results=max_results,
                        fanout_local_shop=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "tavily_local_site_search_failed",
                        extra={
                            "stage": "tavily",
                            "tier": tier,
                            "domain": domain,
                            "sub_query": subq,
                            "reason": str(exc),
                        },
                    )
                    continue
                for sr in batch:
                    try:
                        path_m = urlparse(normalize_url(sr.url)).path or "/"
                    except Exception:
                        path_m = "/"
                    if _product_signal_multiplier(path_m) <= 0.0:
                        continue
                    nk = normalize_url(sr.url)
                    prev = merged.get(nk)
                    if prev is None or sr.score > prev.score:
                        merged[nk] = sr
            return list(merged.values())

        gathered = await asyncio.gather(*[_one(d) for d in domains])

    unique_by_url: dict[str, SearchResult] = {}
    for res_list in gathered:
        for res in res_list:
            if res.score < aggregate_floor:
                continue
            if not is_valid_result(res.url):
                continue
            existing = unique_by_url.get(res.url)
            if existing is None or res.score > existing.score:
                unique_by_url[res.url] = res

    final = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
    logger.info(
        "tavily_local_aggregate",
        extra={
            "stage": "tavily",
            "tier": tier,
            "domains": len(domains),
            "queries_per_domain": queries_per_domain,
            "kept": len(final),
            "top_domains": _top_domains(final, limit=5),
        },
    )
    return final, len(domains) * queries_per_domain


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
