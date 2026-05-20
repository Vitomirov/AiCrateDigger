"""Single-query Tavily search → scored :class:`SearchResult` rows."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer
from app.services.tavily.aggregation import top_domains
from app.services.tavily.client import fetch_tavily_results_body
from app.services.tavily.constants import MAX_RESULTS_PER_QUERY
from app.services.tavily.filtering import is_valid_result
from app.services.tavily.scoring import (
    fanout_single_domain_threshold,
    product_signal_multiplier,
    whitelist_include_domains_threshold,
)
from app.services.tavily.url_utils import (
    host_matches_include_domain,
    include_domains_for_query,
    normalize_url,
)

logger = logging.getLogger(__name__)


async def search_single_query(
    client: httpx.AsyncClient,
    query: str,
    *,
    include_domains: list[str] | None = None,
    max_results: int | None = None,
    fanout_local_shop: bool = False,
    country: str | None = None,
) -> list[SearchResult]:
    settings = get_settings()
    inc = include_domains if include_domains is not None else include_domains_for_query(query)
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
    if country:
        payload["topic"] = "general"
        payload["country"] = country
    if inc:
        payload["include_domains"] = inc

    include_only = inc is not None and len(inc) > 0
    per_query_threshold = min_score_base
    if fanout_local_shop and include_only and len(inc) == 1:
        per_query_threshold = fanout_single_domain_threshold(min_score_base)
    elif include_only:
        per_query_threshold = whitelist_include_domains_threshold(min_score_base)

    with stage_timer("tavily", input={"query": query}) as rec:
        data = await fetch_tavily_results_body(client, payload, query_for_log=query)
        if data is None:
            rec.status = "fail"
            rec.error = "tavily_request_failed_after_retries"
            logger.warning(
                "tavily_http_error",
                extra={
                    "stage": "tavily",
                    "status": "fail",
                    "reason": rec.error,
                    "query": query,
                },
            )
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
            prod = product_signal_multiplier(path_m)
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
                if not host_matches_include_domain(netloc, want_dom):
                    continue
                prod = product_signal_multiplier(path_m)
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
                prod = product_signal_multiplier(path_m)
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
            "top_domains": top_domains(results, limit=5),
        }
        rec.status = "success" if results else "empty"
        return results
