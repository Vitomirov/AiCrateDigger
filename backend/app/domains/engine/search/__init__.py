"""Tavily retrieval — public API surface (lazy).

Importing submodules (e.g. ``domain_batches``) does not eagerly load ``httpx``
or the HTTP client. Use ``from app.domains.engine.search import …`` for the
facade, or import submodules directly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TavilyCircuitBreaker",
    "build_consolidated_query",
    "build_physical_power_query_base",
    "buy_signal_multiplier_for_url",
    "chunk_domains_for_power_queries",
    "chunk_include_domains",
    "dedupe_domains",
    "editorial_discovery_blocked_hosts_from_raw_results",
    "enforce_include_domains_hosts",
    "fetch_tavily_results_body",
    "get_breaker",
    "is_valid_result",
    "normalize_url",
    "prefilter_tavily_results",
    "run_consolidated_tavily_search",
    "run_local_site_searches",
    "run_tavily_for_store_domains",
    "run_tavily_search",
    "tavily_circuit_breaker_scope",
    "tavily_country_from_iso3166_alpha2",
]


def __getattr__(name: str) -> Any:
    if name in ("TavilyCircuitBreaker", "get_breaker", "tavily_circuit_breaker_scope"):
        from app.domains.engine.search import circuit_breaker as m

        return getattr(m, name)
    if name == "fetch_tavily_results_body":
        from app.domains.engine.search.client import fetch_tavily_results_body

        return fetch_tavily_results_body
    if name == "tavily_country_from_iso3166_alpha2":
        from app.domains.engine.search.country_boost import tavily_country_from_iso3166_alpha2

        return tavily_country_from_iso3166_alpha2
    if name == "chunk_include_domains":
        from app.domains.engine.search.domain_batches import chunk_include_domains

        return chunk_include_domains
    if name in (
        "editorial_discovery_blocked_hosts_from_raw_results",
        "enforce_include_domains_hosts",
        "is_valid_result",
    ):
        from app.domains.engine.search import filtering as m

        return getattr(m, name)
    if name == "run_tavily_search":
        from app.domains.engine.search.legacy import run_tavily_search

        return run_tavily_search
    if name == "run_local_site_searches":
        from app.domains.engine.search.local_fanout import run_local_site_searches

        return run_local_site_searches
    if name in ("build_physical_power_query_base", "chunk_domains_for_power_queries"):
        from app.domains.engine.search import power_query as m

        return getattr(m, name)
    if name == "prefilter_tavily_results":
        from app.domains.engine.search.prefilter import prefilter_tavily_results

        return prefilter_tavily_results
    if name == "buy_signal_multiplier_for_url":
        from app.domains.engine.search.scoring import buy_signal_multiplier_for_url

        return buy_signal_multiplier_for_url
    if name in ("build_consolidated_query", "run_consolidated_tavily_search"):
        from app.domains.engine.search import single_call as m

        return getattr(m, name)
    if name == "run_tavily_for_store_domains":
        from app.domains.engine.search.store_domains import run_tavily_for_store_domains

        return run_tavily_for_store_domains
    if name in ("dedupe_domains", "normalize_url"):
        from app.domains.engine.search import url_utils as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
