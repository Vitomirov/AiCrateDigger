"""Tavily search integration — public API surface.

Submodules by concern:
- ``client`` — HTTP transport + retries
- ``search`` — single-query execution (legacy multi-stage helper)
- ``single_call`` — consolidated ``max_results=20`` ``advanced`` call (hot path)
- ``prefilter`` — Python pre-LLM noise filter + per-host dedupe
- ``store_domains`` — batched whitelist search (legacy multi-stage helper)
- ``local_fanout`` — city/country indie power queries (legacy)
- ``filtering`` / ``scoring`` / ``aggregation`` — post-processing
- ``power_query`` / ``domain_batches`` / ``country_boost`` — query planning
"""

from __future__ import annotations

from app.services.tavily.circuit_breaker import (
    TavilyCircuitBreaker,
    get_breaker,
    tavily_circuit_breaker_scope,
)
from app.services.tavily.client import fetch_tavily_results_body
from app.services.tavily.country_boost import tavily_country_from_iso3166_alpha2
from app.services.tavily.domain_batches import chunk_include_domains
from app.services.tavily.filtering import (
    editorial_discovery_blocked_hosts_from_raw_results,
    enforce_include_domains_hosts,
    is_valid_result,
)
from app.services.tavily.legacy import run_tavily_search
from app.services.tavily.local_fanout import run_local_site_searches
from app.services.tavily.power_query import (
    build_physical_power_query_base,
    chunk_domains_for_power_queries,
)
from app.services.tavily.prefilter import prefilter_tavily_results
from app.services.tavily.scoring import buy_signal_multiplier_for_url
from app.services.tavily.single_call import (
    build_consolidated_query,
    run_consolidated_tavily_search,
)
from app.services.tavily.store_domains import run_tavily_for_store_domains
from app.services.tavily.url_utils import dedupe_domains, normalize_url

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
