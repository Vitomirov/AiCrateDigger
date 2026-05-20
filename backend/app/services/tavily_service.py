"""Backward-compatible re-exports — prefer ``app.services.tavily``."""

from __future__ import annotations

from app.services.tavily import (
    buy_signal_multiplier_for_url,
    dedupe_domains,
    editorial_discovery_blocked_hosts_from_raw_results,
    enforce_include_domains_hosts,
    fetch_tavily_results_body,
    is_valid_result,
    normalize_url,
    run_local_site_searches,
    run_tavily_for_store_domains,
    run_tavily_search,
)

# Legacy private alias used by store_discovery and tests.
_fetch_tavily_results_body = fetch_tavily_results_body
_dedupe_domains = dedupe_domains

__all__ = [
    "buy_signal_multiplier_for_url",
    "dedupe_domains",
    "editorial_discovery_blocked_hosts_from_raw_results",
    "enforce_include_domains_hosts",
    "is_valid_result",
    "normalize_url",
    "run_local_site_searches",
    "run_tavily_for_store_domains",
    "run_tavily_search",
]
