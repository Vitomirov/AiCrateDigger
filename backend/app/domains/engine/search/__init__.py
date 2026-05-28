"""Tavily retrieval — public API surface (lazy).

Importing submodules does not eagerly load ``httpx`` or the HTTP client.
Use ``from app.domains.engine.search import …`` for the facade, or import
submodules directly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TavilyCircuitBreaker",
    "build_consolidated_query",
    "fetch_tavily_results_body",
    "get_breaker",
    "prefilter_tavily_results",
    "run_consolidated_tavily_search",
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
    if name == "prefilter_tavily_results":
        from app.domains.engine.search.prefilter import prefilter_tavily_results

        return prefilter_tavily_results
    if name in ("build_consolidated_query", "run_consolidated_tavily_search"):
        from app.domains.engine.search import single_call as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
