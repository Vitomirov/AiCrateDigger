"""Shared Tavily HTTP and policy constants."""

from __future__ import annotations

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS_PER_QUERY = 8
REQUEST_TIMEOUT_SECONDS = 15.0

#: Tavily uses 432/433 for account / rate pressure; 429/503 are generic overload signals.
RETRYABLE_TAVILY_STATUS: frozenset[int] = frozenset({429, 432, 433, 503})

FORBIDDEN_DOMAINS: tuple[str, ...] = (
    "ebay.com",
    "amazon.",
    "acousticsounds.com",
    "kupujemprodajem.com",
    "kupindo.com",
)
