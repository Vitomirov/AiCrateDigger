"""Shared Tavily HTTP and policy constants."""

from __future__ import annotations

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
REQUEST_TIMEOUT_SECONDS = 15.0

#: Tavily uses 432/433 for account / rate pressure; 429/503 are generic overload signals.
RETRYABLE_TAVILY_STATUS: frozenset[int] = frozenset({429, 432, 433, 503})
