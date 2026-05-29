"""Low-level Tavily HTTP client with retry/backoff."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx

from app.core.config import get_settings
from app.core.quota import QuotaKind, assert_quota_available, record_quota_usage
from app.domains.engine.search.circuit_breaker import get_breaker
from app.domains.engine.search.constants import RETRYABLE_TAVILY_STATUS, TAVILY_SEARCH_URL

logger = logging.getLogger(__name__)

#: HTTP status codes signalling account/credit pressure rather than transient overload.
#: Retrying these aggressively wastes wall time — they rarely clear within seconds.
_ACCOUNT_PRESSURE_STATUS: frozenset[int] = frozenset({432, 433})


async def fetch_tavily_results_body(
    client: httpx.AsyncClient,
    payload: dict[str, object],
    *,
    query_for_log: str,
) -> dict[str, object] | None:
    """POST to Tavily search; retry with backoff on transient rate / quota responses.

    Honours the request-scoped :class:`TavilyCircuitBreaker`: once it trips,
    every subsequent call returns ``None`` immediately with zero HTTP cost.
    """
    breaker = get_breaker()
    if breaker.is_open():
        logger.info(
            "tavily_skipped_circuit_open",
            extra={"stage": "tavily", "query": query_for_log[:160]},
        )
        return None

    settings = get_settings()
    max_attempts = int(settings.tavily_http_retry_attempts)
    account_max_attempts = max(1, min(2, max_attempts))
    max_wait = float(settings.tavily_http_retry_max_wait_seconds)

    await assert_quota_available(QuotaKind.TAVILY)

    for attempt in range(max_attempts):
        try:
            response = await client.post(TAVILY_SEARCH_URL, json=payload)
        except httpx.RequestError as exc:
            logger.warning(
                "tavily_request_error",
                extra={
                    "stage": "tavily",
                    "attempt": attempt + 1,
                    "reason": str(exc),
                    "query": query_for_log[:160],
                },
            )
            if attempt < max_attempts - 1:
                delay = min(max_wait, 0.35 * (2**attempt))
                await asyncio.sleep(delay * (0.85 + 0.3 * random.random()))
                continue
            breaker.record_failure(reason="request_error")
            return None

        if response.status_code in RETRYABLE_TAVILY_STATUS:
            logger.warning(
                "tavily_rate_limited",
                extra={
                    "stage": "tavily",
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                    "query": query_for_log[:160],
                },
            )
            cap = (
                account_max_attempts
                if response.status_code in _ACCOUNT_PRESSURE_STATUS
                else max_attempts
            )
            if attempt < cap - 1:
                delay = min(max_wait, 0.45 * (2**attempt))
                await asyncio.sleep(delay * (0.85 + 0.3 * random.random()))
                continue
            breaker.record_failure(reason=f"http_{response.status_code}")
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "tavily_http_error",
                extra={
                    "stage": "tavily",
                    "status_code": response.status_code,
                    "reason": str(exc),
                    "query": query_for_log[:160],
                },
            )
            breaker.record_failure(reason=f"http_{response.status_code}")
            return None

        try:
            data = response.json()
        except Exception:
            logger.exception(
                "tavily_json_decode",
                extra={"stage": "tavily", "query": query_for_log[:160]},
            )
            breaker.record_failure(reason="json_decode")
            return None

        breaker.record_success()
        await record_quota_usage(QuotaKind.TAVILY)
        return data

    breaker.record_failure(reason="retry_exhausted")
    return None
