"""IP-based sliding-window rate limit for anonymous paid API routes."""

from __future__ import annotations

import logging
import time

from fastapi import HTTPException, Request

from app.core.config import get_settings
from app.core.db.redis_cache import get_redis_client

logger = logging.getLogger(__name__)

#: Shared bucket for ``/search``, ``/search-listings``, and ``/parse``.
_RATE_LIMIT_KEY_PREFIX = "rate_limit:api:"

_RATE_LIMIT_UNAVAILABLE_DETAIL = (
    "Service temporarily unavailable. Please try again in a few minutes."
)


def _rate_limit_settings() -> tuple[int, int]:
    settings = get_settings()
    return settings.search_rate_limit_max_requests, settings.search_rate_limit_window_seconds


def _client_ip(request: Request) -> str:
    """Resolve client IP, preferring the first hop in ``X-Forwarded-For``."""
    forwarded = request.headers.get("X-Forwarded-For") or request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _rate_limit_exceeded_detail(max_requests: int, window_seconds: int) -> str:
    hours = max(1, window_seconds // 3600)
    return (
        f"Request limit reached. You can make {max_requests} searches "
        f"every {hours} hours."
    )


def _raise_rate_limit_unavailable(*, reason: str) -> None:
    logger.warning(
        "rate_limiter_unavailable_rejecting",
        extra={"stage": "rate_limiter", "status": "blocked", "reason": reason},
    )
    raise HTTPException(status_code=503, detail=_RATE_LIMIT_UNAVAILABLE_DETAIL)


async def ip_rate_limiter(request: Request) -> None:
    """FastAPI dependency: N paid requests per IP per rolling window (see Settings)."""
    settings = get_settings()
    if not settings.search_rate_limit_enabled:
        logger.debug(
            "rate_limiter_disabled",
            extra={"stage": "rate_limiter", "status": "bypass", "reason": "config"},
        )
        return

    max_requests, window_seconds = _rate_limit_settings()
    fail_closed = settings.search_rate_limit_fail_closed

    client = await get_redis_client()
    if client is None:
        if fail_closed:
            _raise_rate_limit_unavailable(reason="redis_unavailable")
        logger.warning(
            "rate_limiter_redis_unavailable_allowing_request",
            extra={"stage": "rate_limiter", "status": "bypass"},
        )
        return

    ip = _client_ip(request)
    key = f"{_RATE_LIMIT_KEY_PREFIX}{ip}"
    now = time.time()
    window_start = now - window_seconds

    try:
        read_pipe = client.pipeline(transaction=True)
        read_pipe.zremrangebyscore(key, 0, window_start)
        read_pipe.zcard(key)
        _, count = await read_pipe.execute()

        if int(count) >= max_requests:
            logger.info(
                "rate_limit_exceeded",
                extra={
                    "stage": "rate_limiter",
                    "status": "blocked",
                    "ip_head": ip[:32],
                    "count": int(count),
                    "max_requests": max_requests,
                    "window_seconds": window_seconds,
                },
            )
            raise HTTPException(
                status_code=429,
                detail=_rate_limit_exceeded_detail(max_requests, window_seconds),
            )

        member = str(time.time_ns())
        write_pipe = client.pipeline(transaction=True)
        write_pipe.zadd(key, {member: now})
        write_pipe.expire(key, window_seconds)
        await write_pipe.execute()
    except HTTPException:
        raise
    except Exception as exc:
        if fail_closed:
            _raise_rate_limit_unavailable(reason=str(exc)[:200])
        logger.warning(
            "rate_limiter_redis_error_allowing_request",
            extra={"stage": "rate_limiter", "status": "bypass", "reason": str(exc)[:200]},
        )
