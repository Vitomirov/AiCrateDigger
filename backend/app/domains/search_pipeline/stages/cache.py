"""Search response cache lookup and persistence."""

from __future__ import annotations

import logging
from typing import Any

from app.core.db.cache import get_cached_search_payload, set_cached_search_payload
from app.core.db.redis_cache import get_cached_search, set_cached_search
from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.search_pipeline.models.result import ListingResult
from app.domains.search_pipeline.pipeline_context import stage_timer

logger = logging.getLogger(__name__)


def empty_response(
    query: str,
    parsed: ParsedQuery | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "query": query,
        "results": [],
        "parsed": parsed,
        "reason": reason,
    }


async def stage_cache_lookup(
    *,
    query: str,
    parsed: ParsedQuery,
    redis_key: str,
    pg_key: str,
) -> dict[str, Any] | None:
    """Return a full pipeline response on cache hit, or ``None`` to continue live."""
    with stage_timer(
        "redis_cache_lookup",
        input={"cache_key": redis_key, "pg_cache_key_head": pg_key[:16]},
    ) as rec:
        cached = await get_cached_search(redis_key)
        cache_source = "redis" if cached is not None else None
        if cached is None:
            cached = await get_cached_search_payload(pg_key)
            if cached is not None:
                cache_source = "postgres"
        rec.output = {"hit": cached is not None, "source": cache_source}
        rec.status = "success" if cached is not None else "empty"

    if cached is None:
        return None

    try:
        cached_rows = cached.get("results") or []
        hydrated_rows = [ListingResult.model_validate(row) for row in cached_rows]
    except Exception:
        logger.warning(
            "redis_cache_payload_invalid_falling_back_to_live",
            extra={"stage": "redis_cache", "cache_key_head": redis_key[:64]},
        )
        return None

    logger.info(
        "search_cache_hit",
        extra={
            "stage": "search_cache",
            "cache_key_head": redis_key[:64],
            "result_count": len(hydrated_rows),
            "source": cache_source or "redis",
        },
    )
    return {
        "query": query,
        "results": hydrated_rows,
        "parsed": parsed,
        "reason": None,
    }


async def persist_cache_payload(
    *,
    redis_key: str,
    pg_key: str,
    payload: dict[str, Any],
    redis_ttl_seconds: int,
    pg_ttl_seconds: int,
) -> None:
    """Write the response to BOTH Redis (hot read) and Postgres (operator audit).

    Failures on either tier are logged but never raised — the user already has
    the live response in hand by the time this runs.
    """
    try:
        await set_cached_search(redis_key, payload, ttl_seconds=redis_ttl_seconds)
    except Exception:
        logger.exception(
            "redis_cache_write_failed",
            extra={"stage": "redis_cache", "cache_key_head": redis_key[:64]},
        )
    try:
        await set_cached_search_payload(pg_key, payload, ttl_seconds=pg_ttl_seconds)
    except Exception:
        logger.exception(
            "postgres_cache_write_failed",
            extra={"stage": "search_cache", "cache_key_head": pg_key[:16]},
        )
