"""Search pipeline response cache (PostgreSQL row TTL).

Without ``DATABASE_URL``, all functions no-op / return ``None`` so the API
behaviour matches the old in-memory-only stack.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete

from app.core.config import get_settings
from app.core.db.database import SearchResponseCacheORM, is_database_configured, session_factory

logger = logging.getLogger(__name__)


def hydrate_cached_pipeline_dict(cached: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON cache rows back into :class:`ListingResult` instances for the router."""
    from app.domains.search_pipeline.models.result import ListingResult

    rows = cached.get("results") or []
    return {
        "query": cached.get("query", ""),
        "results": [ListingResult.model_validate(x) for x in rows],
    }


def build_search_cache_key(
    *,
    format_token: str | None,
    artist: str | None,
    album_title: str,
    country_code: str | None,
    resolved_city: str | None = None,
    geo_granularity: str | None = None,
) -> str:
    """Postgres cache key — SHA-256 of the canonical Redis cache identity."""
    from app.core.db.search_cache_key import (
        build_pipeline_search_cache_key,
        build_postgres_search_cache_key,
    )

    redis_key = build_pipeline_search_cache_key(
        format_token=format_token,
        artist=artist,
        album=album_title,
        country_code=country_code,
        resolved_city=resolved_city,
        geo_granularity=geo_granularity,
    )
    return build_postgres_search_cache_key(redis_cache_key=redis_key)


async def get_cached_search_payload(cache_key: str) -> dict[str, Any] | None:
    settings = get_settings()
    if settings.debug:
        return None
    if not is_database_configured() or not settings.search_cache_enabled:
        return None
    try:
        sf = session_factory()
    except RuntimeError:
        return None

    now = datetime.now(UTC)
    async with sf() as session:
        row = await session.get(SearchResponseCacheORM, cache_key)
        if row is None:
            return None
        if row.expires_at <= now:
            await session.delete(row)
            await session.commit()
            return None
        try:
            return json.loads(row.payload_json)
        except json.JSONDecodeError:
            await session.delete(row)
            await session.commit()
            return None


async def set_cached_search_payload(cache_key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
    """Persist search responses for DBeaver audit; writes run even when ``DEBUG=true``."""
    settings = get_settings()
    if not is_database_configured() or not settings.search_cache_enabled:
        return
    try:
        sf = session_factory()
    except RuntimeError:
        return

    now = datetime.now(UTC)
    expires = now + timedelta(seconds=max(60, ttl_seconds))
    body = json.dumps(payload, default=str)
    async with sf() as session:
        row = await session.get(SearchResponseCacheORM, cache_key)
        if row is None:
            session.add(
                SearchResponseCacheORM(
                    cache_key=cache_key,
                    payload_json=body,
                    expires_at=expires,
                )
            )
        else:
            row.payload_json = body
            row.expires_at = expires
        await session.commit()

    logger.debug(
        "search_cache_set",
        extra={"stage": "search_cache", "cache_key_head": cache_key[:12], "ttl_seconds": ttl_seconds},
    )


async def purge_expired_search_cache_rows() -> int:
    """Best-effort cleanup; safe to call occasionally from lifespan or a cron."""
    settings = get_settings()
    if not is_database_configured():
        return 0
    try:
        sf = session_factory()
    except RuntimeError:
        return 0
    now = datetime.now(UTC)
    async with sf() as session:
        res = await session.execute(delete(SearchResponseCacheORM).where(SearchResponseCacheORM.expires_at <= now))
        await session.commit()
        return res.rowcount or 0
