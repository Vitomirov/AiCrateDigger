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

from app.config import get_settings
from app.db.database import SearchResponseCacheORM, session_factory

logger = logging.getLogger(__name__)


def hydrate_cached_pipeline_dict(cached: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON cache rows back into :class:`ListingResult` instances for the router."""
    from app.models.result import ListingResult

    rows = cached.get("results") or []
    return {
        "query": cached.get("query", ""),
        "results": [ListingResult.model_validate(x) for x in rows],
    }


def build_tavily_tier_cache_key(*, artist: str | None, album_title: str, tier: str) -> str:
    """Cache raw Tavily hits per geography tier (before LLM extract)."""
    import hashlib

    parts = (
        "tavily_tier_v1",
        (artist or "").strip().lower(),
        (album_title or "").strip().lower(),
        (tier or "").strip().lower(),
    )
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def get_cached_tavily_tier_payload(cache_key: str) -> list[dict[str, Any]] | None:
    """Intermediate Tavily JSON (list of result dicts) or ``None``."""
    settings = get_settings()
    if settings.debug:
        return None
    if not settings.database_url or not settings.search_cache_enabled:
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
            body = json.loads(row.payload_json)
        except json.JSONDecodeError:
            await session.delete(row)
            await session.commit()
            return None
    raw_list = body.get("tavily_raw_results") if isinstance(body, dict) else body
    if not isinstance(raw_list, list):
        return None
    out: list[dict[str, Any]] = []
    for x in raw_list:
        if isinstance(x, dict):
            out.append(x)
    return out


async def set_cached_tavily_tier_payload(
    cache_key: str,
    raw_results: list[dict[str, Any]],
    *,
    ttl_seconds: int,
) -> None:
    settings = get_settings()
    if settings.debug:
        return
    if not settings.database_url or not settings.search_cache_enabled:
        return
    try:
        sf = session_factory()
    except RuntimeError:
        return

    now = datetime.now(UTC)
    expires = now + timedelta(seconds=max(60, ttl_seconds))
    payload = {"tavily_raw_results": raw_results}
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
        "tavily_tier_cache_set",
        extra={"stage": "search_cache", "cache_key_head": cache_key[:12], "ttl_seconds": ttl_seconds},
    )


def build_search_cache_key(*, user_query: str, artist: str | None, album_title: str, debug: bool) -> str:
    """Stable SHA-256 hex for cache identity (debug mode gets a separate slot)."""
    import hashlib

    parts = (
        (user_query or "").strip().lower(),
        (artist or "").strip().lower(),
        (album_title or "").strip().lower(),
        "1" if debug else "0",
    )
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def get_cached_search_payload(cache_key: str) -> dict[str, Any] | None:
    settings = get_settings()
    if settings.debug:
        return None
    if not settings.database_url or not settings.search_cache_enabled:
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
    settings = get_settings()
    if settings.debug:
        return
    if not settings.database_url or not settings.search_cache_enabled:
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
    if not settings.database_url:
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
