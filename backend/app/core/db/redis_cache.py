"""Redis cache layer for the consolidated vinyl-search pipeline.

A 7-day TTL Redis cache that short-circuits the entire pipeline when a request
re-hits a recently computed query. Every Tavily credit + OpenAI token spent on
the cached lookup drops to **zero** until the TTL elapses.

Design contract
---------------
* **Optional** — if ``settings.redis_url`` is unset OR the Redis server is
  unreachable, every helper here returns a safe default (``None`` on read,
  ``False`` on write). The pipeline always falls back to a live search.
* **Debug bypass** — when ``settings.debug=True`` is set, the cache is
  fully bypassed (both reads and writes). Operators using the debug JSON
  inspector / pipeline-stage tracing always see fresh pipeline behaviour
  instead of a stale snapshot from before the latest code change. This
  mirrors :mod:`app.core.db.cache` so the two caching tiers behave the same.
* **Schema-versioned key** — :data:`_PIPELINE_CACHE_SCHEMA_VERSION` is embedded
  in the cache key. Bump it whenever a pipeline change must invalidate every
  cached response (e.g. new stage that alters results). Old entries simply
  cease to be hit and expire via their existing TTL.
* **No business logic** — only cache key shaping + I/O. The pipeline owns the
  decision of *what* to cache.
* **Deterministic, human-readable key** — built by
  :func:`app.core.db.search_cache_key.build_pipeline_search_cache_key` from
  parsed intent (format, artist, album, country, optional city) so paraphrases
  share a slot when the parser resolves the same fields.

Async-first: all I/O is awaited via ``redis.asyncio``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import redis.asyncio as redis_async  # type: ignore[import-not-found]
    from redis.exceptions import RedisError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - module absence is handled gracefully
    redis_async = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment,misc]

from app.core.config import get_settings
from app.core.db.search_cache_key import PIPELINE_CACHE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Back-compat alias for purge logic and external imports.
_PIPELINE_CACHE_SCHEMA_VERSION: int = PIPELINE_CACHE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Client singleton (lazy, async)
# ---------------------------------------------------------------------------

_client: Any | None = None
_client_init_failed: bool = False


async def get_redis_client() -> Any | None:
    """Return a connected async Redis client, or ``None`` if unavailable.

    Connection failures flip a sticky ``_client_init_failed`` flag so we don't
    keep paying the connect timeout on every request when Redis is down.
    """
    global _client, _client_init_failed

    if redis_async is None or _client_init_failed:
        return None
    if _client is not None:
        return _client

    settings = get_settings()
    url = (settings.redis_url or "").strip()
    if not url:
        _client_init_failed = True
        return None

    try:
        client = redis_async.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        await client.ping()
    except (RedisError, OSError, ValueError) as exc:
        _client_init_failed = True
        logger.warning(
            "redis_connect_failed_falling_back_to_live_pipeline",
            extra={"stage": "redis_cache", "reason": str(exc)[:200]},
        )
        return None

    _client = client
    logger.info("redis_cache_connected", extra={"stage": "redis_cache", "status": "success"})
    return _client


async def dispose_redis_client() -> None:
    """Close the singleton (lifespan shutdown hook)."""
    global _client, _client_init_failed
    if _client is None:
        return
    try:
        close_coro = getattr(_client, "aclose", None) or getattr(_client, "close", None)
        if close_coro is not None:
            await close_coro()
    except Exception:
        logger.debug("redis_close_error", extra={"stage": "redis_cache"})
    _client = None
    _client_init_failed = False


# ---------------------------------------------------------------------------
# Public read / write helpers
# ---------------------------------------------------------------------------


async def get_cached_search(cache_key: str) -> dict[str, Any] | None:
    """Return the cached payload dict, or ``None`` on miss / Redis outage.

    Debug bypass: when ``settings.debug=True``, we never read from Redis — the
    operator is iterating on pipeline behaviour and must see fresh results,
    not the 7-day-old snapshot from before the latest code change. This is
    the same contract enforced by :mod:`app.core.db.cache`.
    """
    settings = get_settings()
    if settings.debug:
        logger.info(
            "redis_cache_bypassed_debug_mode",
            extra={
                "stage": "redis_cache",
                "cache_key_head": cache_key[:64],
                "reason": "debug",
            },
        )
        return None
    client = await get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(cache_key)
    except (RedisError, OSError) as exc:
        logger.warning(
            "redis_get_failed",
            extra={
                "stage": "redis_cache",
                "cache_key_head": cache_key[:64],
                "reason": str(exc)[:200],
            },
        )
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "redis_payload_corrupt_evicting",
            extra={"stage": "redis_cache", "cache_key_head": cache_key[:64]},
        )
        try:
            await client.delete(cache_key)
        except (RedisError, OSError):
            pass
        return None
    if not isinstance(data, dict):
        return None
    return data


async def set_cached_search(
    cache_key: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int,
) -> bool:
    """Store ``payload`` JSON with ``EX=ttl_seconds``. Returns success.

    Debug bypass: when ``settings.debug=True``, the write is skipped so a
    debug-time pipeline run never poisons the production cache with a payload
    that may contain dev-only diagnostics or an in-flight schema variant.
    """
    settings = get_settings()
    if settings.debug:
        logger.info(
            "redis_cache_write_skipped_debug_mode",
            extra={
                "stage": "redis_cache",
                "cache_key_head": cache_key[:64],
                "reason": "debug",
            },
        )
        return False
    client = await get_redis_client()
    if client is None:
        return False
    try:
        body = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "redis_payload_serialization_failed",
            extra={"stage": "redis_cache", "reason": str(exc)[:200]},
        )
        return False
    try:
        await client.set(cache_key, body, ex=max(60, int(ttl_seconds)))
    except (RedisError, OSError) as exc:
        logger.warning(
            "redis_set_failed",
            extra={
                "stage": "redis_cache",
                "cache_key_head": cache_key[:64],
                "reason": str(exc)[:200],
            },
        )
        return False
    logger.info(
        "redis_cache_set",
        extra={
            "stage": "redis_cache",
            "cache_key_head": cache_key[:64],
            "ttl_seconds": ttl_seconds,
            "bytes": len(body),
        },
    )
    return True


async def purge_stale_pipeline_cache_versions() -> int:
    """Delete every cached search-response key that predates the current schema.

    Called once from the FastAPI ``lifespan`` so an operator restarting the
    backend after a pipeline-behaviour bump (i.e. a new value of
    :data:`_PIPELINE_CACHE_SCHEMA_VERSION`) does not have to ``FLUSHALL`` Redis
    by hand. Stale keys are also harmless on their own — the new key builder
    just won't read them — but pruning them frees memory immediately.

    Best-effort: returns 0 on any Redis outage / failure.
    """
    client = await get_redis_client()
    if client is None:
        return 0

    current_prefix = f"cratedigger:search:v{_PIPELINE_CACHE_SCHEMA_VERSION}:"
    deleted = 0

    try:
        # ``SCAN`` over the search-response keyspace and drop any prefix that
        # is not the current schema-version. Using ``SCAN`` (not ``KEYS *``)
        # keeps Redis non-blocking for production deployments.
        scan_iter = client.scan_iter(match="cratedigger:search:*", count=200)
        async for raw_key in scan_iter:
            key_str = raw_key if isinstance(raw_key, str) else raw_key.decode("utf-8", "replace")
            if key_str.startswith(current_prefix):
                continue
            try:
                removed = await client.delete(key_str)
            except (RedisError, OSError):
                continue
            if removed:
                deleted += int(removed)
    except (RedisError, OSError) as exc:
        logger.warning(
            "redis_cache_purge_stale_versions_failed",
            extra={"stage": "redis_cache", "reason": str(exc)[:200]},
        )
        return 0

    if deleted:
        logger.info(
            "redis_cache_purged_stale_pipeline_versions",
            extra={
                "stage": "redis_cache",
                "deleted_keys": deleted,
                "current_version": _PIPELINE_CACHE_SCHEMA_VERSION,
            },
        )
    return deleted


__all__ = [
    "dispose_redis_client",
    "get_cached_search",
    "get_redis_client",
    "purge_stale_pipeline_cache_versions",
    "set_cached_search",
]
