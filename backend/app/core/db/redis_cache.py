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
* **Deterministic, human-readable key** — the key format mirrors the spec:
  ``cratedigger:search:v{N}:{format}:{artist}:{album}:{country_code_or_global}``
  with lowercase, trimmed, underscore-collapsed values so the same logical
  query never aliases across casing/whitespace variations.

Async-first: all I/O is awaited via ``redis.asyncio``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    import redis.asyncio as redis_async  # type: ignore[import-not-found]
    from redis.exceptions import RedisError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - module absence is handled gracefully
    redis_async = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment,misc]

from app.core.config import get_settings

logger = logging.getLogger(__name__)


#: Monotonic version stamp embedded in every Redis search-cache key.
#:
#: BUMP this integer whenever a pipeline behaviour change must invalidate
#: every cached response (new stage, prefilter rule change, prompt rewrite,
#: scoring tweak, ListingResult schema change, …). Old keys simply stop
#: being hit by reads and naturally expire via the existing 7-day TTL —
#: operators do not need to ``FLUSHALL`` Redis.
#:
#: Current bump: ``2`` — locks in the prefilter whitelist injection fix +
#: Stage 6.5 opportunistic store discovery + the new debug-bypass behaviour.
#: Anything cached under the ``v1`` (unversioned) key is intentionally
#: orphaned because it predates the local-shop fixes.
_PIPELINE_CACHE_SCHEMA_VERSION: int = 2


# ---------------------------------------------------------------------------
# Client singleton (lazy, async)
# ---------------------------------------------------------------------------

_client: Any | None = None
_client_init_failed: bool = False


def _normalize_token(value: str | None, *, fallback: str = "any") -> str:
    """Lowercase + trim + collapse whitespace to ``_`` so the cache key never
    aliases incorrectly (``" Pink   Floyd "`` and ``"pink_floyd"`` collide).

    Non-alphanumeric characters (other than ``-`` and ``.``) are also folded to
    ``_`` so user typos like commas or accents do not generate noisy keys.
    """
    s = (value or "").strip().lower()
    if not s:
        return fallback
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def build_redis_search_key(
    *,
    format_token: str | None,
    artist: str | None,
    album: str | None,
    country_code: str | None,
) -> str:
    """Deterministic, human-readable, schema-versioned cache key.

    Format: ``cratedigger:search:v{N}:{format}:{artist}:{album}:{country|global}``.
    All segments are lowercase, trimmed, with whitespace replaced by ``_``.

    The ``v{N}`` segment carries :data:`_PIPELINE_CACHE_SCHEMA_VERSION` so a
    pipeline-behaviour bump (new stage, prefilter rule, scoring change) does
    not require operators to ``FLUSHALL`` Redis — stale entries are simply
    orphaned by the new key namespace.
    """
    fmt = _normalize_token(format_token, fallback="vinyl")
    artist_part = _normalize_token(artist, fallback="unknown_artist")
    album_part = _normalize_token(album, fallback="unknown_album")
    country_part = _normalize_token(country_code, fallback="global")
    return (
        f"cratedigger:search:v{_PIPELINE_CACHE_SCHEMA_VERSION}"
        f":{fmt}:{artist_part}:{album_part}:{country_part}"
    )


async def _get_client() -> Any | None:
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
    client = await _get_client()
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
    client = await _get_client()
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
    client = await _get_client()
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
    "build_redis_search_key",
    "dispose_redis_client",
    "get_cached_search",
    "purge_stale_pipeline_cache_versions",
    "set_cached_search",
]
