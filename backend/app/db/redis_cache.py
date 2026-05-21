"""Redis cache layer for the consolidated vinyl-search pipeline.

A 7-day TTL Redis cache that short-circuits the entire pipeline when a request
re-hits a recently computed query. Every Tavily credit + OpenAI token spent on
the cached lookup drops to **zero** until the TTL elapses.

Design contract
---------------
* **Optional** — if ``settings.redis_url`` is unset OR the Redis server is
  unreachable, every helper here returns a safe default (``None`` on read,
  ``False`` on write). The pipeline always falls back to a live search.
* **No business logic** — only cache key shaping + I/O. The pipeline owns the
  decision of *what* to cache.
* **Deterministic, human-readable key** — the key format mirrors the spec:
  ``cratedigger:search:{format}:{artist}:{album}:{country_code_or_global}`` with
  lowercase, trimmed, underscore-collapsed values so the same logical query
  never aliases across casing/whitespace variations.

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

from app.config import get_settings

logger = logging.getLogger(__name__)


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
    """Deterministic, human-readable cache key.

    Format: ``cratedigger:search:{format}:{artist}:{album}:{country|global}``.
    All segments are lowercase, trimmed, with whitespace replaced by ``_``.
    """
    fmt = _normalize_token(format_token, fallback="vinyl")
    artist_part = _normalize_token(artist, fallback="unknown_artist")
    album_part = _normalize_token(album, fallback="unknown_album")
    country_part = _normalize_token(country_code, fallback="global")
    return f"cratedigger:search:{fmt}:{artist_part}:{album_part}:{country_part}"


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
    """Return the cached payload dict, or ``None`` on miss / Redis outage."""
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
    """Store ``payload`` JSON with ``EX=ttl_seconds``. Returns success."""
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


__all__ = [
    "build_redis_search_key",
    "dispose_redis_client",
    "get_cached_search",
    "set_cached_search",
]
