"""UTC daily counters in Redis — account-level spend fuse."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.core.db.redis_cache import get_redis_client
from app.core.quota.exceptions import QuotaExceededError, QuotaUnavailableError
from app.core.quota.kinds import QuotaKind
from app.core.quota.limits import daily_limit_for_kind

logger = logging.getLogger(__name__)

_QUOTA_KEY_PREFIX = "quota"


def _utc_day_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _redis_key(kind: QuotaKind) -> str:
    return f"{_QUOTA_KEY_PREFIX}:{kind.value}:{_utc_day_suffix()}"


def seconds_until_utc_quota_reset() -> int:
    """TTL for quota keys: until next UTC midnight plus one hour buffer."""
    now = datetime.now(timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return max(3600, int((next_midnight - now).total_seconds()) + 3600)


def _quota_policy(settings: Settings) -> tuple[bool, bool]:
    return settings.global_daily_quota_enabled, settings.global_daily_quota_fail_closed


async def _read_count(client: Any, key: str) -> int:
    raw = await client.get(key)
    if raw is None:
        return 0
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


async def assert_quota_available(kind: QuotaKind, *, amount: int = 1) -> None:
    """Pre-flight check before a paid provider call.

    Raises :class:`QuotaExceededError` or :class:`QuotaUnavailableError`.
    No-op when quotas are disabled or the bucket limit is ``0`` (unlimited).
    """
    if amount < 1:
        return

    settings = get_settings()
    enabled, fail_closed = _quota_policy(settings)
    if not enabled:
        return

    limit = daily_limit_for_kind(settings, kind)
    if limit <= 0:
        return

    client = await get_redis_client()
    if client is None:
        if fail_closed:
            logger.warning(
                "global_quota_redis_unavailable",
                extra={"stage": "global_quota", "kind": kind.value, "status": "blocked"},
            )
            raise QuotaUnavailableError(reason="redis_unavailable")
        logger.warning(
            "global_quota_redis_unavailable_bypass",
            extra={"stage": "global_quota", "kind": kind.value, "status": "bypass"},
        )
        return

    key = _redis_key(kind)
    try:
        current = await _read_count(client, key)
    except Exception as exc:
        if fail_closed:
            raise QuotaUnavailableError(reason=str(exc)[:200]) from exc
        logger.warning(
            "global_quota_read_error_bypass",
            extra={"stage": "global_quota", "kind": kind.value, "reason": str(exc)[:200]},
        )
        return

    if current + amount > limit:
        logger.info(
            "global_quota_exceeded",
            extra={
                "stage": "global_quota",
                "kind": kind.value,
                "current": current,
                "limit": limit,
                "requested": amount,
            },
        )
        raise QuotaExceededError(kind, limit=limit, current=current)


async def record_quota_usage(kind: QuotaKind, *, amount: int = 1) -> None:
    """Increment the daily counter after a successful provider call."""
    if amount < 1:
        return

    settings = get_settings()
    enabled, fail_closed = _quota_policy(settings)
    if not enabled:
        return

    limit = daily_limit_for_kind(settings, kind)
    if limit <= 0:
        return

    client = await get_redis_client()
    if client is None:
        if fail_closed:
            raise QuotaUnavailableError(reason="redis_unavailable")
        return

    key = _redis_key(kind)
    ttl = seconds_until_utc_quota_reset()
    try:
        new_value = await client.incrby(key, amount)
        if int(new_value) == amount:
            await client.expire(key, ttl)
        if int(new_value) > limit:
            logger.warning(
                "global_quota_soft_overrun",
                extra={
                    "stage": "global_quota",
                    "kind": kind.value,
                    "count": int(new_value),
                    "limit": limit,
                },
            )
    except Exception as exc:
        if fail_closed:
            raise QuotaUnavailableError(reason=str(exc)[:200]) from exc
        logger.warning(
            "global_quota_write_error",
            extra={"stage": "global_quota", "kind": kind.value, "reason": str(exc)[:200]},
        )
