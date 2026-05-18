"""Early-stop / widening termination policy for validated listing aggregates."""

from __future__ import annotations

import logging

from app.policies.geo_scope import Tier

logger = logging.getLogger(__name__)

# Tiers where a single validated listing is treated as sufficient local intent coverage.
_LOCALIZED_SATISFACTION_TIERS: frozenset[Tier] = frozenset({"city", "country"})


def compute_early_stop(
    *,
    tier: Tier,
    aggregated_validated_count: int,
    stop_floor_need: int,
) -> tuple[bool, str | None]:
    """Whether to stop widening after this tier completes.

    Returns ``(early_stop, reason)`` where ``reason`` is suitable for traces/logs.

    Localized aggressive rule: after ``city`` or ``country``, if we already have
    at least one validated listing in the cross-tier aggregate, do not spend
    credits widening to ``region`` / ``continental`` unless we needed more hits
    for the classic stop-floor (superseded here by the localized rule).
    """
    if tier in _LOCALIZED_SATISFACTION_TIERS and aggregated_validated_count >= 1:
        return True, "localized_satisfied_min_one"
    if aggregated_validated_count >= stop_floor_need:
        return True, "stop_floor_met"
    return False, None


def log_localized_early_exit_if_applicable(
    *,
    tier: Tier,
    aggregated_validated_count: int,
    reason: str | None,
) -> None:
    """Structured log when aggressive localized early exit fires."""
    if reason != "localized_satisfied_min_one":
        return
    logger.info(
        "early_exit_localized_satisfied",
        extra={
            "stage": "geo_widening",
            "tier": tier,
            "aggregated_validated_total": aggregated_validated_count,
            "early_stop_reason": reason,
        },
    )
