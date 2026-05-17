"""Effective early-stop floors for the geo-tier widening loop.

The pipeline keeps widening tiers (city → country → region → continental →
global) until it has collected ``effective_stop_floor()`` validated listings;
this module owns the rules that map ``(tier, geo_confidence, settings)`` to a
concrete numeric floor.
"""

from __future__ import annotations

from typing import Any

from app.policies.geo_scope import Tier


def tier_validated_stop_floor(tier: Tier, *, settings: Any) -> int:
    """Base validated-listing floor for ``tier`` (pre-confidence adjustment)."""
    if tier in ("city", "country"):
        return int(settings.pipeline_geo_stop_country)
    if tier == "region":
        return int(settings.pipeline_geo_stop_region)
    if tier == "continental":
        return int(settings.pipeline_geo_stop_continental)
    return 9999


def effective_stop_floor(tier: Tier, *, settings: Any, confidence: float) -> int:
    """High geo confidence → stop widening sooner; low → require more hits first."""
    base = tier_validated_stop_floor(tier, settings=settings)
    if confidence >= 0.88:
        return max(1, base)
    bump = int((0.88 - confidence) * 6)
    return min(20, max(1, base + max(0, bump)))
