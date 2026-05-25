"""Geo-tier widening loop — one iteration per tier (city → … → global).

The deterministic vinyl-search pipeline widens the geographic store pool until
enough validated listings are gathered (with localized early exit after
``city`` / ``country`` when intent is satisfied).

External callers should import from ``app.domains.search_pipeline.tier_runner`` (this package).
"""

from __future__ import annotations

from app.domains.search_pipeline.tier_runner.context import (
    TierContext,
    TierLoopState,
    TierStoreSelection,
)
from app.domains.search_pipeline.tier_runner.executor import process_tier

__all__ = (
    "TierContext",
    "TierLoopState",
    "TierStoreSelection",
    "process_tier",
)
