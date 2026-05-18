"""Shared dataclasses for the geo-tier widening loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.parse_schema import ParsedQuery
from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import GeoIntent, NormalizedGeoIntent, Tier


@dataclass(slots=True)
class TierContext:
    """Per-request immutables threaded through every tier iteration."""

    parsed: ParsedQuery
    album_title: str
    geo: GeoIntent
    norm: NormalizedGeoIntent
    stores: tuple[StoreEntry, ...]
    settings: Any
    core_query: str
    tavily_relaxation_queries: Any
    curated_city_local_domains: frozenset[str]
    prioritize_physical_locals: bool
    all_allowed: frozenset[str]


@dataclass(slots=True)
class TierLoopState:
    """Mutable state carried across iterations of the geo widening loop."""

    aggregated: dict[str, Any] = field(default_factory=dict)
    listing_tier_map: dict[str, Tier] = field(default_factory=dict)
    store_lookup: dict[str, StoreEntry] = field(default_factory=dict)
    editorial_discovery_blocked: set[str] = field(default_factory=set)
    album_match_by_url: dict[str, bool] = field(default_factory=dict)
    verifier_summary: list[dict[str, Any]] = field(default_factory=list)
    tier_traces: list[dict[str, Any]] = field(default_factory=list)
    tiers_attempted: list[Tier] = field(default_factory=list)
    last_tier: Tier = "continental"


@dataclass(slots=True)
class TierStoreSelection:
    """Per-tier store pool after filtering, sorting, capping and deduping."""

    capped: tuple[StoreEntry, ...]
    pool_size: int
    min_dom: int
    store_domains: list[str]
    domains_for_tavily: list[str]
    tier_lookup: dict[str, StoreEntry]

    @property
    def is_below_floor(self) -> bool:
        return len(self.capped) < self.min_dom
