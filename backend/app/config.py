from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    discogs_token: str | None = None
    database_url: str | None = None
    #: Optional Redis URL (e.g. ``redis://redis:6379/0``). When unset the cache layer
    #: silently no-ops and every request hits the live pipeline.
    redis_url: str | None = None

    debug: bool = False
    log_level: str = "INFO"
    log_format: Literal["human", "json"] = Field(
        default="human",
        description='Terminal-friendly "human" or NDJSON "json" for log aggregation.',
    )
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = Field(default=3600, ge=60)
    #: 7-day TTL for the Redis search-results cache (604_800 s). Configurable so a
    #: human can override at the env layer without code changes.
    redis_search_cache_ttl_seconds: int = Field(default=604_800, ge=60)
    #: Hard cap on candidate URLs passed to the LLM extractor after Python pre-filtering.
    pipeline_prefilter_max_candidates: int = Field(default=10, ge=3, le=20)
    #: Per-host cap for the Python pre-filter (top-N best-scored deep links per shop).
    pipeline_prefilter_max_per_host: int = Field(default=2, ge=1, le=5)
    #: Single consolidated Tavily call: ``max_results`` upper bound.
    tavily_single_call_max_results: int = Field(default=20, ge=5, le=30)
    #: Single consolidated Tavily call: ``search_depth`` value.
    tavily_single_call_depth: str = Field(default="advanced")

    #: Legacy cap for :func:`run_tavily_search` multi-query mode; geo pipeline uses chunking below.
    tavily_max_http_calls: int = Field(default=4, ge=1, le=20)
    #: One request carries up to this many ``include_domains``; only then split (see chunk threshold).
    tavily_max_results_per_batch: int = Field(default=6, ge=5, le=8)
    tavily_max_results_per_domain_aggregate: int = Field(default=2, ge=1, le=5)
    #: ``basic`` reduces Tavily credit burn vs ``advanced``.
    tavily_search_depth: str = Field(default="basic", description='Tavily "search_depth" payload value')
    #: Split ``include_domains`` across parallel requests only when above this count (e.g. 21 → 2 calls).
    tavily_domain_chunk_threshold: int = Field(default=20, ge=8, le=100)
    #: Consolidated ``site:`` power queries truncate before this approximate character budget.
    tavily_power_query_max_chars: int = Field(default=400, ge=200, le=950)
    #: Domains bundled into one textual ``(site: · OR ·)`` tail per Tavily POST (city-tier fan-in).
    tavily_local_power_max_domains_per_chunk: int = Field(default=5, ge=1, le=20)
    #: Drop very low Tavily relevance scores (reduces blog/noise URLs before extraction).
    tavily_min_result_score: float = Field(default=0.16, ge=0.0, le=1.0)
    #: Raw Tavily JSON cached per (artist, album, tier) before extraction; min 12h.
    tavily_intermediate_cache_ttl_seconds: int = Field(default=43200, ge=43200)
    #: Tavily intermittently returns 429 / 432 / 433 under burst traffic; retry before giving up.
    tavily_http_retry_attempts: int = Field(default=5, ge=1, le=10)
    tavily_http_retry_max_wait_seconds: float = Field(default=14.0, ge=1.0, le=120.0)
    #: Pause before city/country fanout (parallel with batched search) to reduce Tavily 432 bursts.
    tavily_fanout_stagger_seconds: float = Field(default=0.45, ge=0.0, le=5.0)
    #: Per-request circuit breaker: trip after N complete retry exhaustions to fail fast
    #: when Tavily hard-throttles the account (avoids minutes of wasted retries).
    tavily_circuit_breaker_failure_threshold: int = Field(default=2, ge=1, le=10)
    #: City-tier local fanout fires one asyncio task per shop; cap concurrency to avoid 432 storms.
    tavily_max_concurrent_domain_fanout: int = Field(default=2, ge=1, le=8)
    pipeline_max_results: int = Field(default=4, ge=1, le=50)
    #: Geo-aware ``include_domains`` caps (query text stays location-free).
    pipeline_geo_local_max_domains: int = Field(default=6, ge=1, le=12)
    pipeline_geo_regional_max_domains: int = Field(default=8, ge=1, le=16)
    pipeline_geo_global_max_domains: int = Field(default=10, ge=1, le=40)
    #: After each tier completes, stop widening if validated PDP count reaches this floor.
    pipeline_geo_stop_country: int = Field(default=2, ge=1, le=20)
    pipeline_geo_stop_region: int = Field(default=3, ge=1, le=20)
    pipeline_geo_stop_continental: int = Field(default=4, ge=1, le=20)

    #: Listing validation: RapidFuzz gates on title vs resolved album/artist (0–100 scale).
    listing_validation_album_fuzz_min: int = Field(
        default=82,
        ge=60,
        le=100,
        description="min max(token_set_ratio, partial_ratio) for album needle vs listing title",
    )
    listing_validation_artist_fuzz_min: int = Field(
        default=75,
        ge=50,
        le=100,
        description="min max(token_set, partial) when validation_artist is set",
    )
    #: Subtracted from album/artist mins when the URL looks like a product detail page.
    listing_validation_pdp_fuzz_relief: int = Field(default=10, ge=0, le=30)
    #: Relax album gate in debug (still rejects non-product URLs/titles).
    listing_validation_debug_album_fuzz_min: int = Field(default=82, ge=60, le=100)

    discogs_base_url: str = "https://api.discogs.com"
    discogs_user_agent: str = "AiCrateDigger/0.1 (+https://github.com/ai-cratedigger)"
    discogs_timeout_seconds: float = 8.0

    @field_validator("log_format", mode="before")
    @classmethod
    def _normalize_log_format(cls, v: object) -> str:
        s = str(v or "human").strip().lower()
        return s if s in ("human", "json") else "human"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()