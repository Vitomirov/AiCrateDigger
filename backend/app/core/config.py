from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_DIR = _BACKEND_DIR.parent


def _discover_env_files() -> tuple[str, ...]:
    """Load ``.env`` from repo root and/or ``backend/`` (whichever exists).

    OS environment variables always win over file values. Compose injects
    ``DATABASE_URL`` directly; local ``poetry run`` from ``backend/`` still
    picks up ``../.env`` when the root file is the only copy.
    """
    explicit = (__import__("os").environ.get("ENV_FILE") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return (str(path),) if path.is_file() else ()

    found: list[str] = []
    for candidate in (_BACKEND_DIR / ".env", _REPO_DIR / ".env"):
        if candidate.is_file():
            found.append(str(candidate))
    return tuple(found)


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    discogs_token: str | None = None
    #: Primary Postgres DSN from env ``DATABASE_URL`` (preferred).
    database_url: str | None = None
    #: Optional components used to build ``DATABASE_URL`` when the full DSN is not set.
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: int | None = None
    #: From env ``REDIS_URL``. When unset the cache layer silently no-ops.
    redis_url: str | None = None
    #: Browser URL for the Next.js UI — used when visitors open the API port by mistake.
    frontend_public_url: str = Field(default="http://localhost:3000")

    debug: bool = False
    #: Env ``SEARCH_RATE_LIMIT_ENABLED``. When ``False``, ``POST /search`` skips IP rate limiting.
    search_rate_limit_enabled: bool = Field(
        default=True,
        description="Enable 5 searches per IP per 24h on POST /search (SEARCH_RATE_LIMIT_ENABLED).",
    )
    search_rate_limit_max_requests: int = Field(default=5, ge=1, le=100)
    search_rate_limit_window_seconds: int = Field(default=86_400, ge=60)
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
    #: Opportunistic post-Tavily store discovery: when ``True``, the pipeline
    #: takes unknown-host snippets from the main consolidated Tavily call and
    #: feeds them into the discovery LLM (gpt-4o-mini) so real local shops
    #: surfaced by the artist/album query are upserted into ``whitelist_stores``
    #: AND added to the current request's prefilter whitelist. Adds ~0.5–1.5s
    #: of latency + one LLM call per uncached city query — disable to revert.
    pipeline_opportunistic_store_discovery_enabled: bool = True
    #: Minimum number of unknown-host snippets required before opportunistic
    #: discovery fires (avoids burning an LLM call on requests where the main
    #: Tavily output is already dominated by curated / blacklisted hosts).
    pipeline_opportunistic_discovery_min_unknown_hosts: int = Field(default=2, ge=1, le=10)
    #: Hard cap on candidate URLs passed to the LLM extractor after Python pre-filtering.
    pipeline_prefilter_max_candidates: int = Field(default=6, ge=3, le=20)
    #: Per-host cap for the Python pre-filter (top-N best-scored deep links per shop).
    pipeline_prefilter_max_per_host: int = Field(default=1, ge=1, le=5)
    #: Single consolidated Tavily call: ``max_results`` upper bound.
    tavily_single_call_max_results: int = Field(default=10, ge=5, le=30)
    #: Single consolidated Tavily call: ``search_depth`` value.
    tavily_single_call_depth: str = Field(default="advanced")

    #: Legacy cap for :func:`run_tavily_search` multi-query mode; geo pipeline uses chunking below.
    tavily_max_http_calls: int = Field(default=4, ge=1, le=20)
    #: One request carries up to this many ``include_domains``; only then split (see chunk threshold).
    tavily_max_results_per_batch: int = Field(default=6, ge=5, le=8)
    tavily_max_results_per_domain_aggregate: int = Field(default=1, ge=1, le=5)
    #: ``basic`` reduces Tavily credit burn vs ``advanced``.
    tavily_search_depth: str = Field(default="basic", description='Tavily "search_depth" payload value')
    #: Split ``include_domains`` across parallel requests only when above this count (e.g. 21 → 2 calls).
    tavily_domain_chunk_threshold: int = Field(default=20, ge=8, le=100)
    #: Consolidated ``site:`` power queries truncate before this approximate character budget.
    tavily_power_query_max_chars: int = Field(default=400, ge=200, le=950)
    #: Domains bundled into one textual ``(site: · OR ·)`` tail per Tavily POST (city-tier fan-in).
    tavily_local_power_max_domains_per_chunk: int = Field(default=5, ge=1, le=20)
    #: Drop very low Tavily relevance scores (reduces blog/noise URLs before extraction).
    tavily_min_result_score: float = Field(default=0.20, ge=0.0, le=1.0)
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

    @property
    def resolved_database_url(self) -> str | None:
        """``DATABASE_URL`` from env, or built from ``POSTGRES_*`` components."""
        if self.database_url:
            return self.database_url
        user = (self.postgres_user or "").strip()
        password = (self.postgres_password or "").strip()
        host = (self.postgres_host or "").strip()
        db = (self.postgres_db or "").strip()
        if not (user and password and host and db):
            return None
        port = self.postgres_port
        port_segment = f":{port}" if port else ""
        return (
            f"postgresql+asyncpg://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}{port_segment}/{quote_plus(db)}"
        )

    @property
    def database_enabled(self) -> bool:
        return bool(self.resolved_database_url)

    @field_validator(
        "debug",
        "search_cache_enabled",
        "search_rate_limit_enabled",
        mode="before",
    )
    @classmethod
    def _parse_bool_env(cls, v: object) -> object:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        return v

    @field_validator("log_format", mode="before")
    @classmethod
    def _normalize_log_format(cls, v: object) -> str:
        s = str(v or "human").strip().lower()
        return s if s in ("human", "json") else "human"

    @field_validator(
        "database_url",
        "redis_url",
        "discogs_token",
        "postgres_user",
        "postgres_password",
        "postgres_db",
        "postgres_host",
        mode="before",
    )
    @classmethod
    def _strip_optional_env_str(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v

    model_config = SettingsConfigDict(
        env_file=_discover_env_files(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
