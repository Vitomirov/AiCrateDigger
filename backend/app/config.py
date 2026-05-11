from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    discogs_token: str | None = None
    database_url: str | None = None

    debug: bool = False
    log_level: str = "INFO"
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = Field(default=3600, ge=60)

    #: Batched-domain Tavily requests per vinyl search (budget ≈ 6 calls → up to 4 pipeline results).
    tavily_max_http_calls: int = Field(default=6, ge=1, le=20)
    tavily_max_results_per_batch: int = Field(default=10, ge=4, le=24)
    tavily_max_results_per_domain_aggregate: int = Field(default=2, ge=1, le=5)
    pipeline_max_results: int = Field(default=4, ge=1, le=50)

    discogs_base_url: str = "https://api.discogs.com"
    discogs_user_agent: str = "AiCrateDigger/0.1 (+https://github.com/ai-cratedigger)"
    discogs_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()