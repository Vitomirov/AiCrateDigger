from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    discogs_token: str
    database_url: str
    # Local persistent directory for Chroma vector store.
    chroma_db_dir: str = "./chroma_db"

    # --- Observability / behavior flags ---
    # When `debug=true`, API handlers attach the full PipelineContext trace under
    # the `debug` key of the response body. Driven by the `DEBUG` env var.
    debug: bool = False
    log_level: str = "INFO"

    # --- Discogs (used by the parser to deterministically resolve "nth album") ---
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
