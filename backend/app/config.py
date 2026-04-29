from functools import lru_cache

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    discogs_token: str | None = None
    database_url: str

    chroma_db_dir: str = "./chroma_db"

    debug: bool = False
    log_level: str = "INFO"

    discogs_base_url: str = "https://api.discogs.com"
    discogs_user_agent: str = "AiCrateDigger/0.1 (+https://github.com/ai-cratedigger)"
    discogs_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def verify_discogs(self) -> bool:
        """
        Minimal runtime check:
        - verifies API reachable
        - verifies auth header works (if token exists)
        """
        headers = {
            "User-Agent": self.discogs_user_agent,
            "Accept": "application/json",
        }

        if self.discogs_token:
            headers["Authorization"] = f"Discogs token={self.discogs_token}"

        try:
            with httpx.Client(timeout=self.discogs_timeout_seconds) as client:
                r = client.get(
                    f"{self.discogs_base_url}/",
                    headers=headers,
                )
                return r.status_code == 200
        except Exception:
            return False


@lru_cache
def get_settings() -> Settings:
    s = Settings()

    # 🔥 QUICK DISC0GS CHECK (runs once per process)
    ok = s.verify_discogs()
    print(f"[DISC0GS CHECK] API OK = {ok}")

    return s