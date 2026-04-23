import logging

from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import setup_logging
from app.routers.search import router as search_router

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
logger.info(
    "startup",
    extra={
        "stage": "startup",
        "status": "success",
        "debug": settings.debug,
        "log_level": settings.log_level,
    },
)

app = FastAPI(title="AiCrateDigg API", version="0.1.0")
app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok", "service": "backend"}
