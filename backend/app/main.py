import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.cache import purge_expired_search_cache_rows
from app.db.database import dispose_engine, init_db
from app.db.store_loader import seed_whitelist_stores_if_empty
from app.logging_config import setup_logging
from app.routers.search import router as search_router

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.database_url:
        await init_db(database_url=settings.database_url, debug=settings.debug)
        inserted = await seed_whitelist_stores_if_empty()
        removed = await purge_expired_search_cache_rows()
        logger.info(
            "lifespan_db",
            extra={
                "stage": "startup",
                "stores_seeded_rows": inserted,
                "cache_expired_purged": removed,
            },
        )
    else:
        logger.warning(
            "lifespan_db_skipped",
            extra={"stage": "startup", "reason": "DATABASE_URL unset — stores + cache use in-process defaults only"},
        )
    yield
    await dispose_engine()


logger.info(
    "startup",
    extra={
        "stage": "startup",
        "status": "success",
        "debug": settings.debug,
        "log_level": settings.log_level,
        "database_configured": bool(settings.database_url),
    },
)

app = FastAPI(title="AiCrateDigg API", version="0.1.0", lifespan=lifespan)
app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok", "service": "backend"}
