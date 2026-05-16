import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.cache import purge_expired_search_cache_rows
from app.db.database import dispose_engine, init_db
from app.db.store_loader import (
    repair_whitelist_store_domains,
    seed_whitelist_stores_if_empty,
    sync_whitelist_store_catalogue,
)
from app.logging_config import setup_logging
from app.routers.search import router as search_router

settings = get_settings()
setup_logging(level=settings.log_level, log_format=settings.log_format)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.database_url:
        await init_db(database_url=settings.database_url, debug=settings.debug)
        inserted = await seed_whitelist_stores_if_empty()
        sync_summary = await sync_whitelist_store_catalogue()
        repair_summary = await repair_whitelist_store_domains()
        removed = await purge_expired_search_cache_rows()
        logger.info(
            "lifespan_db",
            extra={
                "stage": "startup",
                "stores_seeded_rows": inserted,
                "stores_sync": sync_summary,
                "stores_domain_repair": repair_summary,
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
        "log_format": settings.log_format,
        "database_configured": bool(settings.database_url),
    },
)

app = FastAPI(title="AiCrateDigg API", version="0.1.0", lifespan=lifespan)
app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok", "service": "backend"}
