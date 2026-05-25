import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.db.cache import purge_expired_search_cache_rows
from app.core.db.database import dispose_engine, init_db
from app.core.db.redis_cache import dispose_redis_client, purge_stale_pipeline_cache_versions
from app.core.db.store_loader import (
    repair_whitelist_store_domains,
    seed_whitelist_stores_if_empty,
    sync_whitelist_store_catalogue,
)
from app.core.logging_config import setup_logging
from app.api.routers.search import router as search_router

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

    # Sweep Redis search-cache entries that predate the current pipeline
    # schema version. After a pipeline-behaviour bump this prevents the next
    # search from serving a stale snapshot that was computed by the old code
    # path (which is exactly what happened to "Cascet Undead Soil DE" — the
    # `v1` entry kept short-circuiting Stage 6.5 opportunistic discovery).
    stale_purged = await purge_stale_pipeline_cache_versions()
    logger.info(
        "lifespan_redis",
        extra={
            "stage": "startup",
            "stale_pipeline_cache_purged": stale_purged,
        },
    )

    yield
    await dispose_engine()
    await dispose_redis_client()


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
