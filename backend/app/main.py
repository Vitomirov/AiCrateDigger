import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import get_settings
from app.core.quota import QuotaExceededError, QuotaUnavailableError
from app.core.db.cache import purge_expired_search_cache_rows
from app.core.db.database import (
    dispose_engine,
    get_resolved_database_url,
    init_db_from_settings,
    is_database_configured,
    mask_database_url,
)
from app.core.db.redis_cache import dispose_redis_client, purge_stale_pipeline_cache_versions
from app.core.db.store_loader import (
    repair_whitelist_store_domains,
    seed_whitelist_stores_if_empty,
    sync_whitelist_store_catalogue,
)
from app.core.logging_config import setup_logging
from app.core.production_guard import validate_production_settings
from app.api.routers.search import router as search_router

settings = get_settings()
setup_logging(level=settings.log_level, log_format=settings.log_format)
logger = logging.getLogger(__name__)
validate_production_settings(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if await init_db_from_settings():
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
            extra={
                "stage": "startup",
                "reason": "DATABASE_URL / POSTGRES_* unset — stores + cache use in-process defaults only",
            },
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
        "app_env": settings.app_env,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "log_format": settings.log_format,
        "database_configured": is_database_configured(),
        "database_url": mask_database_url(get_resolved_database_url()),
    },
)

_expose_api_schema = settings.app_env != "production"
app = FastAPI(
    title="AiCrateDigg API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _expose_api_schema else None,
    redoc_url="/redoc" if _expose_api_schema else None,
    openapi_url="/openapi.json" if _expose_api_schema else None,
)
app.include_router(search_router)

_QUOTA_UNAVAILABLE_DETAIL = (
    "Service temporarily unavailable. Please try again in a few minutes."
)
_QUOTA_EXCEEDED_DETAIL = (
    "Daily usage limit reached. Please try again tomorrow."
)


@app.exception_handler(QuotaUnavailableError)
async def quota_unavailable_handler(
    _request: Request,
    exc: QuotaUnavailableError,
) -> JSONResponse:
    logger.warning(
        "quota_unavailable",
        extra={"stage": "global_quota", "reason": exc.reason[:200]},
    )
    return JSONResponse(status_code=503, content={"detail": _QUOTA_UNAVAILABLE_DETAIL})


@app.exception_handler(QuotaExceededError)
async def quota_exceeded_handler(
    _request: Request,
    exc: QuotaExceededError,
) -> JSONResponse:
    logger.info(
        "quota_exceeded_response",
        extra={
            "stage": "global_quota",
            "kind": exc.kind.value,
            "current": exc.current,
            "limit": exc.limit,
        },
    )
    return JSONResponse(status_code=503, content={"detail": _QUOTA_EXCEEDED_DETAIL})


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """API has no HTML UI — send browsers to the Next.js app on port 3000."""
    return RedirectResponse(url=get_settings().frontend_public_url, status_code=307)


@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    s = get_settings()
    return {
        "status": "ok",
        "service": "backend",
        "database_configured": s.database_enabled,
    }
