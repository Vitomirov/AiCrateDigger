"""PostgreSQL via SQLAlchemy 2.0 async (optional).

When ``DATABASE_URL`` is unset, engines are not created and callers keep using
in-code policies (e.g. :mod:`app.domains.engine.policies.eu_stores`) only.

Set ``DATABASE_URL`` in the repo-root ``.env`` or the process environment.
Inside Docker Compose use host ``db``; from the host machine use
``localhost`` and the published port (default ``5433``).
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import DateTime, Float, Integer, String, Text, func, inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None

_EXPECTED_TABLES = frozenset({"whitelist_stores", "search_response_cache"})


class Base(DeclarativeBase):
    pass


class WhitelistStoreORM(Base):
    """Mirror of :class:`app.domains.engine.policies.eu_stores.StoreEntry` for DB-backed allowlist."""

    __tablename__ = "whitelist_stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    domain: Mapped[str] = mapped_column(String(160), unique=True, nullable=False, index=True)
    country: Mapped[str] = mapped_column(String(8), nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)
    region: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ships_to_json: Mapped[str] = mapped_column(Text, nullable=False, default='["EU"]')
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    store_type: Mapped[str | None] = mapped_column(String(32), nullable=True)


class SearchResponseCacheORM(Base):
    """Repeat-search TTL cache: keyed by normalized query + resolved album."""

    __tablename__ = "search_response_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)


def _to_async_url(database_url: str) -> str:
    u = database_url.strip()
    if u.startswith("postgresql+asyncpg://"):
        return u
    if u.startswith("postgresql://"):
        return u.replace("postgresql://", "postgresql+asyncpg://", 1)
    raise ValueError(
        "DATABASE_URL must be postgres; use postgresql+asyncpg:// or postgresql:// (we coerce the latter).",
    )


def mask_database_url(database_url: str | None) -> str | None:
    """Redact password for logs."""
    if not database_url:
        return None
    try:
        parsed = urlparse(database_url)
        host = parsed.hostname or "?"
        port = f":{parsed.port}" if parsed.port else ""
        db = (parsed.path or "/").lstrip("/") or "?"
        user = parsed.username or "?"
        return f"{parsed.scheme}://{user}:***@{host}{port}/{db}"
    except Exception:
        return "<invalid DATABASE_URL>"


def is_database_configured() -> bool:
    """True when a Postgres DSN is available from env (``DATABASE_URL`` or ``POSTGRES_*``)."""
    from app.core.config import get_settings

    return get_settings().database_enabled


def get_resolved_database_url() -> str | None:
    """Central accessor: env ``DATABASE_URL`` or DSN built from ``POSTGRES_*``."""
    from app.core.config import get_settings

    return get_settings().resolved_database_url


def session_factory() -> async_sessionmaker[AsyncSession]:
    if _async_session_factory is None:
        raise RuntimeError("Database is not initialised (missing DATABASE_URL or init_db not run).")
    return _async_session_factory


async def init_db(*, database_url: str, debug: bool = False) -> None:
    """Create engine, session factory, and tables (no Alembic yet — ``create_all`` only)."""
    global _engine, _async_session_factory

    async_url = _to_async_url(database_url)
    _engine = create_async_engine(async_url, echo=debug)
    _async_session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        autoflush=False,
    )
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in (
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS country_code VARCHAR(2)",
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS region VARCHAR(32)",
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS city VARCHAR(128)",
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION",
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION",
            "ALTER TABLE whitelist_stores ADD COLUMN IF NOT EXISTS store_type VARCHAR(32)",
        ):
            await conn.execute(text(stmt))

        def _verify_tables(sync_conn: Any) -> list[str]:
            present = set(inspect(sync_conn).get_table_names())
            missing = _EXPECTED_TABLES - present
            if missing:
                logger.error(
                    "database_tables_missing_after_create_all",
                    extra={"stage": "database", "missing": sorted(missing)},
                )
                for table_name in sorted(missing):
                    table = Base.metadata.tables.get(table_name)
                    if table is not None:
                        table.create(sync_conn, checkfirst=True)
            return sorted(present & _EXPECTED_TABLES)

        tables_ready = await conn.run_sync(_verify_tables)

    logger.info(
        "database_init",
        extra={
            "stage": "database",
            "status": "success",
            "database_url": mask_database_url(database_url),
            "tables": tables_ready,
        },
    )


async def init_db_from_settings() -> bool:
    """Initialise Postgres from env (``DATABASE_URL`` or ``POSTGRES_*``). Returns whether DB is active."""
    from app.core.config import get_settings

    settings = get_settings()
    url = settings.resolved_database_url
    if not url:
        return False
    await init_db(database_url=url, debug=settings.debug)
    return True


async def dispose_engine() -> None:
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("database_shutdown", extra={"stage": "database", "status": "disposed"})
    _engine = None
    _async_session_factory = None
