"""PostgreSQL via SQLAlchemy 2.0 async (optional).

When ``DATABASE_URL`` is unset, engines are not created and callers keep using
in-code policies (e.g. :mod:`app.policies.eu_stores`) only.

Docker Compose should set ``DATABASE_URL=postgresql+asyncpg://...@db:5432/...``
with host ``db`` equal to the Postgres service name on the Compose network.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    pass


class WhitelistStoreORM(Base):
    """Mirror of :class:`app.policies.eu_stores.StoreEntry` for DB-backed allowlist."""

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

    logger.info(
        "database_init",
        extra={
            "stage": "database",
            "status": "success",
            "tables": ["whitelist_stores", "search_response_cache"],
        },
    )


async def dispose_engine() -> None:
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("database_shutdown", extra={"stage": "database", "status": "disposed"})
    _engine = None
    _async_session_factory = None
