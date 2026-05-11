"""Load EU vinyl store allowlist from Postgres when configured, else from code.

If ``whitelist_stores`` has rows, **only** those rows are used (full replacement
of the in-code list). If the table is empty or ``DATABASE_URL`` is unset, the
pipeline uses :func:`app.policies.eu_stores.get_active_stores`.
"""

from __future__ import annotations

import json
import logging
from sqlalchemy import func, select

from app.config import get_settings
from app.db.database import WhitelistStoreORM, session_factory
from app.policies.eu_stores import ALLOWED_STORES, StoreEntry, get_active_stores

logger = logging.getLogger(__name__)


def _orm_to_entry(row: WhitelistStoreORM) -> StoreEntry:
    try:
        ships = tuple(json.loads(row.ships_to_json or "[]"))
        if not all(isinstance(x, str) for x in ships):
            ships = ("EU",)
    except json.JSONDecodeError:
        ships = ("EU",)
    return StoreEntry(
        name=row.name,
        domain=row.domain.strip().lower(),
        country=row.country,
        ships_to=ships,
        priority=int(row.priority),
        is_active=bool(row.is_active),
    )


async def seed_whitelist_stores_if_empty() -> int:
    """Insert :data:`ALLOWED_STORES` when the table has zero rows. Returns rows inserted."""
    settings = get_settings()
    if not settings.database_url:
        return 0
    try:
        sf = session_factory()
    except RuntimeError:
        return 0

    async with sf() as session:
        n = await session.scalar(select(func.count()).select_from(WhitelistStoreORM))
        if (n or 0) > 0:
            return 0
        for s in ALLOWED_STORES:
            session.add(
                WhitelistStoreORM(
                    name=s.name,
                    domain=s.domain.strip().lower(),
                    country=s.country,
                    ships_to_json=json.dumps(list(s.ships_to)),
                    priority=s.priority,
                    is_active=s.is_active,
                )
            )
        await session.commit()

    logger.info(
        "whitelist_stores_seeded",
        extra={"stage": "stores", "inserted": len(ALLOWED_STORES)},
    )
    return len(ALLOWED_STORES)


async def load_active_stores() -> tuple[StoreEntry, ...]:
    """DB-backed allowlist when migrations/seed populated the table; else code tuple."""
    settings = get_settings()
    if not settings.database_url:
        return get_active_stores()
    try:
        sf = session_factory()
    except RuntimeError:
        return get_active_stores()

    async with sf() as session:
        result = await session.scalars(
            select(WhitelistStoreORM)
            .where(WhitelistStoreORM.is_active.is_(True))
            .order_by(WhitelistStoreORM.priority.desc(), WhitelistStoreORM.domain.asc())
        )
        rows = result.all()

    if not rows:
        return get_active_stores()

    return tuple(_orm_to_entry(r) for r in rows)
