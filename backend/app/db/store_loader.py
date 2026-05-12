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
from app.policies.eu_stores import ALLOWED_STORES, StoreEntry, StoreType, get_active_stores
from app.policies.store_domain import canonical_store_domain

logger = logging.getLogger(__name__)


def _normalize_country_code(raw: str | None) -> str | None:
    s = (raw or "").strip().upper()
    if s == "UK":
        s = "GB"
    if len(s) == 2 and s.isalpha():
        return s
    return None


def _orm_to_entry(row: WhitelistStoreORM) -> StoreEntry:
    try:
        ships = tuple(json.loads(row.ships_to_json or "[]"))
        if not all(isinstance(x, str) for x in ships):
            ships = ("EU",)
    except json.JSONDecodeError:
        ships = ("EU",)
    cc = _normalize_country_code(getattr(row, "country_code", None) or row.country)
    reg = getattr(row, "region", None)
    if isinstance(reg, str):
        reg = reg.strip() or None
    else:
        reg = None
    lq = int(getattr(row, "listing_quality", 5) or 5)
    lq = max(1, min(10, lq))
    city = getattr(row, "city", None)
    if isinstance(city, str):
        city = city.strip() or None
    else:
        city = None
    lat = getattr(row, "latitude", None)
    lon = getattr(row, "longitude", None)
    try:
        lat_f = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat_f = None
    try:
        lon_f = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon_f = None
    st_raw = getattr(row, "store_type", None) or "regional_ecommerce"
    st_t = str(st_raw).strip().lower()
    if st_t not in ("local_shop", "regional_ecommerce", "marketplace"):
        st_t = "regional_ecommerce"
    store_type: StoreType = st_t  # type: ignore[assignment]
    return StoreEntry(
        name=row.name,
        domain=canonical_store_domain(row.domain),
        country_code=cc,
        region=reg,
        ships_to=ships,
        priority=int(row.priority),
        is_active=bool(row.is_active),
        listing_quality=lq,
        city=city,
        latitude=lat_f,
        longitude=lon_f,
        store_type=store_type,
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
            leg = (s.country_code or "XX")[:8]
            session.add(
                WhitelistStoreORM(
                    name=s.name,
                    domain=canonical_store_domain(s.domain),
                    country=leg,
                    country_code=s.country_code,
                    region=s.region,
                    ships_to_json=json.dumps(list(s.ships_to)),
                    priority=s.priority,
                    is_active=s.is_active,
                    city=s.city,
                    latitude=s.latitude,
                    longitude=s.longitude,
                    store_type=s.store_type,
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
