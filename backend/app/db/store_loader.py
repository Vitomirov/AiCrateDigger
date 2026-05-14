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

ALLOWED_STORE_TYPES: frozenset[str] = frozenset({"local_shop", "regional_ecommerce", "marketplace"})

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


async def sync_whitelist_store_catalogue() -> dict[str, int]:
    """Reconcile DB rows with :data:`ALLOWED_STORES` without overwriting curated edits.

    Inserts rows missing in DB, and back-fills NULL ``city`` / ``store_type`` columns
    from the code catalogue. Never demotes an existing ``local_shop`` to a generic
    ``regional_ecommerce`` (operator may have promoted it deliberately in SQL).
    """
    settings = get_settings()
    summary = {"inserted": 0, "city_filled": 0, "type_filled": 0}
    if not settings.database_url:
        return summary
    try:
        sf = session_factory()
    except RuntimeError:
        return summary

    code_by_domain: dict[str, StoreEntry] = {
        canonical_store_domain(s.domain): s for s in ALLOWED_STORES
    }

    async with sf() as session:
        existing_rows = (
            await session.scalars(select(WhitelistStoreORM))
        ).all()
        existing_by_domain = {canonical_store_domain(r.domain): r for r in existing_rows}

        for dom, entry in code_by_domain.items():
            row = existing_by_domain.get(dom)
            if row is None:
                leg = (entry.country_code or "XX")[:8]
                session.add(
                    WhitelistStoreORM(
                        name=entry.name,
                        domain=dom,
                        country=leg,
                        country_code=entry.country_code,
                        region=entry.region,
                        ships_to_json=json.dumps(list(entry.ships_to)),
                        priority=entry.priority,
                        is_active=entry.is_active,
                        city=entry.city,
                        latitude=entry.latitude,
                        longitude=entry.longitude,
                        store_type=entry.store_type,
                    )
                )
                summary["inserted"] += 1
                continue

            if (row.city is None or not str(row.city).strip()) and entry.city:
                row.city = entry.city
                summary["city_filled"] += 1

            existing_type = (getattr(row, "store_type", None) or "").strip().lower()
            if existing_type not in ALLOWED_STORE_TYPES and entry.store_type:
                row.store_type = entry.store_type
                summary["type_filled"] += 1

        await session.commit()

    logger.info(
        "whitelist_stores_sync",
        extra={"stage": "stores", **summary},
    )
    return summary


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


# Default minimum number of ``local_shop`` rows a resolved city should have before
# the pipeline starts. Below this we trigger on-demand discovery (Tavily + LLM).
# Strict ``< 2`` policy: as soon as the city has only one indie (or none) on file,
# we await discovery so the new domains hit Tavily on the SAME request.
LOCAL_COVERAGE_THRESHOLD: int = 2


async def ensure_local_coverage(
    city: str | None,
    country_code: str | None,
    *,
    threshold: int = LOCAL_COVERAGE_THRESHOLD,
) -> dict[str, object]:
    """Guarantee ``threshold`` ``local_shop`` rows exist for ``(city, country_code)``.

    Returns a small summary dict for the pipeline debug payload. Safe no-op when
    inputs are missing, the DB is unavailable, or discovery returns empty.
    """
    summary: dict[str, object] = {
        "city": city,
        "country_code": country_code,
        "threshold": threshold,
        "before": 0,
        "after": 0,
        "triggered": False,
        "discovery": None,
        "skipped_reason": None,
    }
    settings = get_settings()
    if not city or not country_code:
        summary["skipped_reason"] = "missing_city_or_country_code"
        return summary
    if not settings.database_url:
        summary["skipped_reason"] = "no_database_url"
        return summary

    # Local import — keeps the discovery module optional at cold start and avoids
    # pulling httpx/openai when ``ensure_local_coverage`` is never called.
    from app.services.store_discovery import (
        count_local_shops_in_city,
        discover_new_stores,
    )

    before = await count_local_shops_in_city(city, country_code)
    summary["before"] = before
    if before >= threshold:
        summary["after"] = before
        summary["skipped_reason"] = "coverage_already_sufficient"
        return summary

    summary["triggered"] = True
    report = await discover_new_stores(city=city, country_code=country_code)
    summary["discovery"] = report.as_dict()
    summary["after"] = await count_local_shops_in_city(city, country_code)
    return summary
