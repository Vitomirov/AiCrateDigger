"""PostgreSQL upsert and local-shop coverage queries for discovery."""

from __future__ import annotations

import json
import logging

from sqlalchemy import case, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.db.database import WhitelistStoreORM, session_factory
from app.domains.engine.policies.geo_scope import country_to_region
from app.domains.engine.search.store_discovery.models import (
    DISCOVERED_PRIORITY,
    DiscoveredStoreCandidate,
)

logger = logging.getLogger(__name__)


def dedupe_candidates_by_domain(
    candidates: list[DiscoveredStoreCandidate],
) -> list[DiscoveredStoreCandidate]:
    """Keep one row per canonical ``domain`` (highest ``confidence`` wins).

    Prevents duplicate rows in ``values_list`` that would confuse PostgreSQL
    bulk INSERT or trigger redundant conflict handling.
    """
    best_by_domain: dict[str, DiscoveredStoreCandidate] = {}
    for cand in candidates:
        prev = best_by_domain.get(cand.domain)
        if prev is None or cand.confidence > prev.confidence:
            best_by_domain[cand.domain] = cand
    return list(best_by_domain.values())


async def save_discovered_stores(
    candidates: list[DiscoveredStoreCandidate],
) -> tuple[list[str], list[str]]:
    """Bulk PostgreSQL UPSERT for discovered stores.

    Uses ``INSERT ... ON CONFLICT (domain) DO UPDATE`` so duplicate domains from
    Tavily/LLM never raise ``UniqueViolationError``. Conflicting rows refresh
    empty/null ``city``, ``country_code``, ``region``, ``store_type`` (when not
    yet set to a known type), and ``is_active``. Never updates ``priority``,
    ``name``, or ``ships_to_json``.

    Returns ``(inserted_domains, updated_domains)`` relative to rows that existed
    before this call (domains present only in this batch).
    """
    if not candidates:
        return [], []

    unique = dedupe_candidates_by_domain(candidates)
    domains_in_batch = [c.domain for c in unique]

    try:
        sf = session_factory()
    except RuntimeError:
        logger.warning(
            "store_discovery_no_db",
            extra={"stage": "store_discovery", "reason": "session_factory_unavailable"},
        )
        return [], []

    values_list: list[dict[str, object]] = []
    for cand in unique:
        reg = country_to_region(cand.country_code)
        leg = cand.country_code[:8] if cand.country_code else "XX"
        values_list.append(
            {
                "name": cand.name,
                "domain": cand.domain,
                "country": leg,
                "country_code": cand.country_code,
                "region": reg,
                "ships_to_json": json.dumps(["EU"]),
                "priority": DISCOVERED_PRIORITY,
                "is_active": True,
                "city": cand.city,
                "store_type": "local_shop",
            }
        )

    async with sf() as session:
        existing_rows = (
            await session.scalars(
                select(WhitelistStoreORM.domain).where(
                    WhitelistStoreORM.domain.in_(domains_in_batch)
                )
            )
        ).all()
        existing_before = set(existing_rows)

        stmt = pg_insert(WhitelistStoreORM).values(values_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=["domain"],
            set_={
                "city": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.city), ""),
                    stmt.excluded.city,
                ),
                "country_code": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.country_code), ""),
                    stmt.excluded.country_code,
                ),
                "region": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.region), ""),
                    stmt.excluded.region,
                ),
                "store_type": case(
                    (
                        WhitelistStoreORM.store_type.in_(
                            ("local_shop", "regional_ecommerce", "marketplace")
                        ),
                        WhitelistStoreORM.store_type,
                    ),
                    else_=stmt.excluded.store_type,
                ),
                "is_active": or_(WhitelistStoreORM.is_active, stmt.excluded.is_active),
            },
        )
        await session.execute(stmt)
        await session.commit()

    inserted = [d for d in domains_in_batch if d not in existing_before]
    updated = [d for d in domains_in_batch if d in existing_before]
    return inserted, updated


async def count_local_shops_in_city(city: str, country_code: str) -> int:
    """Count active ``local_shop`` rows whose ``city`` (case-insensitive) matches."""
    try:
        sf = session_factory()
    except RuntimeError:
        return 0
    c = (city or "").strip()
    cc = (country_code or "").strip().upper()
    if cc == "UK":
        cc = "GB"
    if not c or not cc:
        return 0
    async with sf() as session:
        n = await session.scalar(
            select(func.count())
            .select_from(WhitelistStoreORM)
            .where(WhitelistStoreORM.is_active.is_(True))
            .where(WhitelistStoreORM.store_type == "local_shop")
            .where(WhitelistStoreORM.country_code == cc)
            .where(func.lower(WhitelistStoreORM.city) == c.lower())
        )
    return int(n or 0)
