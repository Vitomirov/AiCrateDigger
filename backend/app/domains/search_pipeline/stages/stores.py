"""Load store catalogue and derive prefilter host sets."""

from __future__ import annotations

import logging

from app.core.db.store_loader import load_active_stores
from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.policies.store_domain import canonical_store_domain
from app.domains.query_parser.parse_schema import ParsedQuery

logger = logging.getLogger(__name__)


async def load_active_stores_catalogue() -> tuple[StoreEntry, ...]:
    """Single DB read of the active whitelist for one pipeline request.

    Robustness contract: an unexpected failure must NEVER leave the pipeline
    without a usable catalogue — we degrade to the in-code seed
    (:func:`get_active_stores`) so curated indies are not demoted to unknown hosts.
    """
    try:
        return await load_active_stores()
    except Exception:
        logger.exception(
            "load_active_stores_failed_falling_back_to_code_seed",
            extra={"stage": "stores"},
        )
        from app.domains.engine.policies.eu_stores import get_active_stores

        return get_active_stores()


def known_shop_hosts_from_stores(stores: tuple[StoreEntry, ...]) -> frozenset[str]:
    """All active hosts — positive prefilter signal for curated + discovered shops."""
    hosts: set[str] = set()
    for s in stores:
        dom = canonical_store_domain(getattr(s, "domain", "") or "")
        if dom:
            hosts.add(dom)
    return frozenset(hosts)


def trusted_local_shop_hosts_from_stores(
    stores: tuple[StoreEntry, ...],
    parsed: ParsedQuery,
) -> frozenset[str]:
    """City-matched ``local_shop`` domains from an already-loaded catalogue."""
    city = (parsed.resolved_city or "").strip()
    cc = (parsed.country_code or "").strip().upper()
    if not cc:
        return frozenset()
    from app.domains.engine.policies.geo_proximity import cities_match

    hosts: set[str] = set()
    for s in stores:
        if getattr(s, "store_type", None) != "local_shop" or not s.domain:
            continue
        if (s.country_code or "").strip().upper() != cc:
            continue
        if city:
            store_city = (getattr(s, "city", None) or "").strip()
            if store_city and not cities_match(city, store_city):
                continue
        dom = canonical_store_domain(s.domain)
        if dom:
            hosts.add(dom)
    return frozenset(hosts)


async def load_pipeline_shop_hosts(
    parsed: ParsedQuery,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(known_shop_hosts, trusted_local_shop_hosts)`` from one catalogue load."""
    stores = await load_active_stores_catalogue()
    return (
        known_shop_hosts_from_stores(stores),
        trusted_local_shop_hosts_from_stores(stores, parsed),
    )
