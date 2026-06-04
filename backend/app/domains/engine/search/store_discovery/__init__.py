"""Dynamic indie record-store discovery (Tavily + LLM → DB upsert).

When the curated whitelist has too few ``local_shop`` rows for a resolved city,
:func:`discover_new_stores` widens coverage on demand:

    1. Tavily probes (no ``include_domains``) — multiple shop-shaped queries
       broaden recall in cities where the English "best of" listicles dominate
       the SERP. Each request passes the structured ``country`` field so Tavily
       prioritises geographically-matching results.
    2. LLM (gpt-4o-mini, JSON-only, temperature 0) verifies each candidate as a
       *physical* indie record shop OR a small local vinyl mailorder, extracts
       canonical domain + display name.
    3. Upsert into ``whitelist_stores``: insert new rows with ``store_type='local_shop'``;
       for existing rows only *back-fill* nullable fields (``city``, ``store_type``,
       ``country_code``, …). ``priority`` is NEVER overwritten on existing rows.

:func:`discover_stores_from_snippets` exposes the same LLM verification + upsert
path against arbitrary externally-sourced snippets so the pipeline can also
run **opportunistic discovery** on the main consolidated Tavily call's results
(see :mod:`app.domains.search_pipeline.vinyl_search`).

Implementation modules live under this package (``probe``, ``llm_verify``,
``persistence``, ``coordinator``).

No fallback to web crawling. Deterministic JSON contract. Logs every reject reason.
"""

from app.domains.engine.search.store_discovery.coordinator import (
    discover_new_stores,
    discover_stores_from_snippets,
)
from app.domains.engine.search.store_discovery.models import (
    DiscoveredStoreCandidate,
    DiscoveryReport,
)
from app.domains.engine.search.store_discovery.persistence import (
    count_local_shops_in_city,
    save_discovered_stores,
)

__all__ = [
    "DiscoveredStoreCandidate",
    "DiscoveryReport",
    "count_local_shops_in_city",
    "discover_new_stores",
    "discover_stores_from_snippets",
    "save_discovered_stores",
]
