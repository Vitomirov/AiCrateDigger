"""Tunable constants for the deterministic vinyl-search pipeline.

Centralised here so other pipeline sub-modules can import the same numbers
without circular references through :mod:`app.pipeline.vinyl_search`.
"""

from __future__ import annotations

#: Minimum unique store domains required to actually fire Tavily for a tier.
#: When the per-tier pool falls below this floor the tier is skipped with a
#: structured ``geo_tier`` empty trace.
MIN_STORE_DOMAINS_DEFAULT: int = 2

#: City tier accepts a single curated indie shop on its own — one local
#: brick-and-mortar is still meaningful signal worth Tavily-ing.
MIN_STORE_DOMAINS_CITY: int = 1

#: Result capping (Local-First Strike): when at least one ``local_shop`` row
#: validated in the aggregated pool, hold non-indie store types to these caps
#: in the final response so a single mega-retailer cannot crowd out indies.
MAX_REGIONAL_WHEN_LOCAL_PRESENT: int = 1
MAX_MARKETPLACE_WHEN_LOCAL_PRESENT: int = 0

#: Stricter locality-first cap: when prioritised target-city indie locals
#: validated, regional giants are withheld entirely unless no such locals
#: exist upstream (the fallback path still allows them via the broader
#: pool-level ``MAX_REGIONAL_WHEN_LOCAL_PRESENT`` heuristic).
MAX_REGIONAL_WHEN_PRIORITIZED_PRIMARY_LOCAL_PRESENT: int = 0
