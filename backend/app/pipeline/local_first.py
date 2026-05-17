"""Local-First Strike result-capping policy.

Once at least one validated ``local_shop`` row exists in the aggregated pool
we stream those rows first and tightly bound how many regional / marketplace
rows are allowed to share the final response with them.
"""

from __future__ import annotations

from typing import Any

from app.policies.eu_stores import StoreEntry
from app.policies.listing_rank import resolve_store_for_url

from app.pipeline.constants import (
    MAX_MARKETPLACE_WHEN_LOCAL_PRESENT,
    MAX_REGIONAL_WHEN_LOCAL_PRESENT,
    MAX_REGIONAL_WHEN_PRIORITIZED_PRIMARY_LOCAL_PRESENT,
)


def apply_local_first_caps(
    sorted_listings: list[Any],
    *,
    store_by_domain: dict[str, StoreEntry],
    generic_local_shop_present_in_pool: bool,
    prioritize_physical_locals: bool,
    primary_target_local_shop_present: bool,
    max_results: int,
) -> list[Any]:
    """Local-First result capping.

    If at least one validated ``local_shop`` row exists in the aggregated pool:
      * ``local_shop`` rows are streamed first until ``max_results``,
      * ``regional_ecommerce`` capped to 1 normally, or **0** when prioritized
        target-city indie locals validated (mega-retailer fallback only once
        indie coverage is exhausted),
      * ``marketplace`` capped to ``MAX_MARKETPLACE_WHEN_LOCAL_PRESENT``.

    Without ``generic_local_shop_present_in_pool`` the helper preserves input
    order (subject only to ``max_results``).
    """
    if max_results <= 0:
        return []
    if not generic_local_shop_present_in_pool:
        return sorted_listings[:max_results]

    regional_cap = MAX_REGIONAL_WHEN_LOCAL_PRESENT
    if prioritize_physical_locals and primary_target_local_shop_present:
        regional_cap = MAX_REGIONAL_WHEN_PRIORITIZED_PRIMARY_LOCAL_PRESENT

    out: list[Any] = []
    seen_regional = 0
    seen_market = 0
    for lst in sorted_listings:
        store = resolve_store_for_url(str(getattr(lst, "url", "") or ""), store_by_domain)
        st = (store.store_type if store is not None else "regional_ecommerce") or "regional_ecommerce"
        if st == "local_shop":
            out.append(lst)
        elif st == "regional_ecommerce":
            if seen_regional < regional_cap:
                out.append(lst)
                seen_regional += 1
        elif st == "marketplace":
            if seen_market < MAX_MARKETPLACE_WHEN_LOCAL_PRESENT:
                out.append(lst)
                seen_market += 1
        else:
            # Unknown / default: treat as regional.
            if seen_regional < regional_cap:
                out.append(lst)
                seen_regional += 1
        if len(out) >= max_results:
            break
    return out
