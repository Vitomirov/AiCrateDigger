"""Per-tier store-pool helpers (selection, capping, deduping, diagnostics)."""

from __future__ import annotations

from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.validators.listings import normalize_whitelist_domain


def dedupe_store_entries_by_domain(
    capped: tuple[StoreEntry, ...],
) -> tuple[StoreEntry, ...]:
    """Drop duplicate ``StoreEntry`` rows that share a normalised domain.

    Preserves the first occurrence so the upstream sort order survives.
    """
    seen: set[str] = set()
    out: list[StoreEntry] = []
    for s in capped:
        k = normalize_whitelist_domain(s.domain)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return tuple(out)


def store_type_distribution(stores: tuple[StoreEntry, ...]) -> dict[str, int]:
    """Histogram of ``store_type`` values used in trace payloads."""
    out: dict[str, int] = {}
    for s in stores:
        k = s.store_type or "regional_ecommerce"
        out[k] = out.get(k, 0) + 1
    return out
