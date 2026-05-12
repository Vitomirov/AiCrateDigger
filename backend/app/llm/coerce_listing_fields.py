"""Deterministic coercion of raw LLM listing fields (no I/O, import-light)."""

from __future__ import annotations

from typing import Any


def coerce_in_stock(item: dict[str, Any]) -> bool:
    """Normalize extractor ``in_stock`` for :class:`app.domain.listing_schema.Listing`.

    Explicit booleans, stock phrases, and ``0`` / ``1`` are mapped; ``null``,
    missing, or malformed values default to ``True`` (available / unknown).
    """
    v = item.get("in_stock")
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return True
        if any(
            x in s
            for x in (
                "out of stock",
                "sold out",
                "unavailable",
                "nicht verfügbar",
                "nicht verfuegbar",
                "indisponible",
                "épuisé",
                "rupture",
            )
        ):
            return False
        if any(
            x in s
            for x in (
                "in stock",
                "available",
                "add to cart",
                "en stock",
                "disponible",
            )
        ):
            return True
        return True
    if isinstance(v, (int, float)):
        return v != 0
    return True
