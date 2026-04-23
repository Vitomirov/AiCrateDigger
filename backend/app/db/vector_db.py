"""DEPRECATED — kept as a thin compatibility shim.

The emergent marketplace RAG lives in `app.db.marketplace_db` now. This module
is retained only to avoid import-time breakage if any external code (e.g.
notebooks, old tests, or stale caches) still reaches for `VerifiedStore` /
`add_verified_store`. All writes here are re-routed to
`marketplace_db.record_store_confirmation` with a deprecation warning so the
signal still lands in the right place.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from app.db.marketplace_db import (
    StoreConfirmation,
    get_marketplace_db,
    normalize_domain,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VerifiedStore",
    "VectorDBService",
    "get_vector_db_service",
    "normalize_domain",
]


@dataclass(slots=True)
class VerifiedStore:
    """Legacy dataclass — mirrored onto `StoreConfirmation` on write."""

    title: str
    url: str
    location: str
    domain: str = ""


class VectorDBService:
    """Compat shim. DO NOT extend. Use `marketplace_db` directly."""

    def __init__(self, *_, **__) -> None:
        warnings.warn(
            "VectorDBService is deprecated; use app.db.marketplace_db directly.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def add_verified_store(self, *, store: VerifiedStore) -> None:
        logger.warning(
            "vector_db_shim_add_verified_store_called",
            extra={"stage": "rag_store", "status": "deprecated", "domain": store.domain},
        )
        confirmation = StoreConfirmation(
            url=store.url,
            title=store.title,
            location=store.location,
        )
        await get_marketplace_db().record_store_confirmation(confirmation)


def get_vector_db_service() -> VectorDBService:
    warnings.warn(
        "get_vector_db_service() is deprecated; use app.db.marketplace_db.get_marketplace_db().",
        DeprecationWarning,
        stacklevel=2,
    )
    return VectorDBService()
