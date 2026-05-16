"""Result type for the extract-listings pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.listing_schema import Listing


@dataclass
class ExtractListingsReport:
    listings: list[Listing]
    diagnostic: dict[str, Any] = field(default_factory=dict)
