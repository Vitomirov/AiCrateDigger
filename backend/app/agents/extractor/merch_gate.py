"""Merch / accessory rejection gate."""

from __future__ import annotations

from app.agents.extractor.constants import REJECTION_KEYWORDS


def listing_looks_like_merch(title: str, content: str) -> bool:
    hay = f"{title} {content}".lower()
    return any(kw in hay for kw in REJECTION_KEYWORDS)
