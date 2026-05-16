"""Coerce LLM string outputs into ListingResult-compatible forms."""

from __future__ import annotations


def clean_optional_string(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = " ".join(str(raw).split()).strip(" ,;:-")
    return s or None


def normalize_availability(value: object) -> str:
    v = str(value or "").lower().strip()
    return v if v in {"available", "sold_out", "unknown"} else "unknown"


def normalize_seller(value: object) -> str:
    v = str(value or "").lower().strip()
    return v if v in {"store", "private", "unknown"} else "unknown"
