"""Hostname normalization for listing URLs."""

from __future__ import annotations

from urllib.parse import urlsplit


def normalize_domain(url_or_domain: str) -> str | None:
    """Return a stable lowercase base domain (no `www.`, no port)."""
    if not url_or_domain:
        return None
    try:
        candidate = url_or_domain.strip()
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        netloc = urlsplit(candidate).netloc.lower().strip()
        if not netloc:
            return None
        if netloc.startswith("www."):
            netloc = netloc[4:]
        netloc = netloc.split(":", maxsplit=1)[0]
        return netloc or None
    except Exception:
        return None
