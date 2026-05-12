"""Canonical hostname for store whitelist / Tavily ``include_domains``.

Never pass URL paths or schemes — only registrable host, lowercased, no ``www.``.
"""

from __future__ import annotations

from urllib.parse import urlparse


def canonical_store_domain(raw: str | None) -> str:
    """Return pure hostname for ``include_domains`` and whitelist keys.

    Accepts ``mascom.rs``, ``https://www.mascom.rs/shop/...``, ``mascom.rs/shop``.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = s.lower()
    if "://" in s:
        try:
            netloc = urlparse(s).netloc
        except Exception:
            netloc = ""
        s = netloc or s.split("://", 1)[-1]
    s = s.split("/")[0].split("?", 1)[0].split("#", 1)[0]
    s = s.split(":", 1)[0]
    if s.startswith("www."):
        s = s[4:]
    return s.strip()
