"""Canonical hostname for store whitelist / Tavily ``include_domains``.

Never pass URL paths or schemes — only registrable host, lowercased, no ``www.``.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Persisted typo / drift (LLM, manual SQL) → registrable host that resolves in DNS /
# Tavily ``include_domains``. Not a store catalogue; see :func:`repair_whitelist_store_domains`.
_STORE_DOMAIN_ALIASES: dict[str, str] = {
    "niche-records.ro": "nicherecords.ro",
}


def registrable_host_only(raw: str | None) -> str:
    """Lowercase host, no scheme/path/port, no ``www.`` — **before** typo aliases."""
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


def canonical_store_domain(raw: str | None) -> str:
    """Return pure hostname for ``include_domains`` and whitelist keys.

    Accepts ``mascom.rs``, ``https://www.mascom.rs/shop/...``, ``mascom.rs/shop``.
    """
    base = registrable_host_only(raw)
    if not base:
        return ""
    return _STORE_DOMAIN_ALIASES.get(base, base)
