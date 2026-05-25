"""Canonical hostname for store whitelist / Tavily ``include_domains``.

Never pass URL paths or schemes — only registrable host, lowercased, no ``www.``.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Persisted typo / drift (LLM, manual SQL) → registrable host that resolves in DNS /
# Tavily ``include_domains``. Not a store catalogue; see :func:`repair_whitelist_store_domains`.
_STORE_DOMAIN_ALIASES: dict[str, str] = {
    "niche-records.ro": "nicherecords.ro",
}

# Strings the discovery LLM sometimes emits in place of a real domain when the
# Tavily snippet does not name one explicitly. Without this guard they pass
# :func:`canonical_store_domain` untouched and end up in Tavily ``include_domains``,
# poisoning the request and producing zero hits. Lower-case, trimmed comparison.
_PLACEHOLDER_HOSTS: frozenset[str] = frozenset(
    {
        "none",
        "null",
        "nil",
        "na",
        "n/a",
        "n.a.",
        "unknown",
        "tbd",
        "tba",
        "not provided",
        "notprovided",
        "not_provided",
        "not-provided",
        "not specified",
        "notspecified",
        "not-specified",
        "not_specified",
        "no domain",
        "no-domain",
        "no_domain",
        "missing",
        "example.com",
        "example.org",
        "domain.com",
    }
)

# Minimal hostname syntax — at least one dot, no whitespace, only DNS-legal chars
# (RFC 1035 lite: letters/digits/hyphen/dot). Reject e.g. ``foo bar``, ``a.b``
# (too short), and Unicode-only labels.
_VALID_DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$")
_MIN_TLD_LEN = 2
_MIN_TOTAL_LEN = 4


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


def is_valid_store_host(raw: str | None) -> bool:
    """True iff ``raw`` looks like a real registrable hostname Tavily can use.

    Rejects placeholder strings the discovery LLM occasionally emits (``none``,
    ``unknown``, ``not provided``…), single-label hosts (no dot), and anything
    with non-DNS characters. Used as a defence-in-depth filter at every layer
    that hands domains to Tavily ``include_domains`` so a single bad row in
    ``whitelist_stores`` can never poison the search request.
    """
    host = canonical_store_domain(raw)
    if not host:
        return False
    if host in _PLACEHOLDER_HOSTS:
        return False
    if len(host) < _MIN_TOTAL_LEN:
        return False
    if not _VALID_DOMAIN_RE.match(host):
        return False
    tld = host.rsplit(".", 1)[-1]
    return len(tld) >= _MIN_TLD_LEN and tld.isalpha()
