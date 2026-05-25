"""URL normalization and domain allowlist helpers."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse, urlsplit, urlunsplit

from app.domains.engine.policies.store_domain import canonical_store_domain, is_valid_store_host

logger = logging.getLogger(__name__)

_SITE_TAIL_RE = re.compile(r"\bsite:([^\s]+)\s*$", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Strip query string, fragment, trailing slash. Lowercase host."""
    try:
        stripped = url.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
        parsed = urlsplit(stripped.strip())
        normalized_path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), normalized_path, "", ""))
    except Exception:
        return url


def normalize_store_domain(domain: str) -> str | None:
    d = canonical_store_domain(domain)
    return d or None


def host_matches_include_domain(netloc: str, allowed_domain: str) -> bool:
    """True if URL host is ``allowed_domain`` or a subdomain of it (``www`` stripped)."""
    h = (netloc or "").lower().strip()
    if h.startswith("www."):
        h = h[4:]
    a = (allowed_domain or "").lower().strip()
    if a.startswith("www."):
        a = a[4:]
    if not h or not a:
        return False
    return h == a or h.endswith("." + a)


def host_bucket(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if h.startswith("www."):
        h = h[4:]
    return h


def include_domains_for_query(query: str) -> list[str] | None:
    m = _SITE_TAIL_RE.search(query.strip())
    if not m:
        return None
    dom = m.group(1).strip().lower()
    if dom.startswith("www."):
        dom = dom[4:]
    return [dom] if dom else None


def dedupe_domains(domains: list[str]) -> list[str]:
    """Canonicalise, drop empties and any host that fails :func:`is_valid_store_host`."""
    seen: set[str] = set()
    out: list[str] = []
    skipped_invalid: list[str] = []
    for raw in domains:
        n = normalize_store_domain(raw)
        if n is None or n in seen:
            continue
        if not is_valid_store_host(n):
            skipped_invalid.append(n[:64])
            continue
        seen.add(n)
        out.append(n)
    if skipped_invalid:
        logger.warning(
            "tavily_skipped_invalid_include_domain",
            extra={
                "stage": "tavily",
                "skipped_count": len(skipped_invalid),
                "sample": skipped_invalid[:10],
            },
        )
    return out
