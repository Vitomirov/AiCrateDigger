"""URL / host normalization and allowlist matching."""

from __future__ import annotations

from urllib.parse import urlsplit


def normalize_domain(url_or_domain: str) -> str | None:
    if not url_or_domain:
        return None
    candidate = url_or_domain.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        netloc = urlsplit(candidate).netloc.lower().strip()
    except ValueError:
        return None
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":", 1)[0]


def normalize_allowed_domains(domains: set[str]) -> set[str]:
    out: set[str] = set()
    for d in domains:
        n = normalize_domain(d)
        if n:
            out.add(n)
    return out


def host_matches_whitelist(host: str, allowed: set[str]) -> bool:
    if not host:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    return h in allowed or any(h.endswith("." + d) for d in allowed)
