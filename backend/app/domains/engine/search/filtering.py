"""Post-fetch filtering: forbidden hosts, domain leaks, editorial hubs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from urllib.parse import urlparse

from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.engine.policies.store_domain import canonical_store_domain
from app.domains.engine.search.constants import FORBIDDEN_DOMAINS
from app.domains.engine.search.scoring import product_signal_multiplier
from app.domains.engine.search.url_utils import host_matches_include_domain, normalize_store_domain, normalize_url

logger = logging.getLogger(__name__)


def is_valid_result(url: str) -> bool:
    u = url.lower()
    return not any(domain in u for domain in FORBIDDEN_DOMAINS)


def enforce_include_domains_hosts(
    results: list[SearchResult],
    allowed_domains: Sequence[str],
) -> list[SearchResult]:
    """Drop rows whose registrable host is not under any ``allowed_domains`` entry."""
    if not allowed_domains or not results:
        return results
    normed: list[str] = []
    for d in allowed_domains:
        n = normalize_store_domain(d)
        if n:
            normed.append(n)
    if not normed:
        return results
    kept: list[SearchResult] = []
    leaked = 0
    for r in results:
        try:
            netloc = urlparse(r.url).netloc or ""
        except Exception:
            leaked += 1
            continue
        if any(host_matches_include_domain(netloc, d) for d in normed):
            kept.append(r)
        else:
            leaked += 1
    if leaked:
        logger.info(
            "tavily_include_domains_leak_filtered",
            extra={
                "stage": "tavily",
                "dropped": leaked,
                "kept": len(kept),
                "allowed_count": len(normed),
            },
        )
    return kept


def editorial_discovery_url(url: str) -> bool:
    """URLs that seldom yield deterministic listing rows (blogs, magazine hubs)."""
    try:
        pl = urlparse(url.strip()).path.lower() or "/"
    except Exception:
        pl = "/"
    if product_signal_multiplier(pl) <= 0.0:
        return True
    from app.validators import listings as _lv

    if _lv._url_looks_non_product(url) and not _lv.url_suggests_product_detail_page(url):  # noqa: SLF001
        return True
    return False


def editorial_discovery_blocked_hosts_from_raw_results(
    rows: Sequence[SearchResult],
    *,
    deterministic_failed: bool,
) -> set[str]:
    """Identify hosts that look like SERP hubs when deterministic extraction stalled."""
    out: set[str] = set()
    if not deterministic_failed or not rows:
        return out
    for r in rows:
        if editorial_discovery_url(r.url):
            host = urlparse(normalize_url(r.url)).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            host = canonical_store_domain(host)
            if host:
                out.add(host)
    return out
