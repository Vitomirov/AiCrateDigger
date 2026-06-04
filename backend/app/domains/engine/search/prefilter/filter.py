"""Main Tavily pre-filter orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from app.domains.engine.search.prefilter.hosts import (
    host_in_whitelist,
    is_blacklisted,
    registrable_host,
)
from app.domains.engine.search.prefilter.signals import looks_like_product_url, result_score

logger = logging.getLogger(__name__)


def prefilter_tavily_results(
    raw_results: list[dict[str, Any]],
    *,
    max_candidates: int = 10,
    max_per_host: int = 2,
    known_shop_hosts: Iterable[str] | None = None,
    trusted_local_shop_hosts: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sanitize ``raw_results`` into an LLM-ready candidate list.

    Args:
        raw_results: Tavily result dicts (``url`` / ``title`` / ``content`` / ``score``).
        max_candidates: hard cap on returned candidate count.
        max_per_host: keep at most this many deep links per host (variety).
        known_shop_hosts: hostnames present in the ``whitelist_stores`` table.
            Whitelist hosts always pass the noise gate and get a score boost;
            non-whitelisted hosts must show a PDP-shaped URL to survive.
        trusted_local_shop_hosts: city-matched ``local_shop`` domains from the
            active store pool (including freshly discovered rows). Unioned with
            ``known_shop_hosts`` for whitelist bypass and relaxed thin-path URLs.

    Returns ``(kept_candidates, diagnostic_dict)``.
    """
    whitelist: frozenset[str] = frozenset(
        h.strip().lower().removeprefix("www.")
        for h in (
            *(known_shop_hosts or []),
            *(trusted_local_shop_hosts or []),
        )
        if (h or "").strip()
    )

    diagnostic: dict[str, Any] = {
        "raw_count": len(raw_results),
        "whitelist_size": len(whitelist),
        "missing_url": 0,
        "blacklisted_hosts": 0,
        "rejected_no_pdp_signal": 0,
        "per_host_capped": 0,
        "kept_count": 0,
        "kept_unique_hosts": 0,
        "kept_known_shop": 0,
        "kept_top_hosts": [],
    }

    if not raw_results:
        return [], diagnostic

    blacklist_hosts_seen: set[str] = set()
    rejected_unknown_hosts: set[str] = set()
    annotated: list[tuple[float, str, bool, dict[str, Any]]] = []

    for row in raw_results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            diagnostic["missing_url"] += 1
            continue
        host = registrable_host(url)
        if host is None:
            diagnostic["missing_url"] += 1
            continue
        if is_blacklisted(host):
            diagnostic["blacklisted_hosts"] += 1
            blacklist_hosts_seen.add(host)
            continue

        is_known_shop = host_in_whitelist(host, whitelist)
        if not is_known_shop and not looks_like_product_url(url):
            diagnostic["rejected_no_pdp_signal"] += 1
            rejected_unknown_hosts.add(host)
            continue

        score = result_score(row, is_known_shop=is_known_shop)
        annotated.append((score, host, is_known_shop, row))

    annotated.sort(key=lambda x: x[0], reverse=True)

    per_host_counts: dict[str, int] = {}
    capped: list[dict[str, Any]] = []
    for score, host, is_known_shop, row in annotated:
        if len(capped) >= max_candidates:
            break
        cnt = per_host_counts.get(host, 0)
        if cnt >= max_per_host:
            diagnostic["per_host_capped"] += 1
            continue
        per_host_counts[host] = cnt + 1
        candidate = {
            "url": str(row.get("url") or "").strip(),
            "title": str(row.get("title") or "").strip(),
            "content": str(row.get("content") or "").strip(),
            "score": float(score),
            "host": host,
            "is_known_shop": is_known_shop,
        }
        capped.append(candidate)

    diagnostic["kept_count"] = len(capped)
    diagnostic["kept_unique_hosts"] = len({c["host"] for c in capped})
    diagnostic["kept_known_shop"] = sum(1 for c in capped if c["is_known_shop"])
    diagnostic["kept_top_hosts"] = [c["host"] for c in capped][:10]
    if blacklist_hosts_seen:
        diagnostic["blacklisted_sample"] = sorted(blacklist_hosts_seen)[:10]
    if rejected_unknown_hosts:
        diagnostic["rejected_no_pdp_sample"] = sorted(rejected_unknown_hosts)[:10]

    logger.info(
        "tavily_prefilter",
        extra={"stage": "tavily_prefilter", **diagnostic},
    )

    return capped, diagnostic
