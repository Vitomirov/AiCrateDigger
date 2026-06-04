"""Primary and opportunistic store-discovery hooks on the search hot path."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import BackgroundTasks

from app.core.config import get_settings
from app.domains.engine.search.prefilter import (
    _host_in_whitelist,
    _is_blacklisted,
    _registrable_host,
)
from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.search_pipeline.stages.background import schedule_background_store_discovery

logger = logging.getLogger(__name__)


async def stage_ensure_local_coverage(
    parsed: ParsedQuery,
    *,
    background_tasks: BackgroundTasks | None,
) -> dict[str, object]:
    """Evaluate indie coverage and schedule discovery when the city is thin.

    Runs only when the parser resolved a specific city + country. Internally
    idempotent: when ``count_local_shops_in_city(city, cc) >= LOCAL_COVERAGE_THRESHOLD``
    no Tavily / OpenAI traffic is generated — it is just a single ``COUNT(*)``
    on ``whitelist_stores``. When coverage is insufficient, discovery is
    scheduled in the background so the search hot path is not blocked.

    Returns the coverage summary dict (empty when inputs are missing).
    """
    from app.core.db.store_loader import ensure_local_coverage

    city = (parsed.resolved_city or "").strip()
    cc = (parsed.country_code or "").strip()
    if not city or not cc:
        return {}
    with stage_timer(
        "store_discovery",
        input={"city": city, "country_code": cc},
    ) as rec:
        try:
            coverage = await ensure_local_coverage(city=city, country_code=cc)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never fail the user's search.
            logger.exception(
                "store_discovery_failed",
                extra={
                    "stage": "store_discovery",
                    "city": city,
                    "country_code": cc,
                },
            )
            rec.status = "fail"
            rec.error = str(exc)[:240]
            return {"triggered": False, "error": str(exc)[:240]}
        if coverage.get("scheduled"):
            from app.domains.engine.search.store_discovery import discover_new_stores

            schedule_background_store_discovery(
                background_tasks,
                lambda: discover_new_stores(city=city, country_code=cc),
                label="primary",
            )
        rec.output = coverage
        if not coverage.get("triggered"):
            rec.status = "empty"
        elif coverage.get("scheduled"):
            rec.status = "success"
        return coverage


def primary_discovery_should_skip_opportunistic(
    primary_discovery_summary: dict[str, object] | None,
) -> bool:
    """True when inline store discovery already ran for this request."""
    if not primary_discovery_summary:
        return False
    if primary_discovery_summary.get("triggered"):
        return True
    disc = primary_discovery_summary.get("discovery")
    if isinstance(disc, dict) and int(disc.get("inserted") or 0) > 0:
        return True
    return False


def tavily_city_token(parsed: ParsedQuery) -> str | None:
    """Resolved city for Tavily query injection when geo is city-level."""
    city = (parsed.resolved_city or "").strip()
    if city:
        return city
    if parsed.geo_granularity == "city":
        return (parsed.location or "").strip() or None
    return None


def select_unknown_host_snippets_for_discovery(
    raw_results: list[dict[str, Any]],
    *,
    known_shop_hosts: frozenset[str],
) -> list[dict[str, str]]:
    """Filter main-Tavily results down to *plausible-shop* unknown-host snippets.

    Excludes hosts that are:
      * blacklisted (YouTube, Discogs, news, social, …),
      * already in the active whitelist (curated + previously discovered).

    Keeps one snippet per unique host — the discovery LLM needs a clear shop
    name signal, not 5 deep links from the same domain.
    """
    chosen: dict[str, dict[str, str]] = {}
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        host = _registrable_host(url)
        if host is None or _is_blacklisted(host):
            continue
        if _host_in_whitelist(host, known_shop_hosts):
            continue
        if host in chosen:
            continue
        chosen[host] = {
            "title": str(row.get("title") or "").strip()[:240],
            "url": url,
            "content": str(row.get("content") or "").strip()[:1500],
        }
    return list(chosen.values())


def schedule_opportunistic_store_discovery(
    *,
    background_tasks: BackgroundTasks | None,
    parsed: ParsedQuery,
    raw_results: list[dict[str, Any]],
    known_shop_hosts: frozenset[str],
    primary_discovery_summary: dict[str, object] | None = None,
) -> None:
    """Schedule LLM verification of unknown-host snippets from the main Tavily call.

    Upserts run in the background so the prefilter / extractor stages are not
    blocked. Verified domains benefit subsequent searches via ``whitelist_stores``.
    """
    settings = get_settings()
    if not settings.pipeline_opportunistic_store_discovery_enabled:
        return

    if primary_discovery_should_skip_opportunistic(primary_discovery_summary):
        city = (parsed.resolved_city or "").strip()
        cc = (parsed.country_code or "").strip()
        logger.info(
            "opportunistic_store_discovery_skipped_primary_discovery_triggered",
            extra={
                "stage": "opportunistic_store_discovery",
                "city": city,
                "country_code": cc,
                "skipped_reason": "skipped_primary_discovery_triggered",
                "primary_triggered": bool(
                    (primary_discovery_summary or {}).get("triggered")
                ),
            },
        )
        with stage_timer(
            "opportunistic_store_discovery",
            input={"skipped_reason": "skipped_primary_discovery_triggered"},
        ) as rec:
            rec.status = "empty"
            rec.output = {"skipped_reason": "skipped_primary_discovery_triggered"}
        return

    city = (parsed.resolved_city or "").strip()
    cc = (parsed.country_code or "").strip()
    if not city or not cc:
        return

    snippets = select_unknown_host_snippets_for_discovery(
        raw_results,
        known_shop_hosts=known_shop_hosts,
    )
    min_required = settings.pipeline_opportunistic_discovery_min_unknown_hosts
    if len(snippets) < min_required:
        logger.info(
            "opportunistic_store_discovery_skipped_thin_signal",
            extra={
                "stage": "store_discovery",
                "city": city,
                "country_code": cc,
                "unknown_host_count": len(snippets),
                "min_required": min_required,
            },
        )
        return

    from app.domains.engine.search.store_discovery import discover_stores_from_snippets

    with stage_timer(
        "opportunistic_store_discovery",
        input={
            "city": city,
            "country_code": cc,
            "unknown_host_count": len(snippets),
            "scheduled": True,
        },
    ) as rec:
        schedule_background_store_discovery(
            background_tasks,
            lambda: discover_stores_from_snippets(
                city=city,
                country_code=cc,
                snippets=snippets,
            ),
            label="opportunistic",
        )
        rec.status = "success"
        rec.output = {"scheduled": True, "unknown_host_count": len(snippets)}
