"""Public discovery entry points and shared verify-and-upsert flow."""

from __future__ import annotations

import logging

from app.core.config import get_settings
from app.core.db.database import is_database_configured
from app.domains.engine.search.store_discovery.llm_verify import llm_extract_candidates
from app.domains.engine.search.store_discovery.models import (
    TAVILY_MAX_RESULTS,
    DiscoveryReport,
)
from app.domains.engine.search.store_discovery.persistence import save_discovered_stores
from app.domains.engine.search.store_discovery.probe import tavily_probe

logger = logging.getLogger(__name__)


def normalize_city_country(
    city: str | None, country_code: str | None
) -> tuple[str, str]:
    """Trim + ISO-2 normalize the (city, country_code) inputs. ``UK`` → ``GB``."""
    city_clean = (city or "").strip()
    cc_clean = (country_code or "").strip().upper()
    if cc_clean == "UK":
        cc_clean = "GB"
    return city_clean, cc_clean


async def verify_and_upsert_snippets(
    *,
    city: str,
    country_code: str,
    snippets: list[dict[str, str]],
    report: DiscoveryReport,
) -> None:
    """Run LLM verification on ``snippets`` and upsert verified rows into DB.

    Mutates ``report`` in place — the caller decides whether the inputs came
    from a dedicated probe (:func:`tavily_probe`) or from an external Tavily
    call (e.g. the main consolidated search in
    :mod:`app.domains.search_pipeline.vinyl_search`).
    """
    candidates = await llm_extract_candidates(
        city=city,
        country_code=country_code,
        snippets=snippets,
    )
    report.candidates = len(candidates)
    if not candidates:
        report.error = "llm_no_verified_stores"
        return

    inserted, updated = await save_discovered_stores(candidates)
    report.inserted = len(inserted)
    report.updated = len(updated)
    report.rejected = report.candidates - report.inserted - report.updated
    report.domains_inserted = inserted
    report.domains_updated = updated


async def discover_new_stores(city: str, country_code: str) -> DiscoveryReport:
    """End-to-end discovery: Tavily probe → LLM → DB upsert.

    No-op when the required env keys / DB session are missing.
    """
    report = DiscoveryReport()
    settings = get_settings()

    city_clean, cc_clean = normalize_city_country(city, country_code)
    if not city_clean or not cc_clean or len(cc_clean) != 2:
        report.error = "missing_city_or_country_code"
        return report
    if not settings.tavily_api_key or not settings.openai_api_key:
        report.error = "missing_api_keys"
        return report
    if not is_database_configured():
        report.error = "no_database_url"
        return report

    logger.info(
        "store_discovery_start",
        extra={"stage": "store_discovery", "city": city_clean, "country_code": cc_clean},
    )

    snippets = await tavily_probe(city_clean, cc_clean)
    if not snippets:
        report.error = "tavily_no_results"
        return report

    await verify_and_upsert_snippets(
        city=city_clean,
        country_code=cc_clean,
        snippets=snippets,
        report=report,
    )

    logger.info(
        "store_discovery_done",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "inserted": report.inserted,
            "updated": report.updated,
            "rejected": report.rejected,
            "candidates": report.candidates,
        },
    )
    return report


async def discover_stores_from_snippets(
    *,
    city: str | None,
    country_code: str | None,
    snippets: list[dict[str, str]],
) -> DiscoveryReport:
    """Run LLM verification + DB upsert against arbitrary external snippets.

    Companion to :func:`discover_new_stores`: instead of firing its own Tavily
    probe, this entry-point lets the pipeline reuse snippets from the main
    consolidated Tavily call. That captures cases where Tavily already surfaced
    a real local shop URL (``rockers.de``, ``van-records.com``, …) in the
    answer to the user's artist/album query, but the prefilter would otherwise
    drop it as an unknown host without a PDP-shaped URL.

    No-op when env keys / DB are missing, or the snippets list is empty.
    Returns a populated :class:`DiscoveryReport`. The production pipeline
    schedules this call in the background; upserted domains affect *future*
    searches via ``whitelist_stores``.
    """
    report = DiscoveryReport()
    settings = get_settings()

    city_clean, cc_clean = normalize_city_country(city, country_code)
    if not city_clean or not cc_clean or len(cc_clean) != 2:
        report.error = "missing_city_or_country_code"
        return report
    if not settings.openai_api_key:
        report.error = "missing_api_keys"
        return report
    if not is_database_configured():
        report.error = "no_database_url"
        return report
    if not snippets:
        report.error = "no_snippets"
        return report

    # Bound the LLM payload so a 20-row main Tavily call doesn't push the
    # discovery call into the high-token tier. We already deduplicate by URL
    # downstream when merging insert/update domain lists.
    bounded = snippets[: 2 * TAVILY_MAX_RESULTS]

    logger.info(
        "store_discovery_from_snippets_start",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "snippet_count": len(bounded),
        },
    )

    await verify_and_upsert_snippets(
        city=city_clean,
        country_code=cc_clean,
        snippets=bounded,
        report=report,
    )

    logger.info(
        "store_discovery_from_snippets_done",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "inserted": report.inserted,
            "updated": report.updated,
            "rejected": report.rejected,
            "candidates": report.candidates,
            "error": report.error,
        },
    )
    return report
