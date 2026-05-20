"""City/country-tier local shop power queries and optional open-web fallback."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import get_settings
from app.models.search_query import SearchResult
from app.services.tavily.aggregation import apply_merge_floor_and_collect, top_domains
from app.services.tavily.constants import REQUEST_TIMEOUT_SECONDS
from app.services.tavily.country_boost import tavily_country_from_iso3166_alpha2
from app.services.tavily.filtering import enforce_include_domains_hosts, is_valid_result
from app.services.tavily.power_query import (
    build_physical_power_query_base,
    chunk_domains_for_power_queries,
)
from app.services.tavily.scoring import whitelist_include_domains_threshold
from app.services.tavily.search import search_single_query
from app.services.tavily.url_utils import dedupe_domains

logger = logging.getLogger(__name__)


async def _fanout_stagger_pause(*, reason: str) -> None:
    """Brief pause to avoid Tavily 432 bursts when fanout follows other calls."""
    settings = get_settings()
    stagger = float(settings.tavily_fanout_stagger_seconds)
    if stagger <= 0:
        return
    logger.debug(
        "tavily_fanout_stagger",
        extra={"stage": "tavily", "reason": reason, "seconds": stagger},
    )
    await asyncio.sleep(stagger)


async def run_local_site_searches(
    *,
    local_domains: list[str],
    tier: str | None = None,
    artist: str | None = None,
    album_title: str | None = None,
    fallback_country_iso: str | None = None,
    skip_entry_stagger: bool = False,
) -> tuple[list[SearchResult], int]:
    """Low-credit local shop coverage using consolidated power queries.

    Step 1 — one Tavily POST per domain chunk with ``site:`` + ``include_domains``.
    Step 2 — only when Step 1 returns no kept rows: open-web call with country boost.
    """
    if not local_domains:
        return [], 0

    if not skip_entry_stagger:
        await _fanout_stagger_pause(reason="local_fanout_entry")

    settings = get_settings()
    domains = dedupe_domains(list(local_domains))
    if not domains:
        return [], 0

    power_base = build_physical_power_query_base(
        artist=artist,
        album_title=album_title or "",
    )
    if not (power_base or "").strip():
        return [], 0

    max_results = settings.tavily_max_results_per_batch
    min_score_base = float(settings.tavily_min_result_score)
    merge_floor = whitelist_include_domains_threshold(min_score_base)

    planner = chunk_domains_for_power_queries(
        power_base,
        domains,
        max_chars=int(settings.tavily_power_query_max_chars),
        max_domains_per_chunk=int(settings.tavily_local_power_max_domains_per_chunk),
    )
    logger.info(
        "tavily_local_power_queries",
        extra={
            "stage": "tavily",
            "tier": tier,
            "domain_total": len(domains),
            "chunks": len(planner),
            "power_query_base": power_base[:180],
        },
    )

    http_calls = 0
    aggregated: dict[str, SearchResult] = {}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        if planner:

            async def _one_chunk(dom_chunk: list[str], q_text: str) -> list[SearchResult]:
                logger.info(
                    "tavily_local_site_search",
                    extra={
                        "stage": "tavily",
                        "tier": tier,
                        "domains_in_chunk": len(dom_chunk),
                        "query_head": q_text[:200],
                    },
                )
                return await search_single_query(
                    client,
                    q_text,
                    include_domains=dom_chunk,
                    max_results=max_results,
                    fanout_local_shop=False,
                )

            chunk_lists = await asyncio.gather(*[_one_chunk(dc, qt) for dc, qt in planner])
            chunk_lists = [
                [r for r in bl if is_valid_result(r.url)] for bl in chunk_lists
            ]
            http_calls += len(planner)

            merged_round = apply_merge_floor_and_collect(
                chunk_lists,
                merge_floor=merge_floor,
            )
            for uk, sr in merged_round.items():
                prev = aggregated.get(uk)
                if prev is None or sr.score > prev.score:
                    aggregated[uk] = sr

        merged_primary = sorted(aggregated.values(), key=lambda x: x.score, reverse=True)
        merged_primary = enforce_include_domains_hosts(merged_primary, domains)

        if merged_primary:
            final = merged_primary
        else:
            await _fanout_stagger_pause(reason="local_fanout_fallback")
            tav_country = tavily_country_from_iso3166_alpha2(fallback_country_iso)
            fb = await search_single_query(
                client,
                power_base.strip(),
                include_domains=None,
                max_results=max_results,
                fanout_local_shop=False,
                country=tav_country,
            )
            fb = [r for r in fb if is_valid_result(r.url)]
            http_calls += 1
            uniq_fb = apply_merge_floor_and_collect([fb], merge_floor=float(min_score_base))
            aggregated = dict(uniq_fb)
            final = sorted(aggregated.values(), key=lambda x: x.score, reverse=True)

    logger.info(
        "tavily_local_aggregate",
        extra={
            "stage": "tavily",
            "tier": tier,
            "domains_requested": len(domains),
            "http_calls": http_calls,
            "kept": len(final),
            "top_domains": top_domains(final, limit=5),
        },
    )
    return final, http_calls
