"""Tavily batched search + optional city/country-tier parallel fanout."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import get_settings
from app.core.db.cache import (
    build_tavily_tier_cache_key,
    get_cached_tavily_tier_payload,
)
from app.domains.search_pipeline.models.search_query import SearchResult
from app.domains.search_pipeline.tier_runner.context import TierContext, TierStoreSelection
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.policies.geo_scope import Tier
from app.domains.engine.search import (
    normalize_url,
    run_local_site_searches,
    run_tavily_for_store_domains,
)
from app.domains.engine.validators.listings import normalize_whitelist_domain

logger = logging.getLogger(__name__)

_FANOUT_TIERS: frozenset[Tier] = frozenset({"city", "country"})


def merge_fanout_into_raw(
    raw_results: list[SearchResult],
    fanout_results: list[SearchResult],
) -> list[SearchResult]:
    """URL-dedup merge; higher Tavily score wins."""
    existing_by_url: dict[str, SearchResult] = {
        normalize_url(r.url): r for r in raw_results
    }
    for r in fanout_results:
        k = normalize_url(r.url)
        prev = existing_by_url.get(k)
        if prev is None or r.score > prev.score:
            existing_by_url[k] = r
    return list(existing_by_url.values())


def _tier_has_local_shop_fanout(capped: tuple[StoreEntry, ...]) -> bool:
    return any(s.domain and s.store_type == "local_shop" for s in capped)


def _local_domains_for_fanout(capped: tuple[StoreEntry, ...]) -> list[str]:
    domains = [
        normalize_whitelist_domain(s.domain)
        for s in capped
        if s.domain and s.store_type == "local_shop"
    ]
    return [d for d in domains if d]


async def run_tavily_stage(
    ctx: TierContext,
    tier: Tier,
    domains_for_tavily: list[str],
) -> tuple[list[SearchResult], bool, str]:
    """Batched Tavily call (with per-tier cache) — wraps the ``tavily`` stage."""
    tkey = build_tavily_tier_cache_key(
        artist=ctx.parsed.artist,
        album_title=ctx.album_title or "",
        tier=tier,
        core_query=ctx.core_query,
        include_domains=domains_for_tavily,
    )
    cached_raw = await get_cached_tavily_tier_payload(tkey)
    cache_hit = cached_raw is not None

    logger.info(
        "tavily_include_domains",
        extra={
            "stage": "tavily",
            "tier": tier,
            "core_query": ctx.core_query,
            "include_domains_count": len(domains_for_tavily),
            "include_domains": domains_for_tavily,
            "cache_hit": cache_hit,
        },
    )

    with stage_timer("tavily") as rec:
        rec.input = {
            "tier": tier,
            "include_domains": domains_for_tavily,
            "cache_hit": cache_hit,
        }
        if cache_hit:
            raw_results = [SearchResult.model_validate(x) for x in (cached_raw or [])]
        else:
            raw_results, _ = await run_tavily_for_store_domains(
                ctx.core_query,
                domains_for_tavily,
                tier=tier,
                relaxation_queries=ctx.tavily_relaxation_queries,
            )

    return raw_results, cache_hit, tkey


async def tier_local_fanout_fetch_only(
    ctx: TierContext,
    tier: Tier,
    capped: tuple[StoreEntry, ...],
    *,
    skip_entry_stagger: bool = False,
) -> tuple[list[SearchResult], int, int]:
    """City/country-tier local_shop power queries (no merge into batched SERP)."""
    if tier not in _FANOUT_TIERS:
        return [], 0, 0

    local_domains_for_fanout = _local_domains_for_fanout(capped)
    if not local_domains_for_fanout:
        return [], 0, 0

    logger.info(
        "tavily_local_fanout_start",
        extra={
            "stage": "tavily",
            "tier": tier,
            "core_query": ctx.core_query,
            "local_domains_count": len(local_domains_for_fanout),
            "local_domains": local_domains_for_fanout,
        },
    )

    with stage_timer("tavily_local_fanout") as rec_fan:
        fanout_results, n_calls = await run_local_site_searches(
            local_domains=local_domains_for_fanout,
            tier=tier,
            artist=ctx.parsed.artist,
            album_title=ctx.album_title or "",
            fallback_country_iso=ctx.norm.resolved_country,
            skip_entry_stagger=skip_entry_stagger,
        )
        rec_fan.input = {
            "tier": tier,
            "local_domains": local_domains_for_fanout,
        }
        rec_fan.output = {
            "http_calls": n_calls,
            "kept": len(fanout_results),
        }
        rec_fan.status = "success" if fanout_results else "empty"

    return fanout_results, n_calls, len(fanout_results)


async def _staggered_tier_local_fanout(
    ctx: TierContext,
    tier: Tier,
    capped: tuple[StoreEntry, ...],
) -> tuple[list[SearchResult], int, int]:
    """Micro-delay fanout so batched Tavily can land before the next burst."""
    settings = get_settings()
    stagger = float(settings.tavily_fanout_stagger_seconds)
    if stagger > 0:
        logger.debug(
            "tavily_fanout_stagger",
            extra={"stage": "tavily", "tier": tier, "seconds": stagger},
        )
        await asyncio.sleep(stagger)
    return await tier_local_fanout_fetch_only(
        ctx,
        tier,
        capped,
        skip_entry_stagger=True,
    )


def _unpack_gather_tavily(
    result: tuple[list[SearchResult], bool, str] | Exception,
) -> tuple[list[SearchResult], bool, str]:
    if isinstance(result, Exception):
        logger.error(
            "tavily_batched_branch_failed",
            exc_info=result,
            extra={"stage": "tavily"},
        )
        return [], False, ""
    return result


def _unpack_gather_fanout(
    result: tuple[list[SearchResult], int, int] | Exception,
) -> tuple[list[SearchResult], int, int]:
    if isinstance(result, Exception):
        logger.error(
            "tavily_local_fanout_branch_failed",
            exc_info=result,
            extra={"stage": "tavily_local_fanout"},
        )
        return [], 0, 0
    return result


async def run_tier_local_fanout_sequential(
    ctx: TierContext,
    tier: Tier,
    capped: tuple[StoreEntry, ...],
    raw_results: list[SearchResult],
) -> tuple[list[SearchResult], int, int]:
    """Sequential merge path when parallel gather is not used."""
    fanout_results, n_calls, n_kept = await tier_local_fanout_fetch_only(ctx, tier, capped)
    merged = merge_fanout_into_raw(raw_results, fanout_results)
    return merged, n_calls, n_kept


async def run_tavily_and_fanout_merged(
    ctx: TierContext,
    tier: Tier,
    selection: TierStoreSelection,
) -> tuple[list[SearchResult], bool, str, int, int]:
    """Tavily batched + local fanout concurrently when applicable; returns merged SERPs."""
    domains_for_tavily = selection.domains_for_tavily
    use_parallel = tier in _FANOUT_TIERS and _tier_has_local_shop_fanout(selection.capped)

    if use_parallel:
        t_task = asyncio.create_task(
            run_tavily_stage(ctx, tier, domains_for_tavily),
        )
        f_task = asyncio.create_task(
            _staggered_tier_local_fanout(ctx, tier, selection.capped),
        )
        raw_tup, fan_tup = await asyncio.gather(t_task, f_task, return_exceptions=True)
        raw_results, cache_hit, tkey = _unpack_gather_tavily(raw_tup)
        fanout_results, local_site_count, local_site_kept = _unpack_gather_fanout(fan_tup)
        merged = merge_fanout_into_raw(raw_results, fanout_results)
        return merged, cache_hit, tkey, local_site_count, local_site_kept

    raw_results, cache_hit, tkey = await run_tavily_stage(ctx, tier, domains_for_tavily)
    merged, local_site_count, local_site_kept = await run_tier_local_fanout_sequential(
        ctx, tier, selection.capped, raw_results
    )
    return merged, cache_hit, tkey, local_site_count, local_site_kept
