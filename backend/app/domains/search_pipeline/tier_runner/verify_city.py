"""City-tier album-title verification (LLM)."""

from __future__ import annotations

from typing import Any

from app.domains.engine.extraction.verify_album_match import verify_album_match
from app.domains.search_pipeline.tier_runner.context import TierContext, TierLoopState
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.policies.eu_stores import StoreEntry
from app.domains.engine.policies.geo_scope import Tier
from app.domains.engine.policies.listing_rank import resolve_store_for_url
from app.domains.engine.policies.physical_local import qualifies_as_target_city_local_shop
from app.domains.engine.search import normalize_url


async def run_city_verify_stage(
    *,
    ctx: TierContext,
    state: TierLoopState,
    tier: Tier,
    accepted: list[Any],
    tier_lookup: dict[str, StoreEntry],
    rejected_reasons: list[dict[str, str]],
) -> list[Any]:
    """LLM album-title verification on city-tier ``local_shop`` rows.

    City-tier indie rows receive the +500 bonus downstream, so a wrong-album
    title would unfairly dominate. The verifier returns:

    * ``"reject"``    → drop the row entirely from this tier (unless soft-kept
      because it is a prioritised target-city physical-local shop).
    * ``"confirmed"`` → mark ``album_match_by_url[url] = True`` (bonus eligible).
    * ``"unsure"``    → mark ``False`` (no bonus, no drop).
    """
    to_verify: list[Any] = []
    for lst in accepted:
        url = str(getattr(lst, "url", "") or "")
        host_store = resolve_store_for_url(url, tier_lookup)
        if host_store is not None and host_store.store_type == "local_shop":
            to_verify.append(lst)

    if not to_verify:
        return accepted

    with stage_timer("verify_album_match") as rec_v:
        rec_v.input = {
            "tier": tier,
            "candidates": len(to_verify),
            "album_title": ctx.album_title or "",
            "artist": ctx.parsed.artist,
        }
        verdicts = await verify_album_match(
            to_verify,
            artist=ctx.parsed.artist,
            album_title=ctx.album_title or "",
        )
        rec_v.output = {
            "verdicts_returned": len(verdicts),
            "rejected": sum(
                1 for v in verdicts.values() if v.verdict == "reject"
            ),
            "confirmed": sum(
                1 for v in verdicts.values() if v.verdict == "confirmed"
            ),
            "unsure": sum(
                1 for v in verdicts.values() if v.verdict == "unsure"
            ),
        }
        rec_v.status = "success" if verdicts else "empty"

    surviving: list[Any] = []
    for lst in accepted:
        url = str(getattr(lst, "url", "") or "")
        host_store = resolve_store_for_url(url, tier_lookup)
        is_local = (
            host_store is not None
            and host_store.store_type == "local_shop"
        )
        if not is_local:
            surviving.append(lst)
            continue
        v = verdicts.get(url)
        if v is None:
            state.album_match_by_url[url] = False
            state.album_match_by_url[normalize_url(url)] = False
            surviving.append(lst)
            continue
        if v.verdict == "reject":
            keep_physical_local_despite_strict_reject = (
                ctx.prioritize_physical_locals
                and qualifies_as_target_city_local_shop(
                    listing_url=url,
                    store_lookup=state.store_lookup,
                    norm=ctx.norm,
                )
            )
            if keep_physical_local_despite_strict_reject:
                state.album_match_by_url[url] = False
                state.album_match_by_url[normalize_url(url)] = False
                state.verifier_summary.append(
                    {
                        "tier": tier,
                        "url": url[:160],
                        "verdict": "reject_soft_kept",
                        "reason": v.reason[:120],
                    }
                )
                surviving.append(lst)
                continue
            rejected_reasons.append(
                {
                    "url": url[:160],
                    "title": str(getattr(lst, "title", ""))[:120],
                    "reason": f"verify_album_match:{v.reason[:80]}",
                }
            )
            state.verifier_summary.append(
                {
                    "tier": tier,
                    "url": url[:160],
                    "verdict": "reject",
                    "reason": v.reason[:120],
                }
            )
            continue
        state.album_match_by_url[url] = v.verdict == "confirmed"
        state.album_match_by_url[normalize_url(url)] = v.verdict == "confirmed"
        surviving.append(lst)
        state.verifier_summary.append(
            {
                "tier": tier,
                "url": url[:160],
                "verdict": v.verdict,
                "reason": v.reason[:120],
            }
        )
    return surviving
