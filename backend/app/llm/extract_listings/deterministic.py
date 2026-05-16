"""Non-LLM extraction for very small candidate batches."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.domain.listing_schema import Listing
from app.llm.extract_listings.domains import normalize_domain
from app.llm.extract_listings.evidence_alignment import evidence_blob_matches_target_release
from app.llm.extract_listings.price_currency import sniff_price_currency

_SNIPPET_STORE_MAX = 520


def deterministic_listings_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    snippet_relax_hosts: frozenset[str] | None = None,
) -> list[Listing]:
    """Small-batch path: strict SERP evidence (same thresholds as merge/LLM path).

    ``snippet_relax_hosts`` is ignored here on purpose — it only informs intent
    relax upstream in :func:`~app.llm.extract_listings.prefilter.collect_snippet_candidates`.
    Indie-specific fuzz belongs in ``validate_listing(..., relaxed_local_indie=True)``.
    """
    _ = snippet_relax_hosts
    out: list[Listing] = []
    for c in candidates:
        url = str(c.get("url") or "").strip()
        if not url:
            continue
        raw_title = str(c.get("title") or "").strip()
        content = str(c.get("content") or "")
        evidence_lc = (raw_title + " " + content).strip().lower()
        if not evidence_blob_matches_target_release(
            evidence_lc,
            artist=artist,
            album=album,
        ):
            continue
        if raw_title.strip():
            pick_title = raw_title.strip()
        else:
            snippet_line = content.strip().replace("\n", " ")[:260].strip()
            pick_title = snippet_line.split("·")[0].split("|")[0].strip()[:220]
        if len(pick_title) < 3:
            continue
        price_v, currency = sniff_price_currency(content)
        dom = normalize_domain(url) or "store"
        raw_snippet = (raw_title + " · " + content).strip()[:_SNIPPET_STORE_MAX]
        try:
            lst = Listing(
                title=pick_title,
                price=price_v,
                currency=currency,
                in_stock=True,
                url=url,
                store=dom,
                source_snippet=raw_snippet or None,
            )
        except (ValidationError, TypeError, ValueError):
            continue
        out.append(lst)
    return out
