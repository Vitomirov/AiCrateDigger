"""Turn LLM JSON rows into validated listings (dedupe by URL, stock-aware)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.agents.extractor.evidence_alignment import (
    evidence_blob_matches_target_release,
    listing_title_grounded_in_evidence,
    looks_like_pure_query_echo_title,
)
from app.agents.extractor.intent_match import snippet_passes_release_intent
from app.agents.extractor.listing_domains import normalize_domain
from app.agents.extractor.utils.price_currency import coerce_price_currency
from app.domain.listing_schema import Listing
from app.llm.coerce_listing_fields import coerce_in_stock

_SNIPPET_STORE_MAX = 520


def merge_llm_rows_into_listings(
    extracted: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    artist_l: str | None,
    album_l: str,
    diagnostic: dict[str, Any],
    snippet_relax_hosts: frozenset[str] | None = None,
) -> list[Listing]:
    diagnostic.setdefault("drop_evidence_target_miss_pdd", 0)
    diagnostic.setdefault("drop_llm_title_ungrounded", 0)
    diagnostic.setdefault("drop_query_echo_pick", 0)

    by_url_blob = {c["url"]: (c["title"] + " " + c["content"]).lower() for c in candidates}
    by_url_raw_title = {c["url"]: c["title"] for c in candidates}
    by_url_candidate = {c["url"]: c for c in candidates}
    allowed_urls = set(by_url_blob.keys())

    results: dict[str, Listing] = {}

    for item in extracted:
        url = str(item.get("url") or "").strip()
        if url not in allowed_urls:
            diagnostic["drop_url_not_in_candidates"] += 1
            continue

        evidence_blob_lc = by_url_blob.get(url, "").strip().lower()

        raw_title = (by_url_raw_title.get(url) or "").strip()
        llm_title = str(item.get("title") or "").strip()
        cand = by_url_candidate.get(url) or {}

        host = normalize_domain(url)
        if not snippet_passes_release_intent(
            url=url,
            blob=evidence_blob_lc,
            artist_l=artist_l,
            album_l=album_l,
            host=host,
            snippet_relax_hosts=snippet_relax_hosts,
        ):
            diagnostic["drop_title_gate"] += 1
            continue

        if not evidence_blob_matches_target_release(
            evidence_blob_lc,
            artist=artist,
            album=album,
        ):
            diagnostic["drop_evidence_target_miss_pdd"] += 1
            continue

        if llm_title and looks_like_pure_query_echo_title(
            llm_title,
            artist=artist,
            album=album,
            evidence_blob_lc=evidence_blob_lc,
        ):
            llm_title = ""

        pick_title = ""
        if raw_title:
            pick_title = raw_title
            if llm_title and listing_title_grounded_in_evidence(llm_title, evidence_blob_lc, min_ratio=55.0):
                if album_l in llm_title.lower() and album_l not in raw_title.lower():
                    pick_title = llm_title
        elif llm_title:
            if listing_title_grounded_in_evidence(llm_title, evidence_blob_lc, min_ratio=48.0) and (
                not looks_like_pure_query_echo_title(
                    llm_title,
                    artist=artist,
                    album=album,
                    evidence_blob_lc=evidence_blob_lc,
                )
            ):
                pick_title = llm_title
            else:
                diagnostic["drop_llm_title_ungrounded"] += 1
                continue
        else:
            diagnostic["drop_title_gate"] += 1
            continue

        if looks_like_pure_query_echo_title(
            pick_title,
            artist=artist,
            album=album,
            evidence_blob_lc=evidence_blob_lc,
        ):
            diagnostic["drop_query_echo_pick"] += 1
            continue

        domain = host or ""
        store_raw = item.get("store")
        store = (str(store_raw).strip() if store_raw else "") or domain

        in_stock = coerce_in_stock(item)
        price_v, currency = coerce_price_currency(item)

        raw_snippet = ((cand.get("title") or "") + " · " + (cand.get("content") or "")).strip()[
            :_SNIPPET_STORE_MAX
        ]

        try:
            listing = Listing(
                title=pick_title,
                price=price_v,
                currency=currency,
                in_stock=in_stock,
                url=url,
                store=store,
                source_snippet=raw_snippet or None,
            )
        except (ValidationError, TypeError, ValueError):
            diagnostic["drop_pydantic"] += 1
            continue

        existing = results.get(url)
        if existing is None or (listing.in_stock and not existing.in_stock):
            results[url] = listing

    return list(results.values())
