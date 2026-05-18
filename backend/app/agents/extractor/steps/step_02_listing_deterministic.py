"""Non-LLM extraction for very small candidate batches."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.agents.extractor.evidence_alignment import (
    evidence_blob_matches_target_release,
    url_path_evidence_text,
)
from app.agents.extractor.intent_match import intent_matches_snippet, snippet_passes_release_intent
from app.agents.extractor.listing_constants import SNIPPET_CHAR_CAP
from app.agents.extractor.listing_domains import normalize_domain
from app.agents.extractor.utils.price_currency import sniff_price_currency
from app.domain.listing_schema import Listing
from app.validators.listings import url_suggests_product_detail_page

_SNIPPET_STORE_MAX = 520


def candidate_has_extractable_evidence_signal(
    *,
    url: str,
    raw_title: str,
    raw_content: str,
    artist: str | None,
    album: str,
    artist_l: str | None,
    album_l: str,
    snippet_relax_hosts: frozenset[str] | None = None,
) -> bool:
    """Whether snippet + URL slug plausibly references the target release.

    Aligns deterministic extraction with :func:`snippet_passes_release_intent`
    PDP-aware branches so we do not burn an LLM round when the SERP row already
    passed the prefilter but ``evidence_blob_matches_target_release`` was overly
    strict on thin blobs.

    Used by the orchestrator to short-circuit hopeless LLM calls.
    """
    slug_text = url_path_evidence_text(url)
    evidence_lc = (raw_title + " " + raw_content + " " + slug_text).strip().lower()
    if evidence_blob_matches_target_release(evidence_lc, artist=artist, album=album):
        return True

    domain = normalize_domain(url)
    capped_blob = f"{raw_title} {raw_content}".lower()
    if snippet_relax_hosts and domain and domain in snippet_relax_hosts:
        if snippet_passes_release_intent(
            url=url,
            blob=capped_blob[:SNIPPET_CHAR_CAP],
            artist_l=artist_l,
            album_l=album_l,
            host=domain,
            snippet_relax_hosts=snippet_relax_hosts,
        ):
            return True

    if url_suggests_product_detail_page(url):
        return intent_matches_snippet(
            url=url,
            blob=evidence_lc,
            artist_l=artist_l,
            album_l=album_l,
            relaxed_indie_candidate=False,
        )
    return False


def deterministic_listings_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    snippet_relax_hosts: frozenset[str] | None = None,
) -> list[Listing]:
    """Small-batch path: strict SERP evidence (same thresholds as merge/LLM path).

    ``snippet_relax_hosts`` is forwarded to :func:`candidate_has_extractable_evidence_signal`
    so curated locals stay aligned with ``step_01_snippet_prefilter``.
    """
    artist_l = (artist or "").strip().lower() or None
    album_l = album.strip().lower()

    out: list[Listing] = []
    for c in candidates:
        url = str(c.get("url") or "").strip()
        if not url:
            continue
        raw_title = str(c.get("title") or "").strip()
        content = str(c.get("content") or "")
        if not candidate_has_extractable_evidence_signal(
            url=url,
            raw_title=raw_title,
            raw_content=content,
            artist=artist,
            album=album,
            artist_l=artist_l,
            album_l=album_l,
            snippet_relax_hosts=snippet_relax_hosts,
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
