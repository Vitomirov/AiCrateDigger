"""Build snippet candidates: domain allowlist + intent pre-gate."""

from __future__ import annotations

from typing import Any

from app.domains.engine.extraction.intent_match import (
    snippet_passes_artist_catalog_intent,
    snippet_passes_release_intent,
)
from app.domains.search_pipeline.search_intent import SearchIntent
from app.domains.engine.extraction.listing_constants import (
    SNIPPET_CHAR_CAP,
    blob_suggests_merch_or_digital_only,
)
from app.domains.engine.extraction.listing_domains import host_matches_whitelist, normalize_domain


def _snippet_passes_intent(
    *,
    url: str,
    blob: str,
    artist_l: str | None,
    album_l: str | None,
    host: str | None,
    search_intent: SearchIntent,
    snippet_relax_hosts: frozenset[str] | None,
) -> bool:
    if search_intent == "artist_catalog":
        return snippet_passes_artist_catalog_intent(
            url=url,
            blob=blob,
            artist_l=artist_l,
            host=host,
            snippet_relax_hosts=snippet_relax_hosts,
        )
    if not album_l:
        return False
    return snippet_passes_release_intent(
        url=url,
        blob=blob,
        artist_l=artist_l,
        album_l=album_l,
        host=host,
        snippet_relax_hosts=snippet_relax_hosts,
    )


def collect_snippet_candidates(
    raw_results: list[dict[str, Any]],
    *,
    artist_l: str | None,
    album_l: str | None,
    search_intent: SearchIntent,
    allowed_hosts: set[str],
    snippet_relax_hosts: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return candidates ``{url, title, content}`` and count dropped by intent mismatch."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped_intent = 0

    for r in raw_results:
        url = str(r.get("url") or "").strip()
        if not url or url in seen:
            continue

        domain = normalize_domain(url)
        if not domain or not host_matches_whitelist(domain, allowed_hosts):
            continue

        raw_title = str(r.get("title") or "").strip()
        raw_content = str(r.get("content") or "")[:SNIPPET_CHAR_CAP]
        blob = f"{raw_title} {raw_content}".lower()

        # Checked unconditionally — even a verified/whitelisted local shop's own
        # merch or digital-download page is not a buyable physical album, and
        # verified hosts otherwise bypass the intent gate below entirely.
        if blob_suggests_merch_or_digital_only(blob):
            dropped_intent += 1
            continue

        is_verified_local_shop = domain in allowed_hosts

        if not is_verified_local_shop and not _snippet_passes_intent(
            url=url,
            blob=blob,
            artist_l=artist_l,
            album_l=album_l,
            host=domain,
            search_intent=search_intent,
            snippet_relax_hosts=snippet_relax_hosts,
        ):
            dropped_intent += 1
            continue

        seen.add(url)
        candidates.append(
            {
                "url": url,
                "title": raw_title,
                "content": raw_content,
            }
        )

    return candidates, dropped_intent
