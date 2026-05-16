"""Build snippet candidates: domain allowlist + intent pre-gate."""

from __future__ import annotations

from typing import Any

from app.llm.extract_listings.constants import SNIPPET_CHAR_CAP
from app.llm.extract_listings.domains import host_matches_whitelist, normalize_domain
from app.llm.extract_listings.intent_match import snippet_passes_release_intent


def collect_snippet_candidates(
    raw_results: list[dict[str, Any]],
    *,
    artist_l: str | None,
    album_l: str,
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

        is_verified_local_shop = domain in allowed_hosts

        if not is_verified_local_shop and not snippet_passes_release_intent(
            url=url,
            blob=blob,
            artist_l=artist_l,
            album_l=album_l,
            host=domain,
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
