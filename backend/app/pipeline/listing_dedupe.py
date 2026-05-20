"""Listing-level dedupe helpers (URL-level and store-domain level).

Both helpers are intentionally side-effect-free: they take a ranked listing
sequence and return a filtered one plus a small diagnostic dict. The pipeline
relies on the upstream rank order to break ties for groups whose composite
score is identical.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from rapidfuzz import fuzz

from app.policies.eu_stores import StoreEntry
from app.policies.listing_rank import resolve_store_for_url
from app.policies.store_domain import canonical_store_domain
from app.services.tavily import normalize_url
from app.validators.listings import normalize_whitelist_domain


def title_album_fuzzy_score(lst: Any, album_title: str) -> float:
    """Best of (partial_ratio, token_set_ratio) between listing title and album."""
    if not (album_title or "").strip():
        return 0.0
    ttl = str(getattr(lst, "title", "") or "").strip().lower()
    alb = album_title.strip().lower()
    if not ttl or not alb:
        return 0.0
    return float(max(fuzz.partial_ratio(alb, ttl), fuzz.token_set_ratio(alb, ttl)))


def verdict_confirmed_rank(
    lst: Any, album_match_by_url: dict[str, bool] | None
) -> int:
    """Prefer explicit True verifier keys; missing keys neutral; explicit False suppressed."""
    if not album_match_by_url:
        return 1
    u = str(getattr(lst, "url", "") or "")
    nk = normalize_url(u)
    if u in album_match_by_url:
        return 2 if album_match_by_url[u] else 0
    if nk in album_match_by_url:
        return 2 if album_match_by_url[nk] else 0
    return 1


def pick_best_same_domain_rows(
    rows: list[Any],
    *,
    album_title: str | None,
    album_match_by_url: dict[str, bool] | None,
) -> Any:
    """Choose the listing on a single host that best evidences the target album."""
    if len(rows) == 1:
        return rows[0]

    def key(lst: Any) -> tuple[int, float, str]:
        u = str(getattr(lst, "url", "") or "")
        return (
            verdict_confirmed_rank(lst, album_match_by_url),
            title_album_fuzzy_score(lst, album_title or ""),
            u,
        )

    return max(rows, key=key)


def dedupe_listings_by_normalized_url(listings: list[Any]) -> list[Any]:
    """Stable URL-level dedupe using :func:`normalize_url` as the equality key."""
    order: list[str] = []
    by_key: dict[str, Any] = {}
    for lst in listings:
        u = getattr(lst, "url", None)
        if not u:
            continue
        key = normalize_url(str(u))
        if key not in by_key:
            by_key[key] = lst
            order.append(key)
    return [by_key[k] for k in order]


def dedupe_listings_by_domain(
    sorted_listings: list[Any],
    *,
    store_by_domain: dict[str, StoreEntry],
    album_title: str | None = None,
    artist: str | None = None,
    album_match_by_url: dict[str, bool] | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Pick the best listing per store domain after scoring, not simply "first in list".

    Groups hosts using the same keying policy as before, but when several URLs
    share a domain (e.g. multiple Groovie PDPs) we keep the row that best matches
    the target album title and any explicit ``verify_album_match`` confirmation
    flags. ``artist`` is accepted for forward-compatible tie metadata (unused for
    now — title overlap + verifier rank carry the signal).

    ``sorted_listings`` should still be ranked best-first for ordering of *groups*
    and as a prior when composite scores are tied.
    """
    _ = artist  # reserved for future token checks
    groups: "OrderedDict[str, list[Any]]" = OrderedDict()
    for lst in sorted_listings:
        url = str(getattr(lst, "url", "") or "")
        store = resolve_store_for_url(url, store_by_domain)
        if store is not None and store.domain:
            key = normalize_whitelist_domain(store.domain)
        else:
            key = canonical_store_domain(url)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(lst)

    out: list[Any] = []
    dropped: dict[str, int] = {}
    for key, rows in groups.items():
        if len(rows) == 1:
            out.append(rows[0])
            continue
        kept = pick_best_same_domain_rows(
            rows,
            album_title=album_title,
            album_match_by_url=album_match_by_url,
        )
        out.append(kept)
        dropped[key] = len(rows) - 1
    return out, dropped
