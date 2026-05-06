"""Discogs Studio-Album Canonical Index.

Builds a deterministic, ordered list of an artist's STUDIO albums from the
Discogs REST API. Resolution is strict: only **masters** whose **main release**
formats indicate a full **Album** or **LP** survive — singles/EPs posing as
masters (e.g. "Jump In The Fire") are dropped.

Index rules: ``album_index`` 1 = debut studio album (by year), 2 = second, …;
``-1`` = latest.

Rules are deterministic — no LLM, no fuzzy matching; prefer precision over recall.

Usage::

    albums = await get_studio_albums("Metallica")
    resolution = await resolve_album_by_index(artist="Metallica", album_index=2)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Album:
    """One canonical studio album in an artist's chronological discography."""

    title: str
    year: int | None
    discogs_id: str  # master id


@dataclass(frozen=True, slots=True)
class AlbumResolution:
    """Return envelope from ``resolve_album_by_index``."""

    album: Album | None
    index: int | None
    confidence: float
    detail: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Paths & limits
# ---------------------------------------------------------------------------

_SEARCH_PATH = "/database/search"
_RELEASES_PATH = "/artists/{artist_id}/releases"
_MASTER_PATH = "/masters/{master_id}"
_RELEASE_PATH = "/releases/{release_id}"

_PER_PAGE = 100
_MAX_PAGES = 10
# Cap concurrent master↔release enrich calls to stay polite to Discogs.
_ENRICH_CONCURRENCY = 8
_MIN_ALBUM_YEAR = 1900

# Section 1 — titles must NOT contain (word tokens / phrases where noted).
_TITLE_PHRASE_REJECT: tuple[str, ...] = (
    "best of",
    "greatest hits",
    "live at",
    "live in",
    "live from",
    "compilation",
    "the collection",
)

_TITLE_TOKEN_REJECT: frozenset[str] = frozenset(
    {
        "single",
        "ep",
        "live",
        "compilation",
        "bootleg",
        "demo",
        "radio",
        "broadcast",
        "session",
        "sessions",
        "soundtrack",
        "ost",
        "promo",
        "unofficial",
        "concert",
        "mixtape",
        "anthology",
        "split",
        "remix",
        "remixes",
        "remixed",
        "demos",
        "rarities",
    }
)

# Section 2 — scoring penalty tokens (same as aggressive list; word tokens).
_SCORE_PENALTY_TITLE_TOKENS: frozenset[str] = frozenset(
    {
        "single",
        "ep",
        "live",
        "bootleg",
        "demo",
        "radio",
        "broadcast",
    }
)

# Format description/name tokens that forbid treating the main release as a studio LP.
_FORMAT_REJECT: frozenset[str] = frozenset(
    {
        "single",
        "ep",
        "compilation",
        "maxi-single",
        "mixtape",
        "promo",
    }
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_DISCOGRAPHY_CACHE: dict[str, list[Album]] = {}


def _cache_key(artist: str) -> str:
    return (artist or "").strip().lower()


def clear_discography_cache() -> None:
    _DISCOGRAPHY_CACHE.clear()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "User-Agent": settings.discogs_user_agent,
        "Accept": "application/json",
    }
    token = (settings.discogs_token or "").strip()
    if token:
        headers["Authorization"] = f"Discogs token={token}"
    return headers


async def _get_json(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    try:
        resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        out = resp.json()
        return out if isinstance(out, dict) else None
    except httpx.HTTPError:
        return None


async def _resolve_artist_id(client: httpx.AsyncClient, artist: str) -> int | None:
    settings = get_settings()
    params = {"q": artist, "type": "artist", "per_page": 5}
    try:
        resp = await client.get(
            f"{settings.discogs_base_url}{_SEARCH_PATH}",
            params=params,
            headers=_headers(),
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "discogs_artist_search_error",
            extra={"stage": "discogs", "status": "fail", "reason": str(exc), "artist": artist},
        )
        return None

    results = (resp.json() or {}).get("results") or []
    if not results:
        return None

    aid = results[0].get("id")
    try:
        return int(aid) if aid is not None else None
    except (TypeError, ValueError):
        return None


async def _fetch_releases(client: httpx.AsyncClient, artist_id: int) -> list[dict[str, Any]]:
    settings = get_settings()
    rows: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        params = {
            "sort": "year",
            "sort_order": "asc",
            "per_page": _PER_PAGE,
            "page": page,
        }
        try:
            resp = await client.get(
                f"{settings.discogs_base_url}{_RELEASES_PATH.format(artist_id=artist_id)}",
                params=params,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json() or {}
        except httpx.HTTPError as exc:
            logger.warning(
                "discogs_releases_error",
                extra={
                    "stage": "discogs",
                    "status": "fail",
                    "reason": str(exc),
                    "artist_id": artist_id,
                    "page": page,
                },
            )
            break

        page_rows = data.get("releases") or []
        if not page_rows:
            break
        rows.extend(page_rows)

        pagination = data.get("pagination") or {}
        try:
            total_pages = int(pagination.get("pages", 1))
        except (TypeError, ValueError):
            total_pages = 1
        if page >= total_pages:
            break
    return rows


# ---------------------------------------------------------------------------
# Normalization & title rules
# ---------------------------------------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)


def _normalize_title_key(title: str) -> str:
    t = title.lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return " ".join(t.split())


def _title_tokens(title_lower: str) -> set[str]:
    cleaned = title_lower.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    return set(cleaned.split())


def _title_rejected_hard(title: str) -> bool:
    tl = title.lower()
    for phrase in _TITLE_PHRASE_REJECT:
        if phrase in tl:
            return True
    return bool(_title_tokens(tl) & _TITLE_TOKEN_REJECT)


def _too_short_probable_track(norm_key: str) -> bool:
    """Reject very short one-word titles (e.g. 'One'); strict, not fuzzy."""
    parts = norm_key.split()
    return bool(len(parts) == 1 and len(parts[0]) <= 3)


def _normalize_year_from_row(row: dict[str, Any]) -> int | None:
    raw = row.get("year")
    try:
        year = int(raw) if raw is not None and raw != "" else 0
    except (TypeError, ValueError):
        return None
    if year < _MIN_ALBUM_YEAR:
        return None
    return year


# ---------------------------------------------------------------------------
# Main release format + scoring
# ---------------------------------------------------------------------------


def _format_vocab_from_release(release_data: dict[str, Any]) -> set[str]:
    """Lowercased tokens from format names, text, and descriptions."""
    out: set[str] = set()
    for fmt in release_data.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        for key in ("name", "text"):
            raw = fmt.get(key)
            if not raw or not isinstance(raw, str):
                continue
            for part in raw.replace(";", ",").replace("/", " ").split():
                p = part.strip().lower()
                if p:
                    out.add(p)
        for d in fmt.get("descriptions") or []:
            if isinstance(d, str) and d.strip():
                out.add(d.strip().lower())
    return out


def _main_release_id(master_blob: dict[str, Any]) -> int | None:
    mr = master_blob.get("main_release")
    if isinstance(mr, dict):
        rid = mr.get("id")
    else:
        rid = mr
    try:
        return int(rid) if rid is not None else None
    except (TypeError, ValueError):
        return None


def _master_release_type_album_ok(master_blob: dict[str, Any]) -> bool:
    """When Discogs provides release_type on the master, require 'album'."""
    raw = master_blob.get("release_type")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return True
    return str(raw).strip().lower() == "album"


def _format_studio_ok(vocab: set[str]) -> bool:
    """True when descriptions indicate a full-length album / LP release."""
    if _FORMAT_REJECT & vocab:
        return False
    if "mini-album" in vocab or "mini album" in vocab:
        return False
    return "album" in vocab or "lp" in vocab


def _score_candidate(title: str, vocab: set[str]) -> int:
    score = 0
    if "album" in vocab:
        score += 2
    if "lp" in vocab:
        score += 1
    if _title_tokens(title.lower()) & _SCORE_PENALTY_TITLE_TOKENS:
        score -= 3
    return score


@dataclass
class _Candidate:
    album: Album
    norm_title: str
    score: int


async def _enrich_master_row(
    client: httpx.AsyncClient,
    base_url: str,
    rel: dict[str, Any],
) -> _Candidate | None:
    """Map one artist ``releases`` master row → candidate or None."""
    if str(rel.get("type") or "").lower() != "master":
        return None
    role = str(rel.get("role") or "").lower()
    if role and role != "main":
        return None

    title = str(rel.get("title") or "").strip()
    if not title or _title_rejected_hard(title):
        return None

    year = _normalize_year_from_row(rel)
    if year is None:
        return None

    norm = _normalize_title_key(title)
    if _too_short_probable_track(norm):
        return None

    try:
        mid = int(rel.get("id"))
    except (TypeError, ValueError):
        return None

    master_url = f"{base_url}{_MASTER_PATH.format(master_id=mid)}"
    master_blob = await _get_json(client, master_url)
    if not master_blob or not _master_release_type_album_ok(master_blob):
        return None

    rid = _main_release_id(master_blob)
    if rid is None:
        return None

    release_url = f"{base_url}{_RELEASE_PATH.format(release_id=rid)}"
    release_blob = await _get_json(client, release_url)
    if not release_blob:
        return None

    vocab = _format_vocab_from_release(release_blob)
    if not _format_studio_ok(vocab):
        return None

    sc = _score_candidate(title, vocab)
    if sc < 2:
        return None

    album = Album(title=title, year=year, discogs_id=str(mid))
    return _Candidate(album=album, norm_title=norm, score=sc)


def _dedupe_same_norm_and_year(cands: list[_Candidate]) -> list[_Candidate]:
    """Duplicate normalized title + same year → keep lowest master id."""
    best: dict[tuple[str, int], _Candidate] = {}
    for c in cands:
        key = (c.norm_title, c.album.year or 0)
        existing = best.get(key)
        if existing is None or int(c.album.discogs_id) < int(existing.album.discogs_id):
            best[key] = c
    return list(best.values())


def _dedupe_norm_earliest_year(cands: list[_Candidate]) -> list[_Candidate]:
    """One row per normalized title — keep the earliest calendar year."""
    by_norm: dict[str, _Candidate] = {}
    for c in sorted(
        cands,
        key=lambda x: (x.album.year or 99999, int(x.album.discogs_id)),
    ):
        if c.norm_title not in by_norm:
            by_norm[c.norm_title] = c
    return list(by_norm.values())


def _finalize_candidates(raw: list[_Candidate]) -> list[Album]:
    step1 = _dedupe_same_norm_and_year(raw)
    step2 = _dedupe_norm_earliest_year(step1)
    step2.sort(key=lambda x: (x.album.year or 99999, x.album.title.lower()))
    return [c.album for c in step2]


async def _build_studio_albums(client: httpx.AsyncClient, releases: list[dict[str, Any]]) -> list[Album]:
    settings = get_settings()
    base = settings.discogs_base_url.rstrip("/")

    master_rows = [r for r in releases if str(r.get("type") or "").lower() == "master"]
    sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)

    async def _bounded(row: dict[str, Any]) -> _Candidate | None:
        async with sem:
            return await _enrich_master_row(client, base, row)

    enriched = await asyncio.gather(*[_bounded(r) for r in master_rows])
    candidates = [c for c in enriched if c is not None]
    return _finalize_candidates(candidates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_studio_albums(artist: str) -> list[Album]:
    key = _cache_key(artist)
    if not key:
        return []

    cached = _DISCOGRAPHY_CACHE.get(key)
    if cached is not None:
        return list(cached)

    settings = get_settings()
    timeout = settings.discogs_timeout_seconds

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            artist_id = await _resolve_artist_id(client, artist)
            if not artist_id:
                _DISCOGRAPHY_CACHE[key] = []
                logger.info(
                    "discogs_artist_miss",
                    extra={"stage": "discogs", "status": "empty", "artist": artist},
                )
                return []
            releases = await _fetch_releases(client, artist_id)
            albums = await _build_studio_albums(client, releases)
    except Exception:
        logger.exception(
            "get_studio_albums_failed",
            extra={"stage": "discogs", "status": "fail", "artist": artist},
        )
        return []

    _DISCOGRAPHY_CACHE[key] = albums
    first_ten = [{"year": a.year, "title": a.title, "id": a.discogs_id} for a in albums[:10]]
    logger.info(
        "discogs_studio_filter_debug",
        extra={
            "stage": "discogs",
            "status": "success" if albums else "empty",
            "artist": artist,
            "total_releases_fetched": len(releases),
            "total_after_filter": len(albums),
            "ordered_album_preview_first_10": first_ten,
        },
    )
    logger.info(
        "discogs_studio_albums_built",
        extra={
            "stage": "discogs",
            "status": "success" if albums else "empty",
            "artist": artist,
            "count": len(albums),
        },
    )
    return list(albums)


def _resolution_detail(ordered: list[Album], selected: Album | None, confidence: float) -> dict[str, Any]:
    return {
        "albums": [{"title": a.title, "year": a.year, "discogs_id": a.discogs_id} for a in ordered],
        "selected_album": selected.title if selected else None,
        "confidence": confidence,
    }


async def resolve_album_by_index(*, artist: str, album_index: int) -> AlbumResolution:
    if not (artist or "").strip() or album_index == 0:
        return AlbumResolution(
            album=None,
            index=None,
            confidence=0.0,
            detail=_resolution_detail([], None, 0.0),
        )

    albums = await get_studio_albums(artist)
    if not albums:
        return AlbumResolution(
            album=None,
            index=None,
            confidence=0.0,
            detail=_resolution_detail([], None, 0.0),
        )

    if album_index == -1:
        picked = albums[-1]
        return AlbumResolution(
            album=picked,
            index=len(albums),
            confidence=1.0,
            detail=_resolution_detail(albums, picked, 1.0),
        )

    if 1 <= album_index <= len(albums):
        picked = albums[album_index - 1]
        return AlbumResolution(
            album=picked,
            index=album_index,
            confidence=1.0,
            detail=_resolution_detail(albums, picked, 1.0),
        )

    return AlbumResolution(
        album=None,
        index=None,
        confidence=0.5,
        detail=_resolution_detail(albums, None, 0.5),
    )


# ---------------------------------------------------------------------------
# Legacy compatibility surface
# ---------------------------------------------------------------------------

DiscogsAlbum = Album
DiscogsResolution = AlbumResolution


async def get_artist_discography(artist: str) -> list[Album]:
    return await get_studio_albums(artist)


async def search_release_by_track(artist: str, track: str) -> list[Album]:
    return []
