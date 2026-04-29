"""Discogs REST API client. Used by the Parser agent to resolve relative album
references ("2nd album", "latest", "debut") DETERMINISTICALLY instead of letting
the LLM hallucinate a title.

The parser also uses `resolve_album_by_index`, `get_artist_discography`,
and `search_release_by_track`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import httpx

from app.config import get_settings
from app.pipeline_context import get_context

logger = logging.getLogger(__name__)

_ARTIST_SEARCH_PATH = "/database/search"
_ARTIST_RELEASES_PATH = "/artists/{artist_id}/releases"

# Types that Discogs uses for a "Master" (canonical studio album) are sorted
# into the releases list when sorted by year. We filter to albums only.
_ALBUM_FORMATS: tuple[str, ...] = ("Vinyl", "CD", "Cassette", "Album", "LP")


@dataclass(frozen=True, slots=True)
class DiscogsAlbum:
    title: str
    year: int | None
    artist_id: int
    master_id: int | None


@dataclass(frozen=True, slots=True)
class DiscogsResolution:
    """Return envelope from `resolve_album_by_index`."""

    album: DiscogsAlbum | None
    # Position in the (year-sorted) discography: 1-based. None if no resolution.
    index: int | None
    # "high" if Discogs returned a confident single match, "medium" if we picked
    # from multiple candidates, "low" if nothing actionable came back.
    confidence: str


def _headers() -> dict[str, str]:
    settings = get_settings()
    token = (settings.discogs_token or "").strip()
    headers = {"User-Agent": settings.discogs_user_agent, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Discogs token={token}"
    return headers


@lru_cache(maxsize=256)
def _cache_key(artist: str, album_index: int) -> str:
    return f"{artist.strip().lower()}::{album_index}"


async def _search_artist(client: httpx.AsyncClient, artist: str) -> int | None:
    settings = get_settings()
    params = {"q": artist, "type": "artist", "per_page": 5}
    resp = await client.get(
        f"{settings.discogs_base_url}{_ARTIST_SEARCH_PATH}", params=params, headers=_headers()
    )
    resp.raise_for_status()
    data = resp.json() or {}
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    artist_id = first.get("id")
    try:
        return int(artist_id) if artist_id is not None else None
    except (TypeError, ValueError):
        return None


async def _fetch_releases(client: httpx.AsyncClient, artist_id: int) -> list[dict]:
    settings = get_settings()
    # Discogs will paginate; for our purposes the main-release sort on year, asc
    # is enough to pick "nth album". We fetch 100 — plenty for any real artist.
    params = {"sort": "year", "sort_order": "asc", "per_page": 100, "page": 1}
    url = f"{settings.discogs_base_url}{_ARTIST_RELEASES_PATH.format(artist_id=artist_id)}"
    resp = await client.get(url, params=params, headers=_headers())
    resp.raise_for_status()
    data = resp.json() or {}
    return data.get("releases", []) or []


def _filter_studio_albums(releases: list[dict]) -> list[DiscogsAlbum]:
    """Keep only main studio albums; drop singles/comps/bootlegs/live etc.

    Discogs flags releases with role="Main" + type="master" for canonical studio
    releases. We also defensively skip titles that look like live/compilation.
    """
    noisy_keywords = (
        "live", "bootleg", "compilation", "greatest hits", "best of",
        "single", "ep", "sampler", "promo", "soundtrack",
    )
    albums: list[DiscogsAlbum] = []
    seen_titles: set[str] = set()

    for rel in releases:
        role = (rel.get("role") or "").lower()
        rel_type = (rel.get("type") or "").lower()
        title = str(rel.get("title") or "").strip()
        if not title or rel_type not in {"master", "release"}:
            continue
        # Skip non-main contributions (e.g., compilations where the artist appears).
        if role and role not in {"main", "artist"}:
            continue
        lower_title = title.lower()
        if any(kw in lower_title for kw in noisy_keywords):
            continue
        # Same studio album can appear multiple times (reissues). Dedup by lowercase title.
        if lower_title in seen_titles:
            continue
        seen_titles.add(lower_title)

        year = rel.get("year")
        try:
            year_int = int(year) if year else None
        except (TypeError, ValueError):
            year_int = None

        artist_id = rel.get("artist_id") or rel.get("main_release_id") or 0
        master_id = rel.get("master_id") or rel.get("id")
        try:
            albums.append(
                DiscogsAlbum(
                    title=title,
                    year=year_int,
                    artist_id=int(artist_id) if artist_id else 0,
                    master_id=int(master_id) if master_id else None,
                )
            )
        except (TypeError, ValueError):
            continue

    # Stable sort oldest-first so index 1 = debut, index -1 = latest.
    albums.sort(key=lambda a: (a.year or 10**6, a.title.lower()))
    return albums


async def resolve_album_by_index(*, artist: str, album_index: int) -> DiscogsResolution:
    """Return the nth studio album for `artist` according to Discogs.

    `album_index`:
        1, 2, 3, ... -> 1-based from debut
        -1           -> latest studio album

    Always returns a `DiscogsResolution`; .album is None if nothing could be resolved.
    Errors are swallowed and reported as `confidence="low"` so the parser can fall
    back to the LLM.
    """
    settings = get_settings()
    ctx = get_context()
    request_id = ctx.request_id if ctx else None

    if not artist.strip() or album_index == 0:
        return DiscogsResolution(album=None, index=None, confidence="low")

    log_extra = {
        "stage": "discogs",
        "request_id": request_id,
        "artist": artist,
        "album_index": album_index,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.discogs_timeout_seconds) as client:
            artist_id = await _search_artist(client, artist)
            if not artist_id:
                logger.warning("discogs_artist_miss", extra={**log_extra, "status": "empty"})
                return DiscogsResolution(album=None, index=None, confidence="low")

            releases = await _fetch_releases(client, artist_id)
    except httpx.HTTPError as exc:
        logger.warning(
            "discogs_http_error",
            extra={**log_extra, "status": "fail", "reason": f"{type(exc).__name__}: {exc}"},
        )
        return DiscogsResolution(album=None, index=None, confidence="low")
    except Exception:
        logger.exception("discogs_unexpected_error", extra={**log_extra, "status": "fail"})
        return DiscogsResolution(album=None, index=None, confidence="low")

    albums = _filter_studio_albums(releases)
    if not albums:
        logger.warning(
            "discogs_no_studio_albums",
            extra={**log_extra, "status": "empty", "count": len(releases)},
        )
        return DiscogsResolution(album=None, index=None, confidence="low")

    if album_index == -1:
        picked = albums[-1]
        idx = len(albums)
    elif 1 <= album_index <= len(albums):
        picked = albums[album_index - 1]
        idx = album_index
    else:
        logger.warning(
            "discogs_index_out_of_range",
            extra={**log_extra, "status": "empty", "count": len(albums)},
        )
        return DiscogsResolution(album=None, index=None, confidence="low")

    # "high" confidence iff the artist query returned an unambiguous hit AND the
    # index is within the bounds of the studio-album list. Otherwise "medium".
    confidence = "high" if 1 <= idx <= len(albums) else "medium"
    logger.info(
        "discogs_resolved",
        extra={
            **log_extra,
            "status": "success",
            "output": {"title": picked.title, "year": picked.year, "index": idx},
        },
    )
    return DiscogsResolution(album=picked, index=idx, confidence=confidence)


async def get_artist_discography(artist: str) -> list[DiscogsAlbum]:
    """Filtered studio-main release rows for ``artist``, chronological order."""

    settings = get_settings()
    if not artist.strip():
        return []
    try:
        async with httpx.AsyncClient(timeout=settings.discogs_timeout_seconds) as client:
            artist_id = await _search_artist(client, artist)
            if not artist_id:
                return []
            releases = await _fetch_releases(client, artist_id)
        return _filter_studio_albums(releases)
    except Exception:
        logger.exception("get_artist_discography_failed")
        return []


def _album_from_track_search_hit(hit: dict, track: str) -> DiscogsAlbum | None:
    """Map one Discogs database/search row to a candidate album title."""

    tit = str(hit.get("title") or "").strip()
    ht = hit.get("type")
    if not tit:
        return None
    hid = hit.get("id")
    try:
        hid_i = int(hid) if hid is not None else None
    except (TypeError, ValueError):
        hid_i = None

    if ht == "master":
        return DiscogsAlbum(
            title=tit,
            year=None,
            artist_id=0,
            master_id=hid_i,
        )
    if ht != "release":
        return None

    tr_l = track.lower()
    for sep in (" – ", " - "):
        if sep not in tit:
            continue
        _left, right = [p.strip() for p in tit.split(sep, maxsplit=1)]
        rl = right.lower()
        if len(right) > 2 and (tr_l not in rl or rl != tr_l):
            return DiscogsAlbum(
                title=right,
                year=None,
                artist_id=0,
                master_id=None,
            )

    for sep in (" – ", " - "):
        if sep in tit:
            sub = tit.split(sep, maxsplit=1)[1].strip()
            if len(sub) > 2:
                return DiscogsAlbum(
                    title=sub,
                    year=None,
                    artist_id=0,
                    master_id=None,
                )

    return DiscogsAlbum(
        title=tit,
        year=None,
        artist_id=0,
        master_id=None,
    )


async def search_release_by_track(artist: str, track: str) -> list[DiscogsAlbum]:
    """Discogs database/search hits for ``artist`` + ``track`` (deduped by title)."""

    settings = get_settings()
    artist = artist.strip()
    track = track.strip()
    if not artist or not track:
        return []

    queries = (f'{artist} "{track}"', f"{artist} {track}")
    collected: list[DiscogsAlbum] = []
    seen_titles: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=settings.discogs_timeout_seconds) as client:
            for q in queries:
                for kind in ("master", "release"):
                    try:
                        resp = await client.get(
                            f"{settings.discogs_base_url}/database/search",
                            params={"q": q, "type": kind, "per_page": 25, "page": 1},
                            headers=_headers(),
                        )
                        resp.raise_for_status()
                        blob = resp.json() or {}
                    except httpx.HTTPError:
                        continue
                    except Exception:
                        logger.exception("search_release_by_track_http")
                        continue

                    for hit in blob.get("results") or []:
                        album = _album_from_track_search_hit(hit, track)
                        if album is None:
                            continue
                        key = album.title.strip().lower()
                        if key in seen_titles:
                            continue
                        seen_titles.add(key)
                        collected.append(album)
    except Exception:
        logger.exception("search_release_by_track_failed")
        return []
    return collected
