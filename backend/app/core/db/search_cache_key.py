"""Canonical search-cache identity from parsed intent (not raw user text).

Both Redis (hot read) and Postgres (audit / fallback) derive keys from the
same normalized ``{format, artist, album, country, city?}`` tuple so paraphrases
like *"Division Bell by Pink Floyd in Barselona"* and *"Pink Floyd's Division
Bell in Barcelona, Spain"* collide when :func:`parse_user_query` resolves the
same structured fields.

Redis keys are human-readable and schema-versioned; Postgres keys are a
SHA-256 hex digest of the Redis key so they fit ``String(64)`` primary keys.
"""

from __future__ import annotations

import hashlib
import re

#: Bump when pipeline behaviour changes must invalidate every cached response.
PIPELINE_CACHE_SCHEMA_VERSION: int = 3


def normalize_cache_token(value: str | None, *, fallback: str = "any") -> str:
    """Lowercase + trim + collapse whitespace to ``_`` for stable cache segments."""
    s = (value or "").strip().lower()
    if not s:
        return fallback
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def _city_cache_segment(
    resolved_city: str | None,
    geo_granularity: str | None,
) -> str | None:
    """Return a city token for city-level geo, else ``None`` (omit from key)."""
    city = (resolved_city or "").strip()
    if not city:
        return None
    gran = (geo_granularity or "").strip().lower()
    if gran and gran not in ("city", "none"):
        return None
    if gran == "none":
        return None
    return normalize_cache_token(city, fallback="any")


def build_pipeline_search_cache_key(
    *,
    format_token: str | None,
    artist: str | None,
    album: str | None,
    country_code: str | None,
    resolved_city: str | None = None,
    geo_granularity: str | None = None,
) -> str:
    """Human-readable Redis cache key from parsed search intent.

    Format::

        cratedigger:search:v{N}:{format}:{artist}:{album}:{country|global}[:{city}]

    The optional ``:{city}`` segment is present only for city-level queries so
    Bucharest and Cluj-Napoca do not share a cache slot within Romania.
    """
    fmt = normalize_cache_token(format_token, fallback="vinyl")
    artist_part = normalize_cache_token(artist, fallback="unknown_artist")
    album_part = normalize_cache_token(album, fallback="unknown_album")
    country_part = normalize_cache_token(country_code, fallback="global")
    key = (
        f"cratedigger:search:v{PIPELINE_CACHE_SCHEMA_VERSION}"
        f":{fmt}:{artist_part}:{album_part}:{country_part}"
    )
    city_part = _city_cache_segment(resolved_city, geo_granularity)
    if city_part is not None:
        key = f"{key}:{city_part}"
    return key


def build_postgres_search_cache_key(*, redis_cache_key: str) -> str:
    """Deterministic SHA-256 hex of the Redis key (64 chars, fits Postgres PK)."""
    return hashlib.sha256(redis_cache_key.encode("utf-8")).hexdigest()


def build_pipeline_search_cache_keys(
    *,
    format_token: str | None,
    artist: str | None,
    album: str | None,
    country_code: str | None,
    resolved_city: str | None = None,
    geo_granularity: str | None = None,
) -> tuple[str, str]:
    """Return ``(redis_key, postgres_key)`` for the same parsed intent."""
    redis_key = build_pipeline_search_cache_key(
        format_token=format_token,
        artist=artist,
        album=album,
        country_code=country_code,
        resolved_city=resolved_city,
        geo_granularity=geo_granularity,
    )
    return redis_key, build_postgres_search_cache_key(redis_cache_key=redis_key)


__all__ = [
    "PIPELINE_CACHE_SCHEMA_VERSION",
    "build_pipeline_search_cache_key",
    "build_pipeline_search_cache_keys",
    "build_postgres_search_cache_key",
    "normalize_cache_token",
]
