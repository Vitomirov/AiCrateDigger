"""Marketplace Discovery Layer — emergent, data-driven RAG memory.

Design principles (NON-NEGOTIABLE):
- Zero hardcoded marketplace lists, zero regional heuristics.
- Every domain we observe in the wild gets one row, keyed by its base domain.
- Per-request writes MERGE counters (upsert-with-merge) rather than overwriting.
- Retrieval = semantic similarity (Chroma) combined with a deterministic
  "emergent score" derived from the counters. The score is the only ranker —
  we never second-guess it with a hand-written rule about a specific TLD.

Metadata schema per document (all fields are Chroma-compatible primitives):
    domain              str   — stable primary key (e.g. "metropolismusic.rs")
    total_hits          int   — times this domain appeared in Tavily results
    store_hits          int   — times the extractor confirmed it as a seller "store"
    artist_mentions     int   — aggregate artist-token matches across all hits
    album_mentions      int   — aggregate album-token matches
    format_mentions     int   — aggregate format-token matches
    tavily_score_sum    float — sum of Tavily relevance scores
    last_seen_ts        float — Unix timestamp of most recent write (for recency)
    sample_title        str   — most recent listing title observed
    sample_location     str   — most recent location hint observed
    sample_url          str   — most recent URL observed

Embedding text: freeform blob (domain + sample_title + sample_location +
concatenated recent queries). Updated on every write so semantic retrieval
keeps improving as signal accumulates.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma  # type: ignore
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

from app.config import get_settings

logger = logging.getLogger(__name__)

# Emergent score weights. Intentionally soft — the system self-balances
# because higher-hit domains naturally dominate through volume.
W_FREQUENCY = 0.30  # log-scaled total hits
W_CONVERSION = 0.25  # store confirmations / total hits
W_RECENCY = 0.15  # exponential decay on last_seen
W_ARTIST_DENSITY = 0.20  # artist mentions / total hits
W_QUALITY = 0.10  # mean Tavily relevance score

RECENCY_DECAY_SECONDS = 60 * 60 * 24 * 30  # ~30-day e-folding time

# Length cap on the free-form sample context we embed so Chroma doesn't blow up.
_SAMPLE_CONTEXT_CAP = 512


# ---------------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------------

def normalize_domain(url_or_domain: str) -> str | None:
    """Return a stable lowercase base domain. No TLD/region heuristics."""
    if not url_or_domain:
        return None
    try:
        candidate = url_or_domain.strip()
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        netloc = urlsplit(candidate).netloc.lower().strip()
        if not netloc:
            return None
        if netloc.startswith("www."):
            netloc = netloc[4:]
        netloc = netloc.split(":", maxsplit=1)[0]
        return netloc or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MarketplaceSignal:
    """Read-side materialized view of a row + its emergent score."""

    domain: str
    total_hits: int = 0
    store_hits: int = 0
    artist_mentions: int = 0
    album_mentions: int = 0
    format_mentions: int = 0
    tavily_score_sum: float = 0.0
    last_seen_ts: float = 0.0
    sample_title: str = ""
    sample_location: str = ""
    sample_url: str = ""
    similarity: float = 0.0
    emergent_score: float = 0.0

    def as_debug(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "total_hits": self.total_hits,
            "store_hits": self.store_hits,
            "artist_mentions": self.artist_mentions,
            "album_mentions": self.album_mentions,
            "format_mentions": self.format_mentions,
            "emergent_score": round(self.emergent_score, 3),
            "similarity": round(self.similarity, 3),
            "last_seen_ts": self.last_seen_ts,
        }


@dataclass(slots=True)
class TavilyObservation:
    """Write-side payload when ingesting a Tavily result."""

    url: str
    title: str
    content: str
    tavily_score: float
    artist: str
    album: str
    music_format: str
    location_hint: str | None = None


@dataclass(slots=True)
class StoreConfirmation:
    url: str
    title: str
    location: str | None = None
    tavily_score: float = 0.0
    artist: str = ""
    album: str = ""
    music_format: str = ""
    content: str = ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MarketplaceDBService:
    """Emergent marketplace memory backed by Chroma + deterministic scoring."""

    def __init__(self, *, persist_directory: str, collection_name: str = "marketplaces") -> None:
        settings = get_settings()
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._lock = threading.Lock()

        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory,
        )

    # ------------------------------------------------------------------ writes

    async def record_tavily_hit(self, observation: TavilyObservation) -> str | None:
        """Upsert a row reflecting one Tavily result. Returns the domain written,
        or None if we couldn't resolve one. Idempotent on redundant replays."""
        domain = normalize_domain(observation.url)
        if not domain:
            return None
        await asyncio.to_thread(self._apply_tavily_hit_sync, domain, observation)
        return domain

    async def record_store_confirmation(self, confirmation: StoreConfirmation) -> str | None:
        """Upsert a strong-signal confirmation — extractor flagged this URL as a
        professional store listing. Bumps `store_hits`."""
        domain = normalize_domain(confirmation.url)
        if not domain:
            return None
        await asyncio.to_thread(self._apply_store_confirmation_sync, domain, confirmation)
        return domain

    # ------------------------------------------------------------------ reads

    async def retrieve_candidates(
        self, *, intent_text: str, k: int = 20
    ) -> list[MarketplaceSignal]:
        """Return up to `k` marketplace candidates for the given intent text,
        sorted by a blend of semantic similarity and emergent score."""
        if not intent_text.strip():
            return []
        return await asyncio.to_thread(self._retrieve_sync, intent_text, k)

    async def collection_size(self) -> int:
        return await asyncio.to_thread(self._collection_size_sync)

    # -------------------------------------------------------------- sync impls

    def _apply_tavily_hit_sync(self, domain: str, obs: TavilyObservation) -> None:
        with self._lock:
            existing = self._read_existing(domain)
            merged = _merge_tavily(existing, obs)
            self._write_row(merged, embedding_text=self._build_embedding_text(merged, obs.content))

    def _apply_store_confirmation_sync(self, domain: str, conf: StoreConfirmation) -> None:
        with self._lock:
            existing = self._read_existing(domain)
            merged = _merge_store(existing, conf)
            self._write_row(
                merged, embedding_text=self._build_embedding_text(merged, conf.content)
            )

    def _retrieve_sync(self, intent_text: str, k: int) -> list[MarketplaceSignal]:
        try:
            # Fetch a wider pool than k; we'll re-rank by emergent score below.
            fetch_k = max(k * 2, 20)
            pairs = self._vectorstore.similarity_search_with_score(intent_text, k=fetch_k)
        except Exception:
            logger.exception(
                "marketplace_retrieve_failed",
                extra={"stage": "rag_retrieve", "status": "fail"},
            )
            return []

        now = time.time()
        out: list[MarketplaceSignal] = []
        seen_domains: set[str] = set()
        for doc, chroma_score in pairs:
            metadata = getattr(doc, "metadata", None) or {}
            signal = _signal_from_metadata(metadata)
            if not signal.domain or signal.domain in seen_domains:
                continue
            seen_domains.add(signal.domain)
            # Chroma returns distance (lower=better) for most backends; convert
            # to a [0,1] similarity. If the store already returned similarity we
            # still get something monotone, which is all we need for ranking.
            signal.similarity = max(0.0, 1.0 - float(chroma_score))
            signal.emergent_score = _compute_emergent_score(signal, now=now)
            out.append(signal)

        # Final rank: blend similarity 0.4 + emergent 0.6. The emergent side
        # dominates because that's what the system is LEARNING over time.
        out.sort(
            key=lambda s: (0.4 * s.similarity + 0.6 * s.emergent_score),
            reverse=True,
        )
        return out[:k]

    def _collection_size_sync(self) -> int:
        try:
            # Pull only IDs to keep it cheap.
            got = self._vectorstore.get(include=[])
            ids = got.get("ids") if isinstance(got, dict) else None
            return len(ids or [])
        except Exception:
            return 0

    # -------------------------------------------------------------- internals

    def _read_existing(self, domain: str) -> MarketplaceSignal | None:
        try:
            got = self._vectorstore.get(ids=[domain])
        except Exception:
            return None
        if not isinstance(got, dict):
            return None
        metadatas = got.get("metadatas") or []
        if not metadatas:
            return None
        return _signal_from_metadata(metadatas[0] or {})

    def _write_row(self, signal: MarketplaceSignal, *, embedding_text: str) -> None:
        metadata = _metadata_from_signal(signal)
        # add_texts with an existing ID upserts (langchain-chroma wrapper).
        self._vectorstore.add_texts(
            texts=[embedding_text],
            metadatas=[metadata],
            ids=[signal.domain],
        )

    @staticmethod
    def _build_embedding_text(signal: MarketplaceSignal, recent_content: str) -> str:
        # Keep the embedding focused on what a query would look for: domain name,
        # a couple of recent titles/locations, and a snippet of surrounding content.
        # We intentionally do NOT add any region/language tokens — those emerge
        # from the content itself.
        parts = [
            signal.domain,
            signal.sample_title,
            signal.sample_location,
            (recent_content or "")[: _SAMPLE_CONTEXT_CAP],
        ]
        return "\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Metadata <-> signal adapters
# ---------------------------------------------------------------------------

def _signal_from_metadata(metadata: dict[str, Any]) -> MarketplaceSignal:
    def _i(key: str) -> int:
        v = metadata.get(key, 0) or 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def _f(key: str) -> float:
        v = metadata.get(key, 0.0) or 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _s(key: str) -> str:
        v = metadata.get(key, "") or ""
        return str(v)

    return MarketplaceSignal(
        domain=_s("domain"),
        total_hits=_i("total_hits"),
        store_hits=_i("store_hits"),
        artist_mentions=_i("artist_mentions"),
        album_mentions=_i("album_mentions"),
        format_mentions=_i("format_mentions"),
        tavily_score_sum=_f("tavily_score_sum"),
        last_seen_ts=_f("last_seen_ts"),
        sample_title=_s("sample_title"),
        sample_location=_s("sample_location"),
        sample_url=_s("sample_url"),
    )


def _metadata_from_signal(signal: MarketplaceSignal) -> dict[str, Any]:
    return {
        "domain": signal.domain,
        "total_hits": int(signal.total_hits),
        "store_hits": int(signal.store_hits),
        "artist_mentions": int(signal.artist_mentions),
        "album_mentions": int(signal.album_mentions),
        "format_mentions": int(signal.format_mentions),
        "tavily_score_sum": float(signal.tavily_score_sum),
        "last_seen_ts": float(signal.last_seen_ts),
        "sample_title": signal.sample_title[:256],
        "sample_location": signal.sample_location[:128],
        "sample_url": signal.sample_url[:512],
    }


# ---------------------------------------------------------------------------
# Merge logic (pure functions — easy to reason about)
# ---------------------------------------------------------------------------

def _count_mentions(needle: str, haystack: str) -> int:
    if not needle:
        return 0
    n = needle.strip().lower()
    h = haystack.lower()
    if not n or not h:
        return 0
    return h.count(n)


def _merge_tavily(existing: MarketplaceSignal | None, obs: TavilyObservation) -> MarketplaceSignal:
    base = existing or MarketplaceSignal(domain=normalize_domain(obs.url) or "")
    haystack = f"{obs.title}\n{obs.content}"
    return MarketplaceSignal(
        domain=base.domain or (normalize_domain(obs.url) or ""),
        total_hits=base.total_hits + 1,
        store_hits=base.store_hits,
        artist_mentions=base.artist_mentions + _count_mentions(obs.artist, haystack),
        album_mentions=base.album_mentions + _count_mentions(obs.album, haystack),
        format_mentions=base.format_mentions + _count_mentions(obs.music_format, haystack),
        tavily_score_sum=base.tavily_score_sum + float(obs.tavily_score or 0.0),
        last_seen_ts=time.time(),
        sample_title=(obs.title or base.sample_title)[:256],
        sample_location=(obs.location_hint or base.sample_location)[:128],
        sample_url=(obs.url or base.sample_url)[:512],
    )


def _merge_store(existing: MarketplaceSignal | None, conf: StoreConfirmation) -> MarketplaceSignal:
    base = existing or MarketplaceSignal(domain=normalize_domain(conf.url) or "")
    haystack = f"{conf.title}\n{conf.content}"
    return MarketplaceSignal(
        domain=base.domain or (normalize_domain(conf.url) or ""),
        total_hits=max(base.total_hits, 1),  # a confirmation implies a hit
        store_hits=base.store_hits + 1,
        artist_mentions=base.artist_mentions + _count_mentions(conf.artist, haystack),
        album_mentions=base.album_mentions + _count_mentions(conf.album, haystack),
        format_mentions=base.format_mentions + _count_mentions(conf.music_format, haystack),
        tavily_score_sum=base.tavily_score_sum + float(conf.tavily_score or 0.0),
        last_seen_ts=time.time(),
        sample_title=(conf.title or base.sample_title)[:256],
        sample_location=(conf.location or base.sample_location or "")[:128],
        sample_url=(conf.url or base.sample_url)[:512],
    )


# ---------------------------------------------------------------------------
# Emergent scoring
# ---------------------------------------------------------------------------

def _compute_emergent_score(s: MarketplaceSignal, *, now: float) -> float:
    """Deterministic emergent score ∈ [0, 1] — see module docstring.

    Intentionally monotone in: total_hits, store_hits, artist_mentions, recency,
    mean tavily score. No domain/TLD-specific adjustments.
    """
    if s.total_hits <= 0:
        return 0.0

    # 1. frequency (log-scaled so a few prolific domains don't dominate)
    freq = math.log1p(s.total_hits) / math.log1p(s.total_hits + 10.0)

    # 2. conversion — probability we converted a raw hit to a listing.
    conv = (s.store_hits / float(s.total_hits)) if s.total_hits else 0.0
    conv = min(conv, 1.0)

    # 3. recency — exp decay
    age = max(now - s.last_seen_ts, 0.0)
    recency = math.exp(-age / RECENCY_DECAY_SECONDS) if s.last_seen_ts > 0 else 0.0

    # 4. artist density — mentions per hit, clipped to [0, 1]
    artist_density = min((s.artist_mentions / float(s.total_hits)) / 2.0, 1.0)

    # 5. mean Tavily quality
    mean_quality = min(s.tavily_score_sum / float(s.total_hits), 1.0)

    return (
        W_FREQUENCY * freq
        + W_CONVERSION * conv
        + W_RECENCY * recency
        + W_ARTIST_DENSITY * artist_density
        + W_QUALITY * mean_quality
    )


# ---------------------------------------------------------------------------
# Bulk ingest helper (used post-Tavily)
# ---------------------------------------------------------------------------

async def ingest_tavily_batch(
    service: MarketplaceDBService,
    observations: list[TavilyObservation],
) -> list[str]:
    """Concurrency-friendly fan-out over a batch. Returns the list of domains
    written (deduped, order preserved)."""
    if not observations:
        return []

    tasks = [service.record_tavily_hit(o) for o in observations]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    domains: list[str] = []
    seen: set[str] = set()
    for res in results:
        if isinstance(res, Exception):
            logger.warning(
                "marketplace_ingest_error",
                extra={"stage": "rag_store", "status": "fail", "reason": str(res)},
            )
            continue
        if res and res not in seen:
            seen.add(res)
            domains.append(res)
    return domains


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache
def get_marketplace_db() -> MarketplaceDBService:
    settings = get_settings()
    persist_directory = getattr(settings, "chroma_db_dir", "./chroma_db")
    return MarketplaceDBService(persist_directory=persist_directory)
