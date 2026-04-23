"""Agent 3 — Extractor (hardened).

Pipeline per candidate:
    1. Deterministic pre-filter  (gates: url / title-length / artist fuzzy match / merch keywords)
    2. LLM structured extraction (price, location, availability, seller_type, reason)
    3. Pydantic validation       (ListingResult contract; bad shapes rejected)
    4. RAG feedback loop         (high-score "store" seller -> marketplace_db.record_store_confirmation)

Hard rules (NON-NEGOTIABLE):
- `price` is explicitly set to null when absent (never omitted).
- `title` length < 5                                         -> reject.
- `url` missing                                              -> reject.
- fuzzy artist match below threshold AGAINST TITLE+CONTENT   -> reject.
- Merch keyword in title/content                             -> reject.
- Every rejection is logged with a structured `reason`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import ValidationError
from rapidfuzz import fuzz

from app.config import get_settings
from app.db.marketplace_db import (
    StoreConfirmation,
    get_marketplace_db,
    normalize_domain,
)
from app.models.result import ListingResult
from app.models.search_query import SearchResult
from app.pipeline_context import stage_timer

logger = logging.getLogger(__name__)

MAX_AI_BATCH_SIZE = 10
MAX_FINAL_RESULTS = 10

# ---- Deterministic gates ----
MIN_TITLE_LEN = 5
# RapidFuzz partial_ratio ∈ [0, 100]. Against title+content this threshold eliminates
# Taylor-Swift-when-searching-EKV cases while tolerating diacritic variants (EKV vs E.K.V.).
MIN_ARTIST_FUZZY = 60
# Stores below this LLM-reported score (after artist-match gating) are not promoted to RAG memory.
STORE_PERSIST_SCORE_THRESHOLD = 0.5

# Keywords that indicate non-music merch / accessories. Checked against the lowercased
# concatenation of title + first 1500 chars of snippet content.
_REJECTION_KEYWORDS: tuple[str, ...] = (
    "cutting board", "tote bag", "t-shirt", "tshirt", "hoodie", "mug ",
    "poster", "sticker", "keychain", "turntable", "stylus", "cartridge",
    "slipmat", "cleaning kit", "record sleeve", "vinyl sleeves",
    "daska za seckanje", "majica", "šolja", "šoljica", "torba",
    "einkaufstasche", "kissen", "kochmesser",
)


EXTRACTOR_SYSTEM_PROMPT = """
### ROLE
You are Agent 3 (Extractor) for AiCrateDigger. You DO NOT decide what to keep — the caller has
already fuzzy-matched the artist against each snippet and only sends you survivors. Your job is
pure structured extraction.

### STRICT OUTPUT SCHEMA (JSON)
{
  "scores": [
    {
      "url": "string",
      "title": "string",
      "score": 0.0,
      "price": "string|null",
      "location": "string|null",
      "availability": "available|sold_out|unknown",
      "seller_type": "store|private|unknown",
      "match_reason": "string"
    }
  ]
}

### RULES
- Emit one entry per listing in the input. Preserve `url` verbatim.
- `price`: local-currency string with symbol/code if visible (e.g. "1.890 RSD", "€24,90").
  If not visible in the snippet -> `null` (MUST NOT omit).
- `location`: "City[, District]" cleaned of marketplace boilerplate. If not visible -> `null`.
- `availability`: detect sold-out signals across languages:
    en: sold out / out of stock / unavailable
    sr/hr/bs: nema na stanju / rasprodato / prodato
    de: ausverkauft / nicht verfügbar
    fr: épuisé / indisponible
    es: agotado
    it: esaurito
  Mark `sold_out` if any appear; `available` if explicitly in stock; else `unknown`.
- `seller_type`:
    "store"   = professional webshop / record store with its own domain.
    "private" = individual seller on a generic marketplace.
    "unknown" = cannot tell.
- `score` ∈ [0.0, 1.0]: your confidence the snippet represents a genuine listing of the target
  release on the requested physical format. Start at 0.6, bump up for strong evidence (clear
  price + format + city), bump down for vague / index pages (without eliminating them — the
  deterministic scorer handles final ranking).
- `match_reason`: short free-form string. If you spot a listing that slipped past the pre-filter
  but is still clearly wrong (e.g. CD when Vinyl was requested), lower the score and write the
  reason, e.g. `"wrong_format:cd_not_vinyl"`.
""".strip()


def _looks_like_merch(title: str, content: str) -> bool:
    hay = f"{title} {content}".lower()
    return any(kw in hay for kw in _REJECTION_KEYWORDS)


def _clean(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = " ".join(str(raw).split()).strip(" ,;:-")
    return s or None


def _artist_fuzzy_score(artist: str, title: str, content: str) -> float:
    """RapidFuzz partial ratio of `artist` against (title + first 400 chars of content)."""
    haystack = f"{title} {content[:400]}"
    return float(fuzz.partial_ratio(artist.lower(), haystack.lower()))


def _album_fuzzy_score(album: str, title: str, content: str) -> float:
    if not album:
        return 0.0
    haystack = f"{title} {content[:400]}"
    return float(fuzz.partial_ratio(album.lower(), haystack.lower()))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def extract_and_score_results(
    candidates: list[SearchResult],
    artist: str | None,
    album: str | None,
    music_format: str | None,
    country: str | None,
    city: str | None,
) -> list[ListingResult]:
    # Defensive normalization — upstream agents may now pass None for any of
    # these tokens in the partial-intent / bootstrap paths. The extractor is
    # the last gate and must never crash on a missing field.
    artist = (artist or "").strip()
    album = (album or "").strip()
    music_format = (music_format or "").strip()
    country = (country or "").strip()
    with stage_timer(
        "extractor",
        input={"artist": artist, "album": album, "candidates_in": len(candidates)},
    ) as rec:
        if not candidates:
            rec.status = "empty"
            return []

        batch = candidates[:MAX_AI_BATCH_SIZE]
        location_fallback = ", ".join(filter(None, [city, country])) or country

        # -------- PASS 1: deterministic pre-filter --------
        survivors, pre_artist_scores, pre_album_scores = _pre_filter(
            batch=batch, artist=artist, album=album
        )

        if not survivors:
            rec.status = "empty"
            rec.extra["rejected_all"] = True
            return []

        # -------- PASS 2: LLM structured extraction on survivors --------
        llm_output = await _llm_extract(
            survivors=survivors,
            artist=artist,
            album=album,
            music_format=music_format,
            city=city,
            country=country,
        )

        # Map LLM output by URL for quick lookup; anything the LLM silently dropped is rejected.
        llm_by_url: dict[str, dict] = {}
        for item in llm_output:
            u = str(item.get("url") or "").strip()
            if u:
                llm_by_url[u] = item

        # -------- PASS 3: assemble validated ListingResult objects --------
        validated: list[ListingResult] = []
        persist_tasks: list[asyncio.Task[None]] = []
        seen_persist_domains: set[str] = set()
        rejections_post = 0

        for survivor, artist_match, album_match in zip(
            survivors, pre_artist_scores, pre_album_scores, strict=True
        ):
            item = llm_by_url.get(survivor.url)
            if item is None:
                _log_reject(survivor.url, "llm_dropped_silently", artist_match=artist_match)
                rejections_post += 1
                continue

            try:
                llm_score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                llm_score = 0.0

            domain = normalize_domain(survivor.url)

            try:
                listing = ListingResult(
                    url=survivor.url,
                    title=(item.get("title") or survivor.title).strip(),
                    score=min(max(llm_score, 0.0), 1.0),
                    price=_clean(item.get("price")),
                    location=_clean(item.get("location")) or None,
                    availability=_normalize_availability(item.get("availability")),
                    seller_type=_normalize_seller(item.get("seller_type")),
                    domain=domain,
                    artist_match=round(artist_match / 100.0, 3),
                    album_match=round(album_match / 100.0, 3),
                    match_reason=str(item.get("match_reason") or "accepted").strip() or "accepted",
                )
            except ValidationError as exc:
                _log_reject(
                    survivor.url,
                    "validation_failed",
                    artist_match=artist_match,
                    detail=str(exc.errors()[:2]),
                )
                rejections_post += 1
                continue

            validated.append(listing)

            # -------- RAG feedback loop (store confirmation) --------
            if (
                listing.seller_type == "store"
                and listing.score >= STORE_PERSIST_SCORE_THRESHOLD
                and domain
                and domain not in seen_persist_domains
            ):
                seen_persist_domains.add(domain)
                confirmation = StoreConfirmation(
                    url=listing.url,
                    title=listing.title,
                    location=listing.location or location_fallback,
                    tavily_score=survivor.score,
                    artist=artist,
                    album=album,
                    music_format=music_format,
                    content=survivor.content,
                )
                logger.info(
                    "rag_store_queued",
                    extra={
                        "stage": "rag_store",
                        "status": "success",
                        "domain": domain,
                        "url": listing.url,
                    },
                )
                persist_tasks.append(asyncio.create_task(_safe_persist(confirmation)))

        # -------- Await persist tasks so writes are confirmed before return --------
        if persist_tasks:
            persist_results = await asyncio.gather(*persist_tasks, return_exceptions=True)
            persisted = sum(1 for r in persist_results if not isinstance(r, Exception))
            logger.info(
                "rag_store_done",
                extra={
                    "stage": "rag_store",
                    "status": "success" if persisted else "empty",
                    "count": persisted,
                },
            )

        final = sorted(validated, key=lambda x: x.score, reverse=True)[:MAX_FINAL_RESULTS]
        rec.output = {
            "kept": len(final),
            "rejected_pre": len(batch) - len(survivors),
            "rejected_post": rejections_post,
        }
        rec.status = "success" if final else "empty"
        return final


# ---------------------------------------------------------------------------
# Pre-filter / LLM / helpers
# ---------------------------------------------------------------------------

def _pre_filter(
    *,
    batch: list[SearchResult],
    artist: str,
    album: str,
) -> tuple[list[SearchResult], list[float], list[float]]:
    survivors: list[SearchResult] = []
    artist_scores: list[float] = []
    album_scores: list[float] = []
    for cand in batch:
        if not cand.url.strip():
            _log_reject(cand.url, "missing_url")
            continue
        title = (cand.title or "").strip()
        if len(title) < MIN_TITLE_LEN:
            _log_reject(cand.url, "title_too_short", detail=title)
            continue
        if _looks_like_merch(title, cand.content):
            _log_reject(cand.url, "merch_keyword", detail=title)
            continue

        # Partial-intent safety: without an artist anchor we cannot run the
        # fuzzy-match gate, so we let Tavily's own score be the only signal.
        # This path corresponds to `intent_completeness in {"partial","unknown"}`
        # where artist failed to extract. We still reject obvious garbage (no url,
        # short title, merch keywords) upstream.
        if artist.strip():
            a_score = _artist_fuzzy_score(artist, title, cand.content)
            if a_score < MIN_ARTIST_FUZZY:
                _log_reject(
                    cand.url, "artist_mismatch", artist_match=a_score, detail=f"title={title!r}"
                )
                continue
        else:
            # No artist to match against — assign neutral mid-score so ranking
            # still works downstream but no quality assertion is made.
            a_score = 50.0

        b_score = _album_fuzzy_score(album, title, cand.content)
        survivors.append(cand)
        artist_scores.append(a_score)
        album_scores.append(b_score)
    return survivors, artist_scores, album_scores


async def _llm_extract(
    *,
    survivors: list[SearchResult],
    artist: str,
    album: str,
    music_format: str,
    city: str | None,
    country: str,
) -> list[dict]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    payload = {
        "target_artist": artist,
        "target_album": album,
        "target_format": music_format,
        "target_city": city,
        "target_country": country,
        "listings": [
            {"url": r.url, "title": r.title, "content": r.content[:1500]} for r in survivors
        ],
    }
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        logger.exception("extractor_llm_failed", extra={"stage": "extractor", "status": "fail"})
        return []

    scores_list = data.get("scores", []) or []
    return [item for item in scores_list if isinstance(item, dict)]


def _normalize_availability(value) -> str:
    v = str(value or "").lower().strip()
    return v if v in {"available", "sold_out", "unknown"} else "unknown"


def _normalize_seller(value) -> str:
    v = str(value or "").lower().strip()
    return v if v in {"store", "private", "unknown"} else "unknown"


def _log_reject(url: str, reason: str, *, artist_match: float | None = None, detail: str | None = None) -> None:
    domain = urlparse(url).netloc.lower() if url else ""
    extras: dict = {
        "stage": "extractor",
        "status": "fail",
        "reason": reason,
        "url": url,
        "domain": domain,
    }
    if artist_match is not None:
        extras["artist_match"] = round(artist_match, 2)
    if detail:
        extras["detail"] = detail
    logger.info("extractor_reject", extra=extras)


async def _safe_persist(confirmation: StoreConfirmation) -> None:
    """Background task wrapper — must never propagate to the request scope."""
    try:
        await get_marketplace_db().record_store_confirmation(confirmation)
    except Exception:
        logger.exception(
            "rag_store_failed",
            extra={
                "stage": "rag_store",
                "status": "fail",
                "url": confirmation.url,
            },
        )
