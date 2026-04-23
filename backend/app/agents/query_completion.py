"""Query Completion Layer — sits between Parser (Agent 1) and Query Generator (Agent 2).

RESPONSIBILITY:
Best-effort enrichment of a partial `ParsedQuery`. NEVER raises, NEVER blocks
the pipeline. If everything fails, the original ParsedQuery is returned and
Agent 2 composes fallback queries from whatever tokens remain.

ENRICHMENT ORDER (per spec):
    1. RAG lookup      — past successful listings that matched the same artist.
    2. Discogs API     — catalog lookup (invoked only when parser already tried the
                         deterministic path; no duplicate calls here).
    3. Tavily probe    — contextual expansion (reserved hook, currently no-op).

HONESTY RULE:
We do NOT guess album titles. Enrichment only promotes an album when we have
deterministic, consensual evidence (e.g. multiple high-score RAG hits naming
the same title). Otherwise we leave `album` null and rely on the downstream
bootstrap search to surface results for the user to narrow down.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from app.db.marketplace_db import get_marketplace_db
from app.models.search_query import ParsedQuery
from app.pipeline_context import stage_timer

logger = logging.getLogger(__name__)

# Consensus thresholds for honest enrichment (tuned conservatively — we'd rather
# leave `album` null than hallucinate).
RAG_CONSENSUS_MIN_AGREEMENT = 2  # at least this many RAG hits must name the same album
RAG_CONSENSUS_MIN_SIMILARITY = 0.35  # filter out weakly-related domains


async def complete_query(parsed: ParsedQuery) -> ParsedQuery:
    """Enrich a partial `ParsedQuery`. Returns the (possibly updated) ParsedQuery.

    Never raises — the pipeline continues regardless of enrichment outcome.
    """
    with stage_timer(
        "query_completion",
        input={
            "intent_completeness": parsed.intent_completeness,
            "missing": parsed.missing_fields(),
        },
    ) as rec:
        # Nothing to do.
        if parsed.intent_completeness == "complete":
            rec.status = "skipped_complete"
            return parsed

        attempts: dict[str, Any] = {}
        enriched = parsed

        # --- Step 1: RAG lookup ----------------------------------------------
        try:
            rag_album, rag_meta = await _rag_album_consensus(parsed)
            attempts["rag"] = rag_meta
            if rag_album and not enriched.has_album:
                enriched = enriched.model_copy(
                    update={
                        "resolved_album": rag_album,
                        "resolution_confidence": "medium",
                        "intent_completeness": "complete" if enriched.has_artist else "partial",
                    }
                )
        except Exception as exc:
            logger.warning(
                "query_completion_rag_error",
                extra={
                    "stage": "query_completion",
                    "status": "fail",
                    "source": "rag",
                    "reason": str(exc),
                    "fallback_triggered": True,
                },
            )
            attempts["rag"] = {"error": str(exc)}

        # --- Step 2: Discogs -------------------------------------------------
        # The parser already invokes Discogs when an album_index is available; doing
        # it again here without fresh signal would just repeat the same miss. We
        # document the hook and skip.
        attempts["discogs"] = {"skipped": "handled_by_parser"}

        # --- Step 3: Tavily contextual expansion -----------------------------
        # Reserved for a future "probe Tavily with an artist-only query, then
        # frequency-rank candidate album titles across snippets". Skipped today
        # because the cost is non-trivial and results are noisy without a
        # per-artist listings memory.
        attempts["tavily"] = {"skipped": "not_implemented"}

        rec.output = {
            "intent_before": parsed.intent_completeness,
            "intent_after": enriched.intent_completeness,
            "enriched": enriched is not parsed,
            "attempts": attempts,
        }
        rec.status = "enriched" if enriched is not parsed else "passthrough"
        logger.info(
            "query_completion_done",
            extra={
                "stage": "query_completion",
                "status": rec.status,
                "intent_completeness": enriched.intent_completeness,
                "missing_fields": enriched.missing_fields(),
                "fallback_triggered": enriched is parsed and parsed.intent_completeness != "complete",
                "output": rec.output,
            },
        )
        return enriched


# ---------------------------------------------------------------------------
# RAG consensus (honest — only enriches on agreement)
# ---------------------------------------------------------------------------

async def _rag_album_consensus(parsed: ParsedQuery) -> tuple[str | None, dict[str, Any]]:
    """Return an album title ONLY if multiple sufficiently-similar RAG entries
    agree on the same value. Otherwise `(None, meta)`.

    We use `sample_title` from each candidate as a weak proxy. This is an honest,
    conservative proxy — it promotes an album only when several domains observed
    listings with the same album title in their snippets.
    """
    if not parsed.has_artist:
        return None, {"skipped": "no_artist_anchor"}

    intent_parts = [parsed.artist or "", parsed.format or "", parsed.city or "", parsed.country or ""]
    intent_text = " ".join(p for p in intent_parts if p).strip()
    if not intent_text:
        return None, {"skipped": "empty_intent_text"}

    service = get_marketplace_db()
    candidates = await service.retrieve_candidates(intent_text=intent_text, k=20)

    # Keep only candidates above the similarity floor so we don't vote on noise.
    filtered = [c for c in candidates if c.similarity >= RAG_CONSENSUS_MIN_SIMILARITY]
    if len(filtered) < RAG_CONSENSUS_MIN_AGREEMENT:
        return None, {
            "hits": len(candidates),
            "above_sim_floor": len(filtered),
            "consensus": None,
        }

    artist_lc = (parsed.artist or "").strip().lower()
    votes: Counter[str] = Counter()
    for c in filtered:
        title = (c.sample_title or "").strip()
        if not title:
            continue
        lc = title.lower()
        # We only accept a candidate album title if the artist name appears in
        # the same sample_title — that's our weak "this title belongs to this
        # artist" anchor. Without it, titles are ambiguous.
        if artist_lc and artist_lc not in lc:
            continue
        # Strip the artist + common separators to isolate the album segment.
        album_guess = _extract_album_segment(title, artist_lc)
        if album_guess:
            votes[album_guess.lower()] += 1

    if not votes:
        return None, {
            "hits": len(candidates),
            "above_sim_floor": len(filtered),
            "consensus": None,
            "reason": "no_artist_anchored_titles",
        }

    top_key, top_count = votes.most_common(1)[0]
    if top_count < RAG_CONSENSUS_MIN_AGREEMENT:
        return None, {
            "hits": len(candidates),
            "above_sim_floor": len(filtered),
            "top_vote": top_count,
            "consensus": None,
        }

    # Return the most-voted key in a reasonably-cased form (first original occurrence).
    for c in filtered:
        title = (c.sample_title or "").strip()
        guess = _extract_album_segment(title, artist_lc)
        if guess and guess.lower() == top_key:
            return guess, {
                "hits": len(candidates),
                "above_sim_floor": len(filtered),
                "top_vote": top_count,
                "consensus": guess,
            }
    return top_key, {
        "hits": len(candidates),
        "above_sim_floor": len(filtered),
        "top_vote": top_count,
        "consensus": top_key,
    }


def _extract_album_segment(title: str, artist_lc: str) -> str | None:
    """Cheap artist-strip heuristic. Returns the non-artist segment, cleaned up.

    Intentionally conservative — if we can't confidently isolate a segment, we
    return None rather than guessing. This keeps us on the "do not hallucinate"
    side of the line.
    """
    if not title:
        return None
    if not artist_lc:
        return None
    lc = title.lower()
    idx = lc.find(artist_lc)
    if idx < 0:
        return None
    # Remove the artist prefix and common separators.
    tail = title[idx + len(artist_lc):]
    for sep in [" - ", " – ", " — ", ": ", " | ", " / "]:
        if tail.startswith(sep):
            tail = tail[len(sep):]
            break
    else:
        # No separator -> we aren't confident where the album actually starts.
        tail = tail.strip()
    tail = tail.strip(" -–—:|/\"'()[]")
    # Trim trailing format/condition chatter that often follows an album.
    for noise in (" vinyl", " lp", " cd", " cassette", " mint", " new", " ep"):
        pos = tail.lower().find(noise)
        if pos > 0:
            tail = tail[:pos].strip()
    # A legitimate album title is typically ≥ 2 chars and ≤ 80.
    if 2 <= len(tail) <= 80:
        return tail
    return None
