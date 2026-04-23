"""Agent 2 — Global Marketplace Discovery Compiler.

The generator is NOT a writer, classifier, or country router.
It is a retrieval + composition pipeline that:

    1. Embeds the user's intent.
    2. Queries the emergent `marketplaces` RAG (marketplace_db).
    3. If RAG coverage ≥ threshold        -> emit deterministic `site:{domain}` queries.
    4. If RAG coverage below threshold    -> emit neutral bootstrap/discovery queries
                                            so that Tavily can DISCOVER new domains,
                                            which get ingested back into RAG.
    5. Optionally asks a minimal LLM composer to phrase one or two additional
       queries using ONLY tokens provided by the user's original input (artist,
       album, format, city, country) in the user's detected language.

Hard constraints enforced here (validated by tests / logs, not comments alone):
    - NO hardcoded countries, marketplaces, domains, regional lists.
    - NO US-giant exclusions. Excluding domains is a regional assumption.
    - NO local-language dictionaries. The LLM receives the language name only.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.db.marketplace_db import MarketplaceSignal, get_marketplace_db
from app.models.search_query import ParsedQuery, QueryGenDebug, SearchQueries
from app.pipeline_context import stage_timer

logger = logging.getLogger(__name__)

# Scheduling constants.
MAX_TOTAL_QUERIES = 5
MIN_TOTAL_QUERIES = 3
# Retrieval width — we pull a wide pool from RAG and let the emergent score pick.
RAG_FETCH_K = 20
# Below this effective signal count we consider ourselves in BOOTSTRAP mode.
RAG_BOOTSTRAP_THRESHOLD = 2
# How many deterministic site: queries we emit when RAG is rich.
MAX_SITE_QUERIES = 3
# An emergent-score floor below which a RAG hit is not strong enough to drive a
# `site:` query on its own. Everything below goes into the discovery pool.
MIN_EMERGENT_SCORE_FOR_SITE = 0.10


# ---------------------------------------------------------------------------
# Neutral LLM composer prompt (kept deliberately minimal).
# ---------------------------------------------------------------------------

COMPOSER_SYSTEM_PROMPT = """
You are a neutral search-phrase composer. You will receive a set of tokens and a language name.
Return `N` natural-sounding search phrases in that language, EACH combining the provided tokens.

YOU MUST:
- Use ONLY the tokens explicitly provided. Do not introduce brand names, domains, or marketplaces
  that were not in the input.
- Do not use `site:` or any other search operators.
- Do not add negative exclusions (`-site:`, `-foo`, etc).
- Write in the language name given to you. If you do not know the language, fall back to the
  tokens exactly as provided.

Return strict JSON of the form:
  {"queries": ["string", "string", ...]}
""".strip()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def generate_search_queries(parsed_data: ParsedQuery) -> SearchQueries:
    with stage_timer(
        "query_gen",
        input=parsed_data.model_dump(exclude={"original_query"}),
    ) as rec:
        # Normalize every field to a safe string up front. Partial intents are
        # the default contract now — NOTHING in this function is allowed to
        # crash on a missing token.
        artist = (parsed_data.artist or "").strip()
        album_label = parsed_data.effective_album  # "" when truly unknown
        music_format = (parsed_data.format or "").strip()
        city = (parsed_data.city or "").strip()
        country = (parsed_data.country or "").strip()
        language = parsed_data.language
        original_query = (parsed_data.original_query or "").strip()
        intent_completeness = parsed_data.intent_completeness

        # 1. Retrieve emergent marketplace candidates. Intent text uses whatever
        #    tokens we have — this is NEVER empty because we fall back to the
        #    user's raw query text when every structured field is missing.
        intent_text = _build_intent_text(
            artist=artist,
            album_label=album_label,
            music_format=music_format,
            city=city,
            country=country,
        ) or original_query
        candidates = await _retrieve_rag_candidates(intent_text=intent_text)

        # 2. Pick top candidates by emergent score; decide bootstrap vs RAG mode.
        top_candidates = [
            c for c in candidates if c.emergent_score >= MIN_EMERGENT_SCORE_FOR_SITE
        ][:MAX_SITE_QUERIES]
        bootstrap_used = len(top_candidates) < RAG_BOOTSTRAP_THRESHOLD

        # 3. Build queries — purely deterministic + optional minimal LLM composition.
        deterministic_q, deterministic_m = _build_deterministic_queries(
            candidates=top_candidates,
            artist=artist,
            album_label=album_label,
            music_format=music_format,
            bootstrap=bootstrap_used,
        )

        remaining_slots = max(MAX_TOTAL_QUERIES - len(deterministic_q), 0)
        llm_target = max(remaining_slots, MIN_TOTAL_QUERIES - len(deterministic_q), 0)

        llm_q: list[str] = []
        if llm_target > 0:
            llm_q = await _compose_neutral_queries(
                n=llm_target,
                artist=artist,
                album_label=album_label,
                music_format=music_format,
                city=city,
                country=country,
                language=language,
            )

        # 4. Merge + dedup + enforce [3, 5]. If we're still short, fill with
        #    deterministic token-only floors and finally fall back to the raw
        #    original query (the last-ditch "search the user's own phrase").
        merged_q, merged_m = _merge_unique(
            deterministic_q, deterministic_m, llm_q, [""] * len(llm_q)
        )

        if len(merged_q) < MIN_TOTAL_QUERIES:
            floors = _build_floor_queries(
                artist=artist,
                album_label=album_label,
                music_format=music_format,
                city=city,
                country=country,
                original_query=original_query,
            )
            merged_q, merged_m = _merge_unique(
                merged_q, merged_m, floors, [""] * len(floors), cap=MAX_TOTAL_QUERIES
            )

        # Absolute floor: if EVERYTHING above failed to produce even one query,
        # emit the user's raw text three times with trivial variations. This is
        # the "pipeline must never crash on incomplete intent" contract.
        if len(merged_q) < MIN_TOTAL_QUERIES:
            last_ditch = _build_last_ditch_queries(original_query)
            merged_q, merged_m = _merge_unique(
                merged_q, merged_m, last_ditch, [""] * len(last_ditch), cap=MIN_TOTAL_QUERIES
            )

        # 5. Wrap in SearchQueries with a full debug envelope.
        debug = QueryGenDebug(
            rag_hits=[c.as_debug() for c in candidates[:10]],
            discovered_marketplaces=[c.domain for c in top_candidates],
            bootstrap_used=bootstrap_used,
        )
        result = SearchQueries(
            queries=merged_q[:MAX_TOTAL_QUERIES],
            marketplaces=merged_m[:MAX_TOTAL_QUERIES],
            debug=debug,
        )

        rec.output = {
            "queries": result.queries,
            "marketplaces": result.marketplaces,
            "bootstrap_used": bootstrap_used,
            "rag_candidates": len(candidates),
            "top_domains": [c.domain for c in top_candidates],
            "intent_completeness": intent_completeness,
        }
        rec.status = "success" if result.queries else "empty"
        logger.info(
            "queries_ready_for_tavily",
            extra={
                "stage": "query_gen",
                "status": rec.status,
                "count": len(result.queries),
                "output": rec.output,
            },
        )
        return result


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

async def _retrieve_rag_candidates(*, intent_text: str) -> list[MarketplaceSignal]:
    with stage_timer(
        "rag_retrieve",
        input={"intent_text": intent_text[:200]},
    ) as rec:
        try:
            service = get_marketplace_db()
            candidates = await service.retrieve_candidates(intent_text=intent_text, k=RAG_FETCH_K)
        except Exception as exc:
            rec.status = "fail"
            rec.error = str(exc)
            logger.exception(
                "rag_retrieve_failed",
                extra={"stage": "rag_retrieve", "status": "fail"},
            )
            return []
        rec.output = {
            "count": len(candidates),
            "top": [c.as_debug() for c in candidates[:5]],
        }
        rec.status = "success" if candidates else "empty"
        return candidates


# ---------------------------------------------------------------------------
# Deterministic query building (no LLM, no regional logic)
# ---------------------------------------------------------------------------

def _build_intent_text(
    *,
    artist: str,
    album_label: str,
    music_format: str,
    city: str,
    country: str,
) -> str:
    parts = [artist, album_label, music_format, city, country]
    return " ".join(p for p in parts if p).strip()


def _build_deterministic_queries(
    *,
    candidates: list[MarketplaceSignal],
    artist: str,
    album_label: str,
    music_format: str,
    bootstrap: bool,
) -> tuple[list[str], list[str]]:
    """Emit `site:{domain}` queries for each top emergent candidate.
    In bootstrap mode, return empty — the LLM composer + floor queries handle the
    discovery phase without any `site:` bias.

    Partial intents allowed: any combination of artist/album/format may be missing.
    We only emit the tokens that actually exist, so we never produce orphan `""` quotes.
    """
    if bootstrap or not candidates:
        return [], []

    body = _compose_body(artist=artist, album_label=album_label, music_format=music_format)
    if not body:
        # No structured tokens at all; don't try to pin site: queries to nothing.
        return [], []

    queries: list[str] = []
    marketplaces: list[str] = []
    for signal in candidates:
        if not signal.domain:
            continue
        queries.append(f"site:{signal.domain} {body}")
        marketplaces.append(signal.domain)
    return queries, marketplaces


def _build_floor_queries(
    *,
    artist: str,
    album_label: str,
    music_format: str,
    city: str,
    country: str,
    original_query: str,
) -> list[str]:
    """Minimal deterministic fallbacks that use ONLY the user's own tokens.

    Contain no marketplace names, no domains, no regional vocabulary.
    Robust to ANY combination of missing fields.
    """
    body = _compose_body(artist=artist, album_label=album_label, music_format=music_format)
    floors: list[str] = []

    if body:
        if city and country:
            floors.append(f"{body} {city} {country}")
        elif country:
            floors.append(f"{body} {country}")
        floors.append(body)
        if city:
            floors.append(f"{body} {city}")

    # Artist-only fallback for partial intents. This is the "The Doors vinyl
    # Belgrade" case — no album known, but we still produce a sensible query.
    if artist and not album_label:
        tokens = [artist]
        if music_format:
            tokens.append(music_format)
        if city:
            tokens.append(city)
        floors.append(" ".join(tokens))

    # If we STILL have nothing (e.g. parser returned a shell), fall back to the
    # user's own input as a final honest probe.
    if not floors and original_query:
        floors.append(original_query)

    return floors


def _build_last_ditch_queries(original_query: str) -> list[str]:
    """Absolute bottom of the barrel — produce SOMETHING so the pipeline runs.

    If we got here, parsing failed at every level. We'll probe Tavily with the
    user's raw query plus two mild variations. This still feeds the emergent
    RAG so the next run does better.
    """
    q = (original_query or "").strip()
    if not q:
        return []
    return [q, f"{q} vinyl", f"{q} buy"]


def _compose_body(
    *, artist: str, album_label: str, music_format: str
) -> str:
    """Build a query body from whichever tokens exist. Empty string when nothing."""
    parts: list[str] = []
    if artist:
        parts.append(f'"{artist}"')
    if album_label:
        parts.append(f'"{album_label}"')
    if music_format:
        parts.append(music_format)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Neutral LLM composer (minimal role)
# ---------------------------------------------------------------------------

async def _compose_neutral_queries(
    *,
    n: int,
    artist: str,
    album_label: str,
    music_format: str,
    city: str,
    country: str,
    language: str,
) -> list[str]:
    if n <= 0:
        return []

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Only pass tokens that actually have values. If the user gave no album,
    # the composer MUST NOT see an "album" key at all — otherwise a sloppy model
    # may hallucinate one.
    tokens: dict[str, str] = {}
    if artist:
        tokens["artist"] = artist
    if album_label:
        tokens["album"] = album_label
    if music_format:
        tokens["format"] = music_format
    if city:
        tokens["city"] = city
    if country:
        tokens["country"] = country

    if not tokens:
        # Nothing structured to compose over — skip the LLM entirely.
        return []

    user_content = (
        f"Language for output: {language}\n"
        f"Number of phrases to produce: {n}\n"
        f"Tokens (use ONLY these; do not introduce anything else): "
        f"{json.dumps(tokens, ensure_ascii=False)}\n"
        f"Produce {n} natural search phrases in the specified language. "
        f"Return strict JSON with key 'queries'."
    )

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        data = json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning(
            "query_composer_llm_failed",
            extra={"stage": "query_gen", "status": "fail", "reason": str(exc)},
        )
        return []

    raw = data.get("queries", []) or []
    out: list[str] = []
    for q in raw:
        s = str(q or "").strip()
        if not s:
            continue
        # Safety guard: the composer is forbidden from emitting `site:` or `-site:`.
        # If it sneaks one in (model drift, prompt violation), we reject the query
        # outright rather than trust an unverified domain reference.
        if _has_any_site_operator(s):
            logger.warning(
                "query_composer_emitted_site_operator",
                extra={
                    "stage": "query_gen",
                    "status": "fail",
                    "reason": "composer_violation",
                    "output": s,
                },
            )
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def _has_any_site_operator(query: str) -> bool:
    tokens = query.lower().split()
    return any(tok.startswith("site:") or tok.startswith("-site:") for tok in tokens)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_unique(
    a_q: list[str],
    a_m: list[str],
    b_q: list[str],
    b_m: list[str],
    *,
    cap: int = MAX_TOTAL_QUERIES,
) -> tuple[list[str], list[str]]:
    merged_q: list[str] = []
    merged_m: list[str] = []
    seen: set[str] = set()
    for q, m in zip([*a_q, *b_q], [*a_m, *b_m], strict=True):
        key = q.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged_q.append(q)
        merged_m.append(m)
        if len(merged_q) >= cap:
            break
    return merged_q, merged_m
