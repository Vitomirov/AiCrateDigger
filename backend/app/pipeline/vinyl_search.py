"""Deterministic vinyl-search pipeline orchestrator.

Owns the full request lifecycle. Each stage is wrapped in `stage_timer` for
structured tracing. NO business logic in this module — every transformation
lives in the called sub-module. No fallbacks, no error swallowing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.llm.extract_listings import extract_listings
from app.llm.parse_user_query import parse_user_query
from app.models.result import ListingResult
from app.pipeline_context import stage_timer
from app.policies.eu_stores import get_active_stores
from app.policies.search_dsl import build_query_core, build_tavily_core_query
from app.services.discogs_service import resolve_album_by_index
from app.services.tavily_service import run_tavily_for_store_domains
from app.validators.listings import validate_listing

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120
_PREVIEW_N = 3


def _trunc(value: str | None, limit: int = _PREVIEW_LEN) -> str:
    if value is None:
        return ""
    s = str(value)
    return s if len(s) <= limit else f"{s[: limit - 3]}..."


def _sample_urls(raw_results: list[Any], *, n: int = _PREVIEW_N) -> list[str]:
    out: list[str] = []
    for item in raw_results[:n]:
        u = getattr(item, "url", None)
        if u is None and isinstance(item, dict):
            u = item.get("url")
        out.append(_trunc(str(u) if u else "", 200))
    return out


def _preview_listings_dicts(items: list[Any], *, n: int = _PREVIEW_N) -> list[dict[str, str]]:
    previews: list[dict[str, str]] = []
    for it in items[:n]:
        if hasattr(it, "title") and hasattr(it, "url"):
            previews.append(
                {"title": _trunc(it.title), "url": _trunc(it.url, 200)},
            )
        elif isinstance(it, dict):
            previews.append(
                {
                    "title": _trunc(it.get("title")),
                    "url": _trunc(str(it.get("url") or ""), 200),
                },
            )
    return previews


def _listing_to_api_row(listing: Any) -> ListingResult:
    """Map domain ``Listing`` to API ``ListingResult`` (stable for /search)."""
    title = str(listing.title or "")
    if len(title) < 5:
        title = f"{title} · shop"
    price_str = None
    cur = listing.currency or "EUR"
    try:
        if listing.price is not None and float(listing.price) > 0:
            price_str = f"{float(listing.price):.2f} {cur}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        price_str = f"{listing.price} {cur}" if listing.price else None
    return ListingResult(
        url=listing.url,
        title=title,
        score=1.0,
        price=price_str,
        location=None,
        availability="available" if listing.in_stock else "unknown",
        seller_type="store",
        domain=listing.store,
        artist_match=1.0,
        album_match=1.0,
        match_reason="vinyl_pipeline",
    )


def _log_stage(
    stage: str,
    *,
    counts: dict[str, int] | None = None,
    preview: Any = None,
    artist: str | None = None,
    album_title: str | None = None,
    queries: list[str] | None = None,
    status: str = "success",
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "counts": counts or {},
    }
    if preview is not None:
        payload["preview"] = preview
    if artist is not None or album_title is not None or queries is not None:
        payload["keys"] = {
            "artist": artist,
            "album_title": album_title,
            "queries_head": [_trunc(q, 200) for q in (queries or [])[:_PREVIEW_N]],
        }
    logger.info("vinyl_search_pipeline", extra=payload)


async def run_vinyl_search(query: str) -> dict[str, Any]:
    """Run the deterministic vinyl-search pipeline end-to-end.

    Strict step order:
        1. parse        — LLM parses the user query into ParsedQuery.
        2. discogs      — deterministic album resolution (ordinal or literal).
        3. stores       — load active EU vinyl stores from the whitelist.
        4. query_build  — one shared intent string + batched ``include_domains`` (not one Tavily call per store).
        5. tavily       — constrained web search backend.
        6. extract      — LLM extracts Listing JSON from the survivors.
        7. validate     — hard rules: domain allowlist, title match, price.
    """
    settings = get_settings()
    debug_enabled = settings.debug

    queries: list[str] = []
    core_query: str = ""
    store_domains: list[str] = []
    raw_results: list[Any] = []
    listings: list[Any] = []
    validated: list[Any] = []
    album_title: str | None = None
    stores: tuple[Any, ...] = ()

    # 1. Parse
    with stage_timer("parse", input={"query": query}) as rec:
        parsed = await parse_user_query(query)
        rec.output = parsed.model_dump()

    _log_stage(
        "after_parse",
        counts={"fields": 4},
        preview={"parsed": {k: _trunc(str(v) if v is not None else "") for k, v in parsed.model_dump().items()}},
        artist=parsed.artist,
        album_title=parsed.album,
    )

    # 2. Discogs resolution
    with stage_timer(
        "discogs",
        input={
            "artist": parsed.artist,
            "album": parsed.album,
            "album_index": parsed.album_index,
        },
    ) as rec:
        if parsed.album_index is not None:
            resolution = await resolve_album_by_index(
                artist=parsed.artist,
                album_index=parsed.album_index,
            )
            album_title = resolution.album.title if resolution.album else None
            rec.output = {"album": album_title, "confidence": resolution.confidence}
        elif parsed.album:
            album_title = parsed.album
            rec.output = {"album": album_title, "confidence": "literal"}
        else:
            album_title = None
            rec.status = "empty"
            rec.output = {"album": None}

    _log_stage(
        "after_discogs",
        counts={"has_album_title": 1 if album_title else 0},
        preview={"album_title": _trunc(album_title)},
        artist=parsed.artist,
        album_title=album_title,
    )

    if album_title is None:
        out: dict[str, Any] = {"query": query, "results": []}
        if debug_enabled:
            out["debug"] = {
                "queries": [],
                "raw_results_count": 0,
                "extracted_count": 0,
                "validated_count": 0,
            }
        _log_stage(
            "after_early_exit",
            status="empty",
            counts={"raw_results_count": 0, "extracted_count": 0, "validated_count": 0},
            artist=parsed.artist,
            album_title=None,
            queries=[],
        )
        return out

    # 3. Load stores
    with stage_timer("stores") as rec:
        stores = get_active_stores()
        rec.output = {"count": len(stores)}

    _log_stage(
        "after_stores",
        counts={"stores": len(stores)},
        preview={"stores": [{"domain": _trunc(s.domain), "name": _trunc(s.name)} for s in stores[:_PREVIEW_N]]},
        artist=parsed.artist,
        album_title=album_title,
    )

    # 4. Build search intent + allowlisted domains (compression: one core query for Tavily)
    with stage_timer("query_build", input={"album": album_title}) as rec:
        # Tavily: quoted album, no location (avoids skew + junk pages; domains are EU allowlist).
        core_query = build_tavily_core_query(parsed.artist, album_title)
        store_domains = [s.domain for s in stores]
        rec.output = {
            "core_query_len": len(core_query),
            "domains": len(store_domains),
            "user_location_omitted_from_tavily": bool((parsed.location or "").strip()),
        }

    queries = [core_query]

    _log_stage(
        "after_query_build",
        counts={"domains": len(store_domains)},
        preview={
            "core_query": _trunc(core_query, 200),
            "domains_head": [_trunc(d, 80) for d in store_domains[:_PREVIEW_N]],
        },
        artist=parsed.artist,
        album_title=album_title,
        queries=queries,
    )

    # 5. Search (batched Tavily: ceil(|domains| / chunk) HTTP calls)
    with stage_timer("tavily", input={"domains": len(store_domains)}) as rec:
        raw_results = await run_tavily_for_store_domains(core_query, store_domains)
        rec.output = {"count": len(raw_results)}

    _log_stage(
        "after_tavily",
        counts={"raw_results_count": len(raw_results)},
        preview={"sample_urls": _sample_urls(raw_results)},
        artist=parsed.artist,
        album_title=album_title,
        queries=queries,
    )

    # 6. Extract
    with stage_timer("extract", input={"raw_count": len(raw_results)}) as rec:
        extract_report = await extract_listings(
            [r.model_dump() for r in raw_results],
            artist=parsed.artist,
            album=album_title,
            allowed_domains=set(store_domains),
        )
        listings = extract_report.listings
        rec.output = {"count": len(listings), **extract_report.diagnostic}

    if len(listings) == 0:
        diag = extract_report.diagnostic
        reason = diag.get("empty_reason") or "unknown"
        parts = [
            f"empty_reason={reason}",
            f"prefilter_candidates={diag.get('prefilter_candidates')}",
            f"llm_rows={diag.get('llm_rows_returned')}",
            f"json_parse_ok={diag.get('json_parse_ok')}",
            f"drop_url={diag.get('drop_url_not_in_candidates')}",
            f"drop_title_gate={diag.get('drop_title_gate')}",
            f"drop_pydantic={diag.get('drop_pydantic')}",
        ]
        logger.info(
            "vinyl_search_extract_zero_results",
            extra={
                "stage": "vinyl_search",
                "album_title": album_title,
                "diagnostic": diag,
                "summary": "; ".join(str(p) for p in parts),
            },
        )

    _log_stage(
        "after_extract_listings",
        counts={"extracted_count": len(listings)},
        preview={"listings": _preview_listings_dicts(listings)},
        artist=parsed.artist,
        album_title=album_title,
        queries=queries,
    )

    # 7. Validate
    with stage_timer("validate", input={"input_count": len(listings)}) as rec:
        validated = [
            listing
            for listing in listings
            if validate_listing(
                listing.model_copy(
                    update={
                        "validation_artist": parsed.artist,
                        "validation_album": album_title,
                    }
                )
            )
        ]
        rec.output = {"count": len(validated)}

    _log_stage(
        "after_validate",
        counts={"validated_count": len(validated)},
        preview={"validated": _preview_listings_dicts(validated)},
        artist=parsed.artist,
        album_title=album_title,
        queries=queries,
    )

    out = {
        "query": query,
        "results": [_listing_to_api_row(listing) for listing in validated],
    }
    if debug_enabled:
        out["debug"] = {
            "search_core": core_query,
            "search_core_with_location": build_query_core(parsed.artist, album_title, parsed.location),
            "store_domains": list(store_domains),
            "allowed_domains_head": sorted(set(store_domains))[:25],
            "stores_count": len(stores),
            "raw_results_count": len(raw_results),
            "raw_results_preview": [
                {
                    "url": _trunc(str(getattr(r, "url", "") or ""), 240),
                    "title": _trunc(str(getattr(r, "title", "") or "")),
                    "score": float(getattr(r, "score", 0.0) or 0.0),
                }
                for r in raw_results[:8]
            ],
            "extracted_count": len(listings),
            "extract_diagnostic": extract_report.diagnostic,
            "listings_preview": _preview_listings_dicts(listings, n=8),
            "validated_count": len(validated),
            "validated_preview": _preview_listings_dicts(validated, n=8),
        }

    return out
