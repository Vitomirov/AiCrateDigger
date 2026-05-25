"""Single high-yield Tavily request: 1 credit, up to ``max_results`` rows.

Replaces the legacy multi-stage Tavily loop (general → city tier → site-specific
fanout, ~4+ HTTP calls per request) with **one** advanced-depth call capped by
``Settings.tavily_single_call_max_results`` (default **10**) that returns a
diverse European candidate pool for one Tavily credit.

* No ``include_domains`` restriction — diversity across European shops is then
  enforced in Python by :mod:`app.domains.engine.search.prefilter`.
* Optional ``country`` field hints Tavily towards the resolved country without
  collapsing the result set to a single geography.

Honours the request-scoped :class:`app.domains.engine.search.TavilyCircuitBreaker` —
when Tavily hard-throttles the account, the breaker trips and the helper
returns an empty list immediately so the caller can degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.engine.search.client import fetch_tavily_results_body
from app.domains.engine.search.constants import REQUEST_TIMEOUT_SECONDS
from app.domains.engine.search.country_boost import tavily_country_from_iso3166_alpha2

logger = logging.getLogger(__name__)


def _quote_phrase(value: str | None) -> str | None:
    """Return ``"value"`` (phrase-quoted) when ``value`` is non-empty, else ``None``."""
    s = (value or "").strip()
    if not s:
        return None
    return f'"{s}"' if '"' not in s else s


def build_consolidated_query(
    *,
    artist: str | None,
    album: str | None,
    format_token: str,
    country_code: str | None,
    resolved_city: str | None = None,
) -> str:
    """Build the single Tavily query string.

    Spec: ``"{artist}" "{album}" {format_type} shop {city?} {country_code?}``.

    Both artist and album are double-quoted for phrase match so Tavily ranks
    PDPs that literally name the release first. ``shop`` is appended as a
    storefront-intent keyword. When ``resolved_city`` is set (city-level geo),
    the city token is injected before the ISO country code so web search
    respects local constraints. The structured ``country`` field on the Tavily
    payload provides an additional ranking nudge.
    """
    parts: list[str] = []
    artist_q = _quote_phrase(artist)
    album_q = _quote_phrase(album)
    if artist_q:
        parts.append(artist_q)
    if album_q:
        parts.append(album_q)
    fmt = (format_token or "vinyl").strip().lower() or "vinyl"
    parts.append(fmt)
    parts.append("shop")
    city = (resolved_city or "").strip()
    if city:
        parts.append(city)
    cc = (country_code or "").strip().upper()
    if cc:
        parts.append(cc)
    return " ".join(parts).strip()


async def run_consolidated_tavily_search(
    *,
    artist: str | None,
    album: str | None,
    format_token: str,
    country_code: str | None,
    resolved_city: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """One advanced-depth Tavily call → raw result dicts (no Python filtering yet).

    The caller (the pipeline) is expected to feed the return value into
    :func:`app.domains.engine.search.prefilter.prefilter_tavily_results` before the
    LLM extractor.
    """
    settings = get_settings()
    query = build_consolidated_query(
        artist=artist,
        album=album,
        format_token=format_token,
        country_code=country_code,
        resolved_city=resolved_city,
    )
    if not query:
        return []

    max_r = int(max_results or settings.tavily_single_call_max_results)
    depth = (settings.tavily_single_call_depth or "advanced").strip().lower() or "advanced"

    payload: dict[str, str | int] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": depth,
        "max_results": max_r,
    }
    tav_country = tavily_country_from_iso3166_alpha2(country_code)
    if tav_country:
        payload["topic"] = "general"
        payload["country"] = tav_country

    with stage_timer(
        "tavily",
        input={
            "query": query,
            "search_depth": depth,
            "max_results": max_r,
            "country": tav_country,
        },
    ) as rec:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            data = await fetch_tavily_results_body(
                client,
                payload,
                query_for_log=query,
            )
        if data is None:
            rec.status = "fail"
            rec.error = "tavily_request_failed_or_circuit_open"
            logger.warning(
                "tavily_single_call_failed",
                extra={"stage": "tavily", "query": query[:200]},
            )
            return []

        raw_items = data.get("results", []) or []
        cleaned: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            cleaned.append(
                {
                    "url": url,
                    "title": str(item.get("title") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "score": float(item.get("score") or 0.0),
                }
            )

        rec.output = {"raw_count": len(raw_items), "kept_count": len(cleaned)}
        rec.status = "success" if cleaned else "empty"
        logger.info(
            "tavily_single_call_done",
            extra={
                "stage": "tavily",
                "query": query[:200],
                "raw_count": len(raw_items),
                "kept_count": len(cleaned),
                "country": tav_country,
            },
        )
        return cleaned


__all__ = [
    "build_consolidated_query",
    "run_consolidated_tavily_search",
]
