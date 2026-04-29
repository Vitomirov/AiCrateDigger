"""Agent 2 — Deterministic EU city-local physical record-shop query strings."""

from __future__ import annotations

import logging

from app.models.search_query import ParsedQuery, QueryGenDebug, SearchQueries
from app.pipeline_context import stage_timer

logger = logging.getLogger(__name__)

# TEMP: disable RAG influence completely (hard override).


def _strip_site_operator(query: str) -> str:
    s = query
    for needle in ("site:", "Site:", "SITE:"):
        s = s.replace(needle, "")
    return " ".join(s.split()).strip()


def generate_local_queries(
    artist: str,
    album: str,
    music_fmt: str,
    city: str,
    country: str,
) -> list[str]:
    """City-local storefront search strings; France template vs English EU fallback."""

    fmt = (music_fmt or "vinyl").strip().lower()
    loc_city = (city or "").strip()
    loc_country = (country or "").strip()

    geo = loc_city or loc_country

    if loc_country.lower() == "france":
        return [
            f"acheter {fmt} {artist} {album} {geo}".strip(),
            f"disquaire {geo} {artist} {album} {fmt}".strip(),
            f"magasin disque {geo} {artist} {album}".strip(),
            f"{artist} {album} {fmt} {geo} magasin".strip(),
        ]

    return [
        f"{artist} {album} {fmt} {geo} record store".strip(),
        f"{artist} {album} {fmt} {geo} cd shop".strip(),
        f"buy {fmt} {artist} {album} {geo}".strip(),
        f"{artist} {album} {geo} vinyl shop".strip(),
    ]


async def generate_search_queries(parsed_data: ParsedQuery) -> SearchQueries:
    with stage_timer(
        "query_gen",
        input=parsed_data.model_dump(exclude={"original_query"}),
    ) as rec:
        rag_domains: list[str] = []
        marketplaces: list[str] = []

        artist = (parsed_data.artist or "").strip()
        album = (parsed_data.effective_album or "").strip()
        music_format = (parsed_data.format or "vinyl").strip()
        city = (parsed_data.city or "").strip()
        country = (parsed_data.country or "").strip()

        queries = generate_local_queries(
            artist=artist,
            album=album,
            music_fmt=music_format,
            city=city,
            country=country,
        )

        queries = [_strip_site_operator(q) for q in queries]
        queries = [q for q in queries if q][:4]

        if len(queries) < 3:
            geo_fallback = (city or country or "").strip()
            fill_pool = (
                " ".join(
                    filter(
                        None,
                        [
                            artist or None,
                            album or None,
                            (music_format or "").strip().lower() or None,
                            geo_fallback,
                            "record store",
                        ],
                    )
                ).strip(),
                _strip_site_operator(
                    f"{artist} {album or ''} {geo_fallback} music shop".strip(),
                ),
                _strip_site_operator(f"{album or artist} {geo_fallback} vinyl shop".strip()),
            )
            for raw in fill_pool:
                if len(queries) >= 3:
                    break
                if raw and raw not in queries:
                    queries.append(raw)

        debug = QueryGenDebug(
            rag_hits=[],
            discovered_marketplaces=[],
            bootstrap_used=False,
        )
        mkt = ([""] * len(queries))[:5]
        result = SearchQueries(
            queries=queries[:5],
            marketplaces=mkt,
            debug=debug,
        )

        rec.output = {
            "queries": result.queries,
            "marketplaces": result.marketplaces,
            "rag_candidates": len(rag_domains),
            "top_domains": marketplaces,
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


__all__ = [
    "generate_local_queries",
    "generate_search_queries",
]
