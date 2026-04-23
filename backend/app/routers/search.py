import logging

from fastapi import APIRouter, HTTPException

from app.agents.extractor import extract_and_score_results
from app.agents.parser import parse_user_input
from app.agents.query_completion import complete_query
from app.agents.query_generator import generate_search_queries
from app.config import get_settings
from app.models.search_query import (
    ParseRequest,
    ParsedQuery,
    SearchQueries,
    SearchResponse,
)
from app.pipeline_context import start_pipeline
from app.services.tavily_service import TavilyIntent, run_tavily_search

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.post("/parse", response_model=ParsedQuery)
async def parse_search_query(payload: ParseRequest) -> ParsedQuery:
    with start_pipeline(debug=get_settings().debug):
        try:
            return await parse_user_input(query=payload.query)
        except Exception as exc:
            logger.exception("parse_endpoint_failed", extra={"stage": "parser", "status": "fail"})
            raise HTTPException(status_code=502, detail=f"Parse failed: {exc}") from exc


@router.post("/generate-queries", response_model=SearchQueries)
async def generate_queries(payload: ParseRequest) -> SearchQueries:
    with start_pipeline(debug=get_settings().debug):
        try:
            parsed_data = await parse_user_input(query=payload.query)
            enriched = await complete_query(parsed_data)
            return await generate_search_queries(parsed_data=enriched)
        except Exception as exc:
            logger.exception(
                "generate_queries_endpoint_failed",
                extra={"stage": "query_gen", "status": "fail"},
            )
            raise HTTPException(status_code=502, detail=f"Query generation failed: {exc}") from exc


@router.post("/search-listings", response_model=SearchResponse)
async def search_listings(payload: ParseRequest) -> SearchResponse:
    with start_pipeline(debug=get_settings().debug) as ctx:
        try:
            parsed_data = await parse_user_input(query=payload.query)
            enriched = await complete_query(parsed_data)
            gen_queries = await generate_search_queries(parsed_data=enriched)

            # Tavily feedback needs at least an artist OR some tokens to be
            # useful as an intent anchor. If artist is missing we pass a
            # location-only hint — the RAG still ingests whatever domains show
            # up, just with lower mention-density signal.
            tavily_intent = TavilyIntent(
                artist=enriched.artist or "",
                album=enriched.effective_album,
                music_format=enriched.format or "",
                location_hint=enriched.city or enriched.country,
            )
            raw_candidates = await run_tavily_search(
                queries=gen_queries.queries,
                intent=tavily_intent,
            )
            final_results = await extract_and_score_results(
                candidates=raw_candidates,
                artist=enriched.artist,
                album=enriched.effective_album,
                music_format=enriched.format,
                country=enriched.country,
                city=enriched.city,
            )
            logger.info(
                "pipeline_done",
                extra={
                    "stage": "pipeline",
                    "status": "success",
                    "count": len(final_results),
                    "request_id": ctx.request_id,
                    "intent_completeness": enriched.intent_completeness,
                    "missing_fields": enriched.missing_fields(),
                },
            )
            return SearchResponse(results=final_results)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "search_pipeline_failed",
                extra={"stage": "pipeline", "status": "fail", "request_id": ctx.request_id},
            )
            raise HTTPException(status_code=502, detail=f"Search pipeline failed: {exc}") from exc
