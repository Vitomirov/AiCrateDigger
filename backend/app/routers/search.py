from fastapi import APIRouter, HTTPException

from app.agents.parser import parse_user_input
from app.agents.query_generator import generate_search_queries
from app.agents.searcher import run_tavily_search
from app.models.search_query import (
    ParseRequest,
    ParsedQuery,
    SearchQueries,
    SearchResponse,
)

router = APIRouter(tags=["search"])


@router.post("/parse", response_model=ParsedQuery)
async def parse_search_query(payload: ParseRequest) -> ParsedQuery:
    try:
        return await parse_user_input(query=payload.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate-queries", response_model=SearchQueries)
async def generate_queries(payload: ParseRequest) -> SearchQueries:
    try:
        parsed_data = await parse_user_input(query=payload.query)
        return await generate_search_queries(parsed_data=parsed_data)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/search-listings", response_model=SearchResponse)
async def search_listings(payload: ParseRequest) -> SearchResponse:
    try:
        parsed_data = await parse_user_input(query=payload.query)
        generated_queries = await generate_search_queries(parsed_data=parsed_data)
        results = await run_tavily_search(
            queries=generated_queries.queries,
            artist=parsed_data.artist,
            album=parsed_data.album,
            music_format=parsed_data.format,
            source_query=parsed_data.original_query,
            country=parsed_data.country,
            city=parsed_data.city,
        )
        return SearchResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
