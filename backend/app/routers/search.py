from fastapi import APIRouter, HTTPException

from app.agents.parser import parse_user_input
from app.agents.query_generator import generate_search_queries
from app.models.search_query import ParseRequest, ParsedQuery, SearchQueries

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
