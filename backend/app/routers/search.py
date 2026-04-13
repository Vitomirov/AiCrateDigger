from fastapi import APIRouter, HTTPException

from app.agents.parser import parse_user_input
from app.models.search_query import ParsedQuery, ParseRequest

router = APIRouter(tags=["search"])


@router.post("/parse", response_model=ParsedQuery)
async def parse_search_query(payload: ParseRequest) -> ParsedQuery:
    try:
        return await parse_user_input(query=payload.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
