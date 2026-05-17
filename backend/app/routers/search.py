import logging

from fastapi import APIRouter, HTTPException

from app.agents.parser.parse_user_query import parse_user_query
from app.config import get_settings
from app.domain.parse_schema import ParsedQuery
from app.models.search_query import ParseRequest, SearchResponse
from app.pipeline_context import start_pipeline
from app.pipeline.vinyl_search import run_vinyl_search

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


# -------------------------
# DEBUG / PARSE ONLY ENDPOINT
# -------------------------
@router.post("/parse", response_model=ParsedQuery)
async def parse_search_query(payload: ParseRequest) -> ParsedQuery:
    with start_pipeline(debug=get_settings().debug):
        try:
            parsed = await parse_user_query(payload.query)
            return parsed

        except Exception as exc:
            logger.exception(
                "parse_endpoint_failed",
                extra={"stage": "parser", "status": "fail"},
            )
            raise HTTPException(
                status_code=502,
                detail=f"Parse failed: {exc}",
            ) from exc


# -------------------------
# MAIN PIPELINE ENDPOINT (REAL)
# -------------------------
@router.post("/search", response_model=SearchResponse)
async def search(payload: ParseRequest) -> SearchResponse:
    """Single round-trip search endpoint.

    Returns the validated listings AND the parser output that drove the
    pipeline, so callers (today: the Next.js UI, including the dev JSON
    inspector) never need a second `/parse` request. The standalone `/parse`
    endpoint remains available for parse-only debug tooling.
    """
    with start_pipeline(debug=get_settings().debug) as ctx:
        try:
            result = await run_vinyl_search(payload.query)

            settings = get_settings()
            logger.info(
                "pipeline_done",
                extra={
                    "stage": "pipeline",
                    "status": "success",
                    "request_id": ctx.request_id,
                    "count": len(result.get("results", [])),
                    "reason": result.get("reason"),
                },
            )

            return SearchResponse(
                results=result.get("results", []),
                parsed=result.get("parsed"),
                reason=result.get("reason"),
                debug=ctx.as_debug_payload() if settings.debug else None,
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "search_pipeline_failed",
                extra={
                    "stage": "pipeline",
                    "status": "fail",
                    "request_id": ctx.request_id,
                },
            )
            raise HTTPException(
                status_code=502,
                detail=f"Search pipeline failed: {exc}",
            ) from exc


# -------------------------
# LEGACY COMPATIBILITY (OLD NAME)
# -------------------------
@router.post("/search-listings", response_model=SearchResponse)
async def search_listings(payload: ParseRequest) -> SearchResponse:
    return await search(payload)