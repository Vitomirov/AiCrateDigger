import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.internal_auth import require_internal_api_secret
from app.core.rate_limiter import ip_rate_limiter
from app.domains.query_parser.parse_user_query import parse_user_query
from app.core.config import get_settings
from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.search_pipeline.models.search_query import ParseRequest, SearchResponse
from app.domains.search_pipeline.pipeline_context import start_pipeline
from app.domains.search_pipeline.vinyl_search import run_vinyl_search

logger = logging.getLogger(__name__)
router = APIRouter(
    tags=["search"],
    dependencies=[
        Depends(require_internal_api_secret),
        Depends(ip_rate_limiter),
    ],
)

_PARSE_FAILED_DETAIL = "Parse could not be completed. Please try again later."
_SEARCH_FAILED_DETAIL = "Search could not be completed. Please try again later."


# -------------------------
# DEBUG / PARSE ONLY ENDPOINT
# -------------------------
@router.post("/parse", response_model=ParsedQuery)
async def parse_search_query(payload: ParseRequest) -> ParsedQuery:
    with start_pipeline(debug=get_settings().debug):
        try:
            parsed = await parse_user_query(payload.query)
            return parsed

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "parse_endpoint_failed",
                extra={"stage": "parser", "status": "fail", "reason": str(exc)[:200]},
            )
            raise HTTPException(
                status_code=502,
                detail=_PARSE_FAILED_DETAIL,
            ) from exc


# -------------------------
# MAIN PIPELINE ENDPOINT (REAL)
# -------------------------
async def _execute_search(
    payload: ParseRequest,
    background_tasks: BackgroundTasks,
) -> SearchResponse:
    """Single round-trip search endpoint.

    Returns the validated listings AND the parser output that drove the
    pipeline, so callers (today: the Next.js UI, including the dev JSON
    inspector) never need a second `/parse` request. The standalone `/parse`
    endpoint remains available for parse-only debug tooling.
    """
    with start_pipeline(debug=get_settings().debug) as ctx:
        try:
            result = await run_vinyl_search(
                payload.query,
                background_tasks=background_tasks,
            )

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
                    "reason": str(exc)[:200],
                },
            )
            raise HTTPException(
                status_code=502,
                detail=_SEARCH_FAILED_DETAIL,
            ) from exc


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: ParseRequest,
    background_tasks: BackgroundTasks,
) -> SearchResponse:
    return await _execute_search(payload, background_tasks)


# -------------------------
# LEGACY COMPATIBILITY (OLD NAME)
# -------------------------
@router.post("/search-listings", response_model=SearchResponse)
async def search_listings(
    payload: ParseRequest,
    background_tasks: BackgroundTasks,
) -> SearchResponse:
    return await _execute_search(payload, background_tasks)
