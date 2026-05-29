from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.domains.query_parser.parse_schema import ParsedQuery as PipelineParsedQuery
from app.domains.search_pipeline.models.result import ListingResult

#: Machine-readable reasons emitted by `/search` when the pipeline returned
#: zero results without running Tavily (so the UI can render a precise hint
#: instead of guessing from an empty list). Extend conservatively — every new
#: code is a public API surface for the frontend / clients.
SearchEmptyReason = Literal["album_unresolved"]


class ParseRequest(BaseModel):
    # Example surfaces in /docs so manual testers don't post `{}` and get a 422.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "The Doors Strange Days vinyl in Belgrade"},
                {"query": "Tool 10,000 Days vinyl in Belgrade"},
            ]
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Raw natural-language user query (any language).",
    )

    @field_validator("query")
    @classmethod
    def _validate_query_length(cls, value: str) -> str:
        max_len = get_settings().search_query_max_length
        if len(value) > max_len:
            msg = f"query must be at most {max_len} characters"
            raise ValueError(msg)
        return value


class SearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "results": [],
                    "parsed": {
                        "artist": "Tool",
                        "album": None,
                        "album_index": 3,
                        "resolved_album": "Lateralus",
                        "resolution_confidence": "high",
                        "location": None,
                        "country_code": None,
                        "search_scope": "global",
                        "resolved_city": None,
                        "geo_confidence": None,
                        "geo_granularity": None,
                        "language": "unknown",
                        "original_query": "third Tool studio album vinyl",
                    },
                    "reason": "album_unresolved",
                    "debug": None,
                }
            ]
        }
    )

    results: list[ListingResult] = Field(
        default_factory=list,
        description="Aggregated deduplicated search results (strict ListingResult contract).",
    )
    parsed: PipelineParsedQuery | None = Field(
        default=None,
        description=(
            "Parser output that drove this search. Always populated on a successful "
            "response so the UI can render parser debug AND the result list from a "
            "single round-trip (no separate `/parse` call needed)."
        ),
    )
    reason: SearchEmptyReason | None = Field(
        default=None,
        description=(
            "Machine-readable explanation when `results` is empty for a structural "
            "reason (e.g. album_unresolved). Null on a normal Tavily-backed search "
            "regardless of hit count."
        ),
    )
    debug: dict | None = Field(
        default=None,
        description=(
            "Populated only when `DEBUG=true` (env var). Contains the full PipelineContext "
            "trace (parser, tavily, prefilter, extractor) for that request."
        ),
    )
