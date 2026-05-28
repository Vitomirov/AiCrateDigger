"""Pydantic models for the evaluation dataset and per-case results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ParseExpectation(BaseModel):
    """Subset match on :class:`app.domains.query_parser.parse_schema.ParsedQuery`."""

    artist: str | None = None
    album: str | None = None
    album_index: int | None = None
    resolved_album: str | None = None
    resolution_confidence: Literal["high", "medium", "low", "unknown"] | None = None
    location: str | None = None
    country_code: str | None = None
    search_scope: Literal["local", "regional", "global"] | None = None
    resolved_city: str | None = None
    geo_granularity: Literal["city", "country", "region", "none"] | None = None
    fields_present: list[str] = Field(
        default_factory=list,
        description="Keys that must be non-null after parse.",
    )
    fields_absent: list[str] = Field(
        default_factory=list,
        description="Keys that must be null after parse.",
    )


class StageExpectation(BaseModel):
    """Assert a pipeline stage appeared in the DEBUG trace with allowed status."""

    required: bool = True
    status_in: list[Literal["success", "fail", "empty"]] = Field(
        default_factory=lambda: ["success", "empty"],
    )


class PipelineExpectation(BaseModel):
    """Assertions on the full ``run_vinyl_search`` outcome."""

    reason: str | None = None
    min_results: int | None = None
    max_results: int | None = None
    stages: dict[str, StageExpectation] = Field(default_factory=dict)


class EvalCase(BaseModel):
    id: str
    query: str
    description: str
    tags: list[str] = Field(default_factory=list)
    mode: Literal["parse", "full"] = "full"
    expect: dict[str, Any] = Field(default_factory=dict)

    def parse_expectation(self) -> ParseExpectation | None:
        raw = self.expect.get("parse")
        if raw is None:
            return None
        return ParseExpectation.model_validate(raw)

    def pipeline_expectation(self) -> PipelineExpectation | None:
        raw = self.expect.get("pipeline")
        if raw is None:
            return None
        return PipelineExpectation.model_validate(raw)


class EvalDataset(BaseModel):
    version: str = "1"
    description: str = ""
    cases: list[EvalCase]


class StageResult(BaseModel):
    name: str
    passed: bool
    required: bool
    status: str | None = None
    message: str | None = None
    skipped: bool = False


class CaseResult(BaseModel):
    case_id: str
    query: str
    mode: str
    passed: bool
    stages: list[StageResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    result_count: int | None = None
    reason: str | None = None


class EvalReport(BaseModel):
    dataset_version: str
    mode: str
    total: int
    passed: int
    failed: int
    cases: list[CaseResult]
