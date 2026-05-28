"""Assertion helpers for parse output and pipeline stage traces."""

from __future__ import annotations

from typing import Any

from app.domains.query_parser.parse_schema import ParsedQuery

from eval.schema import ParseExpectation, PipelineExpectation, StageExpectation


def _parsed_dump(parsed: ParsedQuery) -> dict[str, Any]:
    return parsed.model_dump()


def check_parse(parsed: ParsedQuery, expect: ParseExpectation) -> list[str]:
    """Return a list of error strings (empty when all checks pass)."""
    errors: list[str] = []
    data = _parsed_dump(parsed)

    for key, expected in expect.model_dump(exclude={"fields_present", "fields_absent"}).items():
        if expected is None:
            continue
        actual = data.get(key)
        if actual != expected:
            errors.append(f"parse.{key}: expected {expected!r}, got {actual!r}")

    for key in expect.fields_present:
        if data.get(key) is None:
            errors.append(f"parse.{key}: expected non-null, got null")

    for key in expect.fields_absent:
        if data.get(key) is not None:
            errors.append(f"parse.{key}: expected null, got {data.get(key)!r}")

    return errors


def _stage_status_map(trace: list[dict[str, Any]]) -> dict[str, str]:
    """Last recorded status per stage name in the request trace."""
    out: dict[str, str] = {}
    for row in trace:
        name = str(row.get("stage") or "")
        status = str(row.get("status") or "")
        if name:
            out[name] = status
    return out


def check_pipeline(
    *,
    pipeline_result: dict[str, Any],
    trace: list[dict[str, Any]],
    expect: PipelineExpectation,
) -> list[str]:
    errors: list[str] = []
    results = pipeline_result.get("results") or []
    count = len(results)
    reason = pipeline_result.get("reason")

    if expect.reason is not None and reason != expect.reason:
        errors.append(f"pipeline.reason: expected {expect.reason!r}, got {reason!r}")

    if expect.min_results is not None and count < expect.min_results:
        errors.append(f"pipeline.results: expected >= {expect.min_results}, got {count}")

    if expect.max_results is not None and count > expect.max_results:
        errors.append(f"pipeline.results: expected <= {expect.max_results}, got {count}")

    status_by_stage = _stage_status_map(trace)
    for stage_name, stage_expect in expect.stages.items():
        errors.extend(
            _check_one_stage(
                stage_name,
                stage_expect,
                status_by_stage=status_by_stage,
            )
        )

    return errors


def _check_one_stage(
    stage_name: str,
    expect: StageExpectation,
    *,
    status_by_stage: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    status = status_by_stage.get(stage_name)

    if status is None:
        if expect.required:
            errors.append(f"stage.{stage_name}: required but missing from trace")
        return errors

    if status not in expect.status_in:
        errors.append(
            f"stage.{stage_name}: status {status!r} not in {expect.status_in!r}",
        )
    return errors


def default_full_pipeline_stages() -> dict[str, StageExpectation]:
    """Stages recorded by the consolidated ``run_vinyl_search`` hot path."""
    return {
        "parse": StageExpectation(required=True, status_in=["success"]),
        "album_resolve": StageExpectation(required=True, status_in=["success", "empty"]),
        "redis_cache_lookup": StageExpectation(
            required=True,
            status_in=["success", "empty"],
        ),
        "tavily": StageExpectation(required=True, status_in=["success", "empty", "fail"]),
        "prefilter": StageExpectation(required=False, status_in=["success", "empty"]),
        "extractor": StageExpectation(required=False, status_in=["success", "empty"]),
    }
