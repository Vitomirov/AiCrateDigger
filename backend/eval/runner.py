"""Execute evaluation cases against parse-only or the full vinyl-search pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from app.core.config import get_settings
from app.core.db.cache import purge_expired_search_cache_rows
from app.core.db.database import dispose_engine, init_db
from app.core.db.redis_cache import purge_stale_pipeline_cache_versions
from app.core.db.store_loader import (
    repair_whitelist_store_domains,
    seed_whitelist_stores_if_empty,
    sync_whitelist_store_catalogue,
)
from app.core.logging_config import setup_logging
from app.domains.query_parser.parse_user_query import parse_user_query
from app.domains.search_pipeline.pipeline_context import start_pipeline
from app.domains.search_pipeline.vinyl_search import run_vinyl_search

from eval.checks import (
    check_parse,
    check_pipeline,
    default_full_pipeline_stages,
)
from eval.schema import (
    CaseResult,
    EvalCase,
    EvalDataset,
    EvalReport,
    ParseExpectation,
    PipelineExpectation,
    StageResult,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset" / "edge_cases.json"


def load_dataset(path: Path) -> EvalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)


async def bootstrap_eval_env() -> None:
    """Mirror API startup so store whitelist + DB are available in Docker."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_format=settings.log_format)
    if not settings.database_url:
        logger.warning(
            "eval_db_skipped",
            extra={"stage": "eval", "reason": "DATABASE_URL unset"},
        )
        return
    await init_db(database_url=settings.database_url, debug=settings.debug)
    await seed_whitelist_stores_if_empty()
    await sync_whitelist_store_catalogue()
    await repair_whitelist_store_domains()
    await purge_expired_search_cache_rows()
    await purge_stale_pipeline_cache_versions()


async def shutdown_eval_env() -> None:
    await dispose_engine()


def _stage_results_from_trace(
    trace: list[dict[str, Any]],
    *,
    pipeline_expect: PipelineExpectation | None,
) -> list[StageResult]:
    status_by_stage: dict[str, str] = {}
    for row in trace:
        name = str(row.get("stage") or "")
        if name:
            status_by_stage[name] = str(row.get("status") or "")

    expected_stages = (
        pipeline_expect.stages if pipeline_expect and pipeline_expect.stages else {}
    )
    if not expected_stages:
        expected_stages = default_full_pipeline_stages()

    results: list[StageResult] = []
    for name, stage_expect in expected_stages.items():
        status = status_by_stage.get(name)
        if status is None:
            results.append(
                StageResult(
                    name=name,
                    passed=not stage_expect.required,
                    required=stage_expect.required,
                    status=None,
                    message="missing from trace" if stage_expect.required else None,
                    skipped=not stage_expect.required,
                ),
            )
            continue
        ok = status in stage_expect.status_in
        results.append(
            StageResult(
                name=name,
                passed=ok,
                required=stage_expect.required,
                status=status,
                message=None if ok else f"status {status!r} not allowed",
            ),
        )
    return results


async def run_parse_case(case: EvalCase) -> CaseResult:
    errors: list[str] = []
    stages: list[StageResult] = []

    with start_pipeline(debug=True) as ctx:
        try:
            parsed = await parse_user_query(case.query)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                query=case.query,
                mode="parse",
                passed=False,
                errors=[f"parse raised {type(exc).__name__}: {exc}"],
            )

        parse_expect = case.parse_expectation()
        if parse_expect is not None:
            errors.extend(check_parse(parsed, parse_expect))

        stages.append(
            StageResult(
                name="parse",
                passed=not errors,
                required=True,
                status="success",
            ),
        )

        trace = ctx.as_debug_payload().get("trace") or []

    return CaseResult(
        case_id=case.id,
        query=case.query,
        mode="parse",
        passed=len(errors) == 0,
        stages=stages,
        errors=errors,
        result_count=None,
        reason=None,
    )


async def run_full_case(case: EvalCase, *, bypass_cache: bool) -> CaseResult:
    errors: list[str] = []
    pipeline_result: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    cache_patch = (
        patch(
            "app.domains.search_pipeline.vinyl_search.get_cached_search",
            new_callable=AsyncMock,
            return_value=None,
        )
        if bypass_cache
        else None
    )

    try:
        if cache_patch is not None:
            cache_patch.start()

        with start_pipeline(debug=True) as ctx:
            pipeline_result = await run_vinyl_search(case.query, background_tasks=None)
            trace = ctx.as_debug_payload().get("trace") or []
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            query=case.query,
            mode="full",
            passed=False,
            errors=[f"pipeline raised {type(exc).__name__}: {exc}"],
        )
    finally:
        if cache_patch is not None:
            cache_patch.stop()

    parsed = pipeline_result.get("parsed")
    if parsed is not None:
        parse_expect = case.parse_expectation()
        if parse_expect is not None:
            errors.extend(check_parse(parsed, parse_expect))

    pipeline_expect = case.pipeline_expectation()
    if pipeline_expect is None:
        pipeline_expect = PipelineExpectation(
            stages=default_full_pipeline_stages(),
        )
    elif not pipeline_expect.stages:
        pipeline_expect = pipeline_expect.model_copy(
            update={"stages": default_full_pipeline_stages()},
        )

    errors.extend(
        check_pipeline(
            pipeline_result=pipeline_result,
            trace=trace,
            expect=pipeline_expect,
        ),
    )

    stages = _stage_results_from_trace(trace, pipeline_expect=pipeline_expect)
    results = pipeline_result.get("results") or []

    return CaseResult(
        case_id=case.id,
        query=case.query,
        mode="full",
        passed=len(errors) == 0,
        stages=stages,
        errors=errors,
        result_count=len(results),
        reason=pipeline_result.get("reason"),
    )


async def run_case(case: EvalCase, *, bypass_cache: bool) -> CaseResult:
    if case.mode == "parse":
        return await run_parse_case(case)
    return await run_full_case(case, bypass_cache=bypass_cache)


async def run_dataset(
    dataset: EvalDataset,
    *,
    case_ids: set[str] | None = None,
    mode_filter: str | None = None,
    bypass_cache: bool = True,
) -> EvalReport:
    cases = dataset.cases
    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
    if mode_filter and mode_filter != "all":
        cases = [c for c in cases if c.mode == mode_filter]

    results: list[CaseResult] = []
    for case in cases:
        logger.info("eval_case_start", extra={"case_id": case.id, "mode": case.mode})
        result = await run_case(case, bypass_cache=bypass_cache)
        results.append(result)
        logger.info(
            "eval_case_done",
            extra={
                "case_id": case.id,
                "passed": result.passed,
                "errors": result.errors,
            },
        )

    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        dataset_version=dataset.version,
        mode=mode_filter or "all",
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        cases=results,
    )
