"""Per-request pipeline trace + stage timer.

Every request creates one `PipelineContext`, stored in a `contextvars.ContextVar`
so agents/services can read/write it without passing it as a function arg.

Design notes:
- The context is *request-scoped*: FastAPI awaits the handler inside a fresh
  contextvar scope per request, so no leaking between requests.
- The `stage_timer` helper is the canonical way to record a stage — it logs a
  start record and a finish record (with duration, status), and pushes a
  compact `StageRecord` into the context so DEBUG mode can echo the whole
  trace back in the HTTP response.
- Emit conventions (aligned with the task spec):
      stage     ∈ {parser, tavily, extractor, pipeline}
      status    ∈ {success, fail, empty}
- "empty" is logged at WARNING so stages that return nothing never silently
  slip past a grep.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

Stage = str  # one of: parser | tavily | extractor | pipeline
Status = str  # one of: success | fail | empty


@dataclass(slots=True)
class StageRecord:
    stage: Stage
    status: Status
    duration_ms: float
    input: Any = None
    output: Any = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.input is not None:
            d["input"] = self.input
        if self.output is not None:
            d["output"] = self.output
        if self.error:
            d["error"] = self.error
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass(slots=True)
class PipelineContext:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    debug: bool = False
    stages: list[StageRecord] = field(default_factory=list)

    def record(self, rec: StageRecord) -> None:
        self.stages.append(rec)

    def as_debug_payload(self) -> dict[str, Any]:
        """Structure returned in HTTP response when DEBUG mode is on."""
        grouped: dict[str, Any] = {}
        for rec in self.stages:
            grouped.setdefault(rec.stage, []).append(rec.as_dict())
        return {
            "request_id": self.request_id,
            "stages": grouped,
            "trace": [r.as_dict() for r in self.stages],
        }


_current: ContextVar[PipelineContext | None] = ContextVar("pipeline_context", default=None)


def get_context() -> PipelineContext | None:
    return _current.get()


def require_context() -> PipelineContext:
    ctx = _current.get()
    if ctx is None:
        # Rather than error out (which would crash unrelated tests that don't set up
        # a context), we lazily create a throwaway one. The log line surfaces the fact.
        ctx = PipelineContext()
        _current.set(ctx)
        logger.warning(
            "pipeline context auto-created; caller forgot to start_pipeline()",
            extra={"stage": "context", "status": "fail", "request_id": ctx.request_id},
        )
    return ctx


@contextlib.contextmanager
def start_pipeline(*, debug: bool = False) -> Iterator[PipelineContext]:
    """Open a per-request pipeline context. Use as a `with` block in the router."""
    ctx = PipelineContext(debug=debug)
    token = _current.set(ctx)
    logger.info(
        "pipeline_start",
        extra={"stage": "pipeline", "status": "success", "request_id": ctx.request_id},
    )
    try:
        yield ctx
    finally:
        logger.info(
            "pipeline_end",
            extra={
                "stage": "pipeline",
                "status": "success",
                "request_id": ctx.request_id,
                "count": len(ctx.stages),
            },
        )
        _current.reset(token)


@contextlib.contextmanager
def stage_timer(stage: Stage, *, input: Any = None, logger_: logging.Logger | None = None) -> Iterator[StageRecord]:
    """Instrument a pipeline stage. Emits start/finish structured logs and records the
    timing in the request's PipelineContext. The caller mutates the yielded record to
    attach `output`, `status`, `extra`, etc.

    Example:
        with stage_timer("parser", input={"query": q}) as rec:
            parsed = await parse(q)
            rec.output = parsed.model_dump()
            rec.status = "success"
    """
    ctx = require_context()
    log = logger_ or logger
    start = time.perf_counter()

    rec = StageRecord(stage=stage, status="success", duration_ms=0.0, input=input)
    log.debug(
        "stage_start",
        extra={"stage": stage, "status": "start", "request_id": ctx.request_id, "input": input},
    )
    try:
        yield rec
    except Exception as exc:
        rec.status = "fail"
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.duration_ms = (time.perf_counter() - start) * 1000.0
        ctx.record(rec)
        log.exception(
            "stage_fail",
            extra={
                "stage": stage,
                "status": "fail",
                "duration_ms": round(rec.duration_ms, 2),
                "request_id": ctx.request_id,
                "reason": rec.error,
            },
        )
        raise
    else:
        rec.duration_ms = (time.perf_counter() - start) * 1000.0
        ctx.record(rec)
        end_level = logging.WARNING if rec.status == "empty" else logging.DEBUG
        log.log(
            end_level,
            "stage_end",
            extra={
                "stage": stage,
                "status": rec.status,
                "duration_ms": round(rec.duration_ms, 2),
                "request_id": ctx.request_id,
                "output": rec.output,
            },
        )
