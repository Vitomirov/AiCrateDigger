"""Centralized structured (JSON) logging for AiCrateDigger.

Every log record includes a stable set of top-level fields:
    timestamp, level, logger, message, stage, request_id, duration_ms, status

Agents/services should emit events through `logger.info(...)` (or the helpers in
`pipeline_context`) with an `extra={"stage": ..., "status": ..., ...}` dict; the
formatter below promotes those keys to first-class fields and folds the rest
into `payload`.

This formatter has ZERO external dependencies so it works even if the image
was built without `python-json-logger`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# Keys that the formatter will promote to top-level of the JSON record if
# present in `record.__dict__`. Anything else ends up under "payload".
_PROMOTED_KEYS: tuple[str, ...] = (
    "stage",
    "status",
    "duration_ms",
    "request_id",
    "input",
    "output",
    "count",
    "url",
    "domain",
    "reason",
)

# Keys owned by the `logging` module itself — we must NOT re-emit these under
# `payload`, or we'd get huge noisy records.
_RESERVED_LOG_KEYS: frozenset[str] = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Log formatter that emits newline-delimited JSON. One object per record."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in _PROMOTED_KEYS:
            if key in record.__dict__ and record.__dict__[key] is not None:
                data[key] = record.__dict__[key]

        # Fold any remaining user-supplied extras into `payload` for debugging.
        payload: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_KEYS or key in _PROMOTED_KEYS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = _safe(value)
        if payload:
            data["payload"] = payload

        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False, default=_safe)


def _safe(value: Any) -> Any:
    """Best-effort JSON-safe coercion. Falls back to repr() for exotic types."""
    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return repr(value)
        except Exception:
            return "<unrepr-able>"


_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Idempotent: configure the root logger to emit structured JSON on stdout.

    Safe to call multiple times (e.g. under uvicorn reload).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Wipe any handlers uvicorn / FastAPI installed so we always emit JSON.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)

    # Tame noisy third-party loggers; we still want WARN/ERROR from them.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # uvicorn's own access log stays at INFO but flows through our JSON handler.
    for uv in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uv).handlers = []
        logging.getLogger(uv).propagate = True

    _CONFIGURED = True
