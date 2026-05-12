"""Centralized logging for AiCrateDigger.

- ``log_format=json``: newline-delimited JSON (good for Loki/Datadog).
- ``log_format=human``: multi-line, indents ``payload`` for local/Docker terminal use.

Set via env ``LOG_FORMAT`` or :attr:`app.config.Settings.log_format`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Literal

# Keys that the formatter will promote to top-level if present in `record.__dict__`.
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

_RESERVED_LOG_KEYS: frozenset[str] = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }
)

_MAX_PAYLOAD_CHARS = 12000


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return repr(value)
        except Exception:
            return "<unrepr-able>"


def structured_log_dict(record: logging.LogRecord) -> dict[str, Any]:
    """Build the same structured object used for JSON and human formatting."""
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
        data["exc"] = logging.Formatter().formatException(record.exc_info)

    return data


class JsonFormatter(logging.Formatter):
    """Newline-delimited JSON, one object per record."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(structured_log_dict(record), ensure_ascii=False, default=_safe)


def _promo_line(data: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in _PROMOTED_KEYS:
        if key not in data or data[key] is None:
            continue
        val = data[key]
        if isinstance(val, (dict, list)):
            continue
        parts.append(f"{key}={val}")
    return "  " + "  ".join(parts) if parts else None


class HumanReadableFormatter(logging.Formatter):
    """Stack-friendly logs: first line summary, then promoted fields, pretty ``payload``."""

    def format(self, record: logging.LogRecord) -> str:
        d = structured_log_dict(record)
        ts = d.get("ts", "")
        level = str(d.get("level", ""))
        logg = str(d.get("logger", ""))
        msg = str(d.get("message", ""))
        line1 = f"{ts}  {level:7}  {logg}  {msg}"

        lines: list[str] = [line1]
        pl = _promo_line(d)
        if pl:
            lines.append(pl)

        p = d.get("payload")
        if p:
            try:
                blob = json.dumps(p, ensure_ascii=False, indent=2, default=_safe)
            except Exception:
                blob = repr(p)
            if len(blob) > _MAX_PAYLOAD_CHARS:
                blob = blob[:_MAX_PAYLOAD_CHARS] + "\n  ... [truncated]"
            lines.append("  payload:")
            for row in blob.splitlines():
                lines.append(f"    {row}")

        if "exc" in d and d["exc"]:
            lines.append("  exc:")
            for row in str(d["exc"]).strip().splitlines():
                lines.append(f"    {row}")

        return "\n".join(lines)


LogFormat = Literal["json", "human"]

_CONFIGURED = False


def setup_logging(
    level: str | None = None,
    log_format: LogFormat | str | None = None,
) -> None:
    """Configure the root logger. Idempotent (first call wins during uvicorn reload)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    fmt_raw = log_format or os.getenv("LOG_FORMAT") or "human"
    fmt = str(fmt_raw).strip().lower()
    if fmt not in ("json", "human"):
        fmt = "human"

    handler = logging.StreamHandler(stream=sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)

    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for uv in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uv).handlers = []
        logging.getLogger(uv).propagate = True

    _CONFIGURED = True
