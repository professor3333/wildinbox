"""Process-wide logging: one JSON object per line in staging, plain text locally.

Structured fields travel in `extra={"fields": {...}}` and become top-level
keys of the JSON record, so logs can be filtered by job, batch, route, or
principal without parsing messages.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_RESERVED = {"ts", "level", "logger", "message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            out.update({k: v for k, v in fields.items() if k not in _RESERVED})
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            line += " " + " ".join(f"{k}={v}" for k, v in fields.items())
        return line


def configure(fmt: str = "text", level: str = "INFO") -> None:
    """Route every logger, including uvicorn's and RQ's, through one handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "rq.worker"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
