"""Logging helpers for console output and JSONL run logs."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_EXTRA_FIELDS = {
    "command",
    "count",
    "duration_ms",
    "event",
    "path",
    "run_id",
    "status",
}


class JsonLineFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in sorted(SAFE_EXTRA_FIELDS):
            if hasattr(record, field):
                payload[field] = _json_safe_value(getattr(record, field))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, sort_keys=True)


def configure_logging(
    *,
    level: str | int = "INFO",
    jsonl_path: Path | None = None,
    console: bool = True,
    logger_name: str = "uni_rag_agent",
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(_logging_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def build_run_log_path(
    runs_dir: Path,
    command_name: str,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", command_name.strip().lower()).strip("-")
    return runs_dir / f"{timestamp}-{slug or 'run'}.jsonl"


def _logging_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.strip().upper())
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unknown logging level: {level}")


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)
