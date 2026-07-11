from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from uni_rag_agent.logging_config import build_run_log_path, configure_logging


def test_jsonl_logging_writes_valid_json_objects(tmp_path) -> None:
    log_path = tmp_path / "runs" / "test.jsonl"
    logger = configure_logging(jsonl_path=log_path, console=False)

    logger.info(
        "inventory started",
        extra={
            "event": "inventory_started",
            "command": "inventory run",
            "count": 3,
            "api_key": "secret-value",
        },
    )
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["message"] == "inventory started"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "uni_rag_agent"
    assert payload["event"] == "inventory_started"
    assert payload["command"] == "inventory run"
    assert payload["count"] == 3
    assert "api_key" not in payload


def test_build_run_log_path_sanitizes_command_name_and_is_unique(tmp_path) -> None:
    now = datetime(2026, 6, 23, 12, 30, 0, tzinfo=timezone.utc)
    path = build_run_log_path(tmp_path, "inventory run", now=now)
    second_path = build_run_log_path(tmp_path, "inventory run", now=now)

    assert path.parent == tmp_path
    assert re.fullmatch(r"20260623T123000Z-[0-9a-f]{8}-inventory-run\.jsonl", path.name)
    assert second_path != path
