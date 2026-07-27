from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from tests.support import make_config
from uni_rag_agent.app.ask_audit import AskAuditPage
from uni_rag_agent.app.audit_dashboard import create_audit_dashboard


class _DashboardStore:
    def __init__(self) -> None:
        self.record = {
            "audit_id": "a" * 32,
            "received_at": datetime(2026, 7, 27, 12, 0, tzinfo=UTC),
            "status": "completed",
            "query": "Where is MapReduce covered?",
            "client": {"ip_address": "203.0.113.9"},
            "response": {"answer_text": "In the distributed systems course."},
        }

    def list_recent(self, *, limit: int, before: datetime | None):
        assert limit == 100
        assert before is None
        return AskAuditPage(records=(self.record,), next_before=None)

    def get(self, audit_id: str):
        return self.record if audit_id == "a" * 32 else None


def test_local_dashboard_lists_and_opens_complete_records(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config = replace(
        config,
        firestore_project_id="offline-project",
        firestore_database="(default)",
    )
    client = TestClient(
        create_audit_dashboard(
            config_loader=lambda: config,
            store=_DashboardStore(),
            geoip_database=tmp_path / "missing.mmdb",
        )
    )

    page = client.get("/")
    meta = client.get("/api/meta")
    records = client.get("/api/asks")
    detail = client.get(f"/api/asks/{'a' * 32}")

    assert page.status_code == 200
    assert "Ask audit" in page.text
    assert meta.json()["project"] == "offline-project"
    assert meta.json()["geoip"]["ready"] is False
    assert records.json()["records"][0]["query"] == "Where is MapReduce covered?"
    assert records.json()["records"][0]["location"] is None
    assert detail.json()["response"]["answer_text"].startswith("In the")
    assert client.get("/api/asks/not-valid").status_code == 404
