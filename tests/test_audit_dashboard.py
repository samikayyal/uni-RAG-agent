from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from tests.support import make_config
from uni_rag_agent.app.audit_cache import AuditCache, AuditSourcePage, GeoIpResolver
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

    def iter_all_pages(self, *, page_size: int):
        assert page_size == 250
        yield AuditSourcePage(records=(self.record,))

    def iter_updated_pages(self, *, updated_at_or_after: datetime, page_size: int):
        assert page_size == 250
        if self.record["received_at"] >= updated_at_or_after:
            yield AuditSourcePage(records=(self.record,))


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
            cache_database=tmp_path / "ask_audit_cache.sqlite",
        )
    )

    page = client.get("/")
    meta = client.get("/api/meta")
    records = client.get("/api/asks")
    detail = client.get(f"/api/asks/{'a' * 32}")

    assert page.status_code == 200
    assert "Ask audit" in page.text
    assert meta.json()["project"] == "offline-project"
    assert meta.json()["cached_count"] == 1
    assert meta.json()["geoip"]["ready"] is False
    assert records.json()["records"][0]["query"] == "Where is MapReduce covered?"
    assert records.json()["records"][0]["location"]["status"] == "private"
    assert detail.json()["response"]["answer_text"].startswith("In the")
    assert client.get("/api/asks/not-valid").status_code == 404


def test_offline_dashboard_never_initializes_firestore_and_sync_is_local(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "ask_audit_cache.sqlite"
    cache = AuditCache(cache_path)
    cache.initialize()
    cache.sync(_DashboardStore(), GeoIpResolver(tmp_path / "missing.mmdb"))
    called = False

    def config_loader():
        nonlocal called
        called = True
        raise AssertionError("offline must not load Firestore configuration")

    client = TestClient(
        create_audit_dashboard(
            config_loader=config_loader,
            cache_database=cache_path,
            geoip_database=tmp_path / "missing.mmdb",
            offline=True,
        )
    )

    assert client.get("/api/meta").json()["offline"] is True
    assert client.post("/api/sync").status_code == 200
    assert client.get("/api/asks").json()["records"][0]["audit_id"] == "a" * 32
    assert called is False


def test_dashboard_starts_stale_cache_but_empty_cache_failure_is_actionable(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "ask_audit_cache.sqlite"
    cache = AuditCache(cache_path)
    cache.initialize()
    cache.sync(_DashboardStore(), GeoIpResolver(tmp_path / "missing.mmdb"))

    class FailingStore:
        def iter_updated_pages(self, **_kwargs):
            raise RuntimeError("offline")

        def iter_all_pages(self, **_kwargs):
            raise RuntimeError("offline")

    config = replace(make_config(tmp_path), firestore_project_id="offline-project")
    client = TestClient(
        create_audit_dashboard(
            config_loader=lambda: config,
            store=FailingStore(),
            cache_database=cache_path,
            geoip_database=tmp_path / "missing.mmdb",
        )
    )
    assert client.get("/api/meta").json()["sync_state"] == "stale"
    assert client.get("/api/asks").json()["records"][0]["audit_id"] == "a" * 32

    with pytest.raises(RuntimeError, match="cache is empty"):
        create_audit_dashboard(
            config_loader=lambda: config,
            store=FailingStore(),
            cache_database=tmp_path / "empty.sqlite",
            geoip_database=tmp_path / "missing.mmdb",
        )


def test_conflicting_dashboard_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used"):
        create_audit_dashboard(
            cache_database=tmp_path / "cache.sqlite",
            offline=True,
            rebuild_cache=True,
        )
