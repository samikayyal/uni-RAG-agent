from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from uni_rag_agent.app.audit_cache import (
    AuditCache,
    AuditCacheError,
    AuditSourcePage,
    GeoIpResolver,
    LocationSnapshot,
)


def _record(
    audit_id: str,
    *,
    received_at: datetime,
    updated_at: datetime | None = None,
    status: str = "received",
    ip_address: str = "198.51.100.8",
) -> dict[str, object]:
    record: dict[str, object] = {
        "audit_id": audit_id,
        "schema_version": 2 if updated_at is not None else 1,
        "received_at": received_at,
        "status": status,
        "query": f"question {audit_id}",
        "request_id": f"request-{audit_id}",
        "session_id": "session-1",
        "client": {"ip_address": ip_address},
    }
    if updated_at is not None:
        record["updated_at"] = updated_at
    return record


class _Source:
    def __init__(
        self, records: list[dict[str, object]], *, fail_after: int | None = None
    ) -> None:
        self.records = records
        self.fail_after = fail_after

    def iter_all_pages(self, *, page_size: int):
        yield from self._pages(self.records, page_size=page_size)

    def iter_updated_pages(self, *, updated_at_or_after: datetime, page_size: int):
        records = [
            record
            for record in self.records
            if isinstance(record.get("updated_at"), datetime)
            and record["updated_at"] >= updated_at_or_after
        ]
        records.sort(key=lambda record: (record["updated_at"], record["audit_id"]))
        yield from self._pages(records, page_size=page_size)

    def _pages(self, records: list[dict[str, object]], *, page_size: int):
        for page_number, start in enumerate(range(0, len(records), page_size)):
            if self.fail_after is not None and page_number >= self.fail_after:
                raise RuntimeError("unavailable")
            yield AuditSourcePage(
                records=tuple(deepcopy(records[start : start + page_size]))
            )


class _Geo:
    def __init__(self, country: str = "Jordan", *, unavailable: bool = False) -> None:
        self.country = country
        self.unavailable = unavailable
        self.ready = not unavailable
        self.error = None if self.ready else "GeoLite unavailable"
        self.database_path = Path("fake.mmdb")
        self.build_epoch = 100 if country == "Jordan" else 200

    def resolve(self, address: object) -> LocationSnapshot:
        if self.unavailable:
            return LocationSnapshot("unavailable", {}, None, None)
        if address == "private":
            return LocationSnapshot(
                "private",
                {"country": "Private/local address"},
                "2026-07-27T00:00:00+00:00",
                self.build_epoch,
            )
        if address == "not-found":
            return LocationSnapshot(
                "not_found", {}, "2026-07-27T00:00:00+00:00", self.build_epoch
            )
        return LocationSnapshot(
            "resolved",
            {
                "country_code": self.country[:2].upper(),
                "country": self.country,
                "region": "region",
                "city": "city",
                "latitude": 1.0,
                "longitude": 2.0,
                "time_zone": "UTC",
            },
            "2026-07-27T00:00:00+00:00",
            self.build_epoch,
        )


def _cache(tmp_path: Path) -> AuditCache:
    cache = AuditCache(tmp_path / "ask_audit_cache.sqlite")
    cache.initialize()
    return cache


def test_bootstrap_accepts_legacy_records_and_complete_json(tmp_path: Path) -> None:
    received = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    legacy = _record("legacy", received_at=received)
    cache = _cache(tmp_path)

    cache.sync(_Source([legacy]), _Geo())

    loaded = cache.get("legacy")
    assert loaded is not None
    assert loaded["schema_version"] == 1
    assert loaded["received_at"] == received.isoformat()
    assert cache.state()["bootstrap_complete"] == 1
    assert cache.state()["firestore_cursor_updated_at"] == received.isoformat()


def test_incremental_equal_timestamp_upserts_terminal_state_and_pages(
    tmp_path: Path,
) -> None:
    timestamp = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    cache = _cache(tmp_path)
    source = _Source([_record("one", received_at=timestamp, updated_at=timestamp)])
    cache.sync(source, _Geo())
    terminal = _record(
        "one", received_at=timestamp, updated_at=timestamp, status="completed"
    )
    terminal["response"] = {"answer_text": "complete"}
    source.records = [terminal] + [
        _record(
            f"page-{index}",
            received_at=timestamp,
            updated_at=timestamp,
            status="accepted",
        )
        for index in range(260)
    ]

    cache.sync(source, _Geo())

    updated = cache.get("one")
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["response"] == {"answer_text": "complete"}
    assert cache.count() == 261


def test_failed_incremental_transaction_rolls_back_records_and_cursor(
    tmp_path: Path,
) -> None:
    first = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    later = first + timedelta(minutes=1)
    cache = _cache(tmp_path)
    cache.sync(_Source([_record("one", received_at=first, updated_at=first)]), _Geo())
    before = cache.state()["firestore_cursor_updated_at"]
    failed = _Source(
        [_record("one", received_at=first, updated_at=later, status="completed")],
        fail_after=1,
    )

    # A second empty page forces a source failure after the upsert has started.
    failed.records.extend(
        [
            _record(f"zzz-{index}", received_at=first, updated_at=later)
            for index in range(250)
        ]
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        cache.sync(failed, _Geo())

    assert cache.get("one")["status"] == "received"  # type: ignore[index]
    assert cache.state()["firestore_cursor_updated_at"] == before
    assert (
        cache.state()["last_error"]
        == "Firestore synchronization failed. Verify Application Default Credentials and connectivity."
    )


def test_location_snapshots_freeze_refresh_retry_and_ip_change(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    cache = _cache(tmp_path)
    source = _Source([_record("one", received_at=timestamp, updated_at=timestamp)])
    cache.sync(source, _Geo("Jordan"))
    assert cache.get("one")["location"]["country"] == "Jordan"  # type: ignore[index]

    source.records = [
        _record("one", received_at=timestamp, updated_at=timestamp, status="completed")
    ]
    cache.sync(source, _Geo("Canada"))
    assert cache.get("one")["location"]["country"] == "Jordan"  # type: ignore[index]
    cache.refresh_locations(_Geo("Canada"))
    assert cache.get("one")["location"]["country"] == "Canada"  # type: ignore[index]

    unavailable = _cache(tmp_path / "retry")
    retry_source = _Source(
        [_record("retry", received_at=timestamp, updated_at=timestamp)]
    )
    unavailable.sync(retry_source, _Geo(unavailable=True))
    retry_source.records = [
        _record(
            "retry", received_at=timestamp, updated_at=timestamp, status="completed"
        )
    ]
    unavailable.sync(retry_source, _Geo("Jordan"))
    assert unavailable.get("retry")["location"]["country"] == "Jordan"  # type: ignore[index]

    source.records = [
        _record(
            "one", received_at=timestamp, updated_at=timestamp, ip_address="203.0.113.4"
        )
    ]
    cache.sync(source, _Geo("Amman"))
    assert cache.get("one")["location"]["country"] == "Amman"  # type: ignore[index]


def test_geoip_private_and_not_found_are_frozen(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    cache = _cache(tmp_path)
    source = _Source(
        [
            _record(
                "private",
                received_at=timestamp,
                updated_at=timestamp,
                ip_address="private",
            ),
            _record(
                "missing",
                received_at=timestamp,
                updated_at=timestamp,
                ip_address="not-found",
            ),
        ]
    )
    cache.sync(source, _Geo())
    assert cache.get("private")["location"]["status"] == "private"  # type: ignore[index]
    assert cache.get("missing")["location"]["status"] == "not_found"  # type: ignore[index]


def test_missing_geo_database_is_retryable_for_real_resolver(tmp_path: Path) -> None:
    resolver = GeoIpResolver(tmp_path / "missing.mmdb")
    result = resolver.resolve("8.8.8.8")
    assert result.status == "unavailable"
    assert result.resolved_at is None
    assert resolver.resolve("127.0.0.1").status == "private"


def test_rebuild_is_exact_and_corrupt_cache_is_not_deleted(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    cache = _cache(tmp_path)
    cache.sync(
        _Source([_record("old", received_at=timestamp, updated_at=timestamp)]), _Geo()
    )
    cache.sync(
        _Source([_record("new", received_at=timestamp, updated_at=timestamp)]),
        _Geo(),
        rebuild=True,
    )
    assert cache.get("old") is None
    assert cache.get("new") is not None

    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("not a sqlite database", encoding="utf-8")
    with pytest.raises(AuditCacheError, match="--rebuild-cache"):
        AuditCache(corrupt).initialize()
    assert corrupt.read_text(encoding="utf-8") == "not a sqlite database"
