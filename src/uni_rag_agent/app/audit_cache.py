"""Generated SQLite cache and incremental synchronizer for public ask audits.

The Firestore ledger is authoritative.  This module deliberately keeps its
copy separate from the application SQLite database because it contains raw
questions, complete responses, and client IP addresses.
"""

from __future__ import annotations

import ipaddress
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = 1


class AuditCacheError(RuntimeError):
    """Raised when the generated audit cache cannot safely be used."""


@dataclass(frozen=True)
class AuditSourcePage:
    """One bounded page from the authoritative Firestore ledger."""

    records: tuple[dict[str, object], ...]


class AskAuditSource(Protocol):
    def iter_all_pages(self, *, page_size: int) -> Iterable[AuditSourcePage]: ...

    def iter_updated_pages(
        self,
        *,
        updated_at_or_after: datetime,
        page_size: int,
    ) -> Iterable[AuditSourcePage]: ...


@dataclass(frozen=True)
class LocationSnapshot:
    status: str
    values: dict[str, object]
    resolved_at: str | None
    geoip_build_epoch: int | None


def _iso(value: object) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _json_safe(value: object) -> object:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AuditCacheError(
        "A Firestore audit record contains a value that cannot be stored as JSON. "
        "Rebuild the cache after updating the record schema."
    )


class GeoIpResolver:
    """Resolve addresses locally and distinguish retryable from frozen results."""

    def __init__(self, database_path: Path | None) -> None:
        self.database_path = database_path
        self._reader: Any = None
        self.error: str | None = None
        self.build_epoch: int | None = None
        if database_path is None or not database_path.is_file():
            self.error = (
                "GeoLite2 City database not found; IP addresses are still available."
            )
            return
        try:
            import geoip2.database

            self._reader = geoip2.database.Reader(str(database_path))
            self.build_epoch = int(self._reader.metadata().build_epoch)
        except ImportError:
            self.error = (
                "The dashboard extra is not installed; run "
                "'uv sync --extra public-demo --extra dashboard'."
            )
        except Exception:
            self.error = "The GeoLite2 City database could not be opened."

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def resolve(self, address: object) -> LocationSnapshot:
        if not isinstance(address, str):
            return LocationSnapshot(
                "invalid", {}, _iso(datetime.now(UTC)), self.build_epoch
            )
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return LocationSnapshot(
                "invalid", {}, _iso(datetime.now(UTC)), self.build_epoch
            )
        resolved_at = _iso(datetime.now(UTC))
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_unspecified
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
        ):
            return LocationSnapshot(
                "private",
                {
                    "country_code": None,
                    "country": "Private/local address",
                    "region": None,
                    "city": None,
                    "latitude": None,
                    "longitude": None,
                    "time_zone": None,
                },
                resolved_at,
                self.build_epoch,
            )
        if self._reader is None:
            return LocationSnapshot("unavailable", {}, None, self.build_epoch)
        try:
            result = self._reader.city(address)
        except Exception as exc:
            # geoip2 exposes a dedicated deterministic "not found" exception.
            if exc.__class__.__name__ == "AddressNotFoundError":
                return LocationSnapshot("not_found", {}, resolved_at, self.build_epoch)
            return LocationSnapshot("unavailable", {}, None, self.build_epoch)
        subdivision = result.subdivisions.most_specific
        return LocationSnapshot(
            "resolved",
            {
                "country_code": result.country.iso_code,
                "country": result.country.name,
                "region": subdivision.name,
                "city": result.city.name,
                "latitude": result.location.latitude,
                "longitude": result.location.longitude,
                "time_zone": result.location.time_zone,
            },
            resolved_at,
            self.build_epoch,
        )


class AuditCache:
    """SQLite projection with transactional, idempotent Firestore synchronization."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
        except sqlite3.DatabaseError as exc:
            raise AuditCacheError(
                f"Audit cache is corrupt or incompatible at {self.path}; use --rebuild-cache."
            ) from exc
        try:
            yield connection
        except sqlite3.DatabaseError as exc:
            raise AuditCacheError(
                f"Audit cache is corrupt or incompatible at {self.path}; use --rebuild-cache."
            ) from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        existed = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not existed and not tables:
                    self._create_schema(connection)
                    return
                required = {"audit_records", "audit_sync_state"}
                if (
                    not required.issubset(tables)
                    or connection.execute("PRAGMA user_version").fetchone()[0]
                    != SCHEMA_VERSION
                ):
                    raise AuditCacheError(
                        f"Audit cache is corrupt or incompatible at {self.path}; use --rebuild-cache."
                    )
            except sqlite3.DatabaseError as exc:
                raise AuditCacheError(
                    f"Audit cache is corrupt or incompatible at {self.path}; use --rebuild-cache."
                ) from exc

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN;
            CREATE TABLE audit_records (
                audit_id TEXT PRIMARY KEY,
                received_at TEXT,
                updated_at TEXT,
                status TEXT,
                request_id TEXT,
                session_id TEXT,
                query TEXT,
                ip_address TEXT,
                country_code TEXT,
                country TEXT,
                region TEXT,
                city TEXT,
                latitude REAL,
                longitude REAL,
                time_zone TEXT,
                geoip_status TEXT,
                geoip_resolved_at TEXT,
                geoip_build_epoch INTEGER,
                record_json TEXT NOT NULL
            );
            CREATE INDEX audit_records_received_at_idx ON audit_records(received_at DESC);
            CREATE INDEX audit_records_updated_at_idx ON audit_records(updated_at ASC);
            CREATE INDEX audit_records_status_idx ON audit_records(status);
            CREATE INDEX audit_records_request_id_idx ON audit_records(request_id);
            CREATE INDEX audit_records_session_id_idx ON audit_records(session_id);
            CREATE INDEX audit_records_query_idx ON audit_records(query);
            CREATE INDEX audit_records_ip_address_idx ON audit_records(ip_address);
            CREATE TABLE audit_sync_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                bootstrap_complete INTEGER NOT NULL DEFAULT 0,
                firestore_cursor_updated_at TEXT,
                last_attempt_at TEXT,
                last_success_at TEXT,
                cached_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );
            INSERT INTO audit_sync_state(singleton) VALUES (1);
            PRAGMA user_version = 1;
            COMMIT;
            """
        )

    def count(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0]
            )

    def state(self) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM audit_sync_state WHERE singleton = 1"
            ).fetchone()
            assert row is not None
            return dict(row)

    def mark_sync_error(self, message: str) -> None:
        with self._connection() as connection, connection:
            connection.execute(
                "UPDATE audit_sync_state SET last_attempt_at = ?, last_error = ? WHERE singleton = 1",
                (_iso(datetime.now(UTC)), message),
            )

    def sync(
        self, source: AskAuditSource, geoip: GeoIpResolver, *, rebuild: bool = False
    ) -> None:
        """Synchronize from Firestore, advancing the cursor only on commit."""

        state = self.state()
        try:
            if rebuild or not bool(state["bootstrap_complete"]):
                self._bootstrap(source, geoip, rebuild=rebuild)
            else:
                cursor = _parse_iso(state["firestore_cursor_updated_at"])
                if cursor is not None:
                    self._incremental(source, geoip, cursor)
                else:
                    # A malformed persisted cursor is never silently skipped.
                    raise AuditCacheError(
                        f"Audit cache is corrupt or incompatible at {self.path}; use --rebuild-cache."
                    )
        except Exception as exc:
            self.mark_sync_error(_sanitized_sync_error(exc))
            raise

    def _bootstrap(
        self, source: AskAuditSource, geoip: GeoIpResolver, *, rebuild: bool
    ) -> None:
        latest: datetime | None = None
        with self._connection() as connection, connection:
            if rebuild:
                connection.execute("DELETE FROM audit_records")
                connection.execute(
                    "UPDATE audit_sync_state SET bootstrap_complete = 0, firestore_cursor_updated_at = NULL, last_success_at = NULL, cached_count = 0, last_error = NULL WHERE singleton = 1"
                )
            for page in source.iter_all_pages(page_size=250):
                for record in page.records:
                    updated = self._upsert(connection, record, geoip)
                    if updated is not None and (latest is None or updated > latest):
                        latest = updated
            now = _iso(datetime.now(UTC))
            connection.execute(
                """
                UPDATE audit_sync_state
                SET bootstrap_complete = 1,
                    firestore_cursor_updated_at = ?,
                    last_attempt_at = ?, last_success_at = ?,
                    cached_count = (SELECT COUNT(*) FROM audit_records),
                    last_error = NULL
                WHERE singleton = 1
                """,
                (
                    _iso(latest) if latest else _iso(datetime(1970, 1, 1, tzinfo=UTC)),
                    now,
                    now,
                ),
            )

    def _incremental(
        self,
        source: AskAuditSource,
        geoip: GeoIpResolver,
        cursor: datetime,
    ) -> None:
        latest = cursor
        with self._connection() as connection, connection:
            for page in source.iter_updated_pages(
                updated_at_or_after=cursor,
                page_size=250,
            ):
                for record in page.records:
                    updated = self._upsert(connection, record, geoip)
                    if updated is not None and updated > latest:
                        latest = updated
            now = _iso(datetime.now(UTC))
            connection.execute(
                """
                UPDATE audit_sync_state
                SET firestore_cursor_updated_at = ?, last_attempt_at = ?, last_success_at = ?,
                    cached_count = (SELECT COUNT(*) FROM audit_records), last_error = NULL
                WHERE singleton = 1
                """,
                (_iso(latest), now, now),
            )

    def _upsert(
        self,
        connection: sqlite3.Connection,
        record: dict[str, object],
        geoip: GeoIpResolver,
    ) -> datetime | None:
        audit_id = record.get("audit_id")
        if not isinstance(audit_id, str) or not audit_id:
            raise AuditCacheError("A Firestore audit record is missing its audit_id.")
        received_at = _iso(record.get("received_at"))
        updated_at = _iso(record.get("updated_at")) or received_at
        if updated_at is None:
            raise AuditCacheError(
                f"Firestore audit record {audit_id} has no usable received_at or updated_at."
            )
        client = record.get("client")
        ip_address = client.get("ip_address") if isinstance(client, dict) else None
        if not isinstance(ip_address, str):
            ip_address = None
        existing = connection.execute(
            "SELECT ip_address, geoip_status, country_code, country, region, city, latitude, longitude, time_zone, geoip_resolved_at, geoip_build_epoch FROM audit_records WHERE audit_id = ?",
            (audit_id,),
        ).fetchone()
        location = self._location_for_upsert(existing, ip_address, geoip)
        payload = _json_safe(record)
        assert isinstance(payload, dict)
        connection.execute(
            """
            INSERT INTO audit_records (
                audit_id, received_at, updated_at, status, request_id, session_id, query,
                ip_address, country_code, country, region, city, latitude, longitude,
                time_zone, geoip_status, geoip_resolved_at, geoip_build_epoch, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(audit_id) DO UPDATE SET
                received_at = excluded.received_at, updated_at = excluded.updated_at,
                status = excluded.status, request_id = excluded.request_id,
                session_id = excluded.session_id, query = excluded.query,
                ip_address = excluded.ip_address, country_code = excluded.country_code,
                country = excluded.country, region = excluded.region, city = excluded.city,
                latitude = excluded.latitude, longitude = excluded.longitude,
                time_zone = excluded.time_zone, geoip_status = excluded.geoip_status,
                geoip_resolved_at = excluded.geoip_resolved_at,
                geoip_build_epoch = excluded.geoip_build_epoch,
                record_json = excluded.record_json
            """,
            (
                audit_id,
                received_at,
                updated_at,
                record.get("status"),
                record.get("request_id"),
                record.get("session_id"),
                record.get("query"),
                ip_address,
                location.values.get("country_code"),
                location.values.get("country"),
                location.values.get("region"),
                location.values.get("city"),
                location.values.get("latitude"),
                location.values.get("longitude"),
                location.values.get("time_zone"),
                location.status,
                location.resolved_at,
                location.geoip_build_epoch,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        return _parse_iso(updated_at)

    @staticmethod
    def _location_for_upsert(
        existing: sqlite3.Row | None,
        ip_address: str | None,
        geoip: GeoIpResolver,
    ) -> LocationSnapshot:
        if existing is None or existing["ip_address"] != ip_address:
            return geoip.resolve(ip_address)
        if existing["geoip_status"] == "unavailable":
            return geoip.resolve(ip_address)
        return LocationSnapshot(
            str(existing["geoip_status"] or "unavailable"),
            {
                "country_code": existing["country_code"],
                "country": existing["country"],
                "region": existing["region"],
                "city": existing["city"],
                "latitude": existing["latitude"],
                "longitude": existing["longitude"],
                "time_zone": existing["time_zone"],
            },
            existing["geoip_resolved_at"],
            existing["geoip_build_epoch"],
        )

    def refresh_locations(self, geoip: GeoIpResolver) -> None:
        with self._connection() as connection, connection:
            rows = connection.execute(
                "SELECT audit_id, ip_address FROM audit_records"
            ).fetchall()
            for row in rows:
                location = geoip.resolve(row["ip_address"])
                connection.execute(
                    """
                    UPDATE audit_records SET country_code = ?, country = ?, region = ?, city = ?,
                        latitude = ?, longitude = ?, time_zone = ?, geoip_status = ?,
                        geoip_resolved_at = ?, geoip_build_epoch = ? WHERE audit_id = ?
                    """,
                    (
                        location.values.get("country_code"),
                        location.values.get("country"),
                        location.values.get("region"),
                        location.values.get("city"),
                        location.values.get("latitude"),
                        location.values.get("longitude"),
                        location.values.get("time_zone"),
                        location.status,
                        location.resolved_at,
                        location.geoip_build_epoch,
                        row["audit_id"],
                    ),
                )

    def list_recent(
        self, *, limit: int, before: datetime | None
    ) -> tuple[list[dict[str, object]], str | None]:
        with self._connection() as connection:
            clauses = ""
            parameters: list[object] = []
            if before is not None:
                clauses = "WHERE received_at < ?"
                parameters.append(_iso(before))
            rows = connection.execute(
                f"SELECT * FROM audit_records {clauses} ORDER BY received_at DESC, audit_id DESC LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            selected = rows[:limit]
            next_before = (
                selected[-1]["received_at"] if len(rows) > limit and selected else None
            )
            return [self._record_from_row(row) for row in selected], next_before

    def get(self, audit_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM audit_records WHERE audit_id = ?", (audit_id,)
            ).fetchone()
            return self._record_from_row(row) if row is not None else None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> dict[str, object]:
        try:
            record = json.loads(row["record_json"])
        except json.JSONDecodeError as exc:
            raise AuditCacheError(
                "Audit cache JSON is corrupt; use --rebuild-cache."
            ) from exc
        if not isinstance(record, dict):
            raise AuditCacheError("Audit cache JSON is corrupt; use --rebuild-cache.")
        record["location"] = {
            "country_code": row["country_code"],
            "country": row["country"],
            "region": row["region"],
            "city": row["city"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "time_zone": row["time_zone"],
            "status": row["geoip_status"],
            "resolved_at": row["geoip_resolved_at"],
            "geoip_build_epoch": row["geoip_build_epoch"],
        }
        return record


def _sanitized_sync_error(exc: Exception) -> str:
    if isinstance(exc, AuditCacheError):
        return str(exc)
    return "Firestore synchronization failed. Verify Application Default Credentials and connectivity."


__all__ = [
    "AskAuditSource",
    "AuditCache",
    "AuditCacheError",
    "AuditSourcePage",
    "GeoIpResolver",
    "LocationSnapshot",
]
