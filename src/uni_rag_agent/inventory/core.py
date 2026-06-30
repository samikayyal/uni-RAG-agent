"""Course archive inventory and file classification."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from uni_rag_agent.config import Config
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema

from .classification import (
    EXTRACTABLE_CATEGORIES,
    METADATA_ONLY_CATEGORIES,
    classify_file,
)
from .file_io import (
    discover_course_entries as _discover_course_entries,
    relative_path as _relative_path,
    sha256_file,
    timestamp_from_epoch as _timestamp_from_epoch,
    utc_now as _utc_now,
    walk_files as _walk_files,
)
from .models import (
    CourseRecord,
    FileClassification,
    FileRecord,
    InventoryCourseSummary,
    InventoryError,
    InventoryRunResult,
    InventorySummary,
)

MISSING_REASON = "missing from latest inventory run"
TRANSIENT_INVENTORY_FAILURE_PREFIXES = (
    "metadata read failed:",
    "hashing failed:",
)


def inventory_courses(config: Config) -> InventoryRunResult:
    """Run an idempotent inventory pass over the configured Courses root."""
    ensure_data_dirs(config)
    with closing(connect_sqlite(config)) as connection:
        initialize_schema(connection)
        return _inventory_courses(connection, config)


def load_inventory_summary(config: Config) -> InventorySummary:
    """Read aggregate inventory state from SQLite without traversing Courses."""
    if not config.sqlite_path.is_file():
        raise InventoryError(f"SQLite database does not exist: {config.sqlite_path}")
    with closing(connect_sqlite(config)) as connection:
        latest_run_id, latest_started_at = _latest_inventory_run(connection)
        by_course = _load_course_summaries(connection)
        by_category = _count_by(connection, "files", "category")
        by_extension = _count_by(connection, "files", "extension")
        by_status = _count_by(connection, "files", "index_status")
        by_reason = _count_by_reason(connection)
        row = connection.execute(
            "SELECT COUNT(*) AS file_count, COALESCE(SUM(size_bytes), 0) AS bytes_total "
            "FROM files"
        ).fetchone()

    files_total = int(row["file_count"] if row else 0)
    bytes_total = int(row["bytes_total"] if row else 0)
    return InventorySummary(
        courses_total=len(by_course),
        files_total=files_total,
        files_missing=int(by_reason.get(MISSING_REASON, 0)),
        bytes_total=bytes_total,
        latest_inventory_run_id=latest_run_id,
        latest_inventory_started_at=latest_started_at,
        by_course=by_course,
        by_category=by_category,
        by_extension=by_extension,
        by_status=by_status,
        by_reason=by_reason,
    )


def upsert_course(connection: sqlite3.Connection, course: CourseRecord) -> int:
    connection.execute(
        """
        INSERT INTO courses (name, path, file_count, total_bytes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            path = excluded.path,
            file_count = excluded.file_count,
            total_bytes = excluded.total_bytes,
            updated_at = excluded.updated_at
        """,
        (
            course.name,
            course.path,
            course.file_count,
            course.total_bytes,
            course.timestamp,
            course.timestamp,
        ),
    )
    row = connection.execute(
        "SELECT id FROM courses WHERE name = ?",
        (course.name,),
    ).fetchone()
    if row is None:
        raise InventoryError(f"Failed to upsert course: {course.name}")
    return int(row["id"])


def update_course_totals(
    connection: sqlite3.Connection,
    *,
    course_id: int,
    file_count: int,
    total_bytes: int,
    timestamp: str,
) -> None:
    connection.execute(
        """
        UPDATE courses
        SET file_count = ?, total_bytes = ?, updated_at = ?
        WHERE id = ?
        """,
        (file_count, total_bytes, timestamp, course_id),
    )


def upsert_file(connection: sqlite3.Connection, file_record: FileRecord) -> int:
    connection.execute(
        """
        INSERT INTO files (
            course_id,
            path,
            relative_path,
            filename,
            extension,
            size_bytes,
            modified_at,
            content_hash,
            category,
            index_status,
            reason_not_indexed,
            discovered_at,
            last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            course_id = excluded.course_id,
            relative_path = excluded.relative_path,
            filename = excluded.filename,
            extension = excluded.extension,
            size_bytes = excluded.size_bytes,
            modified_at = excluded.modified_at,
            content_hash = excluded.content_hash,
            category = excluded.category,
            index_status = excluded.index_status,
            reason_not_indexed = excluded.reason_not_indexed,
            last_seen_at = excluded.last_seen_at
        """,
        (
            file_record.course_id,
            file_record.path,
            file_record.relative_path,
            file_record.filename,
            file_record.extension,
            file_record.size_bytes,
            file_record.modified_at,
            file_record.content_hash,
            file_record.category,
            file_record.index_status,
            file_record.reason_not_indexed,
            file_record.timestamp,
            file_record.timestamp,
        ),
    )
    row = connection.execute(
        "SELECT id FROM files WHERE path = ?",
        (file_record.path,),
    ).fetchone()
    if row is None:
        raise InventoryError(f"Failed to upsert file: {file_record.path}")
    return int(row["id"])


def mark_missing_files(
    connection: sqlite3.Connection,
    seen_paths: set[str],
    inventory_timestamp: str,
) -> int:
    """Mark files not touched by this inventory run as soft-deleted."""
    del seen_paths
    cursor = connection.execute(
        """
        UPDATE files
        SET index_status = 'skipped',
            reason_not_indexed = ?
        WHERE last_seen_at IS NULL OR last_seen_at <> ?
        """,
        (MISSING_REASON, inventory_timestamp),
    )
    return int(cursor.rowcount)


def reset_missing_course_totals(
    connection: sqlite3.Connection,
    seen_course_ids: set[int],
    inventory_timestamp: str,
) -> int:
    """Clear current inventory totals for course folders absent from this run."""
    if seen_course_ids:
        placeholders = ",".join("?" for _ in seen_course_ids)
        cursor = connection.execute(
            f"""
            UPDATE courses
            SET file_count = 0,
                total_bytes = 0,
                updated_at = ?
            WHERE id NOT IN ({placeholders})
            """,
            (inventory_timestamp, *sorted(seen_course_ids)),
        )
    else:
        cursor = connection.execute(
            """
            UPDATE courses
            SET file_count = 0,
                total_bytes = 0,
                updated_at = ?
            """,
            (inventory_timestamp,),
        )
    return int(cursor.rowcount)


def _inventory_courses(
    connection: sqlite3.Connection,
    config: Config,
) -> InventoryRunResult:
    started_at = _utc_now()
    run_id = _start_inventory_run(connection, config, started_at)
    connection.commit()
    diagnostics: list[str] = []
    seen_paths: set[str] = set()
    by_category: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_course: list[InventoryCourseSummary] = []
    seen_course_ids: set[int] = set()
    files_seen = 0
    files_pending = 0
    files_metadata_only = 0
    files_failed = 0
    bytes_seen = 0

    try:
        root_files, course_dirs = _discover_course_entries(
            config.courses_root, diagnostics
        )

        for root_file in root_files:
            file_record = _build_file_record(
                connection=connection,
                config=config,
                path=root_file,
                course_id=None,
                timestamp=started_at,
                diagnostics=diagnostics,
            )
            upsert_file(connection, file_record)
            seen_paths.add(file_record.path)
            files_seen += 1
            bytes_seen += file_record.size_bytes
            _add_file_counts(
                file_record,
                by_category=by_category,
                by_extension=by_extension,
                by_status=by_status,
                by_reason=by_reason,
            )
            if file_record.index_status == "pending":
                files_pending += 1
            elif file_record.index_status == "metadata_only":
                files_metadata_only += 1
            elif file_record.index_status == "failed":
                files_failed += 1

        for course_path in course_dirs:
            course_id = upsert_course(
                connection,
                CourseRecord(
                    name=course_path.name,
                    path=str(course_path),
                    file_count=0,
                    total_bytes=0,
                    timestamp=started_at,
                ),
            )
            seen_course_ids.add(course_id)
            course_file_count = 0
            course_total_bytes = 0
            for file_path in _walk_files(course_path, diagnostics):
                file_record = _build_file_record(
                    connection=connection,
                    config=config,
                    path=file_path,
                    course_id=course_id,
                    timestamp=started_at,
                    diagnostics=diagnostics,
                )
                upsert_file(connection, file_record)
                seen_paths.add(file_record.path)
                files_seen += 1
                course_file_count += 1
                bytes_seen += file_record.size_bytes
                course_total_bytes += file_record.size_bytes
                _add_file_counts(
                    file_record,
                    by_category=by_category,
                    by_extension=by_extension,
                    by_status=by_status,
                    by_reason=by_reason,
                )
                if file_record.index_status == "pending":
                    files_pending += 1
                elif file_record.index_status == "metadata_only":
                    files_metadata_only += 1
                elif file_record.index_status == "failed":
                    files_failed += 1

            update_course_totals(
                connection,
                course_id=course_id,
                file_count=course_file_count,
                total_bytes=course_total_bytes,
                timestamp=started_at,
            )
            by_course.append(
                InventoryCourseSummary(
                    course_id=course_id,
                    name=course_path.name,
                    path=str(course_path),
                    file_count=course_file_count,
                    total_bytes=course_total_bytes,
                )
            )

        reset_missing_course_totals(connection, seen_course_ids, started_at)
        files_missing = mark_missing_files(connection, seen_paths, started_at)
        finished_at = _utc_now()
        status = "completed"
        _finish_inventory_run(
            connection,
            run_id=run_id,
            finished_at=finished_at,
            status=status,
            files_seen=files_seen,
            files_indexed=0,
            files_metadata_only=files_metadata_only,
            files_failed=files_failed,
            error=None,
        )
        connection.commit()
    except Exception as exc:
        connection.rollback()
        with connection:
            _finish_inventory_run(
                connection,
                run_id=run_id,
                finished_at=_utc_now(),
                status="failed",
                files_seen=files_seen,
                files_indexed=0,
                files_metadata_only=files_metadata_only,
                files_failed=files_failed,
                error=str(exc),
            )
        raise

    return InventoryRunResult(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        courses_seen=len(course_dirs),
        files_seen=files_seen,
        files_pending=files_pending,
        files_metadata_only=files_metadata_only,
        files_failed=files_failed,
        files_missing=files_missing,
        bytes_seen=bytes_seen,
        by_course=tuple(by_course),
        by_category=dict(sorted(by_category.items())),
        by_extension=dict(sorted(by_extension.items())),
        by_status=dict(sorted(by_status.items())),
        by_reason=dict(sorted(by_reason.items())),
        diagnostics=tuple(diagnostics),
    )


def _build_file_record(
    *,
    connection: sqlite3.Connection,
    config: Config,
    path: Path,
    course_id: int | None,
    timestamp: str,
    diagnostics: list[str],
) -> FileRecord:
    classification = classify_file(path)
    path_text = str(path)
    relative_path = _relative_path(path, config.courses_root)
    existing = connection.execute(
        """
        SELECT modified_at, size_bytes, content_hash, index_status, reason_not_indexed
        FROM files
        WHERE path = ?
        """,
        (path_text,),
    ).fetchone()

    try:
        stat_result = path.stat()
    except OSError as exc:
        diagnostics.append(f"Could not stat file {path_text}: {exc}")
        return FileRecord(
            course_id=course_id,
            path=path_text,
            relative_path=relative_path,
            filename=path.name,
            extension=classification.extension,
            size_bytes=0,
            modified_at=None,
            content_hash=None,
            category=classification.category,
            index_status="failed",
            reason_not_indexed=f"metadata read failed: {exc}",
            timestamp=timestamp,
        )

    size_bytes = int(stat_result.st_size)
    modified_at = _timestamp_from_epoch(stat_result.st_mtime)
    needs_hash = _needs_hash(existing, modified_at, size_bytes)
    content_hash = existing["content_hash"] if existing is not None else None
    hash_failed = False

    if needs_hash:
        try:
            content_hash = sha256_file(path)
        except OSError as exc:
            diagnostics.append(f"Could not hash file {path_text}: {exc}")
            hash_failed = True

    if hash_failed:
        index_status = "failed"
        reason_not_indexed = f"hashing failed: {path_text}"
    else:
        index_status, reason_not_indexed = _next_inventory_status(
            existing=existing,
            classification=classification,
            modified_at=modified_at,
            size_bytes=size_bytes,
            content_hash=content_hash,
        )

    return FileRecord(
        course_id=course_id,
        path=path_text,
        relative_path=relative_path,
        filename=path.name,
        extension=classification.extension,
        size_bytes=size_bytes,
        modified_at=modified_at,
        content_hash=content_hash,
        category=classification.category,
        index_status=index_status,
        reason_not_indexed=reason_not_indexed,
        timestamp=timestamp,
    )


def _needs_hash(
    existing: sqlite3.Row | None,
    modified_at: str | None,
    size_bytes: int,
) -> bool:
    if existing is None:
        return True
    if existing["modified_at"] != modified_at:
        return True
    if int(existing["size_bytes"]) != size_bytes:
        return True
    return existing["content_hash"] is None


def _next_inventory_status(
    *,
    existing: sqlite3.Row | None,
    classification: FileClassification,
    modified_at: str | None,
    size_bytes: int,
    content_hash: str | None,
) -> tuple[str, str | None]:
    if existing is None:
        return classification.index_status, classification.reason_not_indexed

    old_hash = existing["content_hash"]
    unchanged_metadata = (
        existing["modified_at"] == modified_at
        and int(existing["size_bytes"]) == size_bytes
    )
    unchanged_content = old_hash is not None and old_hash == content_hash
    unchanged = unchanged_metadata or unchanged_content

    if classification.category in METADATA_ONLY_CATEGORIES:
        return classification.index_status, classification.reason_not_indexed

    existing_status = existing["index_status"]
    existing_reason = existing["reason_not_indexed"]
    if _is_transient_inventory_failure(existing_status, existing_reason):
        return classification.index_status, classification.reason_not_indexed

    if not unchanged:
        return classification.index_status, classification.reason_not_indexed

    if existing_status == "skipped" and existing_reason == MISSING_REASON:
        return classification.index_status, classification.reason_not_indexed
    if existing_status in {"pending", "indexed", "failed", "skipped"}:
        return str(existing_status), existing_reason
    return classification.index_status, classification.reason_not_indexed


def _is_transient_inventory_failure(status: str, reason: str | None) -> bool:
    if status != "failed" or reason is None:
        return False
    return reason.startswith(TRANSIENT_INVENTORY_FAILURE_PREFIXES)


def _add_file_counts(
    file_record: FileRecord,
    *,
    by_category: Counter[str],
    by_extension: Counter[str],
    by_status: Counter[str],
    by_reason: Counter[str],
) -> None:
    by_category[file_record.category] += 1
    by_extension[file_record.extension or "<none>"] += 1
    by_status[file_record.index_status] += 1
    if file_record.reason_not_indexed:
        by_reason[file_record.reason_not_indexed] += 1


def _start_inventory_run(
    connection: sqlite3.Connection,
    config: Config,
    started_at: str,
) -> int:
    config_json = json.dumps(
        {
            "run_type": "inventory",
            "courses_root": str(config.courses_root),
            "data_dir": str(config.data_dir),
            "sqlite_path": str(config.sqlite_path),
            "classification_categories": sorted(
                EXTRACTABLE_CATEGORIES | METADATA_ONLY_CATEGORIES
            ),
        },
        sort_keys=True,
    )
    cursor = connection.execute(
        """
        INSERT INTO extraction_runs (
            started_at,
            status,
            config_json,
            files_seen,
            files_indexed,
            files_metadata_only,
            files_failed
        )
        VALUES (?, 'running', ?, 0, 0, 0, 0)
        """,
        (started_at, config_json),
    )
    return int(cursor.lastrowid)


def _finish_inventory_run(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    finished_at: str,
    status: str,
    files_seen: int,
    files_indexed: int,
    files_metadata_only: int,
    files_failed: int,
    error: str | None,
) -> None:
    connection.execute(
        """
        UPDATE extraction_runs
        SET finished_at = ?,
            status = ?,
            files_seen = ?,
            files_indexed = ?,
            files_metadata_only = ?,
            files_failed = ?,
            error = ?
        WHERE id = ?
        """,
        (
            finished_at,
            status,
            files_seen,
            files_indexed,
            files_metadata_only,
            files_failed,
            error,
            run_id,
        ),
    )


def _latest_inventory_run(
    connection: sqlite3.Connection,
) -> tuple[int | None, str | None]:
    rows = connection.execute(
        """
        SELECT id, started_at, config_json
        FROM extraction_runs
        ORDER BY id DESC
        """
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["config_json"])
        except json.JSONDecodeError:
            continue
        if payload.get("run_type") == "inventory":
            return int(row["id"]), str(row["started_at"])
    return None, None


def _load_course_summaries(
    connection: sqlite3.Connection,
) -> tuple[InventoryCourseSummary, ...]:
    rows = connection.execute(
        """
        SELECT id, name, path, file_count, total_bytes
        FROM courses
        ORDER BY name COLLATE NOCASE
        """
    ).fetchall()
    return tuple(
        InventoryCourseSummary(
            course_id=int(row["id"]),
            name=str(row["name"]),
            path=str(row["path"]),
            file_count=int(row["file_count"]),
            total_bytes=int(row["total_bytes"]),
        )
        for row in rows
    )


def _count_by(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> dict[str, int]:
    rows = connection.execute(
        f"""
        SELECT {column} AS value, COUNT(*) AS count
        FROM {table}
        GROUP BY {column}
        ORDER BY {column}
        """
    ).fetchall()
    return {str(row["value"]): int(row["count"]) for row in rows}


def _count_by_reason(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT reason_not_indexed AS value, COUNT(*) AS count
        FROM files
        WHERE reason_not_indexed IS NOT NULL AND reason_not_indexed <> ''
        GROUP BY reason_not_indexed
        ORDER BY reason_not_indexed
        """
    ).fetchall()
    return {str(row["value"]): int(row["count"]) for row in rows}
