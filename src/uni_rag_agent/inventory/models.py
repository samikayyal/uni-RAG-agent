"""Inventory data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class InventoryError(RuntimeError):
    """Raised when an inventory run cannot be completed."""


@dataclass(frozen=True)
class FileClassification:
    extension: str
    category: str
    index_status: str
    reason_not_indexed: str | None


@dataclass(frozen=True)
class CourseRecord:
    name: str
    path: str
    file_count: int
    total_bytes: int
    timestamp: str


@dataclass(frozen=True)
class FileRecord:
    course_id: int | None
    path: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    modified_at: str | None
    content_hash: str | None
    category: str
    index_status: str
    reason_not_indexed: str | None
    timestamp: str


@dataclass(frozen=True)
class InventoryCourseSummary:
    course_id: int
    name: str
    path: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class InventoryRunResult:
    run_id: int
    started_at: str
    finished_at: str
    status: str
    courses_seen: int
    files_seen: int
    files_pending: int
    files_metadata_only: int
    files_failed: int
    files_missing: int
    bytes_seen: int
    by_course: tuple[InventoryCourseSummary, ...]
    by_category: Mapping[str, int]
    by_extension: Mapping[str, int]
    by_status: Mapping[str, int]
    by_reason: Mapping[str, int]
    diagnostics: tuple[str, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "courses_seen": self.courses_seen,
            "files_seen": self.files_seen,
            "files_pending": self.files_pending,
            "files_metadata_only": self.files_metadata_only,
            "files_failed": self.files_failed,
            "files_missing": self.files_missing,
            "bytes_seen": self.bytes_seen,
            "by_course": [course.__dict__ for course in self.by_course],
            "by_category": dict(self.by_category),
            "by_extension": dict(self.by_extension),
            "by_status": dict(self.by_status),
            "by_reason": dict(self.by_reason),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class InventorySummary:
    courses_total: int
    files_total: int
    files_missing: int
    bytes_total: int
    latest_inventory_run_id: int | None
    latest_inventory_started_at: str | None
    by_course: tuple[InventoryCourseSummary, ...]
    by_category: Mapping[str, int]
    by_extension: Mapping[str, int]
    by_status: Mapping[str, int]
    by_reason: Mapping[str, int]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "courses_total": self.courses_total,
            "files_total": self.files_total,
            "files_missing": self.files_missing,
            "bytes_total": self.bytes_total,
            "latest_inventory_run_id": self.latest_inventory_run_id,
            "latest_inventory_started_at": self.latest_inventory_started_at,
            "by_course": [course.__dict__ for course in self.by_course],
            "by_category": dict(self.by_category),
            "by_extension": dict(self.by_extension),
            "by_status": dict(self.by_status),
            "by_reason": dict(self.by_reason),
        }
