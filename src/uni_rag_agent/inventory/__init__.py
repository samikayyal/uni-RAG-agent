"""Inventory and file classification helpers."""

from .classification import (
    EXTRACTABLE_CATEGORIES,
    EXTENSION_CATEGORY_MAP,
    METADATA_ONLY_CATEGORIES,
    classify_file,
)
from .core import (
    MISSING_REASON,
    inventory_courses,
    load_inventory_summary,
    mark_missing_files,
    update_course_totals,
    upsert_course,
    upsert_file,
)
from .file_io import sha256_file
from .models import (
    CourseRecord,
    FileClassification,
    FileRecord,
    InventoryCourseSummary,
    InventoryError,
    InventoryRunResult,
    InventorySummary,
)

__all__ = [
    "EXTRACTABLE_CATEGORIES",
    "EXTENSION_CATEGORY_MAP",
    "METADATA_ONLY_CATEGORIES",
    "MISSING_REASON",
    "CourseRecord",
    "FileClassification",
    "FileRecord",
    "InventoryCourseSummary",
    "InventoryError",
    "InventoryRunResult",
    "InventorySummary",
    "classify_file",
    "inventory_courses",
    "load_inventory_summary",
    "mark_missing_files",
    "sha256_file",
    "update_course_totals",
    "upsert_course",
    "upsert_file",
]
