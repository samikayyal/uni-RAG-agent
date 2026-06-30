"""Storage helpers for generated local app data."""

from .core import (
    REQUIRED_TABLES,
    StorageCheckResult,
    StorageError,
    check_fts5_available,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)

__all__ = [
    "REQUIRED_TABLES",
    "StorageCheckResult",
    "StorageError",
    "check_fts5_available",
    "check_storage",
    "connect_sqlite",
    "connect_sqlite_read_only",
    "ensure_data_dirs",
    "initialize_schema",
]
