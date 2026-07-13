"""Storage initialization and health checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from uni_rag_agent.config import Config, ConfigError, validate_config

REQUIRED_TABLES = (
    "courses",
    "files",
    "extraction_runs",
    "extracted_documents",
    "chunks",
    "chunk_fts",
    "embeddings",
    "data_summaries",
    "search_runs",
    "search_result_sets",
    "search_results",
    "evidence_packets",
    "answers",
)


class StorageError(RuntimeError):
    """Raised when local storage cannot be initialized or inspected."""


@dataclass(frozen=True)
class StorageCheckResult:
    data_dir: Path
    sqlite_path: Path
    extracted_dir: Path
    chroma_dir: Path
    runs_dir: Path
    sqlite_exists: bool
    fts5_available: bool
    required_tables_present: tuple[str, ...]
    missing_tables: tuple[str, ...]
    diagnostics: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.sqlite_exists
            and self.fts5_available
            and not self.missing_tables
            and not self.diagnostics
        )

    def as_safe_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "data_dir": str(self.data_dir),
            "sqlite_path": str(self.sqlite_path),
            "extracted_dir": str(self.extracted_dir),
            "chroma_dir": str(self.chroma_dir),
            "runs_dir": str(self.runs_dir),
            "sqlite_exists": self.sqlite_exists,
            "fts5_available": self.fts5_available,
            "required_tables_present": list(self.required_tables_present),
            "missing_tables": list(self.missing_tables),
            "diagnostics": list(self.diagnostics),
        }


def ensure_data_dirs(config: Config) -> None:
    """Create generated app directories without touching the Courses archive."""
    validate_config(config)
    for path in (
        config.data_dir,
        config.extracted_dir,
        config.chroma_dir,
        config.runs_dir,
        config.sqlite_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def connect_sqlite(config: Config) -> sqlite3.Connection:
    """Open the configured SQLite database and enable foreign keys."""
    connection = sqlite3.connect(config.sqlite_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def connect_sqlite_read_only(config: Config) -> sqlite3.Connection:
    """Open the configured SQLite database in read-only/query-only mode."""
    uri = f"{config.sqlite_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the MVP schema from the architecture contract."""
    fts5_available, diagnostic = check_fts5_available(connection)
    if not fts5_available:
        raise StorageError(f"SQLite FTS5 is not available: {diagnostic}")

    try:
        connection.executescript(MVP_SCHEMA_SQL)
        _migrate_search_runs_schema(connection)
        _ensure_search_results_chunk_delete_policy(connection)
        _ensure_embeddings_lifecycle_schema(connection)
        _ensure_search_result_indexes(connection)
        _ensure_evidence_packet_uniqueness(connection)
        connection.commit()
    except StorageError:
        connection.rollback()
        raise
    except sqlite3.Error as exc:
        connection.rollback()
        raise StorageError(f"SQLite schema initialization failed: {exc}") from exc


def check_storage(config: Config) -> StorageCheckResult:
    """Inspect generated storage paths and schema without creating missing files."""
    diagnostics: list[str] = []

    try:
        validate_config(config)
    except ConfigError as exc:
        diagnostics.append(str(exc))

    for label, path in (
        ("data_dir", config.data_dir),
        ("extracted_dir", config.extracted_dir),
        ("chroma_dir", config.chroma_dir),
        ("runs_dir", config.runs_dir),
    ):
        if not path.exists():
            diagnostics.append(f"{label} does not exist: {path}")
        elif not path.is_dir():
            diagnostics.append(f"{label} is not a directory: {path}")

    sqlite_exists = config.sqlite_path.is_file()
    if not sqlite_exists:
        diagnostics.append(f"SQLite database does not exist: {config.sqlite_path}")

    fts5_available, fts5_diagnostic = _check_fts5_in_memory()
    if not fts5_available:
        diagnostics.append(f"SQLite FTS5 is not available: {fts5_diagnostic}")

    present_tables: tuple[str, ...] = ()
    if sqlite_exists:
        try:
            with sqlite3.connect(config.sqlite_path) as connection:
                present_tables = _required_tables_present(connection)
        except sqlite3.Error as exc:
            diagnostics.append(f"SQLite database cannot be inspected: {exc}")

    missing_tables = tuple(
        table for table in REQUIRED_TABLES if table not in set(present_tables)
    )
    if missing_tables:
        diagnostics.append(f"Missing required tables: {', '.join(missing_tables)}")

    return StorageCheckResult(
        data_dir=config.data_dir,
        sqlite_path=config.sqlite_path,
        extracted_dir=config.extracted_dir,
        chroma_dir=config.chroma_dir,
        runs_dir=config.runs_dir,
        sqlite_exists=sqlite_exists,
        fts5_available=fts5_available,
        required_tables_present=present_tables,
        missing_tables=missing_tables,
        diagnostics=tuple(diagnostics),
    )


def check_fts5_available(connection: sqlite3.Connection) -> tuple[bool, str | None]:
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(value)")
        connection.execute("DROP TABLE temp.fts5_probe")
    except sqlite3.Error as exc:
        return False, str(exc)
    return True, None


def _check_fts5_in_memory() -> tuple[bool, str | None]:
    with sqlite3.connect(":memory:") as connection:
        return check_fts5_available(connection)


def _required_tables_present(connection: sqlite3.Connection) -> tuple[str, ...]:
    placeholders = ",".join("?" for _ in REQUIRED_TABLES)
    rows = connection.execute(
        f"""
        SELECT name
        FROM sqlite_master
        WHERE name IN ({placeholders})
        ORDER BY name
        """,
        REQUIRED_TABLES,
    ).fetchall()
    names = {row[0] for row in rows}
    return tuple(table for table in REQUIRED_TABLES if table in names)


def _ensure_search_results_chunk_delete_policy(connection: sqlite3.Connection) -> None:
    """Keep stale chunk cleanup from blocking on historical search result rows."""
    rows = connection.execute("PRAGMA foreign_key_list(search_results)").fetchall()
    for row in rows:
        if row["table"] == "chunks" and row["from"] == "chunk_id":
            if str(row["on_delete"]).upper() == "SET NULL":
                return
            break
    else:
        return

    connection.execute("ALTER TABLE search_results RENAME TO search_results_old")
    connection.execute(
        """
        CREATE TABLE search_results (
            id INTEGER PRIMARY KEY,
            search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
            chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
            file_id INTEGER REFERENCES files(id),
            retrieval_method TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL,
            selected_for_evidence INTEGER NOT NULL DEFAULT 0,
            result_json TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO search_results (
            id,
            search_run_id,
            chunk_id,
            file_id,
            retrieval_method,
            rank,
            score,
            selected_for_evidence,
            result_json
        )
        SELECT
            id,
            search_run_id,
            chunk_id,
            file_id,
            retrieval_method,
            rank,
            score,
            selected_for_evidence,
            result_json
        FROM search_results_old
        """
    )
    connection.execute("DROP TABLE search_results_old")
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_results_run_id
        ON search_results(search_run_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_results_selected
        ON search_results(selected_for_evidence)
        """
    )


def _migrate_search_runs_schema(connection: sqlite3.Connection) -> None:
    """Migrate the pre-Feature-09 search-run columns without losing rows."""
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(search_runs)").fetchall()
    }
    if "query_plan_json" not in columns:
        if "router_output_json" not in columns:
            raise StorageError(
                "Incompatible SQLite schema: search_runs has neither "
                "query_plan_json nor legacy router_output_json; manual review "
                "is required."
            )
        connection.execute(
            "ALTER TABLE search_runs RENAME COLUMN router_output_json TO query_plan_json"
        )
        columns.remove("router_output_json")
        columns.add("query_plan_json")

    if "retrieval_settings_json" not in columns:
        connection.execute(
            """
            ALTER TABLE search_runs
            ADD COLUMN retrieval_settings_json TEXT NOT NULL DEFAULT '{}'
            """
        )


def _ensure_search_result_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_results_run_id
        ON search_results(search_run_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_results_selected
        ON search_results(selected_for_evidence)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_results_run_method_rank
        ON search_results(search_run_id, retrieval_method, rank)
        """
    )


def _ensure_evidence_packet_uniqueness(connection: sqlite3.Connection) -> None:
    duplicate = connection.execute(
        """
        SELECT search_run_id, COUNT(*) AS packet_count
        FROM evidence_packets
        GROUP BY search_run_id
        HAVING COUNT(*) > 1
        ORDER BY search_run_id
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        raise StorageError(
            "Cannot create unique index on evidence_packets.search_run_id: "
            f"duplicate packets exist for search_run_id={duplicate['search_run_id']} "
            "in table evidence_packets; manual review is required."
        )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_packets_search_run
        ON evidence_packets(search_run_id)
        """
    )


def _ensure_embeddings_lifecycle_schema(connection: sqlite3.Connection) -> None:
    """Keep embedding mappings aligned with the physical vector profile.

    Each physical vector collection represents one provider/model/dimension/
    metric profile. A chunk may therefore have mappings for more than one
    physical collection. The migration also retains the cascade rule needed
    when extraction deletes stale chunks.
    """
    rows = connection.execute("PRAGMA foreign_key_list(embeddings)").fetchall()
    has_chunk_cascade = False
    for row in rows:
        if row["table"] == "chunks" and row["from"] == "chunk_id":
            if str(row["on_delete"]).upper() == "CASCADE":
                has_chunk_cascade = True
            break

    unique_sets = {
        tuple(
            index_column["name"]
            for index_column in connection.execute(
                f"PRAGMA index_info({row['name']})"
            ).fetchall()
        )
        for row in connection.execute("PRAGMA index_list(embeddings)").fetchall()
        if row["unique"]
    }
    expected_unique_sets = {
        ("vector_backend", "vector_collection", "vector_id"),
        ("chunk_id", "vector_backend", "vector_collection"),
    }
    if has_chunk_cascade and expected_unique_sets.issubset(unique_sets):
        return

    connection.execute("ALTER TABLE embeddings RENAME TO embeddings_old")
    connection.execute(
        """
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            vector_backend TEXT NOT NULL,
            vector_collection TEXT NOT NULL,
            vector_id TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            UNIQUE(vector_backend, vector_collection, vector_id),
            UNIQUE(chunk_id, vector_backend, vector_collection)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO embeddings (
            id,
            chunk_id,
            vector_backend,
            vector_collection,
            vector_id,
            embedding_model,
            embedding_dim,
            embedded_at
        )
        SELECT
            id,
            chunk_id,
            vector_backend,
            vector_collection,
            vector_id,
            embedding_model,
            embedding_dim,
            embedded_at
        FROM embeddings_old
        """
    )
    connection.execute("DROP TABLE embeddings_old")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)"
    )


MVP_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL UNIQUE,
    file_count INTEGER NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id),
    path TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT,
    content_hash TEXT,
    category TEXT NOT NULL,
    index_status TEXT NOT NULL,
    reason_not_indexed TEXT,
    discovered_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_course_id ON files(course_id);
CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_category ON files(category);
CREATE INDEX IF NOT EXISTS idx_files_index_status ON files(index_status);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    files_metadata_only INTEGER NOT NULL DEFAULT 0,
    files_failed INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS extracted_documents (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    extraction_run_id INTEGER REFERENCES extraction_runs(id),
    extractor_name TEXT NOT NULL,
    extractor_version TEXT,
    status TEXT NOT NULL,
    text_length INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    error TEXT,
    extracted_at TEXT NOT NULL,
    UNIQUE(file_id, extractor_name)
);

CREATE INDEX IF NOT EXISTS idx_extracted_documents_file_id
ON extracted_documents(file_id);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    extracted_document_id INTEGER REFERENCES extracted_documents(id),
    chunk_uid TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    token_count INTEGER,
    location_type TEXT,
    location_value TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_location ON chunks(location_type, location_value);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    title,
    course_name,
    file_path,
    source_type UNINDEXED,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    vector_backend TEXT NOT NULL,
    vector_collection TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedded_at TEXT NOT NULL,
    UNIQUE(vector_backend, vector_collection, vector_id),
    UNIQUE(chunk_id, vector_backend, vector_collection)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);

CREATE TABLE IF NOT EXISTS data_summaries (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    format TEXT NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    table_count INTEGER,
    sheet_count INTEGER,
    schema_json TEXT NOT NULL,
    sample_json TEXT,
    summary_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_id)
);

CREATE TABLE IF NOT EXISTS search_runs (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT,
    query_plan_json TEXT NOT NULL,
    retrieval_settings_json TEXT NOT NULL DEFAULT '{}',
    searched_courses_json TEXT NOT NULL,
    searched_indexes_json TEXT NOT NULL,
    keyword_terms_json TEXT NOT NULL,
    semantic_queries_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    weaknesses_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS search_result_sets (
    id INTEGER PRIMARY KEY,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
    result_set_id TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    query TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    completed_at TEXT NOT NULL,
    UNIQUE(search_run_id, result_set_id)
);

CREATE INDEX IF NOT EXISTS idx_search_result_sets_run_id
ON search_result_sets(search_run_id);

CREATE TABLE IF NOT EXISTS search_results (
    id INTEGER PRIMARY KEY,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    file_id INTEGER REFERENCES files(id),
    retrieval_method TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL,
    selected_for_evidence INTEGER NOT NULL DEFAULT 0,
    result_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_search_results_run_id
ON search_results(search_run_id);
CREATE INDEX IF NOT EXISTS idx_search_results_selected
ON search_results(selected_for_evidence);
CREATE INDEX IF NOT EXISTS idx_search_results_run_method_rank
ON search_results(search_run_id, retrieval_method, rank);

CREATE TABLE IF NOT EXISTS evidence_packets (
    id INTEGER PRIMARY KEY,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
    packet_json TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    evidence_packet_id INTEGER NOT NULL REFERENCES evidence_packets(id),
    answer_text TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    limitations_json TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL
);
"""
