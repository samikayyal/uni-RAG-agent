"""SQLite FTS5 keyword indexing and search."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing

from uni_rag_agent.config import Config
from uni_rag_agent.retrieval import RetrievalResult
from uni_rag_agent.storage import (
    StorageError,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)

from .models import KeywordIndexError, KeywordIndexResult, KeywordSearchError

ELIGIBLE_SOURCE_TYPES = (
    "document",
    "slides",
    "notebook",
    "code",
    "data_schema",
    "transcript",
)

INDEX_TO_SOURCE_TYPE = {
    "document_index": "document",
    "slides_index": "slides",
    "notebook_index": "notebook",
    "code_index": "code",
    "data_schema_index": "data_schema",
    "transcript_index": "transcript",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def sync_keyword_index(config: Config, rebuild: bool = False) -> KeywordIndexResult:
    """Rebuild the current SQLite FTS5 projection from indexed chunks.

    The MVP implementation always rebuilds `chunk_fts`; the `rebuild` argument
    remains for CLI/API compatibility with the explicit `--rebuild` alias.
    """
    try:
        ensure_data_dirs(config)
        with closing(connect_sqlite(config)) as connection:
            initialize_schema(connection)
            rows_removed = _count_fts_rows(connection)
            chunks_seen = _count_current_eligible_chunks(connection)
            connection.execute("DELETE FROM chunk_fts")
            connection.execute(_insert_keyword_rows_sql(), ELIGIBLE_SOURCE_TYPES)
            rows_indexed = _count_fts_rows(connection)
            by_source_type = _count_fts_by_source_type(connection)
            connection.commit()
    except sqlite3.Error as exc:
        raise KeywordIndexError(f"Keyword index sync failed: {exc}") from exc

    diagnostics: list[str] = []
    if rows_indexed == 0:
        diagnostics.append("No eligible indexed chunks found for keyword indexing.")
    if rows_indexed != chunks_seen:
        diagnostics.append(
            f"Indexed {rows_indexed} FTS rows from {chunks_seen} current "
            "eligible chunks; blank chunk text is skipped."
        )

    return KeywordIndexResult(
        rebuild=True,
        rows_removed=rows_removed,
        chunks_seen=chunks_seen,
        rows_indexed=rows_indexed,
        by_source_type=by_source_type,
        diagnostics=tuple(diagnostics),
    )


def keyword_search(
    config: Config,
    query: str,
    course: str | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    limit = top_k if top_k is not None else config.keyword_top_k
    if limit <= 0:
        raise KeywordSearchError("top_k must be greater than zero")

    match_query = _plain_text_to_fts_query(query)
    source_types = _source_types_for_indexes(indexes)
    if source_types == ():
        return []
    search_source_types = (
        source_types if source_types is not None else ELIGIBLE_SOURCE_TYPES
    )

    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise KeywordSearchError(f"Keyword search storage check failed: {details}")

    sql, params = _build_search_sql(
        match_query=match_query,
        course=course,
        source_types=search_source_types,
        limit=limit,
    )

    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            rows = connection.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        raise KeywordSearchError(f"Keyword query could not be executed: {exc}") from exc
    except sqlite3.Error as exc:
        raise StorageError(f"Keyword search could not inspect SQLite: {exc}") from exc

    return [
        RetrievalResult(
            chunk_id=int(row["chunk_id"]),
            file_id=int(row["file_id"]),
            course=row["course"],
            file_path=row["file_path"],
            source_type=row["source_type"],
            location_type=row["location_type"],
            location_value=row["location_value"],
            rank=rank,
            score=-float(row["raw_score"]),
            snippet=row["snippet"] or "",
        )
        for rank, row in enumerate(rows, start=1)
    ]


def keyword_query_terms(query: str) -> tuple[str, ...]:
    """Return the deduplicated plain-text terms used by keyword search."""
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0)
        normalized = token.casefold()
        if normalized not in seen:
            seen.add(normalized)
            tokens.append(token)
    if not tokens:
        raise KeywordSearchError(
            "Keyword query must contain at least one word or number."
        )
    return tuple(tokens)


def _plain_text_to_fts_query(query: str) -> str:
    tokens = keyword_query_terms(query)
    return " OR ".join(f'"{token}"' for token in tokens)


def _source_types_for_indexes(indexes: Sequence[str] | None) -> tuple[str, ...] | None:
    if indexes is None:
        return None
    if not indexes:
        return ()

    source_types: list[str] = []
    unknown: list[str] = []
    for index_name in indexes:
        source_type = INDEX_TO_SOURCE_TYPE.get(index_name)
        if source_type is None:
            unknown.append(index_name)
        elif source_type not in source_types:
            source_types.append(source_type)

    if unknown:
        allowed = ", ".join(sorted(INDEX_TO_SOURCE_TYPE))
        raise KeywordSearchError(
            f"Unknown logical index name(s): {', '.join(unknown)}. "
            f"Allowed indexes: {allowed}"
        )
    return tuple(source_types)


def _build_search_sql(
    *,
    match_query: str,
    course: str | None,
    source_types: Sequence[str],
    limit: int,
) -> tuple[str, list[object]]:
    where = [
        "chunk_fts MATCH ?",
        _current_chunk_where_sql(source_types, require_non_empty_text=False),
    ]
    params: list[object] = [match_query, *source_types]

    if course is not None:
        where.append("LOWER(courses.name) = LOWER(?)")
        params.append(course)

    params.append(limit)
    return (
        f"""
        SELECT
            chunks.id AS chunk_id,
            files.id AS file_id,
            courses.name AS course,
            files.path AS file_path,
            chunks.source_type AS source_type,
            chunks.location_type AS location_type,
            chunks.location_value AS location_value,
            bm25(chunk_fts) AS raw_score,
            snippet(chunk_fts, -1, '[', ']', '...', 32) AS snippet
        FROM chunk_fts
        JOIN chunks ON chunks.id = chunk_fts.chunk_id
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE {" AND ".join(where)}
        ORDER BY raw_score ASC, chunks.id ASC
        LIMIT ?
        """,
        params,
    )


def _count_fts_rows(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()
    return int(row[0])


def _count_current_eligible_chunks(connection: sqlite3.Connection) -> int:
    where_sql = _current_chunk_where_sql(
        ELIGIBLE_SOURCE_TYPES,
        require_non_empty_text=False,
    )
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        WHERE {where_sql}
        """,
        ELIGIBLE_SOURCE_TYPES,
    ).fetchone()
    return int(row[0])


def _count_fts_by_source_type(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT source_type, COUNT(*) AS count
        FROM chunk_fts
        GROUP BY source_type
        ORDER BY source_type
        """
    ).fetchall()
    return {str(row["source_type"]): int(row["count"]) for row in rows}


def _current_chunk_where_sql(
    source_types: Sequence[str],
    *,
    require_non_empty_text: bool,
) -> str:
    clauses = [
        "files.index_status = 'indexed'",
        f"chunks.source_type IN ({_placeholders(source_types)})",
    ]
    if require_non_empty_text:
        clauses.append("TRIM(chunks.text) <> ''")
    return " AND ".join(clauses)


def _placeholders(values: Sequence[object]) -> str:
    return ", ".join("?" for _ in values)


def _insert_keyword_rows_sql() -> str:
    where_sql = _current_chunk_where_sql(
        ELIGIBLE_SOURCE_TYPES,
        require_non_empty_text=True,
    )
    return f"""
    INSERT INTO chunk_fts (chunk_id, text, title, course_name, file_path, source_type)
    SELECT
        chunks.id,
        chunks.text,
        COALESCE(chunks.title, ''),
        COALESCE(courses.name, ''),
        files.path,
        chunks.source_type
    FROM chunks
    JOIN files ON files.id = chunks.file_id
    LEFT JOIN courses ON courses.id = files.course_id
    WHERE {where_sql}
    ORDER BY chunks.id
    """
