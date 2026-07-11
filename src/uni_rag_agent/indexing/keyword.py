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

from .eligibility import (
    ELIGIBLE_SOURCE_TYPES,
    INDEX_TO_SOURCE_TYPE,
    current_chunk_where_sql,
    source_types_for_indexes,
)
from .models import KeywordIndexError, KeywordIndexResult, KeywordSearchError

__all__ = [
    "ELIGIBLE_SOURCE_TYPES",
    "INDEX_TO_SOURCE_TYPE",
    "keyword_query_terms",
    "keyword_search",
    "keyword_search_terms",
    "sync_keyword_index",
]

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
    *,
    courses: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    if course is not None and courses is not None:
        raise KeywordSearchError("Specify either course or courses, not both")
    if courses is not None and not courses:
        return []
    if courses is None and course is not None:
        courses = (course,)
    return _keyword_search(
        config,
        match_query=_plain_text_to_fts_query(query),
        courses=courses,
        indexes=indexes,
        top_k=top_k,
    )


def keyword_search_terms(
    config: Config,
    terms: Sequence[str],
    *,
    courses: Sequence[str] | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Search routed terms, retaining whitespace-containing phrases."""
    if courses is not None and not courses:
        return []
    normalized_terms = tuple(
        _strip_outer_quotes(term.strip()) for term in terms if term.strip()
    )
    if not normalized_terms:
        raise KeywordSearchError(
            "Keyword query must contain at least one word or number."
        )
    match_query = " OR ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"' for term in normalized_terms
    )
    return _keyword_search(
        config,
        match_query=match_query,
        courses=courses,
        indexes=indexes,
        top_k=top_k,
    )


def _strip_outer_quotes(term: str) -> str:
    if len(term) >= 2 and term[0] == term[-1] == '"':
        return term[1:-1].replace('""', '"').strip()
    return term


def _keyword_search(
    config: Config,
    *,
    match_query: str,
    courses: Sequence[str] | None,
    indexes: Sequence[str] | None,
    top_k: int | None,
) -> list[RetrievalResult]:
    limit = top_k if top_k is not None else config.keyword_top_k
    if limit <= 0:
        raise KeywordSearchError("top_k must be greater than zero")

    source_types = source_types_for_indexes(indexes, error=KeywordSearchError)
    if source_types == ():
        return []
    search_source_types = (
        source_types if source_types is not None else ELIGIBLE_SOURCE_TYPES
    )

    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise KeywordSearchError(f"Keyword search storage check failed: {details}")

    canonical_courses = _canonical_courses(config, courses)
    if courses is not None and not canonical_courses:
        return []
    sql, params = _build_search_sql(
        match_query=match_query,
        courses=canonical_courses,
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
            file_category=row["file_category"],
            file_index_status=row["file_index_status"],
            reason_not_indexed=row["reason_not_indexed"],
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


def _build_search_sql(
    *,
    match_query: str,
    courses: Sequence[str] | None,
    source_types: Sequence[str],
    limit: int,
) -> tuple[str, list[object]]:
    where = [
        "chunk_fts MATCH ?",
        current_chunk_where_sql(source_types, require_non_empty_text=False),
    ]
    params: list[object] = [match_query, *source_types]

    if courses is not None:
        where.append(f"courses.name IN ({', '.join('?' for _ in courses)})")
        params.extend(courses)

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
            files.category AS file_category,
            files.index_status AS file_index_status,
            files.reason_not_indexed AS reason_not_indexed,
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


def _canonical_courses(
    config: Config,
    courses: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if courses is None:
        return None
    requested: list[str] = []
    seen: set[str] = set()
    for course in courses:
        normalized = course.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            requested.append(course.strip())
    if not requested:
        return ()
    placeholders = ", ".join("?" for _ in requested)
    with closing(connect_sqlite_read_only(config)) as connection:
        rows = connection.execute(
            f"SELECT name FROM courses WHERE LOWER(name) IN ({placeholders})",
            [value.casefold() for value in requested],
        ).fetchall()
    canonical_by_key = {str(row["name"]).casefold(): str(row["name"]) for row in rows}
    return tuple(
        canonical_by_key[value.casefold()]
        for value in requested
        if value.casefold() in canonical_by_key
    )


def _count_fts_rows(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()
    return int(row[0])


def _count_current_eligible_chunks(connection: sqlite3.Connection) -> int:
    where_sql = current_chunk_where_sql(
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


def _insert_keyword_rows_sql() -> str:
    where_sql = current_chunk_where_sql(
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
