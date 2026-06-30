from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from uni_rag_agent.config import Config

TEST_TIMESTAMP = "2026-06-27T00:00:00+00:00"


@dataclass(frozen=True)
class MinimalChunkRows:
    file_id: int
    extraction_run_id: int
    extracted_document_id: int
    chunk_id: int


@dataclass(frozen=True)
class SearchResultRows:
    search_run_id: int
    search_result_id: int


def insert_minimal_chunk(
    connection: sqlite3.Connection,
    config: Config,
    *,
    filename: str = "notes.md",
    relative_path: str | None = None,
    extension: str = ".md",
    category: str = "document",
    source_type: str = "document",
    text: str = "BM25",
    timestamp: str = TEST_TIMESTAMP,
) -> MinimalChunkRows:
    relative_path = relative_path or filename
    file_id = connection.execute(
        """
        INSERT INTO files (
            path,
            relative_path,
            filename,
            extension,
            size_bytes,
            category,
            index_status,
            discovered_at,
            last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(config.courses_root / relative_path),
            relative_path,
            filename,
            extension,
            len(text.encode("utf-8")),
            category,
            "indexed",
            timestamp,
            timestamp,
        ),
    ).lastrowid
    extraction_run_id = connection.execute(
        """
        INSERT INTO extraction_runs (started_at, status, config_json)
        VALUES (?, ?, ?)
        """,
        (timestamp, "completed", "{}"),
    ).lastrowid
    extracted_document_id = connection.execute(
        """
        INSERT INTO extracted_documents (
            file_id,
            extraction_run_id,
            extractor_name,
            status,
            text_length,
            chunk_count,
            extracted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_id,
            extraction_run_id,
            "test-extractor",
            "indexed",
            len(text),
            1,
            timestamp,
        ),
    ).lastrowid
    chunk_id = connection.execute(
        """
        INSERT INTO chunks (
            file_id,
            extracted_document_id,
            chunk_uid,
            source_type,
            chunk_index,
            text,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_id,
            extracted_document_id,
            f"file-{file_id}-chunk-0",
            source_type,
            0,
            text,
            timestamp,
        ),
    ).lastrowid
    return MinimalChunkRows(
        file_id=int(file_id),
        extraction_run_id=int(extraction_run_id),
        extracted_document_id=int(extracted_document_id),
        chunk_id=int(chunk_id),
    )


def insert_search_result(
    connection: sqlite3.Connection,
    *,
    chunk_id: int,
    file_id: int,
    query: str = "bm25",
    started_at: str = TEST_TIMESTAMP,
    finished_at: str | None = TEST_TIMESTAMP,
) -> SearchResultRows:
    search_run_id = connection.execute(
        """
        INSERT INTO search_runs (
            query,
            query_type,
            router_output_json,
            searched_courses_json,
            searched_indexes_json,
            keyword_terms_json,
            semantic_queries_json,
            started_at,
            finished_at,
            status,
            weaknesses_json,
            error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query,
            "concept_explanation",
            "{}",
            "[]",
            "[]",
            '["bm25"]',
            "[]",
            started_at,
            finished_at,
            "completed",
            None,
            None,
        ),
    ).lastrowid
    search_result_id = connection.execute(
        """
        INSERT INTO search_results (
            search_run_id,
            chunk_id,
            file_id,
            retrieval_method,
            rank,
            score,
            selected_for_evidence,
            result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            search_run_id,
            chunk_id,
            file_id,
            "keyword",
            1,
            1.0,
            1,
            "{}",
        ),
    ).lastrowid
    return SearchResultRows(
        search_run_id=int(search_run_id),
        search_result_id=int(search_result_id),
    )
