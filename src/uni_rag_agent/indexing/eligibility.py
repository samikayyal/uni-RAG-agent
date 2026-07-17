"""Shared current-chunk eligibility helpers for keyword and vector indexing.

Both keyword (`chunk_fts`) and vector (ChromaDB) indexing must index the same
set of chunks: chunks whose joined source file is currently `index_status =
'indexed'` and whose `source_type` is one of the logical retrieval categories.
This module centralizes that DEC-029 current-file-only contract so the two
indexing paths cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from uni_rag_agent.search_contracts import (
    ELIGIBLE_SOURCE_TYPES,
    INDEX_TO_SOURCE_TYPE,
    SOURCE_TYPE_TO_INDEX,
    source_types_for_indexes,
    validate_logical_index,
)


def current_chunk_where_sql(
    source_types: Sequence[str],
    *,
    require_non_empty_text: bool,
) -> str:
    """Build the current-file-only WHERE predicate shared by both indexes.

    The returned SQL expects the source-type values to be bound positionally in
    the same order as ``source_types``.
    """
    clauses = [
        "files.index_status = 'indexed'",
        f"chunks.source_type IN ({placeholders(source_types)})",
    ]
    if require_non_empty_text:
        clauses.append("TRIM(chunks.text) <> ''")
    return " AND ".join(clauses)


def placeholders(values: Sequence[object]) -> str:
    return ", ".join("?" for _ in values)
