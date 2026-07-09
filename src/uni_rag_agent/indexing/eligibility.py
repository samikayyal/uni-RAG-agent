"""Shared current-chunk eligibility helpers for keyword and vector indexing.

Both keyword (`chunk_fts`) and vector (ChromaDB) indexing must index the same
set of chunks: chunks whose joined source file is currently `index_status =
'indexed'` and whose `source_type` is one of the logical retrieval categories.
This module centralizes that DEC-029 current-file-only contract so the two
indexing paths cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

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

SOURCE_TYPE_TO_INDEX = {
    source_type: index_name for index_name, source_type in INDEX_TO_SOURCE_TYPE.items()
}


def source_types_for_indexes(
    indexes: Sequence[str] | None,
    *,
    error: type[Exception],
) -> tuple[str, ...] | None:
    """Map logical index names to chunk source types.

    Returns ``None`` when ``indexes`` is ``None`` (meaning "all eligible source
    types") and an empty tuple when ``indexes`` is an empty sequence (meaning
    "no indexes selected"). Unknown index names raise ``error`` so callers can
    surface a domain-specific exception (keyword vs semantic search).
    """
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
        raise error(
            f"Unknown logical index name(s): {', '.join(unknown)}. "
            f"Allowed indexes: {allowed}"
        )
    return tuple(source_types)


def validate_logical_index(index_name: str, *, error: type[Exception]) -> str:
    """Return the chunk source type for one logical index or raise ``error``."""
    source_type = INDEX_TO_SOURCE_TYPE.get(index_name)
    if source_type is None:
        allowed = ", ".join(sorted(INDEX_TO_SOURCE_TYPE))
        raise error(
            f"Unknown logical index name: {index_name}. Allowed indexes: {allowed}"
        )
    return source_type


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
