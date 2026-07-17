"""Dependency-light contracts shared by indexing, retrieval, and evaluation.

The logical-index taxonomy is a cross-cutting search contract.  Keep its
canonical mapping here and derive every public view from that one decision so
indexing, planning, evidence coverage, and evaluation cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType


LOGICAL_INDEX_TO_SOURCE_TYPE: Mapping[str, str] = MappingProxyType(
    {
        "document_index": "document",
        "slides_index": "slides",
        "notebook_index": "notebook",
        "code_index": "code",
        "data_schema_index": "data_schema",
        "transcript_index": "transcript",
    }
)

LOGICAL_INDEXES = tuple(LOGICAL_INDEX_TO_SOURCE_TYPE)
ELIGIBLE_SOURCE_TYPES = tuple(LOGICAL_INDEX_TO_SOURCE_TYPE.values())
SOURCE_TYPE_TO_LOGICAL_INDEX: Mapping[str, str] = MappingProxyType(
    {
        source_type: logical_index
        for logical_index, source_type in LOGICAL_INDEX_TO_SOURCE_TYPE.items()
    }
)

# Compatibility aliases for existing callers.  They intentionally point to the
# canonical derived views rather than defining another taxonomy.
INDEX_TO_SOURCE_TYPE = LOGICAL_INDEX_TO_SOURCE_TYPE
SOURCE_TYPE_TO_INDEX = SOURCE_TYPE_TO_LOGICAL_INDEX


def source_types_for_indexes(
    indexes: Sequence[str] | None,
    *,
    error: type[Exception],
) -> tuple[str, ...] | None:
    """Map logical index names to source types for a search operation.

    ``None`` means all eligible source types and an empty sequence means no
    indexes selected.  The caller supplies its domain-specific exception type.
    """

    if indexes is None:
        return None
    if not indexes:
        return ()

    source_types: list[str] = []
    unknown: list[str] = []
    for logical_index in indexes:
        source_type = LOGICAL_INDEX_TO_SOURCE_TYPE.get(logical_index)
        if source_type is None:
            unknown.append(logical_index)
        elif source_type not in source_types:
            source_types.append(source_type)

    if unknown:
        allowed = ", ".join(sorted(LOGICAL_INDEX_TO_SOURCE_TYPE))
        raise error(
            f"Unknown logical index name(s): {', '.join(unknown)}. "
            f"Allowed indexes: {allowed}"
        )
    return tuple(source_types)


def validate_logical_index(index_name: str, *, error: type[Exception]) -> str:
    """Return the source type for one logical index or raise ``error``."""

    source_type = LOGICAL_INDEX_TO_SOURCE_TYPE.get(index_name)
    if source_type is None:
        allowed = ", ".join(sorted(LOGICAL_INDEX_TO_SOURCE_TYPE))
        raise error(
            f"Unknown logical index name: {index_name}. Allowed indexes: {allowed}"
        )
    return source_type


__all__ = [
    "ELIGIBLE_SOURCE_TYPES",
    "INDEX_TO_SOURCE_TYPE",
    "LOGICAL_INDEXES",
    "LOGICAL_INDEX_TO_SOURCE_TYPE",
    "SOURCE_TYPE_TO_INDEX",
    "SOURCE_TYPE_TO_LOGICAL_INDEX",
    "source_types_for_indexes",
    "validate_logical_index",
]
