from __future__ import annotations

import pytest

from uni_rag_agent.indexing.eligibility import (
    ELIGIBLE_SOURCE_TYPES as INDEXING_SOURCE_TYPES,
    INDEX_TO_SOURCE_TYPE as INDEXING_INDEX_TO_SOURCE_TYPE,
)
from uni_rag_agent.retrieval.evidence_models import EVIDENCE_SOURCE_TYPES
from uni_rag_agent.retrieval.models import LOGICAL_INDEXES
from uni_rag_agent.search_contracts import (
    ELIGIBLE_SOURCE_TYPES,
    LOGICAL_INDEX_TO_SOURCE_TYPE,
    SOURCE_TYPE_TO_LOGICAL_INDEX,
    source_types_for_indexes,
)


def test_logical_index_views_are_derived_from_one_canonical_mapping() -> None:
    assert tuple(LOGICAL_INDEX_TO_SOURCE_TYPE) == LOGICAL_INDEXES
    assert tuple(LOGICAL_INDEX_TO_SOURCE_TYPE.values()) == ELIGIBLE_SOURCE_TYPES
    assert dict(SOURCE_TYPE_TO_LOGICAL_INDEX) == {
        source_type: logical_index
        for logical_index, source_type in LOGICAL_INDEX_TO_SOURCE_TYPE.items()
    }
    assert INDEXING_INDEX_TO_SOURCE_TYPE is LOGICAL_INDEX_TO_SOURCE_TYPE
    assert INDEXING_SOURCE_TYPES == ELIGIBLE_SOURCE_TYPES
    assert EVIDENCE_SOURCE_TYPES == ELIGIBLE_SOURCE_TYPES


def test_source_types_for_indexes_preserves_requested_order_and_rejects_unknown() -> (
    None
):
    assert source_types_for_indexes(
        ("transcript_index", "document_index", "transcript_index"),
        error=ValueError,
    ) == ("transcript", "document")

    with pytest.raises(ValueError, match="Unknown logical index name"):
        source_types_for_indexes(("missing_index",), error=ValueError)
