"""Indexing public API.

Keyword indexing and the pure embedding-profile registry import eagerly. The
provider factory and ChromaDB-backed sync/search import lazily so importing
this package for keyword-only use does not pull in optional provider SDKs or
``chromadb``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .eligibility import ELIGIBLE_SOURCE_TYPES, INDEX_TO_SOURCE_TYPE
from .keyword import (
    keyword_query_terms,
    keyword_search,
    keyword_search_terms,
    sync_keyword_index,
)
from .models import (
    KeywordIndexError,
    KeywordIndexResult,
    KeywordSearchError,
    SemanticSearchError,
    VectorIndexError,
    VectorIndexResult,
)
from .profiles import (
    EMBEDDING_PROFILES,
    EmbeddingProfile,
    physical_collection_name,
    resolve_embedding_profile,
)

if TYPE_CHECKING:
    from .embedding_providers.factory import BuiltEmbeddingModel, build_embedding_model
    from .vector import semantic_search, sync_vector_index

_LAZY_EXPORTS = {
    "BuiltEmbeddingModel": "embedding_providers.factory",
    "build_embedding_model": "embedding_providers.factory",
    "semantic_search": "vector",
    "sync_vector_index": "vector",
}

__all__ = [
    "ELIGIBLE_SOURCE_TYPES",
    "INDEX_TO_SOURCE_TYPE",
    "EMBEDDING_PROFILES",
    "EmbeddingProfile",
    "BuiltEmbeddingModel",
    "KeywordIndexError",
    "KeywordIndexResult",
    "KeywordSearchError",
    "SemanticSearchError",
    "VectorIndexError",
    "VectorIndexResult",
    "build_embedding_model",
    "keyword_query_terms",
    "keyword_search",
    "keyword_search_terms",
    "physical_collection_name",
    "resolve_embedding_profile",
    "semantic_search",
    "sync_keyword_index",
    "sync_vector_index",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
