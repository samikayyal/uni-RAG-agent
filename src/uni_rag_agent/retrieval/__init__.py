"""LLM query planning and hybrid retrieval public API.

The model contracts are eager and dependency-light. Retrieval implementations
are loaded lazily so the existing keyword/vector packages can import
``RetrievalResult`` without a package-initialization cycle.
"""

from __future__ import annotations

import importlib

from .models import (
    FusedRetrievalResult,
    RetrievalContribution,
    RetrievalResult,
    RetrievalResultSet,
    RetrievalRun,
    QueryPlan,
)

_LAZY_EXPORTS = {
    "MetadataSearchError": "metadata",
    "RetrievalError": "core",
    "QueryPlanningError": "planner",
    "merge_with_rrf": "rrf",
    "metadata_search": "metadata",
    "retrieve": "core",
    "plan_query": "planner",
}

__all__ = [
    "FusedRetrievalResult",
    "MetadataSearchError",
    "RetrievalContribution",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalResultSet",
    "RetrievalRun",
    "QueryPlan",
    "QueryPlanningError",
    "merge_with_rrf",
    "metadata_search",
    "retrieve",
    "plan_query",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
