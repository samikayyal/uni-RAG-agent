"""Routing and hybrid retrieval public API.

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
    RouterOutput,
)

_LAZY_EXPORTS = {
    "MetadataSearchError": "metadata",
    "RetrievalError": "core",
    "RoutingError": "router",
    "merge_with_rrf": "rrf",
    "metadata_search": "metadata",
    "retrieve": "core",
    "route_query": "router",
    "validate_router_output": "router",
}

__all__ = [
    "FusedRetrievalResult",
    "MetadataSearchError",
    "RetrievalContribution",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalResultSet",
    "RetrievalRun",
    "RouterOutput",
    "RoutingError",
    "merge_with_rrf",
    "metadata_search",
    "retrieve",
    "route_query",
    "validate_router_output",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
