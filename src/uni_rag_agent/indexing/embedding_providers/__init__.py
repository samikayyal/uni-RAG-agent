"""Lazy embedding-provider package.

Only the dependency-light factory API is re-exported here. Provider modules
are intentionally not imported, so this package can be used for configuration
checks without loading any optional SDK or model runtime.
"""

from __future__ import annotations

from .factory import BuiltEmbeddingModel, build_embedding_model

__all__ = ["BuiltEmbeddingModel", "build_embedding_model"]
