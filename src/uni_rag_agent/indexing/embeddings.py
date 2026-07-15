"""Backward-compatible imports for the provider-based embedding factory.

Provider construction lives in ``embedding_providers``. This module contains
no provider implementation; the loader re-export exists only for older tests
and callers that patch the former Hugging Face construction seam.
"""

from __future__ import annotations

from .embedding_providers.factory import BuiltEmbeddingModel, build_embedding_model
from .embedding_providers.huggingface import _require_huggingface

# The factory uses this marker to distinguish the compatibility default from
# an older caller/test that intentionally patches this re-export.
_DEFAULT_HUGGINGFACE_LOADER = _require_huggingface

__all__ = ["BuiltEmbeddingModel", "build_embedding_model"]
