"""Provider-neutral embedding factory and built-model result."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from uni_rag_agent.config import Config

from ..models import VectorIndexError
from ..profiles import EmbeddingProfile, resolve_embedding_profile

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BuiltEmbeddingModel:
    """Embedding object plus its canonical profile and effective dimension."""

    embeddings: object
    profile: EmbeddingProfile
    dimension: int


PROVIDER_MODULES: dict[str, str] = {
    "huggingface": ".huggingface",
    "google_genai": ".google_genai",
    "nebius": ".nebius",
}


def build_embedding_model(
    config: Config,
    model: str | None = None,
    *,
    error: type[Exception] = VectorIndexError,
) -> BuiltEmbeddingModel:
    """Resolve and lazily construct the selected embedding provider."""
    profile = resolve_embedding_profile(config, model, error=error)
    module_path = PROVIDER_MODULES.get(profile.provider)
    if module_path is None:
        supported = ", ".join(sorted(PROVIDER_MODULES))
        raise error(
            f"Unknown embedding provider '{profile.provider}' for model "
            f"'{profile.model_name}'. Supported providers: {supported}."
        )

    try:
        provider_module = importlib.import_module(module_path, __package__)
    except ImportError as exc:  # pragma: no cover - package installation boundary
        raise error(
            f"Embedding provider '{profile.provider}' is unavailable for model "
            f"'{profile.model_name}'."
        ) from exc

    builder = getattr(provider_module, "build_embeddings", None)
    if not callable(builder):  # pragma: no cover - registry/package invariant
        raise error(
            f"Embedding provider '{profile.provider}' has no construction "
            "implementation."
        )

    kwargs: dict[str, object] = {"error": error}
    if profile.provider == "huggingface":
        legacy_loader = _patched_legacy_huggingface_loader(provider_module)
        if legacy_loader is not None:
            kwargs["loader"] = legacy_loader
    try:
        embeddings, dimension = builder(config, profile, **kwargs)
    except error:
        raise
    except Exception as exc:  # pragma: no cover - provider implementation boundary
        raise error(
            f"Could not construct embedding provider '{profile.provider}' for "
            f"model '{profile.model_name}'."
        ) from exc

    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise error(
            f"Embedding provider '{profile.provider}' returned an invalid "
            f"dimension for model '{profile.model_name}'."
        )
    return BuiltEmbeddingModel(
        embeddings=embeddings,
        profile=profile,
        dimension=dimension,
    )


def _patched_legacy_huggingface_loader(
    provider_module: object,
) -> Callable[..., object] | None:
    """Honor the old test-only loader seam without moving ownership backward.

    ``indexing.embeddings`` re-exports the provider loader for compatibility.
    Existing offline vector tests patch that re-export; detecting a changed
    binding here keeps those tests working while the factory remains the sole
    owner of ``BuiltEmbeddingModel`` and provider dispatch.
    """
    compatibility = sys.modules.get("uni_rag_agent.indexing.embeddings")
    if compatibility is None:
        return None
    candidate = getattr(compatibility, "_require_huggingface", None)
    default = getattr(provider_module, "_require_huggingface", None)
    compatibility_default = getattr(
        compatibility,
        "_DEFAULT_HUGGINGFACE_LOADER",
        default,
    )
    if callable(candidate) and candidate is not compatibility_default:
        return candidate
    return None


__all__ = [
    "BuiltEmbeddingModel",
    "PROVIDER_MODULES",
    "build_embedding_model",
]
