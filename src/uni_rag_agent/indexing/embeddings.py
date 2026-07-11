"""Hugging Face embedding model loading with lazy optional dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from .models import VectorIndexError
from .profiles import EmbeddingProfile, resolve_embedding_profile


@dataclass(frozen=True)
class BuiltEmbeddingModel:
    """An embedding object paired with its resolved profile and runtime dim."""

    embeddings: Embeddings
    profile: EmbeddingProfile
    dimension: int


def build_embedding_model(
    config: Config,
    model: str | None = None,
    *,
    error: type[Exception] = VectorIndexError,
) -> BuiltEmbeddingModel:
    """Resolve, load, and probe the selected embedding profile."""
    profile = resolve_embedding_profile(config, model, error=error)
    embeddings = _load_embeddings(profile, error=error)
    dimension = _probe_dimension(embeddings, profile, error=error)
    return BuiltEmbeddingModel(
        embeddings=embeddings,
        profile=profile,
        dimension=dimension,
    )


def _load_embeddings(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> Embeddings:
    huggingface_embeddings = _require_huggingface(profile, error=error)
    model_kwargs: dict[str, object] = {}
    if profile.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    try:
        return huggingface_embeddings(
            model_name=profile.model_name,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:  # pragma: no cover - requires real model + network
        raise error(
            f"Could not load embedding model '{profile.model_name}': {exc}. "
            f"{profile.access_notes or ''}".strip()
        ) from exc


def _require_huggingface(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[Embeddings]:
    """Import ``HuggingFaceEmbeddings`` lazily, mapping failures to ``error``.

    Isolated so tests can monkeypatch it to assert the optional-dependency
    failure path without installing the heavy embeddings extra.
    """
    extra = profile.requires_extra or "embeddings"
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise error(
            f"Embedding model '{profile.model_name}' requires the optional "
            f"'{extra}' extra (langchain-huggingface and sentence-transformers). "
            f"Install it with: uv sync --extra {extra}"
        ) from exc
    return HuggingFaceEmbeddings


def _probe_dimension(
    embeddings: Embeddings,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> int:
    try:
        vector = embeddings.embed_query("dimension probe")
    except Exception as exc:  # pragma: no cover - requires real model + network
        raise error(
            f"Could not determine embedding dimension for '{profile.model_name}': {exc}"
        ) from exc
    dimension = len(vector)
    if dimension <= 0:
        raise error(f"Embedding model '{profile.model_name}' returned an empty vector.")
    return dimension
