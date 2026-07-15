"""Lazy Hugging Face provider for the reviewed local embedding profiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from ..profiles import EmbeddingProfile
from .common import (
    EmbeddingValidationError,
    retry_transient,
    sanitize_provider_error,
    validate_vectors,
)


class HuggingFaceEmbeddingsAdapter(Embeddings):
    """Apply the shared retry and vector contract to a local HF client."""

    def __init__(self, client: object, *, model: str, dimension: int) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            result = retry_transient(
                lambda: self.client.embed_documents(values),  # type: ignore[attr-defined]
                provider="Hugging Face",
            )
            return validate_vectors(
                result,
                expected_count=len(values),
                expected_dimension=self.dimension,
                context="Hugging Face embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Hugging Face returned an invalid embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Hugging Face",
                    operation="document embedding",
                    model=self.model,
                )
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            result = retry_transient(
                lambda: self.client.embed_query(text),  # type: ignore[attr-defined]
                provider="Hugging Face",
            )
            return validate_vectors(
                [result],
                expected_count=1,
                expected_dimension=self.dimension,
                context="Hugging Face query embedding response",
            )[0]
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Hugging Face returned an invalid query embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Hugging Face",
                    operation="query embedding",
                    model=self.model,
                )
            ) from exc


def build_embeddings(
    config: Config,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
    loader: Callable[..., object] | None = None,
) -> tuple[object, int]:
    """Load a reviewed local model and probe its runtime dimension."""
    del config  # Local Hugging Face loading is configured by the profile.
    constructor = (
        loader(profile, error=error)
        if loader is not None
        else _require_huggingface(profile, error=error)
    )
    model_kwargs: dict[str, object] = {}
    if profile.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    try:
        embeddings = constructor(
            model_name=profile.effective_api_model_name,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:  # pragma: no cover - real model/access boundary
        detail = sanitize_provider_error(
            exc,
            "Hugging Face",
            operation="construction",
            model=profile.model_name,
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc

    try:
        probe = retry_transient(
            lambda: embeddings.embed_query("dimension probe"),
            provider="Hugging Face",
        )
        vectors = validate_vectors(
            [probe],
            expected_count=1,
            context=f"Hugging Face model '{profile.model_name}' dimension probe",
        )
    except EmbeddingValidationError as exc:
        raise error(
            f"Hugging Face model '{profile.model_name}' returned an invalid "
            "dimension probe."
        ) from exc
    except Exception as exc:  # pragma: no cover - real model/access boundary
        detail = sanitize_provider_error(
            exc,
            "Hugging Face",
            operation="dimension probe",
            model=profile.model_name,
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc
    runtime_dimension = len(vectors[0])
    return (
        HuggingFaceEmbeddingsAdapter(
            embeddings,
            model=profile.model_name,
            dimension=runtime_dimension,
        ),
        runtime_dimension,
    )


def _require_huggingface(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[object]:
    """Import ``HuggingFaceEmbeddings`` only after a local profile is chosen."""
    extra = profile.requires_extra or "embeddings"
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        detail = (
            f"Hugging Face model '{profile.model_name}' requires the optional "
            f"'{extra}' extra (langchain-huggingface and sentence-transformers). "
            f"Install it with: uv sync --extra {extra}"
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc
    return HuggingFaceEmbeddings


# Provider-specific descriptive aliases make the lazy boundary easy to patch
# in offline tests without importing the optional package.
build_huggingface_embeddings = build_embeddings
