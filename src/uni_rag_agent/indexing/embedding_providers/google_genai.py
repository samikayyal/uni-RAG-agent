"""Lazy direct-Gemini embedding provider."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from ..profiles import EmbeddingProfile
from .common import (
    EmbeddingValidationError,
    retry_transient,
    sanitize_provider_error,
    validate_vectors,
)

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"

# The reported Free-tier RPM for gemini-embedding-001 is 100, but the active
# quota is project/model/account-specific. One second before each request keeps
# this process below that RPM with some headroom for clock/timing variance.
GEMINI_EMBEDDING_REQUEST_DELAY_SECONDS = 1.0


class GoogleGenAIEmbeddings(Embeddings):
    """Validated LangChain Gemini embeddings adapter.

    The wrapped LangChain object performs direct Gemini Developer API calls.
    This adapter supplies the document/query task explicitly and validates
    every returned vector without making a separate dimension probe request.
    """

    def __init__(self, client: object, *, model: str, dimension: int) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            time.sleep(GEMINI_EMBEDDING_REQUEST_DELAY_SECONDS)
            result = retry_transient(
                lambda: self.client.embed_documents(  # type: ignore[attr-defined]
                    values,
                    task_type=DOCUMENT_TASK,
                    output_dimensionality=self.dimension,
                ),
                provider="Google GenAI",
            )
            return validate_vectors(
                result,
                expected_count=len(values),
                expected_dimension=self.dimension,
                context="Google GenAI embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Google GenAI returned an invalid embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Google GenAI",
                    operation="document embedding",
                    model=self.model,
                )
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            time.sleep(GEMINI_EMBEDDING_REQUEST_DELAY_SECONDS)
            result = retry_transient(
                lambda: self.client.embed_query(  # type: ignore[attr-defined]
                    text,
                    task_type=QUERY_TASK,
                    output_dimensionality=self.dimension,
                ),
                provider="Google GenAI",
            )
            return validate_vectors(
                [result],
                expected_count=1,
                expected_dimension=self.dimension,
                context="Google GenAI query embedding response",
            )[0]
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Google GenAI returned an invalid query embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Google GenAI",
                    operation="query embedding",
                    model=self.model,
                )
            ) from exc

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed retrieval queries in one Gemini batch request."""
        values = list(texts)
        if not values:
            return []
        try:
            time.sleep(GEMINI_EMBEDDING_REQUEST_DELAY_SECONDS)
            result = retry_transient(
                lambda: self.client.embed_documents(  # type: ignore[attr-defined]
                    values,
                    task_type=QUERY_TASK,
                    output_dimensionality=self.dimension,
                ),
                provider="Google GenAI",
            )
            return validate_vectors(
                result,
                expected_count=len(values),
                expected_dimension=self.dimension,
                context="Google GenAI query embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Google GenAI returned an invalid query embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Google GenAI",
                    operation="query embedding",
                    model=self.model,
                )
            ) from exc


def build_embeddings(
    config: Config,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> tuple[GoogleGenAIEmbeddings, int]:
    """Construct direct Gemini embeddings without a dimension-probe call."""
    api_key = _google_api_key(config)
    if not api_key:
        raise error(
            "Google GenAI embedding provider requires GOOGLE_API_KEY. "
            "Set it in the merged .env file or environment."
        )

    constructor = _require_google_embeddings(profile, error=error)
    dimension = profile.dimension
    try:
        client = constructor(
            model=profile.effective_api_model_name,
            google_api_key=api_key,
            vertexai=False,
            task_type=DOCUMENT_TASK,
            output_dimensionality=dimension,
        )
    except Exception as exc:  # pragma: no cover - provider construction boundary
        detail = sanitize_provider_error(
            exc,
            "Google GenAI",
            operation="construction",
            model=profile.effective_api_model_name,
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc

    return (
        GoogleGenAIEmbeddings(
            client,
            model=profile.effective_api_model_name,
            dimension=dimension,
        ),
        dimension,
    )


def _google_api_key(config: Config) -> str | None:
    configured = getattr(config, "google_api_key", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    value = os.getenv("GOOGLE_API_KEY")
    return value.strip() if value and value.strip() else None


def _require_google_embeddings(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[object]:
    """Import ``GoogleGenerativeAIEmbeddings`` only for the hosted profile."""
    extra = profile.requires_extra or "embeddings-cloud"
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise error(
            "Google GenAI embedding provider requires the optional "
            f"'{extra}' extra (langchain-google-genai). "
            f"Install it with: uv sync --extra {extra}"
        ) from exc
    return GoogleGenerativeAIEmbeddings


# Additional descriptive name for test seams and callers that use the SDK
# name rather than the provider name.
_require_google_genai = _require_google_embeddings
build_google_genai_embeddings = build_embeddings
