"""Lazy Nebius Token Factory provider for Qwen embeddings."""

from __future__ import annotations

import os
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from ..profiles import EmbeddingProfile
from .common import (
    EmbeddingValidationError,
    MalformedEmbeddingResponse,
    retry_transient,
    sanitize_provider_error,
    validate_response_order,
    validate_vectors,
)


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query"


class NebiusEmbeddings(Embeddings):
    """Validated OpenAI-compatible Nebius embeddings adapter."""

    def __init__(self, client: object, *, model: str, dimension: int) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        return self._embed(values, input_value=values, operation="document embedding")

    def embed_query(self, text: str) -> list[float]:
        formatted = f"{QUERY_INSTRUCTION}\nQuery:{text}"
        return self._embed(
            [formatted],
            input_value=formatted,
            operation="query embedding",
        )[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed retrieval queries in one OpenAI-compatible request."""
        values = [f"{QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]
        if not values:
            return []
        return self._embed(
            values,
            input_value=values,
            operation="query embedding",
        )

    def _embed(
        self,
        expected_inputs: list[str],
        *,
        input_value: str | list[str],
        operation: str,
    ) -> list[list[float]]:
        try:
            response = retry_transient(
                lambda: self.client.embeddings.create(  # type: ignore[attr-defined]
                    model=self.model,
                    input=input_value,
                    dimensions=self.dimension,
                ),
                provider="Nebius",
            )
            items = validate_response_order(
                _response_data(response),
                expected_count=len(expected_inputs),
                context="Nebius embedding response",
            )
            vectors = [_response_embedding(item) for item in items]
            return validate_vectors(
                vectors,
                expected_count=len(expected_inputs),
                expected_dimension=self.dimension,
                context="Nebius embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Nebius returned an invalid embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Nebius",
                    operation=operation,
                    model=self.model,
                )
            ) from exc


def build_embeddings(
    config: Config,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> tuple[NebiusEmbeddings, int]:
    """Construct a fixed-endpoint Nebius client without a probe request."""
    api_key = _nebius_api_key(config)
    if not api_key:
        raise error(
            "Nebius embedding provider requires NEBIUS_API_KEY. "
            "Set it in the merged .env file or environment."
        )

    constructor = _require_openai(profile, error=error)
    dimension = profile.dimension
    try:
        client = constructor(api_key=api_key, base_url=NEBIUS_BASE_URL)
    except Exception as exc:  # pragma: no cover - provider construction boundary
        detail = sanitize_provider_error(
            exc,
            "Nebius",
            operation="construction",
            model=profile.effective_api_model_name,
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc

    return (
        NebiusEmbeddings(
            client,
            model=profile.effective_api_model_name,
            dimension=dimension,
        ),
        dimension,
    )


def _nebius_api_key(config: Config) -> str | None:
    configured = getattr(config, "nebius_api_key", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    value = os.getenv("NEBIUS_API_KEY")
    return value.strip() if value and value.strip() else None


def _require_openai(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[object]:
    """Import the OpenAI SDK only for the Nebius profile."""
    extra = profile.requires_extra or "embeddings-cloud"
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise error(
            "Nebius embedding provider requires the optional "
            f"'{extra}' extra (openai). "
            f"Install it with: uv sync --extra {extra}"
        ) from exc
    return OpenAI


def _response_data(response: object) -> list[object]:
    if isinstance(response, dict):
        value = response.get("data")
    else:
        value = getattr(response, "data", None)
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise MalformedEmbeddingResponse(
            "Nebius embedding response has no data sequence."
        )
    try:
        return list(value)
    except (TypeError, ValueError) as exc:
        raise MalformedEmbeddingResponse(
            "Nebius embedding response data is malformed."
        ) from exc


def _response_embedding(item: object) -> object:
    if isinstance(item, dict):
        value = item.get("embedding")
    else:
        value = getattr(item, "embedding", None)
    if value is None:
        raise MalformedEmbeddingResponse(
            "Nebius embedding response item has no embedding vector."
        )
    return value


# Additional names make the SDK seam explicit to offline tests and callers.
_require_nebius_openai = _require_openai
build_nebius_embeddings = build_embeddings
