"""Subprocess-only Hugging Face constructor shim for CLI integration tests."""

from __future__ import annotations

from tests.embedding_doubles import DeterministicTestEmbeddings, embeddings_for_model


class HuggingFaceEmbeddings(DeterministicTestEmbeddings):
    """Match the constructor surface used by production model loading."""

    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, object] | None = None,
        **_: object,
    ) -> None:
        del model_kwargs
        selected = embeddings_for_model(model_name)
        super().__init__(selected.dimension)
