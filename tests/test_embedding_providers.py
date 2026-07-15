from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import make_config
from uni_rag_agent.config import load_config
from uni_rag_agent.indexing import (
    EMBEDDING_PROFILES,
    BuiltEmbeddingModel,
    VectorIndexError,
    build_embedding_model,
    physical_collection_name,
    resolve_embedding_profile,
)
from uni_rag_agent.indexing import embeddings as compatibility_embeddings
from uni_rag_agent.indexing.embedding_providers import common
from uni_rag_agent.indexing.embedding_providers import google_genai, huggingface, nebius


def _vectors(count: int, dimension: int, *, start: float = 1.0) -> list[list[float]]:
    return [[start + number] * dimension for number in range(count)]


def test_registry_preserves_local_profiles_and_adds_canonical_hosted_profiles() -> None:
    local_names = {
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v5-text-small",
        "google/embeddinggemma-300m",
    }
    assert local_names.issubset(EMBEDDING_PROFILES)
    assert EMBEDDING_PROFILES["BAAI/bge-m3"].dimension == 1024
    assert EMBEDDING_PROFILES["jinaai/jina-embeddings-v3"].trust_remote_code is True
    assert EMBEDDING_PROFILES["jinaai/jina-embeddings-v5-text-small"].gated is True
    assert EMBEDDING_PROFILES["google/embeddinggemma-300m"].dimension == 768
    assert all(
        EMBEDDING_PROFILES[name].provider == "huggingface"
        and EMBEDDING_PROFILES[name].requires_extra == "embeddings"
        for name in local_names
    )

    google_profile = EMBEDDING_PROFILES["google/gemini-embedding-001"]
    assert google_profile.provider == "google_genai"
    assert google_profile.dimension == 3072
    assert google_profile.api_model_name == "gemini-embedding-001"
    assert google_profile.requires_extra == "embeddings-cloud"
    assert google_profile.aliases == ("gemini-embedding-001",)

    nebius_profile = EMBEDDING_PROFILES["Qwen/Qwen3-Embedding-8B"]
    assert nebius_profile.provider == "nebius"
    assert nebius_profile.dimension == 4096
    assert nebius_profile.api_model_name == "Qwen/Qwen3-Embedding-8B"
    assert nebius_profile.requires_extra == "embeddings-cloud"


def test_alias_resolves_to_canonical_profile_before_collection_naming(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    canonical = resolve_embedding_profile(config, "google/gemini-embedding-001")
    alias = resolve_embedding_profile(config, "gemini-embedding-001")

    assert alias == canonical
    assert alias.model_name == "google/gemini-embedding-001"
    assert physical_collection_name(
        "document_index",
        provider=alias.provider,
        model_name=alias.model_name,
        dimension=alias.dimension,
        metric=alias.metric,
    ) == physical_collection_name(
        "document_index",
        provider=canonical.provider,
        model_name=canonical.model_name,
        dimension=canonical.dimension,
        metric=canonical.metric,
    )


def test_factory_and_registry_import_without_optional_provider_sdks() -> None:
    code = """
import sys
import uni_rag_agent.indexing
from uni_rag_agent.indexing.embedding_providers import factory
assert 'langchain_google_genai' not in sys.modules
assert 'langchain_huggingface' not in sys.modules
assert 'openai' not in sys.modules
assert factory.PROVIDER_MODULES['google_genai'] == '.google_genai'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_factory_rejects_unknown_provider_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = dataclasses.replace(
        EMBEDDING_PROFILES["BAAI/bge-m3"],
        model_name="test/unknown-provider",
        provider="not-a-provider",
    )
    monkeypatch.setitem(EMBEDDING_PROFILES, profile.model_name, profile)

    with pytest.raises(VectorIndexError, match="Unknown embedding provider"):
        build_embedding_model(make_config(tmp_path), profile.model_name)


def test_local_huggingface_loader_forwards_options_and_probes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeLocalEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            calls.append(("probe", text))
            return [1.0] * 5

    def constructor(**kwargs: object) -> FakeLocalEmbeddings:
        calls.append(("constructor", kwargs))
        return FakeLocalEmbeddings()

    monkeypatch.setattr(
        compatibility_embeddings, "_require_huggingface", lambda *_a, **_k: constructor
    )
    built = build_embedding_model(
        make_config(tmp_path),
        "jinaai/jina-embeddings-v3",
    )

    assert isinstance(built, BuiltEmbeddingModel)
    assert built.profile.model_name == "jinaai/jina-embeddings-v3"
    assert built.dimension == 5
    assert calls == [
        (
            "constructor",
            {
                "model_name": "jinaai/jina-embeddings-v3",
                "model_kwargs": {"trust_remote_code": True},
            },
        ),
        ("probe", "dimension probe"),
    ]


def test_local_huggingface_documents_and_queries_use_shared_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_attempts = 0
    query_attempts = 0

    class FakeLocalEmbeddings:
        def embed_documents(self, _texts: list[str]) -> list[list[float]]:
            nonlocal document_attempts
            document_attempts += 1
            if document_attempts < 3:
                raise ConnectionError("temporary document failure")
            return [[1.0] * 5]

        def embed_query(self, text: str) -> list[float]:
            nonlocal query_attempts
            if text == "dimension probe":
                return [1.0] * 5
            query_attempts += 1
            if query_attempts < 3:
                raise TimeoutError("temporary query failure")
            return [2.0] * 5

    def constructor(**_kwargs: object) -> FakeLocalEmbeddings:
        return FakeLocalEmbeddings()

    def retry_without_sleep(operation: object, **kwargs: object) -> object:
        return common.retry_transient(
            operation,  # type: ignore[arg-type]
            sleep=lambda _delay: None,
            **kwargs,
        )

    monkeypatch.setattr(
        compatibility_embeddings, "_require_huggingface", lambda *_a, **_k: constructor
    )
    monkeypatch.setattr(huggingface, "retry_transient", retry_without_sleep)
    built = build_embedding_model(make_config(tmp_path), "BAAI/bge-m3")

    assert built.embeddings.embed_documents(["document"]) == [[1.0] * 5]  # type: ignore[attr-defined]
    assert built.embeddings.embed_query("query") == [2.0] * 5  # type: ignore[attr-defined]
    assert document_attempts == 3
    assert query_attempts == 3


def test_google_constructor_tasks_dimension_and_no_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []
    method_calls: list[tuple[str, object, dict[str, object]]] = []

    class FakeGoogleEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)

        def embed_documents(
            self, texts: list[str], **kwargs: object
        ) -> list[list[float]]:
            method_calls.append(("documents", texts, kwargs))
            return _vectors(len(texts), 3072)

        def embed_query(self, text: str, **kwargs: object) -> list[float]:
            method_calls.append(("query", text, kwargs))
            return _vectors(1, 3072)[0]

    monkeypatch.setattr(
        google_genai,
        "_require_google_embeddings",
        lambda *_a, **_k: FakeGoogleEmbeddings,
    )
    config = make_config(tmp_path, google_api_key="google-secret")
    built = build_embedding_model(config, "gemini-embedding-001")

    assert built.profile.model_name == "google/gemini-embedding-001"
    assert built.dimension == 3072
    assert constructor_calls == [
        {
            "model": "gemini-embedding-001",
            "google_api_key": "google-secret",
            "vertexai": False,
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 3072,
        }
    ]
    assert method_calls == []

    assert len(built.embeddings.embed_documents(["doc one", "doc two"])) == 2  # type: ignore[attr-defined]
    assert len(built.embeddings.embed_query("query")) == 3072  # type: ignore[attr-defined]
    assert method_calls == [
        (
            "documents",
            ["doc one", "doc two"],
            {"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 3072},
        ),
        (
            "query",
            "query",
            {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 3072},
        ),
    ]


def test_nebius_constructor_request_shape_query_format_and_response_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []
    request_calls: list[dict[str, object]] = []

    class FakeEmbeddingsEndpoint:
        def create(self, **kwargs: object) -> object:
            request_calls.append(kwargs)
            input_value = kwargs["input"]
            count = len(input_value) if isinstance(input_value, list) else 1
            items = [
                SimpleNamespace(index=index, embedding=[float(index + 1)] * 4096)
                for index in range(count)
            ]
            return SimpleNamespace(data=items)

    class FakeNebiusClient:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)
            self.embeddings = FakeEmbeddingsEndpoint()

    monkeypatch.setattr(nebius, "_require_openai", lambda *_a, **_k: FakeNebiusClient)
    built = build_embedding_model(
        make_config(tmp_path, nebius_api_key="nebius-secret"),
        "Qwen/Qwen3-Embedding-8B",
    )

    assert built.profile.model_name == "Qwen/Qwen3-Embedding-8B"
    assert built.dimension == 4096
    assert constructor_calls == [
        {
            "api_key": "nebius-secret",
            "base_url": "https://api.tokenfactory.nebius.com/v1/",
        }
    ]
    documents = built.embeddings.embed_documents(["unchanged A", "unchanged B"])  # type: ignore[attr-defined]
    query = built.embeddings.embed_query("find this")  # type: ignore[attr-defined]

    assert [vector[0] for vector in documents] == [1.0, 2.0]
    assert query[0] == 1.0
    assert request_calls[0] == {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": ["unchanged A", "unchanged B"],
        "dimensions": 4096,
    }
    assert request_calls[1] == {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:find this"
        ),
        "dimensions": 4096,
    }


def test_nebius_reordered_response_is_rejected_before_returning_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeEmbeddingsEndpoint:
        def create(self, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[1.0] * 4096),
                    SimpleNamespace(index=0, embedding=[2.0] * 4096),
                ]
            )

    class FakeNebiusClient:
        def __init__(self, **_kwargs: object) -> None:
            self.embeddings = FakeEmbeddingsEndpoint()

    monkeypatch.setattr(nebius, "_require_openai", lambda *_a, **_k: FakeNebiusClient)
    built = build_embedding_model(
        make_config(tmp_path, nebius_api_key="nebius-secret"),
        "Qwen/Qwen3-Embedding-8B",
    )

    with pytest.raises(RuntimeError, match="invalid embedding response"):
        built.embeddings.embed_documents(["first", "second"])  # type: ignore[attr-defined]
    assert calls == 1


def test_validation_and_retry_classification() -> None:
    assert common.validate_vectors(
        [[1, 2], [3.0, 4.0]], expected_count=2, expected_dimension=2
    ) == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(common.EmbeddingValidationError, match="expected 2"):
        common.validate_vectors([[1.0]], expected_count=2)
    with pytest.raises(common.EmbeddingValidationError, match="empty"):
        common.validate_vectors([[]])
    with pytest.raises(common.EmbeddingValidationError, match="non-finite"):
        common.validate_vectors([[math.nan]])
    with pytest.raises(common.EmbeddingValidationError, match="dimension"):
        common.validate_vectors([[1.0, 2.0]], expected_dimension=3)

    class HttpFailure(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"provider failure secret-token={status_code}")
            self.status_code = status_code

    attempts = 0

    def eventually_available() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HttpFailure(429)
        return "ok"

    assert (
        common.retry_transient(eventually_available, sleep=lambda _delay: None) == "ok"
    )
    assert attempts == 3

    for status in (401, 403, 404, 400):
        calls = 0

        def fail(status: int = status) -> None:
            nonlocal calls
            calls += 1
            raise HttpFailure(status)

        with pytest.raises(HttpFailure):
            common.retry_transient(fail, sleep=lambda _delay: None)
        assert calls == 1

    calls = 0

    def network_failure() -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("secret-token")

    with pytest.raises(ConnectionError):
        common.retry_transient(network_failure, sleep=lambda _delay: None)
    assert calls == 3
    timeout_attempts = 0

    def timeout_failure() -> None:
        nonlocal timeout_attempts
        timeout_attempts += 1
        raise TimeoutError("timed out after 120 seconds")

    assert common.http_status_code(TimeoutError("timed out after 120 seconds")) is None
    with pytest.raises(TimeoutError):
        common.retry_transient(timeout_failure, sleep=lambda _delay: None)
    assert timeout_attempts == 3
    assert "secret-token" not in common.sanitize_provider_error(
        ConnectionError("secret-token"), "Nebius"
    )


def test_hosted_responses_validate_actual_dimensions_and_do_not_retry_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeGoogleEmbeddings:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed_documents(
            self, _texts: list[str], **_kwargs: object
        ) -> list[list[float]]:
            nonlocal calls
            calls += 1
            return [[1.0]]

    monkeypatch.setattr(
        google_genai,
        "_require_google_embeddings",
        lambda *_a, **_k: FakeGoogleEmbeddings,
    )
    built = build_embedding_model(
        make_config(tmp_path, google_api_key="google-secret"),
        "google/gemini-embedding-001",
    )
    with pytest.raises(RuntimeError, match="invalid embedding response"):
        built.embeddings.embed_documents(["one"])  # type: ignore[attr-defined]
    assert calls == 1


@pytest.mark.parametrize(
    ("model", "field", "message"),
    [
        ("google/gemini-embedding-001", "google_api_key", "GOOGLE_API_KEY"),
        ("Qwen/Qwen3-Embedding-8B", "nebius_api_key", "NEBIUS_API_KEY"),
    ],
)
def test_hosted_missing_credentials_are_provider_specific_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    field: str,
    message: str,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    config = dataclasses.replace(make_config(tmp_path), **{field: None})
    with pytest.raises(VectorIndexError, match=message) as exc_info:
        build_embedding_model(config, model)
    assert "secret" not in str(exc_info.value).casefold()


def test_missing_hosted_extras_have_provider_specific_install_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    config = make_config(tmp_path, google_api_key="google-secret")
    with pytest.raises(VectorIndexError, match="Google GenAI.*embeddings-cloud"):
        build_embedding_model(config, "google/gemini-embedding-001")

    monkeypatch.setitem(sys.modules, "openai", None)
    config = make_config(tmp_path, nebius_api_key="nebius-secret")
    with pytest.raises(VectorIndexError, match="Nebius.*embeddings-cloud"):
        build_embedding_model(config, "Qwen/Qwen3-Embedding-8B")


def test_config_loads_cloud_keys_without_repr_or_safe_projection_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=google-secret\nNEBIUS_API_KEY=nebius-secret\n",
        encoding="utf-8",
    )
    config = load_config(repo_root=tmp_path, env_file=env_file)

    assert config.google_api_key == "google-secret"
    assert config.nebius_api_key == "nebius-secret"
    representation = repr(config)
    safe = config.as_safe_dict()
    assert "google-secret" not in representation
    assert "nebius-secret" not in representation
    assert "google_api_key" not in safe
    assert "nebius_api_key" not in safe
    assert "google-secret" not in safe.values()
    assert "nebius-secret" not in safe.values()
    assert not hasattr(config, "embedding_provider")


def test_cloud_extra_is_separate_from_local_and_llm_extras() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert set(extras["embeddings-cloud"]) == {
        "langchain-google-genai>=4.2.7",
        "openai>=1.0.0",
    }
    assert "langchain-huggingface>=0.1.0" in extras["embeddings"]
