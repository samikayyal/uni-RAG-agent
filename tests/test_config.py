from __future__ import annotations

from pathlib import Path

import pytest

from uni_rag_agent.config import ConfigError, load_config, validate_config

UNI_RAG_ENV_KEYS = {
    "UNI_RAG_CHROMA_DIR",
    "UNI_RAG_COURSES_DIR",
    "UNI_RAG_COURSES_ROOT",
    "UNI_RAG_DATA_DIR",
    "UNI_RAG_EMBEDDING_DIM",
    "UNI_RAG_EMBEDDING_MODEL",
    "UNI_RAG_EMBEDDING_PROVIDER",
    "UNI_RAG_FINAL_TOP_K",
    "UNI_RAG_KEYWORD_TOP_K",
    "UNI_RAG_LLM_MODEL",
    "UNI_RAG_LLM_PROVIDER",
    "UNI_RAG_LOG_LEVEL",
    "UNI_RAG_OCR_ENABLED",
    "UNI_RAG_RRF_K",
    "UNI_RAG_RUNS_DIR",
    "UNI_RAG_SEMANTIC_TOP_K",
    "UNI_RAG_SQLITE_PATH",
    "UNI_RAG_USE_FAKE_EMBEDDINGS",
    "UNI_RAG_USE_FAKE_LLM",
}


@pytest.fixture(autouse=True)
def clear_uni_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in UNI_RAG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_resolve_from_repo_root(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    courses_root.mkdir()

    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    validate_config(config)

    assert config.courses_root == courses_root
    assert config.courses_dir == courses_root
    assert config.data_dir == tmp_path / "data"
    assert config.sqlite_path == tmp_path / "data" / "uni_rag.sqlite"
    assert config.chroma_dir == tmp_path / "data" / "indexes" / "vector"
    assert config.runs_dir == tmp_path / "data" / "runs"
    assert config.keyword_top_k == 20
    assert config.semantic_top_k == 20
    assert config.final_top_k == 10
    assert config.rrf_k == 60
    assert config.embedding_dim == 384
    assert config.use_fake_llm is True
    assert config.use_fake_embeddings is True
    assert config.ocr_enabled is False


def test_env_file_overrides_paths_and_retrieval_settings(tmp_path: Path) -> None:
    (tmp_path / "Archive").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "UNI_RAG_COURSES_ROOT=Archive",
                "UNI_RAG_DATA_DIR=.local-data",
                "UNI_RAG_SQLITE_PATH=.local-data/app.sqlite",
                "UNI_RAG_CHROMA_DIR=.local-data/chroma",
                "UNI_RAG_RUNS_DIR=.local-data/runs",
                "UNI_RAG_KEYWORD_TOP_K=7",
                "UNI_RAG_SEMANTIC_TOP_K=8",
                "UNI_RAG_FINAL_TOP_K=3",
                "UNI_RAG_RRF_K=42",
                "UNI_RAG_EMBEDDING_DIM=12",
                "UNI_RAG_USE_FAKE_LLM=false",
                "UNI_RAG_USE_FAKE_EMBEDDINGS=false",
                "UNI_RAG_OCR_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(repo_root=tmp_path, env_file=env_file)
    validate_config(config)

    assert config.courses_root == tmp_path / "Archive"
    assert config.data_dir == tmp_path / ".local-data"
    assert config.sqlite_path == tmp_path / ".local-data" / "app.sqlite"
    assert config.chroma_dir == tmp_path / ".local-data" / "chroma"
    assert config.runs_dir == tmp_path / ".local-data" / "runs"
    assert config.keyword_top_k == 7
    assert config.semantic_top_k == 8
    assert config.final_top_k == 3
    assert config.rrf_k == 42
    assert config.embedding_dim == 12
    assert config.use_fake_llm is False
    assert config.use_fake_embeddings is False
    assert config.ocr_enabled is True


def test_legacy_courses_dir_env_alias_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "LegacyCourses").mkdir()
    monkeypatch.setenv("UNI_RAG_COURSES_DIR", "LegacyCourses")

    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    validate_config(config)

    assert config.courses_root == tmp_path / "LegacyCourses"


def test_missing_courses_root_fails_validation(tmp_path: Path) -> None:
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")

    with pytest.raises(ConfigError, match="Courses root does not exist"):
        validate_config(config)


def test_generated_paths_cannot_live_under_courses(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    courses_root.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "UNI_RAG_COURSES_ROOT=Courses",
                "UNI_RAG_DATA_DIR=Courses/generated-data",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(repo_root=tmp_path, env_file=env_file)

    with pytest.raises(ConfigError, match="must not point inside the Courses root"):
        validate_config(config)


def test_safe_dict_excludes_api_key_like_fields(tmp_path: Path) -> None:
    (tmp_path / "Courses").mkdir()

    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    safe = config.as_safe_dict()

    assert "api_key" not in " ".join(safe)
    assert "OPENAI_API_KEY" not in safe
    assert safe["courses_root"] == str(tmp_path / "Courses")
