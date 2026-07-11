from __future__ import annotations

from pathlib import Path

import pytest

from uni_rag_agent.config import ConfigError, load_config, validate_config

UNI_RAG_ENV_KEYS = {
    "UNI_RAG_CHROMA_DIR",
    "UNI_RAG_COURSES_DIR",
    "UNI_RAG_COURSES_ROOT",
    "UNI_RAG_DATA_DIR",
    "UNI_RAG_EMBEDDING_MODEL",
    "UNI_RAG_FINAL_TOP_K",
    "UNI_RAG_KEYWORD_TOP_K",
    "UNI_RAG_LLM_MODEL",
    "UNI_RAG_LLM_PROVIDER",
    "UNI_RAG_METADATA_TOP_K",
    "UNI_RAG_SEMANTIC_QUERY_LIMIT",
    "UNI_RAG_ROUTER_MIN_CONFIDENCE",
    "UNI_RAG_COURSE_FUZZY_THRESHOLD",
    "UNI_RAG_FILENAME_FUZZY_THRESHOLD",
    "UNI_RAG_PATH_FUZZY_THRESHOLD",
    "UNI_RAG_LOG_LEVEL",
    "UNI_RAG_OCR_ENABLED",
    "UNI_RAG_RRF_K",
    "UNI_RAG_RUNS_DIR",
    "UNI_RAG_SEMANTIC_TOP_K",
    "UNI_RAG_SQLITE_PATH",
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
    assert config.metadata_top_k == 20
    assert config.semantic_query_limit == 3
    assert config.router_min_confidence == 0.60
    assert config.course_fuzzy_threshold == 90
    assert config.filename_fuzzy_threshold == 85
    assert config.path_fuzzy_threshold == 90
    assert config.embedding_model is None
    assert config.llm_provider is None
    assert config.llm_model is None
    safe = config.as_safe_dict()
    assert safe["embedding_model"] is None
    assert safe["llm_provider"] is None
    assert safe["llm_model"] is None
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
                "UNI_RAG_METADATA_TOP_K=7",
                "UNI_RAG_SEMANTIC_QUERY_LIMIT=4",
                "UNI_RAG_ROUTER_MIN_CONFIDENCE=0.75",
                "UNI_RAG_COURSE_FUZZY_THRESHOLD=91",
                "UNI_RAG_FILENAME_FUZZY_THRESHOLD=86",
                "UNI_RAG_PATH_FUZZY_THRESHOLD=92",
                "UNI_RAG_LLM_PROVIDER=ollama",
                "UNI_RAG_LLM_MODEL=llama3.2",
                "UNI_RAG_EMBEDDING_MODEL=BAAI/bge-m3",
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
    assert config.metadata_top_k == 7
    assert config.semantic_query_limit == 4
    assert config.router_min_confidence == 0.75
    assert config.course_fuzzy_threshold == 91
    assert config.filename_fuzzy_threshold == 86
    assert config.path_fuzzy_threshold == 92
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.llm_provider == "ollama"
    assert config.llm_model == "llama3.2"
    assert config.ocr_enabled is True


def test_blank_optional_model_and_provider_values_are_unset(tmp_path: Path) -> None:
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNI_RAG_EMBEDDING_MODEL=   \nUNI_RAG_LLM_PROVIDER=\nUNI_RAG_LLM_MODEL=  \n",
        encoding="utf-8",
    )

    config = load_config(repo_root=tmp_path, env_file=env_file)

    assert config.embedding_model is None
    assert config.llm_provider is None
    assert config.llm_model is None


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


def test_courses_root_must_be_directory(tmp_path: Path) -> None:
    (tmp_path / "Courses").write_text("not a directory", encoding="utf-8")

    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")

    with pytest.raises(ConfigError, match="Courses root is not a directory"):
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


@pytest.mark.parametrize(
    ("env_line", "expected_message"),
    [
        ("UNI_RAG_OCR_ENABLED=maybe", "UNI_RAG_OCR_ENABLED must be one of"),
        ("UNI_RAG_KEYWORD_TOP_K=abc", "UNI_RAG_KEYWORD_TOP_K must be an integer"),
        (
            "UNI_RAG_KEYWORD_TOP_K=0",
            "UNI_RAG_KEYWORD_TOP_K must be greater than zero",
        ),
        ("UNI_RAG_LOG_LEVEL=TRACE", "UNI_RAG_LOG_LEVEL must be one of"),
        ("UNI_RAG_DATA_DIR=", "UNI_RAG_DATA_DIR cannot be empty"),
    ],
)
def test_invalid_env_values_fail_clearly(
    tmp_path: Path,
    env_line: str,
    expected_message: str,
) -> None:
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(env_line, encoding="utf-8")

    with pytest.raises(ConfigError, match=expected_message):
        load_config(repo_root=tmp_path, env_file=env_file)


def test_safe_dict_excludes_injected_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Courses").mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-anthropic-value")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-value")

    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    safe = config.as_safe_dict()

    unsafe_name_terms = ("api_key", "apikey", "secret", "token")
    for key in safe:
        assert all(term not in key.lower() for term in unsafe_name_terms)
    assert "secret-openai-value" not in safe.values()
    assert "secret-anthropic-value" not in safe.values()
    assert "secret-google-value" not in safe.values()
    assert safe["courses_root"] == str(tmp_path / "Courses")


@pytest.mark.parametrize(
    ("env_line", "expected_message"),
    [
        (
            "UNI_RAG_METADATA_TOP_K=0",
            "UNI_RAG_METADATA_TOP_K must be greater than zero",
        ),
        (
            "UNI_RAG_SEMANTIC_QUERY_LIMIT=-1",
            "UNI_RAG_SEMANTIC_QUERY_LIMIT must be greater than zero",
        ),
        ("UNI_RAG_ROUTER_MIN_CONFIDENCE=1.1", "must be between 0 and 1"),
        ("UNI_RAG_COURSE_FUZZY_THRESHOLD=101", "must be between 0 and 100"),
        ("UNI_RAG_LLM_PROVIDER=openai", "must be set together"),
        ("UNI_RAG_LLM_PROVIDER=unknown\nUNI_RAG_LLM_MODEL=model", "must be one of"),
    ],
)
def test_feature_08_config_validation(
    tmp_path: Path,
    env_line: str,
    expected_message: str,
) -> None:
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(env_line, encoding="utf-8")

    with pytest.raises(ConfigError, match=expected_message):
        config = load_config(repo_root=tmp_path, env_file=env_file)
        validate_config(config)
