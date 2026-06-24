"""Configuration loading for Uni RAG Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when environment configuration cannot be parsed."""


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    courses_dir: Path
    data_dir: Path
    sqlite_path: Path
    runs_dir: Path
    log_level: str
    ocr_enabled: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    keyword_top_k: int
    semantic_top_k: int
    final_top_k: int

    def as_safe_dict(self) -> dict[str, str | int | bool]:
        return {
            "repo_root": str(self.repo_root),
            "courses_dir": str(self.courses_dir),
            "data_dir": str(self.data_dir),
            "sqlite_path": str(self.sqlite_path),
            "runs_dir": str(self.runs_dir),
            "log_level": self.log_level,
            "ocr_enabled": self.ocr_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "keyword_top_k": self.keyword_top_k,
            "semantic_top_k": self.semantic_top_k,
            "final_top_k": self.final_top_k,
        }


def load_settings(repo_root: Path | None = None, env_file: Path | None = None) -> Settings:
    root = (repo_root or find_project_root()).resolve()
    dotenv_path = env_file or root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    data_dir = _path_from_env(root, "UNI_RAG_DATA_DIR", "data")

    return Settings(
        repo_root=root,
        courses_dir=_path_from_env(root, "UNI_RAG_COURSES_DIR", "Courses"),
        data_dir=data_dir,
        sqlite_path=_path_from_env(
            root,
            "UNI_RAG_SQLITE_PATH",
            str(data_dir / "uni_rag.sqlite"),
        ),
        runs_dir=_path_from_env(root, "UNI_RAG_RUNS_DIR", str(data_dir / "runs")),
        log_level=_log_level_from_env("UNI_RAG_LOG_LEVEL", "INFO"),
        ocr_enabled=_bool_from_env("UNI_RAG_OCR_ENABLED", False),
        llm_provider=os.environ.get("UNI_RAG_LLM_PROVIDER", "fake").strip() or "fake",
        llm_model=os.environ.get("UNI_RAG_LLM_MODEL", "fake-chat").strip() or "fake-chat",
        embedding_provider=os.environ.get("UNI_RAG_EMBEDDING_PROVIDER", "fake").strip()
        or "fake",
        embedding_model=os.environ.get("UNI_RAG_EMBEDDING_MODEL", "fake-embedding").strip()
        or "fake-embedding",
        keyword_top_k=_int_from_env("UNI_RAG_KEYWORD_TOP_K", 20),
        semantic_top_k=_int_from_env("UNI_RAG_SEMANTIC_TOP_K", 20),
        final_top_k=_int_from_env("UNI_RAG_FINAL_TOP_K", 10),
    )


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _path_from_env(root: Path, name: str, default: str) -> Path:
    raw_value = os.environ.get(name, default).strip()
    if not raw_value:
        raise ConfigError(f"{name} cannot be empty")
    path = Path(raw_value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return value


def _log_level_from_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().upper()
    if value not in ALLOWED_LOG_LEVELS:
        raise ConfigError(f"{name} must be one of: {', '.join(sorted(ALLOWED_LOG_LEVELS))}")
    return value
