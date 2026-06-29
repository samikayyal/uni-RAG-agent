"""Configuration loading for Uni RAG Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when environment configuration cannot be parsed."""


@dataclass(frozen=True)
class Config:
    repo_root: Path
    courses_root: Path
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    runs_dir: Path
    log_level: str
    keyword_top_k: int
    semantic_top_k: int
    final_top_k: int
    rrf_k: int
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    use_fake_llm: bool
    use_fake_embeddings: bool
    ocr_enabled: bool

    def as_safe_dict(self) -> dict[str, str | int | bool]:
        return {
            "repo_root": str(self.repo_root),
            "courses_root": str(self.courses_root),
            "data_dir": str(self.data_dir),
            "sqlite_path": str(self.sqlite_path),
            "chroma_dir": str(self.chroma_dir),
            "runs_dir": str(self.runs_dir),
            "log_level": self.log_level,
            "keyword_top_k": self.keyword_top_k,
            "semantic_top_k": self.semantic_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "use_fake_llm": self.use_fake_llm,
            "use_fake_embeddings": self.use_fake_embeddings,
            "ocr_enabled": self.ocr_enabled,
        }

    @property
    def courses_dir(self) -> Path:
        """Backward-compatible alias for the Feature 01 settings name."""
        return self.courses_root

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"


Settings = Config


def load_config(repo_root: Path | None = None, env_file: Path | None = None) -> Config:
    root = (repo_root or find_project_root()).resolve()
    dotenv_path = env_file or root / ".env"
    env = _merged_env(dotenv_path)

    data_dir = _path_from_env(root, env, "UNI_RAG_DATA_DIR", "data")
    llm_provider = _str_from_env(env, "UNI_RAG_LLM_PROVIDER", "fake")
    embedding_provider = _str_from_env(env, "UNI_RAG_EMBEDDING_PROVIDER", "fake")

    return Config(
        repo_root=root,
        courses_root=_path_from_env(
            root,
            env,
            "UNI_RAG_COURSES_ROOT",
            "Courses",
            aliases=("UNI_RAG_COURSES_DIR",),
        ),
        data_dir=data_dir,
        sqlite_path=_path_from_env(
            root,
            env,
            "UNI_RAG_SQLITE_PATH",
            str(data_dir / "uni_rag.sqlite"),
        ),
        chroma_dir=_path_from_env(
            root,
            env,
            "UNI_RAG_CHROMA_DIR",
            str(data_dir / "indexes" / "vector"),
        ),
        runs_dir=_path_from_env(
            root,
            env,
            "UNI_RAG_RUNS_DIR",
            str(data_dir / "runs"),
        ),
        log_level=_log_level_from_env(env, "UNI_RAG_LOG_LEVEL", "INFO"),
        keyword_top_k=_int_from_env(env, "UNI_RAG_KEYWORD_TOP_K", 20),
        semantic_top_k=_int_from_env(env, "UNI_RAG_SEMANTIC_TOP_K", 20),
        final_top_k=_int_from_env(env, "UNI_RAG_FINAL_TOP_K", 10),
        rrf_k=_int_from_env(env, "UNI_RAG_RRF_K", 60),
        llm_provider=llm_provider,
        llm_model=_str_from_env(env, "UNI_RAG_LLM_MODEL", "fake-chat"),
        embedding_provider=embedding_provider,
        embedding_model=_str_from_env(
            env,
            "UNI_RAG_EMBEDDING_MODEL",
            "fake-embedding",
        ),
        embedding_dim=_int_from_env(env, "UNI_RAG_EMBEDDING_DIM", 384),
        use_fake_llm=_bool_from_env(
            env,
            "UNI_RAG_USE_FAKE_LLM",
            llm_provider == "fake",
        ),
        use_fake_embeddings=_bool_from_env(
            env,
            "UNI_RAG_USE_FAKE_EMBEDDINGS",
            embedding_provider == "fake",
        ),
        ocr_enabled=_bool_from_env(env, "UNI_RAG_OCR_ENABLED", False),
    )


def load_settings(
    repo_root: Path | None = None, env_file: Path | None = None
) -> Settings:
    """Backward-compatible alias for the Feature 01 loader name."""
    return load_config(repo_root=repo_root, env_file=env_file)


def validate_config(config: Config) -> None:
    if not config.courses_root.exists():
        raise ConfigError(f"Courses root does not exist: {config.courses_root}")
    if not config.courses_root.is_dir():
        raise ConfigError(f"Courses root is not a directory: {config.courses_root}")

    for name, path in {
        "UNI_RAG_DATA_DIR": config.data_dir,
        "UNI_RAG_SQLITE_PATH": config.sqlite_path,
        "UNI_RAG_CHROMA_DIR": config.chroma_dir,
        "UNI_RAG_RUNS_DIR": config.runs_dir,
    }.items():
        if _is_relative_to(path, config.courses_root):
            raise ConfigError(f"{name} must not point inside the Courses root: {path}")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _path_from_env(
    root: Path,
    env: Mapping[str, str],
    name: str,
    default: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Path:
    raw_value = _first_env_value(env, (name, *aliases), default).strip()
    if not raw_value:
        raise ConfigError(f"{name} cannot be empty")
    path = Path(raw_value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _str_from_env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ConfigError(f"{name} cannot be empty")
    return value


def _bool_from_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = env.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return value


def _log_level_from_env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip().upper()
    if value not in ALLOWED_LOG_LEVELS:
        allowed = ", ".join(sorted(ALLOWED_LOG_LEVELS))
        raise ConfigError(f"{name} must be one of: {allowed}")
    return value


def _first_env_value(
    env: Mapping[str, str],
    names: tuple[str, ...],
    default: str,
) -> str:
    for name in names:
        value = env.get(name)
        if value is not None:
            return value
    return default


def _merged_env(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if dotenv_path.exists():
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None:
                values[key] = value
    values.update(os.environ)
    return values


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
