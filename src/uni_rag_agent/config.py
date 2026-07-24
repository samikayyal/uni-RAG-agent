"""Configuration loading for Uni RAG Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
ALLOWED_LLM_PROVIDERS = {"openai", "anthropic", "gemini", "ollama"}
DEFAULT_ANSWER_PROMPT_MAX_TOKENS = 16_000
DEFAULT_ASK_TIMEOUT_SECONDS = 120
PUBLIC_EMBEDDING_PROFILES = (
    "google/embeddinggemma-300m",
    "google/gemini-embedding-001",
    "Qwen/Qwen3-Embedding-8B",
)


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
    llm_provider: str | None
    llm_model: str | None
    embedding_model: str | None
    ocr_enabled: bool
    google_api_key: str | None = field(default=None, repr=False)
    google_api_key_2: str | None = field(default=None, repr=False)
    google_api_key_embedding: str | None = field(default=None, repr=False)
    nebius_api_key: str | None = field(default=None, repr=False)
    metadata_top_k: int = 20
    semantic_query_limit: int = 3
    query_plan_min_confidence: float = 0.60
    filename_fuzzy_threshold: int = 85
    path_fuzzy_threshold: int = 90
    evidence_max_tokens: int = 12_000
    answer_llm_provider: str | None = None
    answer_llm_model: str | None = None
    answer_max_retries: int = 1
    answer_session_message_limit: int = 20
    answer_prompt_max_tokens: int = DEFAULT_ANSWER_PROMPT_MAX_TOKENS
    ask_timeout_seconds: int = DEFAULT_ASK_TIMEOUT_SECONDS
    hosted_mode: bool = False
    public_demo_enabled: bool = False
    public_embedding_profiles: tuple[str, ...] = PUBLIC_EMBEDDING_PROFILES
    public_default_embedding_model: str = "google/gemini-embedding-001"
    public_query_max_chars: int = 4_000
    public_ask_capacity: int = 2
    public_capacity_wait_seconds: int = 10
    public_minute_limit: int = 3
    public_client_daily_limit: int = 10
    public_global_daily_limit: int = 100
    demo_token_ttl_seconds: int = 30 * 60
    turnstile_site_key: str | None = None
    turnstile_secret_key: str | None = field(default=None, repr=False)
    demo_token_signing_secret: str | None = field(default=None, repr=False)
    firestore_project_id: str | None = None
    firestore_database: str = "(default)"
    embeddinggemma_model_path: Path | None = None

    def as_safe_dict(self) -> dict[str, str | int | float | bool | None]:
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
            "embedding_model": self.embedding_model,
            "ocr_enabled": self.ocr_enabled,
            "metadata_top_k": self.metadata_top_k,
            "semantic_query_limit": self.semantic_query_limit,
            "query_plan_min_confidence": self.query_plan_min_confidence,
            "filename_fuzzy_threshold": self.filename_fuzzy_threshold,
            "path_fuzzy_threshold": self.path_fuzzy_threshold,
            "evidence_max_tokens": self.evidence_max_tokens,
            "answer_llm_provider": self.answer_llm_provider,
            "answer_llm_model": self.answer_llm_model,
            "answer_max_retries": self.answer_max_retries,
            "answer_session_message_limit": self.answer_session_message_limit,
            "answer_prompt_max_tokens": self.answer_prompt_max_tokens,
            "ask_timeout_seconds": self.ask_timeout_seconds,
            "hosted_mode": self.hosted_mode,
            "public_demo_enabled": self.public_demo_enabled,
            "public_embedding_profiles": ",".join(self.public_embedding_profiles),
            "public_default_embedding_model": self.public_default_embedding_model,
            "public_query_max_chars": self.public_query_max_chars,
            "public_ask_capacity": self.public_ask_capacity,
            "public_capacity_wait_seconds": self.public_capacity_wait_seconds,
            "public_minute_limit": self.public_minute_limit,
            "public_client_daily_limit": self.public_client_daily_limit,
            "public_global_daily_limit": self.public_global_daily_limit,
            "demo_token_ttl_seconds": self.demo_token_ttl_seconds,
            "turnstile_site_key": self.turnstile_site_key,
            "firestore_project_id": self.firestore_project_id,
            "firestore_database": self.firestore_database,
            "embeddinggemma_model_path": (
                str(self.embeddinggemma_model_path)
                if self.embeddinggemma_model_path is not None
                else None
            ),
        }

    @property
    def courses_dir(self) -> Path:
        """Backward-compatible alias for the Feature 01 settings name."""
        return self.courses_root

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def answer_provider(self) -> str | None:
        """Short alias used by answer integrations."""
        return self.answer_llm_provider

    @property
    def answer_model(self) -> str | None:
        """Short alias used by answer integrations."""
        return self.answer_llm_model


Settings = Config


def load_config(repo_root: Path | None = None, env_file: Path | None = None) -> Config:
    root = (repo_root or find_project_root()).resolve()
    dotenv_path = env_file or root / ".env"
    env = _merged_env(dotenv_path)

    data_dir = _path_from_env(root, env, "UNI_RAG_DATA_DIR", "data")
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
        llm_provider=_optional_str_from_env(env, "UNI_RAG_LLM_PROVIDER"),
        llm_model=_optional_str_from_env(env, "UNI_RAG_LLM_MODEL"),
        embedding_model=_optional_str_from_env(env, "UNI_RAG_EMBEDDING_MODEL"),
        ocr_enabled=_bool_from_env(env, "UNI_RAG_OCR_ENABLED", False),
        google_api_key=_optional_str_from_env(env, "GOOGLE_API_KEY"),
        google_api_key_2=_optional_str_from_env(env, "GOOGLE_API_KEY_2"),
        google_api_key_embedding=_optional_str_from_env(
            env, "GOOGLE_API_KEY_EMBEDDING"
        ),
        nebius_api_key=_optional_str_from_env(env, "NEBIUS_API_KEY"),
        metadata_top_k=_int_from_env(env, "UNI_RAG_METADATA_TOP_K", 20),
        semantic_query_limit=_int_from_env(env, "UNI_RAG_SEMANTIC_QUERY_LIMIT", 3),
        query_plan_min_confidence=_float_from_env(
            env, "UNI_RAG_QUERY_PLAN_MIN_CONFIDENCE", 0.60
        ),
        filename_fuzzy_threshold=_bounded_int_from_env(
            env, "UNI_RAG_FILENAME_FUZZY_THRESHOLD", 85
        ),
        path_fuzzy_threshold=_bounded_int_from_env(
            env, "UNI_RAG_PATH_FUZZY_THRESHOLD", 90
        ),
        evidence_max_tokens=_evidence_max_tokens_from_env(
            env, "UNI_RAG_EVIDENCE_MAX_TOKENS", 12_000
        ),
        answer_llm_provider=_optional_str_from_env(env, "UNI_RAG_ANSWER_LLM_PROVIDER"),
        answer_llm_model=_optional_str_from_env(env, "UNI_RAG_ANSWER_LLM_MODEL"),
        answer_max_retries=_nonnegative_int_from_env(
            env, "UNI_RAG_ANSWER_MAX_RETRIES", 1
        ),
        answer_session_message_limit=_positive_int_from_env(
            env, "UNI_RAG_ANSWER_SESSION_MESSAGE_LIMIT", 20
        ),
        answer_prompt_max_tokens=_strict_positive_int_from_env(
            env,
            "UNI_RAG_ANSWER_PROMPT_MAX_TOKENS",
            DEFAULT_ANSWER_PROMPT_MAX_TOKENS,
        ),
        ask_timeout_seconds=_strict_positive_int_from_env(
            env,
            "UNI_RAG_ASK_TIMEOUT_SECONDS",
            DEFAULT_ASK_TIMEOUT_SECONDS,
        ),
        hosted_mode=_bool_from_env(env, "UNI_RAG_HOSTED_MODE", False),
        public_demo_enabled=_bool_from_env(env, "UNI_RAG_PUBLIC_DEMO_ENABLED", False),
        public_embedding_profiles=_csv_from_env(
            env,
            "UNI_RAG_PUBLIC_EMBEDDING_PROFILES",
            PUBLIC_EMBEDDING_PROFILES,
        ),
        public_default_embedding_model=_str_from_env(
            env,
            "UNI_RAG_PUBLIC_DEFAULT_EMBEDDING_MODEL",
            "google/gemini-embedding-001",
        ),
        public_query_max_chars=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_QUERY_MAX_CHARS", 4_000
        ),
        public_ask_capacity=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_ASK_CAPACITY", 2
        ),
        public_capacity_wait_seconds=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_CAPACITY_WAIT_SECONDS", 10
        ),
        public_minute_limit=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_MINUTE_LIMIT", 3
        ),
        public_client_daily_limit=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT", 10
        ),
        public_global_daily_limit=_strict_positive_int_from_env(
            env, "UNI_RAG_PUBLIC_GLOBAL_DAILY_LIMIT", 100
        ),
        demo_token_ttl_seconds=_strict_positive_int_from_env(
            env, "UNI_RAG_DEMO_TOKEN_TTL_SECONDS", 30 * 60
        ),
        turnstile_site_key=_optional_str_from_env(env, "TURNSTILE_SITE_KEY"),
        turnstile_secret_key=_optional_str_from_env(env, "TURNSTILE_SECRET_KEY"),
        demo_token_signing_secret=_optional_str_from_env(
            env, "UNI_RAG_DEMO_TOKEN_SIGNING_SECRET"
        ),
        firestore_project_id=_optional_str_from_env(
            env, "UNI_RAG_FIRESTORE_PROJECT_ID"
        ),
        firestore_database=_str_from_env(
            env, "UNI_RAG_FIRESTORE_DATABASE", "(default)"
        ),
        embeddinggemma_model_path=_optional_path_from_env(
            root, env, "UNI_RAG_EMBEDDINGGEMMA_MODEL_PATH"
        ),
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

    _validate_positive_value("metadata_top_k", config.metadata_top_k)
    _validate_positive_value("semantic_query_limit", config.semantic_query_limit)
    if not 0.0 <= config.query_plan_min_confidence <= 1.0:
        raise ConfigError("UNI_RAG_QUERY_PLAN_MIN_CONFIDENCE must be between 0 and 1")
    for name, value in {
        "UNI_RAG_FILENAME_FUZZY_THRESHOLD": config.filename_fuzzy_threshold,
        "UNI_RAG_PATH_FUZZY_THRESHOLD": config.path_fuzzy_threshold,
    }.items():
        if not 0 <= value <= 100:
            raise ConfigError(f"{name} must be between 0 and 100")
    if config.evidence_max_tokens <= 0:
        raise ConfigError("UNI_RAG_EVIDENCE_MAX_TOKENS must be greater than zero")

    provider = config.llm_provider
    model = config.llm_model
    if (provider is None) != (model is None):
        raise ConfigError(
            "UNI_RAG_LLM_PROVIDER and UNI_RAG_LLM_MODEL must be set together"
        )
    if provider is not None:
        if provider not in ALLOWED_LLM_PROVIDERS:
            allowed = ", ".join(sorted(ALLOWED_LLM_PROVIDERS))
            raise ConfigError(f"UNI_RAG_LLM_PROVIDER must be one of: {allowed}")
        if not model or not model.strip():
            raise ConfigError("UNI_RAG_LLM_MODEL cannot be empty")

    answer_provider = config.answer_llm_provider
    answer_model = config.answer_llm_model
    if (answer_provider is None) != (answer_model is None):
        raise ConfigError(
            "UNI_RAG_ANSWER_LLM_PROVIDER and UNI_RAG_ANSWER_LLM_MODEL must be set together"
        )
    if answer_provider is not None:
        if answer_provider not in ALLOWED_LLM_PROVIDERS:
            allowed = ", ".join(sorted(ALLOWED_LLM_PROVIDERS))
            raise ConfigError(f"UNI_RAG_ANSWER_LLM_PROVIDER must be one of: {allowed}")
        if not answer_model or not answer_model.strip():
            raise ConfigError("UNI_RAG_ANSWER_LLM_MODEL cannot be empty")
    if config.answer_max_retries < 0:
        raise ConfigError("UNI_RAG_ANSWER_MAX_RETRIES must be nonnegative")
    if config.answer_session_message_limit <= 0:
        raise ConfigError(
            "UNI_RAG_ANSWER_SESSION_MESSAGE_LIMIT must be greater than zero"
        )
    if config.answer_prompt_max_tokens <= 0:
        raise ConfigError("UNI_RAG_ANSWER_PROMPT_MAX_TOKENS must be greater than zero")
    if config.ask_timeout_seconds <= 0:
        raise ConfigError("UNI_RAG_ASK_TIMEOUT_SECONDS must be greater than zero")

    _validate_public_demo_config(config)


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


def _optional_str_from_env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _str_from_env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ConfigError(f"{name} cannot be empty")
    return value


def _csv_from_env(
    env: Mapping[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    raw = env.get(name)
    if raw is None:
        return default
    values = tuple(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    )
    if not values:
        raise ConfigError(f"{name} must contain at least one value")
    return values


def _optional_path_from_env(
    root: Path, env: Mapping[str, str], name: str
) -> Path | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    path = Path(raw.strip())
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _validate_public_demo_config(config: Config) -> None:
    public_profiles = set(PUBLIC_EMBEDDING_PROFILES)
    enabled_profiles = set(config.public_embedding_profiles)
    unknown = enabled_profiles - public_profiles
    if unknown:
        raise ConfigError(
            "UNI_RAG_PUBLIC_EMBEDDING_PROFILES contains undeployed profiles: "
            + ", ".join(sorted(unknown))
        )
    if config.public_default_embedding_model not in enabled_profiles:
        raise ConfigError(
            "UNI_RAG_PUBLIC_DEFAULT_EMBEDDING_MODEL must be one of the enabled "
            "public embedding profiles"
        )
    for name, value in {
        "UNI_RAG_PUBLIC_QUERY_MAX_CHARS": config.public_query_max_chars,
        "UNI_RAG_PUBLIC_ASK_CAPACITY": config.public_ask_capacity,
        "UNI_RAG_PUBLIC_CAPACITY_WAIT_SECONDS": config.public_capacity_wait_seconds,
        "UNI_RAG_PUBLIC_MINUTE_LIMIT": config.public_minute_limit,
        "UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT": config.public_client_daily_limit,
        "UNI_RAG_PUBLIC_GLOBAL_DAILY_LIMIT": config.public_global_daily_limit,
        "UNI_RAG_DEMO_TOKEN_TTL_SECONDS": config.demo_token_ttl_seconds,
    }.items():
        if value <= 0:
            raise ConfigError(f"{name} must be greater than zero")
    if config.public_minute_limit > config.public_client_daily_limit:
        raise ConfigError(
            "UNI_RAG_PUBLIC_MINUTE_LIMIT cannot exceed the client daily limit"
        )
    if config.public_client_daily_limit > config.public_global_daily_limit:
        raise ConfigError(
            "UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT cannot exceed the global daily limit"
        )
    if config.hosted_mode:
        path = config.embeddinggemma_model_path
        if path is None or not path.is_dir():
            raise ConfigError(
                "UNI_RAG_EMBEDDINGGEMMA_MODEL_PATH must be an existing directory "
                "when UNI_RAG_HOSTED_MODE is enabled"
            )
    if not config.public_demo_enabled:
        return
    public_bounds = {
        "UNI_RAG_KEYWORD_TOP_K": (config.keyword_top_k, 40),
        "UNI_RAG_SEMANTIC_TOP_K": (config.semantic_top_k, 40),
        "UNI_RAG_METADATA_TOP_K": (config.metadata_top_k, 40),
        "UNI_RAG_FINAL_TOP_K": (config.final_top_k, 15),
        "UNI_RAG_SEMANTIC_QUERY_LIMIT": (config.semantic_query_limit, 3),
        "UNI_RAG_EVIDENCE_MAX_TOKENS": (config.evidence_max_tokens, 16_000),
        "UNI_RAG_PUBLIC_QUERY_MAX_CHARS": (config.public_query_max_chars, 4_000),
        "UNI_RAG_PUBLIC_ASK_CAPACITY": (config.public_ask_capacity, 2),
        "UNI_RAG_PUBLIC_MINUTE_LIMIT": (config.public_minute_limit, 3),
        "UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT": (
            config.public_client_daily_limit,
            10,
        ),
        "UNI_RAG_PUBLIC_GLOBAL_DAILY_LIMIT": (
            config.public_global_daily_limit,
            100,
        ),
    }
    for name, (value, maximum) in public_bounds.items():
        if value > maximum:
            raise ConfigError(f"{name} must not exceed {maximum} in public mode")
    required = {
        "TURNSTILE_SITE_KEY": config.turnstile_site_key,
        "TURNSTILE_SECRET_KEY": config.turnstile_secret_key,
        "UNI_RAG_DEMO_TOKEN_SIGNING_SECRET": config.demo_token_signing_secret,
        "UNI_RAG_FIRESTORE_PROJECT_ID": config.firestore_project_id,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ConfigError(
            "Public demo mode requires configuration for: " + ", ".join(missing)
        )
    if len(config.demo_token_signing_secret or "") < 32:
        raise ConfigError(
            "UNI_RAG_DEMO_TOKEN_SIGNING_SECRET must contain at least 32 characters"
        )


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


def _bounded_int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    value = _int_from_env_allow_zero(env, name, default)
    if not 0 <= value <= 100:
        raise ConfigError(f"{name} must be between 0 and 100")
    return value


def _evidence_max_tokens_from_env(
    env: Mapping[str, str], name: str, default: int
) -> int:
    return _strict_positive_int_from_env(env, name, default)


def _strict_positive_int_from_env(
    env: Mapping[str, str], name: str, default: int
) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    if not raw_value.strip():
        raise ConfigError(f"{name} must be an integer greater than zero")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer greater than zero") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return value


def _nonnegative_int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    if raw_value.strip() == "":
        raise ConfigError(f"{name} must be a nonnegative integer")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigError(f"{name} must be nonnegative")
    return value


def _positive_int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
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


def _int_from_env_allow_zero(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_from_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw_value = env.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return value


def _validate_positive_value(name: str, value: int) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")


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
