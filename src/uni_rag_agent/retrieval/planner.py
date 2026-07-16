"""Mandatory LLM query planning for the read-only retrieval pipeline."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from typing import Any

from uni_rag_agent.config import ALLOWED_LLM_PROVIDERS, Config, validate_config
from uni_rag_agent.storage import connect_sqlite_read_only

from .models import LOGICAL_INDEXES, QUERY_TYPES, QueryPlan

MAX_QUERY_PLAN_CONTEXT_MESSAGES = 6
MAX_QUERY_PLAN_KEYWORD_TERMS = 20
_JSON_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<payload>\{.*\})\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


class QueryPlanningError(RuntimeError):
    """Raised when mandatory LLM query planning cannot produce a valid plan."""


def normalize_query(query: str) -> str:
    """Normalize required query text without changing the user's wording."""
    if not isinstance(query, str):
        raise QueryPlanningError("Query must be text.")
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise QueryPlanningError("Query must not be empty.")
    return normalized


def plan_query(
    config: Config,
    query: str,
    conversation_context: Sequence[Mapping[str, str]] | None = None,
    *,
    chat_model: object | None = None,
) -> QueryPlan:
    """Call the configured LLM once and validate its structured query plan."""
    validate_config(config)
    _require_llm_configuration(config)
    normalized = normalize_query(query)
    context = _validate_conversation_context(conversation_context)
    courses = _load_courses(config)
    model = chat_model if chat_model is not None else build_chat_model(config)
    try:
        response = model.invoke(  # type: ignore[attr-defined]
            _planner_prompt(
                normalized,
                courses=courses,
                context=context,
                semantic_query_limit=config.semantic_query_limit,
            )
        )
    except Exception as exc:  # noqa: BLE001 - provider failures are fatal
        raise QueryPlanningError(
            f"LLM query-planning invocation failed: {exc}"
        ) from exc
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, Mapping) else str(item)
            for item in content
        )
    content_text = str(content)
    fenced_match = _JSON_CODE_FENCE_RE.fullmatch(content_text)
    if fenced_match is not None:
        content_text = fenced_match.group("payload")
    try:
        payload = json.loads(content_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise QueryPlanningError("LLM query plan must be one JSON object.") from exc
    return _parse_query_plan(config, payload, courses=courses)


def _require_llm_configuration(config: Config) -> None:
    if config.llm_provider not in ALLOWED_LLM_PROVIDERS or not config.llm_model:
        raise QueryPlanningError(
            "retrieve requires UNI_RAG_LLM_PROVIDER and UNI_RAG_LLM_MODEL."
        )


def _load_courses(config: Config) -> tuple[str, ...]:
    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            rows = connection.execute(
                "SELECT name FROM courses ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise QueryPlanningError(f"Could not load canonical courses: {exc}") from exc
    return tuple(str(row["name"]) for row in rows)


def _parse_query_plan(
    config: Config,
    payload: Any,
    *,
    courses: Sequence[str],
) -> QueryPlan:
    required = {
        "query_type",
        "candidate_courses",
        "candidate_indexes",
        "keyword_terms",
        "semantic_queries",
        "needs_file_inspection",
        "needs_python",
        "plan_confidence",
        "plan_reason",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise QueryPlanningError(
            "LLM query plan must contain exactly the QueryPlan fields."
        )

    query_type = payload["query_type"]

    if not isinstance(query_type, str) or query_type not in QUERY_TYPES:
        raise QueryPlanningError("LLM query plan contains an unknown query type.")

    canonical_courses = _canonicalize_courses(payload["candidate_courses"], courses)
    indexes = _validate_strings(
        payload["candidate_indexes"], LOGICAL_INDEXES, "logical index"
    )
    keyword_terms = _validate_strings(payload["keyword_terms"], None, "keyword term")
    semantic_queries = _validate_strings(
        payload["semantic_queries"], None, "semantic query"
    )
    if len(keyword_terms) > MAX_QUERY_PLAN_KEYWORD_TERMS:
        raise QueryPlanningError("LLM query plan contains too many keyword terms.")
    if len(semantic_queries) > config.semantic_query_limit:
        raise QueryPlanningError("LLM query plan contains too many semantic queries.")
    for field in ("needs_file_inspection", "needs_python"):
        if type(payload[field]) is not bool:
            raise QueryPlanningError(f"LLM query-plan field {field} must be boolean.")
    confidence = payload["plan_confidence"]
    if type(confidence) not in {int, float} or not 0.0 <= float(confidence) <= 1.0:
        raise QueryPlanningError("LLM plan_confidence must be between 0 and 1.")
    if float(confidence) < config.query_plan_min_confidence:
        raise QueryPlanningError(
            "LLM plan_confidence is below query_plan_min_confidence."
        )
    reason = payload["plan_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise QueryPlanningError("LLM plan_reason must be a nonblank string.")

    if query_type == "unknown_or_unsupported":
        if canonical_courses or indexes or keyword_terms or semantic_queries:
            raise QueryPlanningError(
                "Unsupported query plans must have empty retrieval scopes."
            )
    elif (
        not canonical_courses
        or not indexes
        or not keyword_terms
        or not semantic_queries
    ):
        raise QueryPlanningError(
            "Supported query plans require course, index, keyword, and semantic scopes."
        )

    return QueryPlan(
        query_type=query_type,
        candidate_courses=canonical_courses,
        candidate_indexes=indexes,
        keyword_terms=keyword_terms,
        semantic_queries=semantic_queries,
        needs_file_inspection=payload["needs_file_inspection"],
        needs_python=payload["needs_python"],
        plan_confidence=float(confidence),
        plan_reason=reason.strip(),
    )


def _canonicalize_courses(value: object, courses: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise QueryPlanningError("LLM course values must be a JSON string array.")
    canonical = {course.casefold(): course for course in courses}
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        resolved = canonical.get(item.strip().casefold())
        if resolved is None:
            raise QueryPlanningError(
                f"LLM query plan contains an unknown course: {item}"
            )
        if resolved.casefold() not in seen:
            seen.add(resolved.casefold())
            result.append(resolved)
    return tuple(result)


def _validate_strings(
    value: object, allowed: Sequence[str] | None, label: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise QueryPlanningError(f"LLM {label} values must be a JSON string array.")
    result: list[str] = []
    seen: set[str] = set()
    allowed_values = set(allowed) if allowed is not None else None
    for item in value:
        text = item.strip()
        if not text:
            raise QueryPlanningError(f"LLM {label} values must be nonblank.")
        if allowed_values is not None and text not in allowed_values:
            raise QueryPlanningError(
                f"LLM query plan contains an unknown {label}: {text}"
            )
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def _validate_conversation_context(
    context: Sequence[Mapping[str, str]] | None,
) -> tuple[dict[str, str], ...]:
    if context is None:
        return ()
    validated: list[dict[str, str]] = []
    for message in context:
        if set(message) != {"role", "content"}:
            raise QueryPlanningError(
                "Conversation messages must contain only role and content."
            )
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise QueryPlanningError(f"Unsupported conversation role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise QueryPlanningError(
                "Conversation message content must be nonblank text."
            )
        validated.append({"role": role, "content": content.strip()})
    return tuple(validated[-MAX_QUERY_PLAN_CONTEXT_MESSAGES:])


def _planner_prompt(
    query: str,
    *,
    courses: Sequence[str],
    context: Sequence[Mapping[str, str]],
    semantic_query_limit: int,
) -> str:
    schema = {
        "query_type": list(QUERY_TYPES),
        "candidate_courses": "canonical course-name array; empty only for unknown_or_unsupported",
        "candidate_indexes": "logical-index array; empty only for unknown_or_unsupported",
        "keyword_terms": f"non-empty array, maximum {MAX_QUERY_PLAN_KEYWORD_TERMS}; empty only for unknown_or_unsupported",
        "semantic_queries": "1 through semantic_query_limit non-empty queries; empty only for unknown_or_unsupported",
        "needs_file_inspection": "boolean",
        "needs_python": "boolean",
        "plan_confidence": "number from 0 through 1",
        "plan_reason": "short nonblank string",
    }
    return json.dumps(
        {
            "task": "Return only one JSON object matching this exact query-plan schema.",
            "query": query,
            "canonical_courses": list(courses),
            "logical_indexes": list(LOGICAL_INDEXES),
            "query_types": list(QUERY_TYPES),
            "semantic_query_limit": semantic_query_limit,
            "schema": schema,
            "recent_conversation": list(context),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_chat_model(config: Config) -> object:
    """Construct exactly the configured LangChain provider lazily."""
    _require_llm_configuration(config)
    try:
        if config.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=config.llm_model, temperature=0)
        if config.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=config.llm_model, temperature=0)
        if config.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=config.llm_model, temperature=0)
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.llm_model, temperature=0)
    except ImportError as exc:
        raise QueryPlanningError(
            f"LLM provider '{config.llm_provider}' requires the optional 'llm' extra. "
            "Install it with: uv sync --extra llm"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - construction failures are fatal
        raise QueryPlanningError(
            f"Could not construct LLM provider '{config.llm_provider}': {exc}"
        ) from exc
