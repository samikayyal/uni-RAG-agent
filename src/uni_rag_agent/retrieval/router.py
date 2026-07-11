"""Deterministic query routing with an optional LangChain fallback."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from typing import Any

from rapidfuzz import fuzz, process

from uni_rag_agent.config import ALLOWED_LLM_PROVIDERS, Config, validate_config
from uni_rag_agent.storage import connect_sqlite_read_only

from .models import LOGICAL_INDEXES, QUERY_TYPES, RouterOutput

ALL_INDEXES = tuple(LOGICAL_INDEXES)
COURSE_FUZZY_LEAD = 5
MAX_ROUTER_CONTEXT_MESSAGES = 6
MAX_LLM_KEYWORD_TERMS = 20

ALIASES = {
    "AI": "Artificial Intelligence",
    "BI": "Business Intelligence",
    "DB": "Database",
    "Database Systems": "Database",
    "Data Engineering": "Data Eng",
    "Digital Image Processing": "DIP",
    "HPC": "High Preformance Computing for Big Data",
    "High Performance Computing": "High Preformance Computing for Big Data",
    "High Performance Computing for Big Data": "High Preformance Computing for Big Data",
    "IR": "Information Retrieval",
    "Intro to Data Science": "Intro to DS",
    "ML": "Machine Learning",
    "NLP": "NLP",
    "Natural Language Processing": "NLP",
    "OOP": "Object Oriented Programming",
    "OS": "Operating Systems",
    "KG": "Special Topics 1 (Knowledge Graphs)",
    "Knowledge Graphs": "Special Topics 1 (Knowledge Graphs)",
    "Technical Writing": "Techincal Writing",
}

TYPE_CUES = {
    "cross_course_comparison": (
        "compare",
        "versus",
        "vs",
        "difference",
        "across",
        "both courses",
    ),
    "study_quiz": (
        "quiz",
        "flashcards",
        "practice questions",
        "exam prep",
        "study guide",
    ),
    "portfolio_resume": (
        "portfolio",
        "resume",
        "cv",
        "project bullet",
        "experience bullet",
    ),
    "find_file": ("find", "locate", "where is"),
    "assignment_or_project_lookup": (
        "assignment",
        "homework",
        "project",
        "capstone",
        "deliverable",
    ),
    "course_summary": (
        "course summary",
        "overview of course",
        "syllabus",
        "what did i study in",
    ),
    "code_question": ("code", "implementation", "function", "class", "script"),
    "data_question": ("dataset", "schema", "columns", "rows", "table"),
    "concept_explanation": ("explain", "define", "what is", "how does"),
}

TYPE_INDEXES = {
    "concept_explanation": ALL_INDEXES,
    "course_summary": ALL_INDEXES,
    "cross_course_comparison": ALL_INDEXES,
    "find_file": ALL_INDEXES,
    "assignment_or_project_lookup": (
        "document_index",
        "slides_index",
        "notebook_index",
        "code_index",
        "data_schema_index",
    ),
    "code_question": ("code_index", "notebook_index"),
    "data_question": ("data_schema_index", "notebook_index", "code_index"),
    "study_quiz": ALL_INDEXES,
    "portfolio_resume": ALL_INDEXES,
    "unknown_or_unsupported": (),
}

EXTENSION_INDEXES = {
    ".pdf": "document_index",
    ".docx": "document_index",
    ".doc": "document_index",
    ".txt": "document_index",
    ".md": "document_index",
    ".pptx": "slides_index",
    ".ppt": "slides_index",
    ".ipynb": "notebook_index",
    ".py": "code_index",
    ".r": "code_index",
    ".cpp": "code_index",
    ".h": "code_index",
    ".m": "code_index",
    ".csv": "data_schema_index",
    ".xlsx": "data_schema_index",
    ".json": "data_schema_index",
    ".jsonl": "data_schema_index",
    ".sqlite": "data_schema_index",
    ".db": "data_schema_index",
    ".vtt": "transcript_index",
}

METADATA_ONLY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".jfif",
    ".zip",
    ".rar",
    ".7z",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4a",
    ".wav",
    ".exe",
    ".msi",
    ".cab",
    ".bin",
    ".joblib",
    ".weights",
    ".tflite",
    ".pt",
    ".pkl",
    ".rdata",
    ".rds",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "course",
    "courses",
    "did",
    "do",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "locate",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "the",
    "to",
    "what",
    "where",
    "with",
    "about",
    "does",
    "this",
    "that",
    "was",
    "were",
    "you",
    "your",
}

PYTHON_CUES = (
    "run",
    "execute",
    "calculate",
    "analyze with python",
    "reproduce",
    "evaluate",
    "test this code",
    "inspect runtime behavior",
)


class RoutingError(RuntimeError):
    """Raised when an enabled LLM router cannot be invoked."""


def normalize_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise RoutingError("Query must not be empty.")
    return normalized


def route_query(
    config: Config,
    query: str,
    conversation_context: Sequence[Mapping[str, str]] | None = None,
    *,
    chat_model: object | None = None,
) -> RouterOutput:
    """Route a query using current SQLite course names and deterministic rules."""
    validate_config(config)
    normalized = normalize_query(query)
    context = _validate_conversation_context(conversation_context)
    courses = _load_courses(config)
    aliases = _valid_aliases(courses)
    course_matches, course_ambiguous = _match_courses(
        normalized, courses, aliases, config
    )
    extensions = _detect_extensions(normalized)
    query_type, type_ambiguous = _detect_query_type(normalized, extensions)
    indexes = _route_indexes(query_type, extensions)
    keyword_terms = _build_keyword_terms(normalized, course_matches)

    unresolved = (
        course_ambiguous
        or not course_matches
        or type_ambiguous
        or not indexes
        or not keyword_terms
    )
    if not unresolved:
        return _rule_output(
            normalized,
            query_type=query_type,
            courses=course_matches,
            indexes=indexes,
            keyword_terms=keyword_terms,
            reason="Deterministic course, intent, extension, and index rules resolved the query.",
        )

    return _llm_fallback(
        config,
        normalized,
        courses=courses,
        aliases=aliases,
        context=context,
        chat_model=chat_model,
        reason=_fallback_reason(
            course_ambiguous=course_ambiguous,
            has_courses=bool(course_matches),
            type_ambiguous=type_ambiguous,
            has_indexes=bool(indexes),
            has_terms=bool(keyword_terms),
        ),
    )


def validate_router_output(
    config: Config,
    output: RouterOutput,
) -> RouterOutput:
    """Validate a caller-supplied route against the same invariants as rules."""
    if not isinstance(output, RouterOutput):
        raise RoutingError("router_output must be a RouterOutput instance")
    if output.route_source not in {"rule", "llm", "unsupported"}:
        raise RoutingError("router_output has an invalid route_source")
    if output.query_type not in QUERY_TYPES:
        raise RoutingError(f"Unknown query type: {output.query_type}")
    if not isinstance(output.route_confidence, (int, float)):
        raise RoutingError("route_confidence must be numeric")
    if not 0.0 <= float(output.route_confidence) <= 1.0:
        raise RoutingError("route_confidence must be between 0 and 1")
    if (
        output.route_source != "unsupported"
        and output.route_confidence < config.router_min_confidence
    ):
        raise RoutingError("route_confidence is below router_min_confidence")
    if any(index not in LOGICAL_INDEXES for index in output.candidate_indexes):
        raise RoutingError("router_output contains an unknown logical index")
    courses = _load_courses(config)
    canonical = {course.casefold(): course for course in courses}
    if any(course.casefold() not in canonical for course in output.candidate_courses):
        raise RoutingError("router_output contains a noncanonical course")
    if output.route_source != "unsupported":
        if not output.candidate_courses or not output.candidate_indexes:
            raise RoutingError(
                "supported router_output requires course and index scope"
            )
        if not output.keyword_terms or not output.semantic_queries:
            raise RoutingError(
                "supported router_output requires keyword and semantic queries"
            )
        if not output.needs_keyword_search or not output.needs_semantic_search:
            raise RoutingError(
                "supported router_output must enable keyword and semantic search"
            )
    if len(output.keyword_terms) > MAX_LLM_KEYWORD_TERMS:
        raise RoutingError("router_output contains too many keyword terms")
    if len(output.semantic_queries) > config.semantic_query_limit:
        raise RoutingError("router_output contains too many semantic queries")
    if any(
        not term.strip() for term in (*output.keyword_terms, *output.semantic_queries)
    ):
        raise RoutingError("router_output contains a blank search term")
    for flag in (
        output.needs_keyword_search,
        output.needs_semantic_search,
        output.needs_file_inspection,
        output.needs_python,
    ):
        if type(flag) is not bool:
            raise RoutingError("router_output search flags must be boolean")
    return output


def _rule_output(
    query: str,
    *,
    query_type: str,
    courses: Sequence[str],
    indexes: Sequence[str],
    keyword_terms: Sequence[str],
    reason: str,
) -> RouterOutput:
    return RouterOutput(
        query_type=query_type,
        candidate_courses=tuple(courses),
        candidate_indexes=tuple(indexes),
        keyword_terms=tuple(keyword_terms),
        semantic_queries=(query,),
        needs_keyword_search=True,
        needs_semantic_search=True,
        needs_file_inspection=_needs_file_inspection(query, query_type),
        needs_python=_needs_python(query),
        route_confidence=1.0,
        route_reason=reason,
        route_source="rule",
    )


def _load_courses(config: Config) -> tuple[str, ...]:
    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            rows = connection.execute(
                "SELECT name FROM courses ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise RoutingError(f"Could not load canonical courses: {exc}") from exc
    return tuple(str(row["name"]) for row in rows)


def _valid_aliases(courses: Sequence[str]) -> dict[str, str]:
    canonical = {course.casefold(): course for course in courses}
    return {
        alias: canonical[target.casefold()]
        for alias, target in ALIASES.items()
        if target.casefold() in canonical
    }


def _match_courses(
    query: str,
    courses: Sequence[str],
    aliases: Mapping[str, str],
    config: Config,
) -> tuple[tuple[str, ...], bool]:
    matches: list[str] = []
    seen: set[str] = set()
    for course in courses:
        if _contains_phrase(query, course):
            if course.casefold() not in seen:
                matches.append(course)
                seen.add(course.casefold())
    for alias, course in aliases.items():
        if _contains_phrase(query, alias):
            if course.casefold() not in seen:
                matches.append(course)
                seen.add(course.casefold())
    if matches:
        return tuple(matches), False

    candidates = list(courses) + list(aliases)
    if not candidates:
        return (), True
    extracted = process.extract(
        query,
        candidates,
        scorer=fuzz.WRatio,
        limit=2,
        score_cutoff=config.course_fuzzy_threshold,
    )
    if not extracted:
        return (), True
    best = extracted[0]
    second_score = extracted[1][1] if len(extracted) > 1 else -1.0
    if best[1] - second_score < COURSE_FUZZY_LEAD:
        return (), True
    candidate = str(best[0])
    canonical = {course.casefold(): course for course in courses}
    resolved = canonical.get(candidate.casefold()) or aliases.get(candidate)
    return ((resolved,) if resolved else ()), not bool(resolved)


def _detect_query_type(
    query: str,
    extensions: Sequence[str] = (),
) -> tuple[str, bool]:
    matches = [
        query_type
        for query_type in (
            "cross_course_comparison",
            "study_quiz",
            "portfolio_resume",
            "find_file",
            "assignment_or_project_lookup",
            "course_summary",
            "code_question",
            "data_question",
            "concept_explanation",
        )
        if any(_cue_matches(query, cue) for cue in TYPE_CUES[query_type])
    ]
    if not matches:
        return "unknown_or_unsupported", False
    if "find_file" in matches and extensions:
        return "find_file", False
    if len(matches) > 1:
        return matches[0], True
    return matches[0], False


def _route_indexes(query_type: str, extensions: Sequence[str]) -> tuple[str, ...]:
    explicit = tuple(
        EXTENSION_INDEXES[extension]
        for extension in extensions
        if extension in EXTENSION_INDEXES
    )
    if explicit:
        return tuple(dict.fromkeys(explicit))
    return tuple(TYPE_INDEXES[query_type])


def _detect_extensions(query: str) -> tuple[str, ...]:
    found: list[str] = []
    for token in re.findall(r"(?i)\.[a-z0-9]+\b", query):
        extension = token.casefold()
        if extension not in found:
            found.append(extension)
    return tuple(found)


def _build_keyword_terms(query: str, courses: Sequence[str]) -> tuple[str, ...]:
    quoted = [match.group(1).strip() for match in re.finditer(r'"([^"\n]+)"', query)]
    without_quotes = re.sub(r'"[^"\n]+"', " ", query)
    raw = quoted + re.findall(r"[^\s]+", without_quotes)
    terms: list[str] = []
    seen: set[str] = set()
    for token in raw:
        cleaned = token.strip(".,;:!?()[]{}")
        if not cleaned or cleaned.casefold() in STOPWORDS:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(cleaned)
    if terms:
        return tuple(terms)
    for course in courses:
        if course.casefold() not in seen:
            terms.append(course)
    return tuple(terms)


def _needs_file_inspection(query: str, query_type: str) -> bool:
    return query_type in {
        "find_file",
        "assignment_or_project_lookup",
        "code_question",
        "data_question",
    } or bool(re.search(r"\b(open|read|inspect)\b", query.casefold()))


def _needs_python(query: str) -> bool:
    lowered = query.casefold()
    return any(_cue_matches(lowered, cue) for cue in PYTHON_CUES)


def _contains_phrase(query: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)"
    return re.search(pattern, query.casefold()) is not None


def _cue_matches(query: str, cue: str) -> bool:
    return _contains_phrase(query, cue)


def _validate_conversation_context(
    context: Sequence[Mapping[str, str]] | None,
) -> tuple[dict[str, str], ...]:
    if context is None:
        return ()
    validated: list[dict[str, str]] = []
    for message in context:
        if set(message) != {"role", "content"}:
            raise RoutingError(
                "Conversation messages must contain only role and content"
            )
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise RoutingError(f"Unsupported conversation role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise RoutingError("Conversation message content must be nonblank text")
        validated.append({"role": role, "content": content.strip()})
    return tuple(validated[-MAX_ROUTER_CONTEXT_MESSAGES:])


def _fallback_reason(
    *,
    course_ambiguous: bool,
    has_courses: bool,
    type_ambiguous: bool,
    has_indexes: bool,
    has_terms: bool,
) -> str:
    reasons: list[str] = []
    if course_ambiguous:
        reasons.append("course scope is ambiguous")
    elif not has_courses:
        reasons.append("course scope is unresolved")
    if type_ambiguous:
        reasons.append("query intent has conflicting signals")
    if not has_indexes:
        reasons.append("logical-index scope is unresolved")
    if not has_terms:
        reasons.append("no usable keyword term was derived")
    return "; ".join(reasons) or "deterministic routing was incomplete"


def _llm_fallback(
    config: Config,
    query: str,
    *,
    courses: Sequence[str],
    aliases: Mapping[str, str],
    context: Sequence[Mapping[str, str]],
    chat_model: object | None,
    reason: str,
) -> RouterOutput:
    if config.llm_provider is None or config.llm_model is None:
        return _unsupported_output(f"LLM fallback unavailable: {reason}")
    model = chat_model or build_chat_model(config)
    prompt = _router_prompt(query, courses=courses, aliases=aliases, context=context)
    try:
        response = model.invoke(prompt)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - provider failures are fatal
        raise RoutingError(f"LLM router invocation failed: {exc}") from exc
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, Mapping) else str(item)
            for item in content
        )
    try:
        payload = json.loads(str(content))
        return _parse_llm_output(config, payload, courses=courses, aliases=aliases)
    except (ValueError, TypeError, json.JSONDecodeError, RoutingError) as exc:
        return _unsupported_output(f"Rejected LLM router output: {exc}")


def _unsupported_output(reason: str) -> RouterOutput:
    return RouterOutput(
        query_type="unknown_or_unsupported",
        candidate_courses=(),
        candidate_indexes=(),
        keyword_terms=(),
        semantic_queries=(),
        needs_keyword_search=False,
        needs_semantic_search=False,
        needs_file_inspection=False,
        needs_python=False,
        route_confidence=0.0,
        route_reason=reason,
        route_source="unsupported",
    )


def _parse_llm_output(
    config: Config,
    payload: Any,
    *,
    courses: Sequence[str],
    aliases: Mapping[str, str],
) -> RouterOutput:
    required = {
        "query_type",
        "candidate_courses",
        "candidate_indexes",
        "keyword_terms",
        "semantic_queries",
        "needs_keyword_search",
        "needs_semantic_search",
        "needs_file_inspection",
        "needs_python",
        "route_confidence",
        "route_reason",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RoutingError("LLM output must contain exactly the RouterOutput fields")
    if payload["query_type"] not in QUERY_TYPES:
        raise RoutingError("LLM output contains an unknown query type")
    canonical = {course.casefold(): course for course in courses}
    alias_map = {alias.casefold(): target for alias, target in aliases.items()}
    candidate_courses = _canonicalize_llm_values(
        payload["candidate_courses"], canonical, alias_map, "course"
    )
    candidate_indexes = _validate_string_sequence(
        payload["candidate_indexes"], LOGICAL_INDEXES, "logical index"
    )
    keyword_terms = _validate_string_sequence(
        payload["keyword_terms"], None, "keyword term"
    )
    semantic_queries = _validate_string_sequence(
        payload["semantic_queries"], None, "semantic query"
    )
    if not candidate_courses or not candidate_indexes:
        raise RoutingError("LLM output must provide nonempty course and index scopes")
    if not keyword_terms or not semantic_queries:
        raise RoutingError("LLM output must provide nonempty search terms")
    if len(keyword_terms) > MAX_LLM_KEYWORD_TERMS:
        raise RoutingError("LLM output contains too many keyword terms")
    if len(semantic_queries) > config.semantic_query_limit:
        raise RoutingError("LLM output contains too many semantic queries")
    for field in (
        "needs_keyword_search",
        "needs_semantic_search",
        "needs_file_inspection",
        "needs_python",
    ):
        if type(payload[field]) is not bool:
            raise RoutingError(f"LLM field {field} must be boolean")
    if payload["query_type"] != "unknown_or_unsupported":
        if not payload["needs_keyword_search"] or not payload["needs_semantic_search"]:
            raise RoutingError(
                "Supported LLM routes must enable keyword and semantic search"
            )
    confidence = payload["route_confidence"]
    if type(confidence) not in {int, float} or not 0.0 <= float(confidence) <= 1.0:
        raise RoutingError("LLM route_confidence must be between 0 and 1")
    if float(confidence) < config.router_min_confidence:
        raise RoutingError("LLM route_confidence is below router_min_confidence")
    route_reason = payload["route_reason"]
    if not isinstance(route_reason, str) or not route_reason.strip():
        raise RoutingError("LLM route_reason must be a nonblank string")
    return RouterOutput(
        query_type=payload["query_type"],
        candidate_courses=candidate_courses,
        candidate_indexes=candidate_indexes,
        keyword_terms=keyword_terms,
        semantic_queries=semantic_queries,
        needs_keyword_search=payload["needs_keyword_search"],
        needs_semantic_search=payload["needs_semantic_search"],
        needs_file_inspection=payload["needs_file_inspection"],
        needs_python=payload["needs_python"],
        route_confidence=float(confidence),
        route_reason=route_reason.strip(),
        route_source="llm",
    )


def _validate_string_sequence(
    value: object,
    allowed: Sequence[str] | None,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RoutingError(f"LLM {label} values must be a JSON string array")
    result: list[str] = []
    seen: set[str] = set()
    allowed_set = set(allowed) if allowed is not None else None
    for item in value:
        text = item.strip()
        if not text:
            raise RoutingError(f"LLM {label} values must be nonblank")
        if allowed_set is not None and text not in allowed_set:
            raise RoutingError(f"LLM output contains an unknown {label}: {text}")
        if text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return tuple(result)


def _canonicalize_llm_values(
    value: object,
    canonical: Mapping[str, str],
    aliases: Mapping[str, str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RoutingError(f"LLM {label} values must be a JSON string array")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        key = item.strip().casefold()
        resolved = canonical.get(key) or aliases.get(key)
        if resolved is None:
            raise RoutingError(f"LLM output contains an unknown {label}: {item}")
        if resolved.casefold() not in seen:
            seen.add(resolved.casefold())
            result.append(resolved)
    return tuple(result)


def _router_prompt(
    query: str,
    *,
    courses: Sequence[str],
    aliases: Mapping[str, str],
    context: Sequence[Mapping[str, str]],
) -> str:
    schema = {
        "query_type": list(QUERY_TYPES),
        "candidate_courses": "non-empty canonical course-name array",
        "candidate_indexes": list(LOGICAL_INDEXES),
        "keyword_terms": "non-empty array, maximum 20",
        "semantic_queries": "non-empty array within the configured limit",
        "needs_keyword_search": "boolean",
        "needs_semantic_search": "boolean",
        "needs_file_inspection": "boolean",
        "needs_python": "boolean",
        "route_confidence": "number from 0 through 1",
        "route_reason": "short string",
    }
    return json.dumps(
        {
            "task": "Return only one JSON object matching this routing schema.",
            "query": query,
            "canonical_courses": list(courses),
            "aliases": dict(aliases),
            "logical_indexes": list(LOGICAL_INDEXES),
            "query_types": list(QUERY_TYPES),
            "schema": schema,
            "recent_conversation": list(context),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_chat_model(config: Config) -> object:
    """Construct exactly the configured LangChain provider lazily."""
    if config.llm_provider not in ALLOWED_LLM_PROVIDERS or not config.llm_model:
        raise RoutingError(
            "A supported LLM provider/model pair is required for fallback"
        )
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
        raise RoutingError(
            f"LLM provider '{config.llm_provider}' requires the optional 'llm' extra. "
            "Install it with: uv sync --extra llm"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - constructor/auth setup is fatal
        raise RoutingError(
            f"Could not construct LLM provider '{config.llm_provider}': {exc}"
        ) from exc
