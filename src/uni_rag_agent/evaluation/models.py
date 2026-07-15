"""Typed contracts for the Feature 12 evaluation harness.

The evaluation package deliberately keeps its persisted shape smaller than the
runtime retrieval and answer objects.  Reports contain scores, identifiers and
timings, but never raw evidence text, model output, environment values, or
credentials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from ..retrieval.models import LOGICAL_INDEXES, QUERY_TYPES
from ..retrieval.evidence_persistence import sanitize_error


class EvaluationError(RuntimeError):
    """Raised when evaluation input, state, or report handling fails."""


class EvalSetError(EvaluationError, ValueError):
    """Raised when the committed evaluation set violates its strict schema."""


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalSetError(f"{field} must be a nonblank string")
    return value.strip()


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvalSetError(f"{field} must be an explicit JSON array")
    values = tuple(_string(item, field) for item in value)
    if len(set(values)) != len(values):
        raise EvalSetError(f"{field} must not contain duplicates")
    return values


def _relative_fixture_path(value: str, field: str) -> str:
    normalized = value.replace("\\", "/")
    if (
        PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ":" in normalized.split("/", 1)[0]
    ):
        raise EvalSetError(f"{field} must contain fixture-root-relative paths")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise EvalSetError(f"{field} must contain normalized relative paths")
    return "/".join(parts)


@dataclass(frozen=True)
class EvalItem:
    """One hand-curated evaluation question and deterministic expectations."""

    id: str
    query: str
    query_type: str
    expected_courses: tuple[str, ...]
    expected_files: tuple[str, ...]
    expected_indexes: tuple[str, ...]
    must_include_terms: tuple[str, ...]
    expected_weaknesses: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _string(self.id, "id"))
        object.__setattr__(self, "query", _string(self.query, "query"))
        object.__setattr__(self, "query_type", _string(self.query_type, "query_type"))
        if self.query_type not in QUERY_TYPES:
            raise EvalSetError(f"query_type must be one of: {', '.join(QUERY_TYPES)}")
        for field in (
            "expected_courses",
            "expected_files",
            "expected_indexes",
            "must_include_terms",
            "expected_weaknesses",
        ):
            values = getattr(self, field)
            if isinstance(values, list):
                values = tuple(values)
                object.__setattr__(self, field, values)
            if not isinstance(values, tuple):
                raise EvalSetError(f"{field} must be a tuple after loading")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise EvalSetError(f"{field} must contain nonblank strings")
        if any(index not in LOGICAL_INDEXES for index in self.expected_indexes):
            raise EvalSetError("expected_indexes contains an unknown logical index")
        normalized_files = tuple(
            _relative_fixture_path(value, "expected_files")
            for value in self.expected_files
        )
        object.__setattr__(self, "expected_files", normalized_files)
        if not isinstance(self.notes, str):
            raise EvalSetError("notes must be a JSON string")
        object.__setattr__(self, "notes", self.notes.strip())
        if (
            not self.expected_courses
            and not self.expected_files
            and not self.expected_indexes
            and not self.expected_weaknesses
        ):
            raise EvalSetError(
                "an eval item must specify expected sources or expected_weaknesses"
            )

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "query_type": self.query_type,
            "expected_courses": list(self.expected_courses),
            "expected_files": list(self.expected_files),
            "expected_indexes": list(self.expected_indexes),
            "must_include_terms": list(self.must_include_terms),
            "expected_weaknesses": list(self.expected_weaknesses),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RetrievalScore:
    """Strict deterministic retrieval/evidence score for one eval item."""

    passed: bool
    expected_courses: tuple[str, ...] = ()
    found_courses: tuple[str, ...] = ()
    missing_courses: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    found_files: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    expected_indexes: tuple[str, ...] = ()
    found_indexes: tuple[str, ...] = ()
    missing_indexes: tuple[str, ...] = ()
    expected_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    expected_weaknesses: tuple[str, ...] = ()
    found_weaknesses: tuple[str, ...] = ()
    missing_weaknesses: tuple[str, ...] = ()
    evidence_count: int = 0
    absence_expected: bool = False
    failures: tuple[str, ...] = ()

    def as_safe_dict(self) -> dict[str, object]:
        safe_failures = [sanitize_failure(value) for value in self.failures]
        return {
            "passed": self.passed,
            "expected_courses": list(self.expected_courses),
            "found_courses": list(self.found_courses),
            "missing_courses": list(self.missing_courses),
            "expected_files": list(self.expected_files),
            "found_files": list(self.found_files),
            "missing_files": list(self.missing_files),
            "expected_indexes": list(self.expected_indexes),
            "found_indexes": list(self.found_indexes),
            "missing_indexes": list(self.missing_indexes),
            "expected_terms": list(self.expected_terms),
            "missing_terms": list(self.missing_terms),
            "expected_weaknesses": list(self.expected_weaknesses),
            "found_weaknesses": list(self.found_weaknesses),
            "missing_weaknesses": list(self.missing_weaknesses),
            "evidence_count": self.evidence_count,
            "absence_expected": self.absence_expected,
            "failures": safe_failures,
        }


@dataclass(frozen=True)
class CitationScore:
    """Citation/answer score for one packet and generated answer."""

    passed: bool
    valid: bool
    citation_count: int = 0
    expected_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    expected_weaknesses: tuple[str, ...] = ()
    missing_weaknesses: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def as_safe_dict(self) -> dict[str, object]:
        safe_failures = [sanitize_failure(value) for value in self.failures]
        return {
            "passed": self.passed,
            "valid": self.valid,
            "citation_count": self.citation_count,
            "expected_terms": list(self.expected_terms),
            "missing_terms": list(self.missing_terms),
            "expected_weaknesses": list(self.expected_weaknesses),
            "missing_weaknesses": list(self.missing_weaknesses),
            "failures": safe_failures,
        }


@dataclass(frozen=True)
class EvalResult:
    """Safe per-item result suitable for JSON and Markdown reports."""

    item_id: str
    query: str
    query_type: str
    status: str
    retrieval: RetrievalScore | None = None
    citations: CitationScore | None = None
    search_run_id: int | None = None
    evidence_packet_id: int | None = None
    answer_id: int | None = None
    timings_ms: Mapping[str, float] | None = None
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures

    @property
    def retrieval_score(self) -> RetrievalScore | None:
        """Explicit alias matching the public ``score_retrieval`` name."""

        return self.retrieval

    @property
    def citation_score(self) -> CitationScore | None:
        """Explicit alias matching the public ``score_citations`` name."""

        return self.citations

    def as_safe_dict(self) -> dict[str, object]:
        safe_failures = [sanitize_failure(value) for value in self.failures]
        return {
            "item_id": self.item_id,
            "query_type": self.query_type,
            "status": self.status,
            "passed": self.passed,
            "retrieval": self.retrieval.as_safe_dict() if self.retrieval else None,
            "citations": self.citations.as_safe_dict() if self.citations else None,
            "retrieval_score": self.retrieval.as_safe_dict()
            if self.retrieval
            else None,
            "citation_score": self.citations.as_safe_dict() if self.citations else None,
            "field_results": {
                "query_type": bool(
                    self.retrieval
                    and not any(
                        failure.startswith("query_type mismatch")
                        for failure in self.retrieval.failures
                    )
                ),
                "retrieval": bool(self.retrieval and self.retrieval.passed),
                "citations": bool(self.citations and self.citations.passed),
                "terms": bool(
                    self.retrieval
                    and not self.retrieval.missing_terms
                    and self.citations
                    and not self.citations.missing_terms
                ),
                "weaknesses": bool(
                    self.retrieval
                    and not self.retrieval.missing_weaknesses
                    and self.citations
                    and not self.citations.missing_weaknesses
                ),
            },
            "trace_ids": {
                "search_run_id": self.search_run_id,
                "evidence_packet_id": self.evidence_packet_id,
                "answer_id": self.answer_id,
            },
            "timings_ms": {
                key: round(float(value), 3)
                for key, value in (self.timings_ms or {}).items()
            },
            "failures": safe_failures,
        }


def normalized_text(value: str) -> str:
    """Normalize only whitespace and case for term/weakness checks."""

    return " ".join(value.split()).casefold()


def substring_present(needle: str, haystack: str) -> bool:
    return normalized_text(needle) in normalized_text(haystack)


def sanitize_failure(value: object) -> str:
    """Keep report failures bounded and redact credential-shaped values."""

    sanitized = sanitize_error(value)
    return sanitized[:500] or "evaluation item failed"


def ensure_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvalSetError(f"{field} must be a JSON object")
    return value


__all__ = [
    "CitationScore",
    "EvalItem",
    "EvalResult",
    "EvalSetError",
    "EvaluationError",
    "RetrievalScore",
    "ensure_mapping",
    "normalized_text",
    "sanitize_failure",
    "substring_present",
]
