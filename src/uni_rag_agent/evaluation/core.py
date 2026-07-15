"""Evaluation loading, scoring, fixture preparation, and reporting.

The harness intentionally calls the existing evidence and answer boundaries.
It does not add evaluation tables or a second retrieval/answer implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from ..answering import (
    AnswerResult,
    generate_answer,
    store_answer,
    validate_answer_citations,
)
from ..config import Config, load_config, validate_config
from ..extraction import extract_pending_files, summarize_data_files
from ..indexing import sync_keyword_index, sync_vector_index
from ..inventory import inventory_courses
from ..retrieval import EvidenceBuildResult, EvidencePacket, build_evidence
from ..storage import connect_sqlite_read_only, ensure_data_dirs
from .models import (
    CitationScore,
    EvalItem,
    EvalResult,
    EvalSetError,
    EvaluationError,
    RetrievalScore,
    substring_present,
)
from ..retrieval.evidence_persistence import sanitize_error

EVAL_SET_FIELDS = {
    "id",
    "query",
    "query_type",
    "expected_courses",
    "expected_files",
    "expected_indexes",
    "must_include_terms",
    "expected_weaknesses",
    "notes",
}
MANIFEST_FIELDS = {
    "manifest_version",
    "fixture_digest",
    "source_digest",
    "embedding_model",
    "files",
    "chunks",
    "keyword_rows",
    "vector_rows",
    "files_identity",
    "chunks_identity",
    "keyword_identity",
    "vector_identity",
    "vector_collections",
    "chroma_digest",
    "prepared_at",
}
_SOURCE_TO_INDEX = {
    "document": "document_index",
    "slides": "slides_index",
    "notebook": "notebook_index",
    "code": "code_index",
    "data_schema": "data_schema_index",
    "transcript": "transcript_index",
}


class _EvalItemFailure(RuntimeError):
    """Carry trace ids through the batch failure boundary."""

    def __init__(
        self,
        cause: Exception,
        *,
        search_run_id: int | None = None,
        evidence_packet_id: int | None = None,
        answer_id: int | None = None,
        evidence_ms: float = 0.0,
        answer_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.search_run_id = search_run_id
        self.evidence_packet_id = evidence_packet_id
        self.answer_id = answer_id
        self.evidence_ms = evidence_ms
        self.answer_ms = answer_ms
        self.total_ms = total_ms


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_eval_set_path() -> Path:
    return _repo_root() / "evals" / "fixtures.json"


def fixture_source_root() -> Path:
    return _repo_root() / "evals" / "sources"


def _fixture_eval_root(config: Config) -> Path:
    """Return the evaluation artifact root used for all fixture-state swaps."""

    return (config.runs_dir / "eval").resolve()


def _guard_fixture_path(path: Path, eval_root: Path) -> Path:
    """Reject fixture paths that escape the configured evaluation root."""

    resolved = path.resolve()
    root = eval_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(
            f"Fixture state path must remain below {root}: {resolved}"
        ) from exc
    if relative == Path("."):
        raise EvaluationError("Fixture state path cannot be the evaluation root itself")
    return resolved


def fixture_state_dir(config: Config) -> Path:
    """Return the dedicated generated fixture state root.

    It is deliberately below ``data/runs/eval`` and never reuses the normal
    SQLite or Chroma locations.
    """

    return config.runs_dir / "eval" / "fixture-state"


def fixture_state_config(config: Config, state_root: Path | None = None) -> Config:
    state = state_root or fixture_state_dir(config)
    return replace(
        config,
        courses_root=fixture_source_root(),
        data_dir=state,
        sqlite_path=state / "uni_rag.sqlite",
        chroma_dir=state / "indexes" / "vector",
        runs_dir=state / "runs",
    )


def real_archive_eval_path(config: Config) -> Path:
    return config.runs_dir / "eval" / "real-archive.json"


def load_eval_set(path: Path) -> list[EvalItem]:
    """Load a UTF-8 strict JSON eval list with an exact item schema."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvalSetError(f"evaluation set must be strict UTF-8: {path}") from exc
    except OSError as exc:
        raise EvalSetError(f"could not read evaluation set {path}: {exc}") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise EvalSetError(f"evaluation set is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise EvalSetError("evaluation set root must be an explicit JSON array")
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(payload, start=1):
        if not isinstance(raw_item, Mapping):
            raise EvalSetError(f"evaluation item {index} must be a JSON object")
        fields = set(raw_item)
        if fields != EVAL_SET_FIELDS:
            missing = sorted(EVAL_SET_FIELDS - fields)
            unknown = sorted(fields - EVAL_SET_FIELDS)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise EvalSetError(
                f"evaluation item {index} has invalid fields ({'; '.join(details)})"
            )
        arrays = {
            field: _strict_array(raw_item[field], field, index)
            for field in (
                "expected_courses",
                "expected_files",
                "expected_indexes",
                "must_include_terms",
                "expected_weaknesses",
            )
        }
        item = EvalItem(
            id=_strict_string(raw_item["id"], "id", index),
            query=_strict_string(raw_item["query"], "query", index),
            query_type=_strict_string(raw_item["query_type"], "query_type", index),
            expected_courses=arrays["expected_courses"],
            expected_files=arrays["expected_files"],
            expected_indexes=arrays["expected_indexes"],
            must_include_terms=arrays["must_include_terms"],
            expected_weaknesses=arrays["expected_weaknesses"],
            notes=_strict_notes(raw_item["notes"], index),
        )
        if item.id in seen_ids:
            raise EvalSetError(f"duplicate evaluation item id: {item.id}")
        seen_ids.add(item.id)
        items.append(item)
    if not items:
        raise EvalSetError("evaluation set must contain at least one item")
    return items


def _strict_string(value: object, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalSetError(f"evaluation item {index} field {field} must be nonblank")
    return value.strip()


def _strict_array(value: object, field: str, index: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvalSetError(
            f"evaluation item {index} field {field} must be an explicit JSON array"
        )
    values = tuple(_strict_string(item, field, index) for item in value)
    if len(set(values)) != len(values):
        raise EvalSetError(f"evaluation item {index} field {field} contains duplicates")
    return values


def _strict_notes(value: object, index: int) -> str:
    if not isinstance(value, str):
        raise EvalSetError(f"evaluation item {index} field notes must be a string")
    return value.strip()


def score_retrieval(
    item: EvalItem,
    packet: EvidencePacket,
    courses_root: Path | None = None,
) -> RetrievalScore:
    """Score packet evidence against one eval item's exact source contract."""

    if not isinstance(item, EvalItem):
        raise TypeError("score_retrieval requires an EvalItem")
    if not isinstance(packet, EvidencePacket):
        raise TypeError("score_retrieval requires an EvidencePacket")

    evidence = packet.evidence
    found_courses = tuple(dict.fromkeys(value.course for value in evidence))
    found_indexes = tuple(
        dict.fromkeys(
            _SOURCE_TO_INDEX[item.source_type]
            for item in evidence
            if item.source_type in _SOURCE_TO_INDEX
        )
    )
    found_files = tuple(
        dict.fromkeys(
            expected
            for expected in item.expected_files
            if any(
                _fixture_relative_match(
                    expected,
                    evidence_item.file,
                    courses_root=courses_root,
                )
                for evidence_item in evidence
            )
        )
    )
    evidence_text = "\n".join(value.text for value in evidence)
    missing_courses = tuple(
        expected for expected in item.expected_courses if expected not in found_courses
    )
    missing_indexes = tuple(
        expected for expected in item.expected_indexes if expected not in found_indexes
    )
    missing_files = tuple(
        expected for expected in item.expected_files if expected not in found_files
    )
    missing_terms = tuple(
        term
        for term in item.must_include_terms
        if not substring_present(term, evidence_text)
    )
    found_weaknesses = tuple(
        expected
        for expected in item.expected_weaknesses
        if any(substring_present(expected, weakness) for weakness in packet.weaknesses)
    )
    missing_weaknesses = tuple(
        expected
        for expected in item.expected_weaknesses
        if expected not in found_weaknesses
    )
    absence_expected = not (
        item.expected_courses or item.expected_files or item.expected_indexes
    ) and bool(item.expected_weaknesses)
    failures: list[str] = []
    if packet.interpreted_intent != item.query_type:
        failures.append(
            f"query_type mismatch: expected {item.query_type}, got {packet.interpreted_intent}"
        )
    if missing_courses:
        failures.append("missing expected courses: " + ", ".join(missing_courses))
    if missing_files:
        failures.append("missing expected files: " + ", ".join(missing_files))
    if missing_indexes:
        failures.append("missing expected indexes: " + ", ".join(missing_indexes))
    if missing_terms:
        failures.append("missing terms in evidence: " + ", ".join(missing_terms))
    if missing_weaknesses:
        failures.append("missing expected weaknesses: " + ", ".join(missing_weaknesses))
    if absence_expected and evidence:
        failures.append("absence case requires zero evidence")
    return RetrievalScore(
        passed=not failures,
        expected_courses=item.expected_courses,
        found_courses=found_courses,
        missing_courses=missing_courses,
        expected_files=item.expected_files,
        found_files=found_files,
        missing_files=missing_files,
        expected_indexes=item.expected_indexes,
        found_indexes=found_indexes,
        missing_indexes=missing_indexes,
        expected_terms=item.must_include_terms,
        missing_terms=missing_terms,
        expected_weaknesses=item.expected_weaknesses,
        found_weaknesses=found_weaknesses,
        missing_weaknesses=missing_weaknesses,
        evidence_count=len(evidence),
        absence_expected=absence_expected,
        failures=tuple(failures),
    )


def score_citations(
    packet: EvidencePacket,
    answer: AnswerResult,
    item: EvalItem | None = None,
) -> CitationScore:
    """Validate packet-relative citations and answer limitations.

    ``item`` is optional to preserve the public two-argument interface.  The
    evaluator passes it to enforce the stricter term/weakness requirements.
    """

    if not isinstance(packet, EvidencePacket):
        raise TypeError("score_citations requires an EvidencePacket")
    if not isinstance(answer, AnswerResult):
        raise TypeError("score_citations requires an AnswerResult")
    validation = validate_answer_citations(answer, packet)
    failures: list[str] = list(validation.errors)
    if packet.evidence and not answer.citations:
        failures.append("evidence-backed answer must contain structured citations")
    if not packet.evidence and answer.citations:
        failures.append("absence/empty packet answer must not contain citations")
    missing_packet_weaknesses = tuple(
        weakness
        for weakness in packet.weaknesses
        if not any(
            substring_present(weakness, limitation) for limitation in answer.limitations
        )
    )
    if missing_packet_weaknesses:
        failures.append(
            "answer limitations omit packet weaknesses: "
            + ", ".join(missing_packet_weaknesses)
        )
    expected_terms = item.must_include_terms if item is not None else ()
    missing_terms = tuple(
        term
        for term in expected_terms
        if not substring_present(term, answer.answer_text)
    )
    if missing_terms:
        failures.append("missing terms in answer: " + ", ".join(missing_terms))
    expected_weaknesses = item.expected_weaknesses if item is not None else ()
    missing_weaknesses = tuple(
        expected
        for expected in expected_weaknesses
        if not any(
            substring_present(expected, limitation) for limitation in answer.limitations
        )
    )
    if missing_weaknesses:
        failures.append(
            "answer limitations omit expected weaknesses: "
            + ", ".join(missing_weaknesses)
        )
    return CitationScore(
        passed=not failures,
        valid=validation.valid,
        citation_count=len(answer.citations),
        expected_terms=expected_terms,
        missing_terms=missing_terms,
        expected_weaknesses=expected_weaknesses,
        missing_weaknesses=missing_weaknesses,
        failures=tuple(dict.fromkeys(failures)),
    )


def run_eval_item(item: EvalItem, config: Config) -> EvalResult:
    """Run one item through persisted evidence and answer boundaries."""

    total_start = time.perf_counter()
    if not isinstance(item, EvalItem):
        raise TypeError("run_eval_item requires an EvalItem")
    try:
        validate_config(config)
    except Exception as exc:  # noqa: BLE001 - retain an actual attempt timing
        raise _EvalItemFailure(
            exc,
            total_ms=_elapsed_ms(total_start),
        ) from exc
    evidence_start = total_start
    try:
        result = build_evidence(config, item.query, model=config.embedding_model)
    except Exception as exc:  # noqa: BLE001 - retain trace id at eval boundary
        run_id = getattr(exc, "search_run_id", None)
        raise _EvalItemFailure(
            exc,
            search_run_id=run_id if isinstance(run_id, int) else None,
            evidence_ms=_elapsed_ms(evidence_start),
            total_ms=_elapsed_ms(total_start),
        ) from exc
    evidence_ms = _elapsed_ms(evidence_start)
    answer_start = time.perf_counter()
    try:
        answer = generate_answer(result.packet, config=config)
        answer_ms = _elapsed_ms(answer_start)
        answer_id = store_answer(result.evidence_packet_id, answer, config=config)
    except Exception as exc:  # noqa: BLE001 - retain evidence trace ids in reports
        raise _EvalItemFailure(
            exc,
            search_run_id=result.search_run_id,
            evidence_packet_id=result.evidence_packet_id,
            evidence_ms=evidence_ms,
            answer_ms=_elapsed_ms(answer_start),
            total_ms=_elapsed_ms(total_start),
        ) from exc
    try:
        retrieval_score = score_retrieval(
            item,
            result.packet,
            courses_root=config.courses_root,
        )
        citation_score = score_citations(result.packet, answer, item)
    except Exception as exc:  # noqa: BLE001 - retain trace ids on scoring errors
        raise _EvalItemFailure(
            exc,
            search_run_id=result.search_run_id,
            evidence_packet_id=result.evidence_packet_id,
            answer_id=answer_id,
            evidence_ms=evidence_ms,
            answer_ms=answer_ms,
            total_ms=_elapsed_ms(total_start),
        ) from exc
    total_ms = _elapsed_ms(total_start)
    failures = tuple(
        dict.fromkeys((*retrieval_score.failures, *citation_score.failures))
    )
    return EvalResult(
        item_id=item.id,
        query=item.query,
        query_type=item.query_type,
        status="passed" if not failures else "failed",
        retrieval=retrieval_score,
        citations=citation_score,
        search_run_id=result.search_run_id,
        evidence_packet_id=result.evidence_packet_id,
        answer_id=answer_id,
        timings_ms={
            "evidence_ms": evidence_ms,
            "answer_ms": answer_ms,
            "total_ms": total_ms,
        },
        failures=failures,
    )


def run_eval_set(
    config: Config,
    *,
    fixtures: bool = True,
    smoke_real_archive: bool = False,
) -> tuple[Path, list[EvalResult]]:
    """Run all items, retaining a failure result for every item exception."""

    if fixtures == smoke_real_archive:
        raise EvaluationError(
            "choose exactly one evaluation mode: fixtures or smoke_real_archive"
        )
    if fixtures:
        validate_fixture_state(config)
        run_config = fixture_state_config(config)
        items = load_eval_set(default_eval_set_path())
    else:
        run_config = config
        path = real_archive_eval_path(config)
        if not path.is_file():
            raise EvaluationError(
                f"Real archive eval set is absent: {path}. "
                "Create data/runs/eval/real-archive.json before using "
                "--smoke-real-archive."
            )
        items = load_eval_set(path)
    results: list[EvalResult] = []
    for item in items:
        total_start = time.perf_counter()
        try:
            results.append(run_eval_item(item, run_config))
        except Exception as exc:  # noqa: BLE001 - one bad item must not abort report
            trace = exc if isinstance(exc, _EvalItemFailure) else None
            results.append(
                EvalResult(
                    item_id=item.id,
                    query=item.query,
                    query_type=item.query_type,
                    status="failed",
                    search_run_id=trace.search_run_id if trace else None,
                    evidence_packet_id=trace.evidence_packet_id if trace else None,
                    answer_id=trace.answer_id if trace else None,
                    timings_ms={
                        "evidence_ms": trace.evidence_ms if trace else 0.0,
                        "answer_ms": trace.answer_ms if trace else 0.0,
                        "total_ms": (
                            trace.total_ms
                            if trace and trace.total_ms > 0.0
                            else _elapsed_ms(total_start)
                        ),
                    },
                    failures=(_safe_failure(exc),),
                )
            )
    report_path = write_eval_report(results, output_dir=config.runs_dir / "eval")
    return report_path, results


def write_eval_report(
    results: Sequence[EvalResult],
    output_dir: Path | None = None,
) -> Path:
    """Write paired timestamped JSON and Markdown reports and return JSON path."""

    if output_dir is None:
        output_dir = load_config().runs_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_results = [
        result if isinstance(result, EvalResult) else _coerce_result(result)
        for result in results
    ]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"eval-{now}-{uuid.uuid4().hex[:8]}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    summary = _report_summary(safe_results)
    payload = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "aggregates": summary,
        "results": [result.as_safe_dict() for result in safe_results],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path


def _coerce_result(value: object) -> EvalResult:
    raise TypeError("write_eval_report requires EvalResult values")


def _report_summary(results: Sequence[EvalResult]) -> dict[str, object]:
    total = len(results)
    passed = sum(result.passed for result in results)
    failed = total - passed
    retrieval_scored = [result for result in results if result.retrieval is not None]
    citation_scored = [result for result in results if result.citations is not None]
    return {
        "total_items": total,
        "passed_items": passed,
        "failed_items": failed,
        "pass_rate": (passed / total if total else 0.0),
        "retrieval_pass_rate": (
            sum(
                bool(result.retrieval and result.retrieval.passed)
                for result in retrieval_scored
            )
            / len(retrieval_scored)
            if retrieval_scored
            else 0.0
        ),
        "citation_pass_rate": (
            sum(
                bool(result.citations and result.citations.passed)
                for result in citation_scored
            )
            / len(citation_scored)
            if citation_scored
            else 0.0
        ),
        "timings_ms": {
            key: {
                "count": len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for key, values in _timing_values(results).items()
        },
    }


def _timing_values(results: Sequence[EvalResult]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for result in results:
        for key, value in (result.timings_ms or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))
    return values


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _markdown_report(payload: Mapping[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    lines = [
        "# Uni RAG evaluation report",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Items: {summary.get('total_items', 0)}",
        f"- Passed: {summary.get('passed_items', 0)}",
        f"- Failed: {summary.get('failed_items', 0)}",
        f"- Pass rate: {float(summary.get('pass_rate', 0.0)):.1%}",
        f"- Retrieval pass rate: {float(summary.get('retrieval_pass_rate', 0.0)):.1%}",
        f"- Citation pass rate: {float(summary.get('citation_pass_rate', 0.0)):.1%}",
        "",
        "## Items",
        "",
        "| Item | Query type | Status | Trace IDs | Failures | Timings (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in payload["results"]:  # type: ignore[index]
        assert isinstance(result, Mapping)
        ids = result.get("trace_ids", {})
        timings = result.get("timings_ms", {})
        failures = result.get("failures", [])
        lines.append(
            "| {item} | {query_type} | {status} | {ids} | {failures} | {timings} |".format(
                item=str(result.get("item_id", "")),
                query_type=str(result.get("query_type", "")),
                status=str(result.get("status", "")),
                ids=json.dumps(ids, sort_keys=True),
                failures="; ".join(str(value) for value in failures),
                timings=json.dumps(timings, sort_keys=True),
            )
        )
    lines.extend(
        (
            "",
            "Reports intentionally omit raw evidence text, model output, environment values, and secrets.",
            "",
        )
    )
    return "\n".join(lines)


def prepare_fixture_state(config: Config) -> Mapping[str, object]:
    """Build and atomically activate isolated inventory/index fixture state."""

    if not config.embedding_model:
        raise EvaluationError(
            "Fixture preparation requires a production embedding model. Set "
            "UNI_RAG_EMBEDDING_MODEL to a reviewed profile before running "
            "eval prepare-fixtures."
        )
    source_root = fixture_source_root()
    if not source_root.is_dir():
        raise EvaluationError(f"Committed fixture source root is absent: {source_root}")
    fixture_path = default_eval_set_path()
    items = load_eval_set(fixture_path)
    if not items:
        raise EvaluationError("fixture eval set is empty")

    eval_root = _fixture_eval_root(config)
    eval_root.mkdir(parents=True, exist_ok=True)
    active = _guard_fixture_path(fixture_state_dir(config), eval_root)
    temporary = _guard_fixture_path(
        eval_root / f".fixture-state-{uuid.uuid4().hex}", eval_root
    )
    try:
        state_config = fixture_state_config(config, temporary)
        ensure_data_dirs(state_config)
        inventory_courses(state_config)
        extract_pending_files(state_config)
        # Data-schema files are intentionally handled by Feature 05's
        # dedicated service; invoke it so all six source types are represented.
        summarize_data_files(state_config)
        sync_keyword_index(state_config, rebuild=True)
        sync_vector_index(
            state_config,
            model=config.embedding_model,
            rebuild=True,
        )
        snapshot = _fixture_state_snapshot(state_config)
        manifest = {
            "manifest_version": 1,
            "fixture_digest": _sha256_file(fixture_path),
            "source_digest": _sha256_tree(source_root),
            "embedding_model": config.embedding_model,
            **snapshot,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Validate before touching the active state.  A failed extraction,
        # indexing run, or manifest write therefore leaves the previous state
        # available for ``eval run``.
        validate_fixture_state(config, state_root=temporary)
        _activate_fixture_state(temporary, active, eval_root)
        return manifest
    except Exception:
        _safe_remove_fixture_tree(temporary, eval_root)
        raise


def validate_fixture_state(
    config: Config,
    *,
    state_root: Path | None = None,
) -> Mapping[str, object]:
    """Validate fixture state identity before allowing ``eval run``."""

    eval_root = _fixture_eval_root(config)
    state_path = _guard_fixture_path(state_root or fixture_state_dir(config), eval_root)
    state = fixture_state_config(config, state_path)
    manifest_path = state_path / "manifest.json"
    if (
        not manifest_path.is_file()
        or not state.sqlite_path.is_file()
        or not state.chroma_dir.is_dir()
    ):
        raise EvaluationError(
            "Fixture state is absent. Run `uv run -m uni_rag_agent eval "
            "prepare-fixtures` before `eval run`."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            "Fixture state manifest is unreadable or invalid. Run `uv run -m "
            "uni_rag_agent eval prepare-fixtures` again."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != MANIFEST_FIELDS:
        raise EvaluationError(
            "Fixture state manifest has stale fields. Run `uv run -m "
            "uni_rag_agent eval prepare-fixtures` again."
        )
    if payload["fixture_digest"] != _sha256_file(default_eval_set_path()) or payload[
        "source_digest"
    ] != _sha256_tree(fixture_source_root()):
        raise EvaluationError(
            "Fixture state is stale relative to committed fixtures. Run `uv run -m "
            "uni_rag_agent eval prepare-fixtures` again."
        )
    if payload["embedding_model"] != config.embedding_model:
        raise EvaluationError(
            "Fixture state embedding model does not match configuration. Run `uv run "
            "-m uni_rag_agent eval prepare-fixtures` again."
        )
    if not _valid_manifest_scalars(payload):
        raise EvaluationError(
            "Fixture state manifest has invalid identity or count fields. Run `uv run -m "
            "uni_rag_agent eval prepare-fixtures` again."
        )
    try:
        snapshot = _fixture_state_snapshot(state)
    except Exception as exc:  # noqa: BLE001 - give setup guidance, not raw DB internals
        raise EvaluationError(
            "Fixture state database is not ready. Run `uv run -m uni_rag_agent eval "
            "prepare-fixtures` again."
        ) from exc
    for key, expected in snapshot.items():
        if payload[key] != expected:
            raise EvaluationError(
                "Fixture state manifest identity is stale. Run `uv run -m "
                "uni_rag_agent eval prepare-fixtures` again."
            )
    if snapshot["files"] <= 0 or snapshot["chunks"] <= 0:
        raise EvaluationError(
            "Fixture state has no indexed content. Run `uv run -m uni_rag_agent eval "
            "prepare-fixtures` again."
        )
    return payload


def _fixture_state_snapshot(state: Config) -> dict[str, object]:
    """Return counts plus deterministic identities for a prepared state."""

    if not state.sqlite_path.is_file() or not state.chroma_dir.is_dir():
        raise EvaluationError("Fixture state storage is incomplete")
    try:
        with closing(connect_sqlite_read_only(state)) as connection:
            files = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id, course_id, path, relative_path, filename, extension,
                           size_bytes, modified_at, content_hash, category,
                           index_status, reason_not_indexed
                    FROM files ORDER BY id
                    """
                ).fetchall()
            ]
            chunks = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id, file_id, extracted_document_id, chunk_uid, source_type,
                           chunk_index, title, text, token_count, location_type,
                           location_value, metadata_json
                    FROM chunks ORDER BY id
                    """
                ).fetchall()
            ]
            keyword_rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT rowid, chunk_id, text, title, course_name, file_path,
                           source_type
                    FROM chunk_fts ORDER BY rowid
                    """
                ).fetchall()
            ]
            embeddings = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT chunk_id, vector_backend, vector_collection, vector_id,
                           embedding_model, embedding_dim
                    FROM embeddings
                    ORDER BY vector_backend, vector_collection, vector_id, chunk_id
                    """
                ).fetchall()
            ]
            vector_collections = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT vector_backend, vector_collection, embedding_model,
                           embedding_dim, COUNT(*) AS row_count
                    FROM embeddings
                    GROUP BY vector_backend, vector_collection,
                             embedding_model, embedding_dim
                    ORDER BY vector_backend, vector_collection,
                             embedding_model, embedding_dim
                    """
                ).fetchall()
            ]
            counts = {
                "files": len(files),
                "chunks": len(chunks),
                "keyword_rows": len(keyword_rows),
                "vector_rows": len(embeddings),
            }
    except Exception as exc:  # noqa: BLE001 - expose setup guidance only
        raise EvaluationError("Fixture state database is not ready") from exc
    return {
        **counts,
        "files_identity": _digest_records(files),
        "chunks_identity": _digest_records(chunks),
        "keyword_identity": _digest_records(keyword_rows),
        "vector_identity": _digest_records(embeddings),
        "vector_collections": vector_collections,
        # The SQLite mapping records the profile; hashing the persisted Chroma
        # tree additionally catches same-count vector/collection drift.
        "chroma_digest": _sha256_tree(state.chroma_dir),
    }


def _digest_records(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_manifest_scalars(payload: Mapping[str, object]) -> bool:
    if payload.get("manifest_version") != 1 or isinstance(
        payload.get("manifest_version"), bool
    ):
        return False
    for key in ("fixture_digest", "source_digest", "chroma_digest"):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef" for character in value.casefold()
            )
        ):
            return False
    if (
        not isinstance(payload.get("embedding_model"), str)
        or not str(payload["embedding_model"]).strip()
    ):
        return False
    for key in ("files", "chunks", "keyword_rows", "vector_rows"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False
    for key in (
        "files_identity",
        "chunks_identity",
        "keyword_identity",
        "vector_identity",
    ):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef" for character in value.casefold()
            )
        ):
            return False
    vector_collections = payload.get("vector_collections")
    if not isinstance(vector_collections, list) or not vector_collections:
        return False
    for profile in vector_collections:
        if not isinstance(profile, Mapping) or set(profile) != {
            "vector_backend",
            "vector_collection",
            "embedding_model",
            "embedding_dim",
            "row_count",
        }:
            return False
        for key in ("vector_backend", "vector_collection", "embedding_model"):
            value = profile.get(key)
            if not isinstance(value, str) or not value.strip():
                return False
        for key in ("embedding_dim", "row_count"):
            value = profile.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                return False
    prepared_at = payload.get("prepared_at")
    if not isinstance(prepared_at, str) or not prepared_at.strip():
        return False
    try:
        timestamp = datetime.fromisoformat(prepared_at)
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _activate_fixture_state(temporary: Path, active: Path, eval_root: Path) -> None:
    """Swap a fully validated temporary state into place with restoration."""

    temporary = _guard_fixture_path(temporary, eval_root)
    active = _guard_fixture_path(active, eval_root)
    backup = _guard_fixture_path(
        eval_root / f".fixture-state-backup-{uuid.uuid4().hex}", eval_root
    )
    if active.exists() and not active.is_dir():
        raise EvaluationError(f"Active fixture state is not a directory: {active}")
    had_active = active.exists()
    moved_active = False
    try:
        if active.exists():
            shutil.move(str(active), str(backup))
            moved_active = True
        shutil.move(str(temporary), str(active))
    except Exception:
        if active.exists() and (moved_active or not had_active):
            _safe_remove_fixture_tree(active, eval_root)
        if moved_active and backup.exists():
            shutil.move(str(backup), str(active))
        raise
    else:
        if backup.exists():
            _safe_remove_fixture_tree(backup, eval_root)


def _safe_remove_fixture_tree(path: Path, eval_root: Path) -> None:
    """Remove only a generated fixture sibling below the eval root."""

    target = _guard_fixture_path(path, eval_root)
    if target.exists():
        if not target.is_dir():
            raise EvaluationError(
                f"Refusing to remove non-directory fixture path: {target}"
            )
        shutil.rmtree(target)


def _fixture_relative_match(
    expected: str,
    actual: str,
    *,
    courses_root: Path | None = None,
) -> bool:
    """Compare an evidence file to an exact root-relative fixture path.

    Absolute evidence paths are accepted only when the caller supplies the
    authoritative ``courses_root``.  The old suffix-only comparison could
    incorrectly match an identically named file in a different course.
    """

    actual_text = str(actual)
    actual_path = Path(actual_text)
    windows_absolute = PureWindowsPath(actual_text).is_absolute()
    if actual_path.is_absolute() or windows_absolute:
        if courses_root is None:
            return False
        root_text = str(courses_root).replace("\\", "/").rstrip("/")
        actual_normalized = actual_text.replace("\\", "/")
        if (
            PureWindowsPath(actual_text).is_absolute()
            and PureWindowsPath(root_text).is_absolute()
        ):
            return actual_normalized == f"{root_text}/{expected}"
        try:
            normalized = actual_path.resolve().relative_to(courses_root.resolve())
        except (OSError, ValueError):
            return False
        return normalized.as_posix() == expected
    normalized = str(actual).replace("\\", "/")
    return normalized == expected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationError(f"could not hash fixture file {path}") from exc
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _safe_failure(exc: Exception) -> str:
    if isinstance(exc, _EvalItemFailure):
        exc = exc.cause
    message = sanitize_error(exc)
    return f"{type(exc).__name__}: {message}"


__all__ = [
    "default_eval_set_path",
    "fixture_source_root",
    "fixture_state_config",
    "fixture_state_dir",
    "load_eval_set",
    "prepare_fixture_state",
    "real_archive_eval_path",
    "run_eval_item",
    "run_eval_set",
    "score_citations",
    "score_retrieval",
    "validate_fixture_state",
    "write_eval_report",
]
