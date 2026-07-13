"""Persisted retrieval execution, evidence selection, and coverage reporting."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass

from uni_rag_agent.config import Config
from uni_rag_agent.storage import (
    StorageError,
    connect_sqlite,
    connect_sqlite_read_only,
)

from .core import _execute_retrieval, _RetrievalExecution
from .evidence_models import (
    ANSWER_CONSTRAINTS,
    EVIDENCE_SOURCE_TYPES,
    EvidenceBuildResult,
    EvidenceItem,
    EvidenceLocation,
    EvidenceModelError,
    EvidencePacket,
    SearchCoverage,
    canonical_json,
    location_label,
)
from .evidence_persistence import (
    _SearchRunRecorder,
    clear_selection_flags,
    finalize_run,
    insert_packet,
    sanitize_error,
)
from .metadata import MISSING_INVENTORY_REASON
from .models import FusedRetrievalResult, QueryPlan, RetrievalResultSet, RetrievalRun

INDEX_TO_SOURCE_TYPE = {
    "document_index": "document",
    "slides_index": "slides",
    "notebook_index": "notebook",
    "code_index": "code",
    "data_schema_index": "data_schema",
    "transcript_index": "transcript",
}


class EvidenceError(RuntimeError):
    """Raised when a packet cannot be assembled or loaded safely."""


@dataclass(frozen=True)
class _HydratedCandidate:
    fused: FusedRetrievalResult
    item: EvidenceItem


def _has_content_contribution(candidate: _HydratedCandidate) -> bool:
    """Return whether a fused candidate came from chunk-search evidence."""
    return any(
        contribution.retrieval_method in {"keyword", "semantic"}
        for contribution in candidate.fused.contributions
    )


def build_evidence(
    config: Config,
    query: str,
    conversation_context: Sequence[dict[str, str]] | None = None,
    model: str | None = None,
    *,
    chat_model: object | None = None,
) -> EvidenceBuildResult:
    """Run the mandatory planner/retriever and persist one evidence packet."""
    recorder = _SearchRunRecorder(config)
    execution = _execute_retrieval(
        config,
        query,
        conversation_context=conversation_context,
        model=model,
        chat_model=chat_model,
        recorder=recorder,
    )
    try:
        return _assemble_packet(config, recorder, execution)
    except (EvidenceError, StorageError) as exc:
        _mark_packet_build_failed(config, recorder.search_run_id, exc)
        raise
    except sqlite3.Error as exc:
        error = EvidenceError(f"Evidence packet assembly failed: {exc}")
        _mark_packet_build_failed(config, recorder.search_run_id, error)
        raise error from exc
    except Exception as exc:  # noqa: BLE001 - packet boundary must be explicit
        error = EvidenceError(f"Evidence packet assembly failed: {exc}")
        _mark_packet_build_failed(config, recorder.search_run_id, error)
        raise error from exc


def _assemble_packet(
    config: Config,
    recorder: _SearchRunRecorder,
    execution: _RetrievalExecution,
) -> EvidenceBuildResult:
    if recorder.search_run_id <= 0:
        raise EvidenceError("Cannot assemble an evidence packet without a search run")
    if recorder.query_plan is None or recorder.retrieval_settings is None:
        raise EvidenceError("Persisted search run is missing plan/settings state")

    run = execution.run
    status = "unsupported" if run.status == "unsupported" else "completed"
    try:
        with closing(connect_sqlite(config)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            hydrated, file_only_count = _hydrate_candidates(
                connection,
                candidates=execution.all_fused_candidates,
            )
            selected, omission_counts = _select_evidence(
                config,
                candidates=hydrated,
            )
            weaknesses = _build_weaknesses(
                run,
                recorder.query_plan,
                hydrated,
                selected,
                omission_counts=omission_counts,
                evidence_max_tokens=config.evidence_max_tokens,
            )
            coverage = _build_coverage(
                run=run,
                query_plan=recorder.query_plan,
                fused_candidates=execution.all_fused_candidates,
                hydrated=hydrated,
                selected=selected,
                file_only_count=file_only_count,
                omission_counts=omission_counts,
                weaknesses=weaknesses,
                status=status,
                search_run_id=recorder.search_run_id,
            )
            packet = EvidencePacket(
                search_run_id=recorder.search_run_id,
                query=run.query,
                interpreted_intent=recorder.query_plan.query_type,
                query_plan=recorder.query_plan,
                retrieval_settings=recorder.retrieval_settings,
                searched={
                    "courses": run.searched_courses,
                    "indexes": run.searched_indexes,
                    "keyword_terms": run.keyword_terms,
                    "semantic_queries": run.semantic_queries,
                },
                coverage=coverage,
                evidence=tuple(item.item for item in selected),
                weaknesses=tuple(weaknesses),
                answer_constraints=ANSWER_CONSTRAINTS,
            )
            _mark_selected_fused_rows(
                connection,
                search_run_id=recorder.search_run_id,
                selected=selected,
            )
            packet_id = insert_packet(
                connection,
                search_run_id=recorder.search_run_id,
                packet_json=canonical_json(packet),
                evidence_count=len(packet.evidence),
            )
            finalize_run(
                connection,
                search_run_id=recorder.search_run_id,
                status=status,
                weaknesses=packet.weaknesses,
            )
            connection.commit()
    except EvidenceError:
        raise
    except sqlite3.IntegrityError as exc:
        raise EvidenceError(f"Evidence packet persistence failed: {exc}") from exc
    except sqlite3.Error as exc:
        raise EvidenceError(f"Evidence packet storage failed: {exc}") from exc

    return EvidenceBuildResult(
        search_run_id=recorder.search_run_id,
        evidence_packet_id=packet_id,
        retrieval_run=run,
        coverage=coverage,
        packet=packet,
    )


def _hydrate_candidates(
    connection: sqlite3.Connection,
    *,
    candidates: Sequence[FusedRetrievalResult],
) -> tuple[tuple[_HydratedCandidate, ...], int]:
    hydrated: list[_HydratedCandidate] = []
    file_only_count = 0
    for candidate in candidates:
        if candidate.chunk_id is None:
            file_only_count += 1
            continue
        item = _hydrate_candidate(connection, candidate)
        hydrated.append(_HydratedCandidate(fused=candidate, item=item))
    return tuple(hydrated), file_only_count


def _hydrate_candidate(
    connection: sqlite3.Connection,
    candidate: FusedRetrievalResult,
) -> EvidenceItem:
    row = connection.execute(
        """
        SELECT
            chunks.id AS chunk_id,
            chunks.file_id AS chunk_file_id,
            chunks.source_type AS source_type,
            chunks.text AS text,
            chunks.token_count AS token_count,
            chunks.location_type AS location_type,
            chunks.location_value AS location_value,
            files.id AS file_id,
            files.path AS file_path,
            files.index_status AS index_status,
            files.reason_not_indexed AS reason_not_indexed,
            courses.name AS course
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE chunks.id = ?
          AND chunks.file_id = ?
          AND files.id = ?
        """,
        (candidate.chunk_id, candidate.file_id, candidate.file_id),
    ).fetchone()
    if row is None:
        raise EvidenceError(
            "Authoritative retrieval drift: chunk/file identity no longer exists "
            f"for chunk_id={candidate.chunk_id}, file_id={candidate.file_id}."
        )
    if row["index_status"] != "indexed":
        raise EvidenceError(
            "Authoritative retrieval drift: required file is not currently indexed "
            f"(file_id={candidate.file_id}, status={row['index_status']})."
        )
    if row["reason_not_indexed"] == MISSING_INVENTORY_REASON:
        raise EvidenceError(
            "Authoritative retrieval drift: required file is missing from the latest inventory "
            f"(file_id={candidate.file_id})."
        )
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise EvidenceError(
            f"Authoritative retrieval drift: chunk_id={candidate.chunk_id} has blank text."
        )
    source_type = row["source_type"]
    if source_type not in EVIDENCE_SOURCE_TYPES:
        raise EvidenceError(
            f"Unsupported authoritative evidence source type: {source_type!r}."
        )
    course = row["course"]
    file_path = str(row["file_path"])
    location_type = _optional_row_text(row["location_type"])
    location_value = _optional_row_text(row["location_value"])
    _require_candidate_identity(
        candidate,
        course=str(course) if course is not None else None,
        file_path=file_path,
        source_type=str(source_type),
        location_type=location_type,
        location_value=location_value,
    )
    stored_tokens = row["token_count"]
    token_count = (
        int(stored_tokens)
        if isinstance(stored_tokens, int)
        and not isinstance(stored_tokens, bool)
        and stored_tokens > 0
        else len(text.split())
    )
    if token_count <= 0:
        raise EvidenceError(
            f"Authoritative retrieval drift: chunk_id={candidate.chunk_id} has no usable tokens."
        )
    if course is None or not str(course).strip():
        raise EvidenceError(
            f"Authoritative retrieval drift: file_id={candidate.file_id} has no course."
        )
    return EvidenceItem(
        course=str(course),
        file_id=int(row["file_id"]),
        chunk_id=int(row["chunk_id"]),
        file=file_path,
        source_type=str(source_type),
        location=EvidenceLocation(
            type=location_type,
            value=location_value,
            label=location_label(location_type, location_value),
        ),
        text=text,
        token_count=token_count,
        rank=candidate.rank,
        score=candidate.score,
        retrieval_method="hybrid",
        contributions=candidate.contributions,
    )


def _require_candidate_identity(
    candidate: FusedRetrievalResult,
    *,
    course: str | None,
    file_path: str,
    source_type: str,
    location_type: str | None,
    location_value: str | None,
) -> None:
    checks = (
        (candidate.course, course, "course"),
        (candidate.file_path, file_path, "file path"),
        (candidate.source_type, source_type, "source type"),
        (candidate.location_type, location_type, "location type"),
        (candidate.location_value, location_value, "location value"),
    )
    for candidate_value, actual_value, label in checks:
        if candidate_value is not None and candidate_value != actual_value:
            raise EvidenceError(
                f"Authoritative retrieval drift: {label} changed for chunk_id={candidate.chunk_id}."
            )


def _select_evidence(
    config: Config,
    *,
    candidates: Sequence[_HydratedCandidate],
) -> tuple[tuple[_HydratedCandidate, ...], dict[str, int]]:
    selected: list[_HydratedCandidate] = []
    total_tokens = 0
    token_budget_omissions = 0
    oversized_omissions = 0
    for candidate in candidates:
        if len(selected) >= config.final_top_k:
            break
        token_count = candidate.item.token_count
        if token_count > config.evidence_max_tokens:
            oversized_omissions += 1
            continue
        if total_tokens + token_count > config.evidence_max_tokens:
            token_budget_omissions += 1
            continue
        selected.append(candidate)
        total_tokens += token_count
    return tuple(selected), {
        "token_budget_omission_count": token_budget_omissions,
        "oversized_evidence_omission_count": oversized_omissions,
        "evidence_token_count": total_tokens,
    }


def _mark_selected_fused_rows(
    connection: sqlite3.Connection,
    *,
    search_run_id: int,
    selected: Sequence[_HydratedCandidate],
) -> None:
    for candidate in selected:
        cursor = connection.execute(
            """
            UPDATE search_results
            SET selected_for_evidence = 1
            WHERE search_run_id = ?
              AND retrieval_method = 'hybrid'
              AND rank = ?
              AND chunk_id = ?
              AND file_id = ?
            """,
            (
                search_run_id,
                candidate.fused.rank,
                candidate.fused.chunk_id,
                candidate.fused.file_id,
            ),
        )
        if cursor.rowcount != 1:
            raise EvidenceError(
                "Could not mark the selected fused result for evidence "
                f"(rank={candidate.fused.rank}, chunk_id={candidate.fused.chunk_id})."
            )


def _build_coverage(
    *,
    run: RetrievalRun,
    query_plan: QueryPlan,
    fused_candidates: Sequence[FusedRetrievalResult],
    hydrated: Sequence[_HydratedCandidate],
    selected: Sequence[_HydratedCandidate],
    file_only_count: int,
    omission_counts: Mapping[str, int],
    weaknesses: Sequence[str],
    status: str,
    search_run_id: int,
) -> SearchCoverage:
    retrieval_run = run
    result_sets = retrieval_run.result_sets
    raw_counts = {
        "metadata": sum(
            len(result_set.results)
            for result_set in result_sets
            if result_set.retrieval_method == "metadata"
        ),
        "keyword": sum(
            len(result_set.results)
            for result_set in result_sets
            if result_set.retrieval_method == "keyword"
        ),
        "semantic": sum(
            len(result_set.results)
            for result_set in result_sets
            if result_set.retrieval_method == "semantic"
        ),
    }
    content_hydrated = tuple(
        candidate for candidate in hydrated if _has_content_contribution(candidate)
    )
    hit_courses = {candidate.item.course for candidate in content_hydrated}
    hit_source_types = {candidate.item.source_type for candidate in content_hydrated}
    planned_source_types = tuple(
        dict.fromkeys(
            INDEX_TO_SOURCE_TYPE[index]
            for index in query_plan.candidate_indexes
            if index in INDEX_TO_SOURCE_TYPE
        )
    )
    hit_indexes = {
        index
        for index in query_plan.candidate_indexes
        if INDEX_TO_SOURCE_TYPE.get(index) in hit_source_types
    }
    courses_with = tuple(
        course for course in query_plan.candidate_courses if course in hit_courses
    )
    indexes_with = tuple(
        index for index in query_plan.candidate_indexes if index in hit_indexes
    )
    source_types_with = tuple(
        source_type
        for source_type in planned_source_types
        if source_type in hit_source_types
    )
    courses_without = tuple(
        course for course in query_plan.candidate_courses if course not in hit_courses
    )
    indexes_without = tuple(
        index for index in query_plan.candidate_indexes if index not in hit_indexes
    )
    semantic_without = tuple(
        result_set.query
        for result_set in result_sets
        if result_set.retrieval_method == "semantic" and not result_set.results
    )
    return SearchCoverage(
        search_run_id=search_run_id,
        status=status,
        searched_courses=tuple(retrieval_run.searched_courses),
        searched_indexes=tuple(retrieval_run.searched_indexes),
        keyword_terms=tuple(retrieval_run.keyword_terms),
        semantic_queries=tuple(retrieval_run.semantic_queries),
        raw_result_count=sum(raw_counts.values()),
        raw_result_counts_by_method=raw_counts,
        fused_candidate_count=len(fused_candidates),
        selectable_candidate_count=len(hydrated),
        evidence_count=len(selected),
        evidence_token_count=omission_counts["evidence_token_count"],
        courses_with_chunk_hits=courses_with,
        indexes_with_chunk_hits=indexes_with,
        source_types_with_chunk_hits=source_types_with,
        courses_without_chunk_hits=courses_without,
        indexes_without_chunk_hits=indexes_without,
        semantic_queries_without_hits=semantic_without,
        missing_capabilities=_missing_capabilities(query_plan),
        file_only_candidate_count=file_only_count,
        token_budget_omission_count=omission_counts["token_budget_omission_count"],
        oversized_evidence_omission_count=omission_counts[
            "oversized_evidence_omission_count"
        ],
        unselected_selectable_candidate_count=max(0, len(hydrated) - len(selected)),
        weaknesses=tuple(weaknesses),
    )


def _build_weaknesses(
    run: RetrievalRun,
    query_plan: QueryPlan,
    hydrated: Sequence[_HydratedCandidate],
    selected: Sequence[_HydratedCandidate],
    *,
    omission_counts: Mapping[str, int],
    evidence_max_tokens: int,
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    for weakness in run.weaknesses:
        _append_unique(weaknesses, weakness)
    if query_plan.query_type == "unknown_or_unsupported":
        _append_unique(weaknesses, query_plan.plan_reason)
    if query_plan.needs_file_inspection:
        _append_unique(
            weaknesses,
            "The query plan requested source-file inspection, but Feature 09 used stored chunks only.",
        )
    if query_plan.needs_python:
        _append_unique(
            weaknesses,
            "The query plan requested Python execution, but Feature 09 does not execute course code.",
        )
    content_hydrated = tuple(
        candidate for candidate in hydrated if _has_content_contribution(candidate)
    )
    hit_courses = {candidate.item.course for candidate in content_hydrated}
    for course in query_plan.candidate_courses:
        if course not in hit_courses:
            _append_unique(
                weaknesses,
                f"No chunk-backed evidence was found in planned course: {course}.",
            )
    hit_indexes = {
        index
        for index in query_plan.candidate_indexes
        if INDEX_TO_SOURCE_TYPE.get(index)
        in {candidate.item.source_type for candidate in content_hydrated}
    }
    for index in query_plan.candidate_indexes:
        if index not in hit_indexes:
            _append_unique(
                weaknesses,
                f"No chunk-backed evidence was found in planned index: {index}.",
            )
    for result_set in run.result_sets:
        if result_set.retrieval_method == "semantic" and not result_set.results:
            _append_unique(
                weaknesses,
                f"Semantic query returned no hits: {result_set.query}",
            )
    for limitation in _ineligible_file_weaknesses(run.result_sets):
        _append_unique(weaknesses, limitation)
    if omission_counts["token_budget_omission_count"]:
        _append_unique(
            weaknesses,
            "Evidence omitted "
            f"{omission_counts['token_budget_omission_count']} candidate(s) because "
            f"the whole chunk would exceed the {evidence_max_tokens}-token budget.",
        )
    if omission_counts["oversized_evidence_omission_count"]:
        _append_unique(
            weaknesses,
            "Evidence omitted "
            f"{omission_counts['oversized_evidence_omission_count']} oversized candidate(s) "
            f"larger than the {evidence_max_tokens}-token budget.",
        )
    if not selected:
        _append_unique(
            weaknesses,
            "No evidence was selected; the indexed material is insufficient to answer safely.",
        )
    return tuple(weaknesses)


def _ineligible_file_weaknesses(
    result_sets: Sequence[RetrievalResultSet],
) -> tuple[str, ...]:
    grouped: list[tuple[str, str]] = []
    for result_set in result_sets:
        if result_set.retrieval_method != "metadata":
            continue
        for result in result_set.results:
            if result.file_index_status == "indexed" and result.file_category not in {
                "image_metadata_only",
                "media_metadata_only",
                "archive_metadata_only",
                "installer_metadata_only",
                "model_metadata_only",
            }:
                continue
            category = result.file_category or "unknown_metadata_only"
            reason = (
                result.reason_not_indexed or result.file_index_status or "not indexed"
            )
            pair = (category, " ".join(str(reason).split())[:160])
            if pair not in grouped:
                grouped.append(pair)
    return tuple(
        f"Matched ineligible files: {category} ({reason})."
        for category, reason in grouped
    )


def _missing_capabilities(query_plan: QueryPlan) -> tuple[str, ...]:
    missing: list[str] = []
    if query_plan.needs_file_inspection:
        missing.append("file_inspection")
    if query_plan.needs_python:
        missing.append("python_execution")
    return tuple(missing)


def load_evidence_packet(
    config: Config,
    evidence_packet_id: int | None = None,
    *,
    search_run_id: int | None = None,
) -> EvidencePacket:
    """Load and strictly validate one immutable stored packet."""
    if (evidence_packet_id is None) == (search_run_id is None):
        raise EvidenceError(
            "Specify exactly one positive evidence_packet_id or search_run_id."
        )
    identifier = evidence_packet_id if evidence_packet_id is not None else search_run_id
    if (
        not isinstance(identifier, int)
        or isinstance(identifier, bool)
        or identifier <= 0
    ):
        raise EvidenceError("Evidence packet identifiers must be positive integers.")
    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            if evidence_packet_id is not None:
                row = connection.execute(
                    """
                    SELECT evidence_packets.id AS packet_id,
                           evidence_packets.search_run_id,
                           evidence_packets.packet_json,
                           evidence_packets.evidence_count,
                           search_runs.status AS run_status
                    FROM evidence_packets
                    JOIN search_runs ON search_runs.id = evidence_packets.search_run_id
                    WHERE evidence_packets.id = ?
                    """,
                    (identifier,),
                ).fetchone()
                if row is None:
                    raise EvidenceError(f"Evidence packet {identifier} does not exist.")
            else:
                row = connection.execute(
                    """
                    SELECT evidence_packets.id AS packet_id,
                           evidence_packets.search_run_id,
                           evidence_packets.packet_json,
                           evidence_packets.evidence_count,
                           search_runs.status AS run_status
                    FROM evidence_packets
                    JOIN search_runs ON search_runs.id = evidence_packets.search_run_id
                    WHERE evidence_packets.search_run_id = ?
                    """,
                    (identifier,),
                ).fetchone()
                if row is None:
                    _raise_missing_packet_for_run(connection, identifier)
            try:
                packet = EvidencePacket.from_dict(json.loads(row["packet_json"]))
            except (json.JSONDecodeError, TypeError, EvidenceModelError) as exc:
                raise EvidenceError(
                    f"Stored evidence packet {row['packet_id']} is invalid: {exc}"
                ) from exc
            if packet.search_run_id != int(row["search_run_id"]):
                raise EvidenceError(
                    f"Stored evidence packet {row['packet_id']} is attached to the wrong search run."
                )
            run_status = str(row["run_status"])
            if run_status not in {"completed", "unsupported"}:
                raise EvidenceError(
                    f"Stored evidence packet {row['packet_id']} is attached to a run "
                    f"with invalid status: {run_status}."
                )
            if packet.coverage.status != run_status:
                raise EvidenceError(
                    f"Stored evidence packet {row['packet_id']} status "
                    f"{packet.coverage.status!r} does not match owning run status "
                    f"{run_status!r}."
                )
            if int(row["evidence_count"]) != len(packet.evidence):
                raise EvidenceError(
                    f"Stored evidence packet {row['packet_id']} has an inconsistent evidence count."
                )
            return packet
    except EvidenceError:
        raise
    except sqlite3.Error as exc:
        raise StorageError(f"Evidence packet could not be loaded: {exc}") from exc


def _raise_missing_packet_for_run(
    connection: sqlite3.Connection,
    search_run_id: int,
) -> None:
    row = connection.execute(
        "SELECT status FROM search_runs WHERE id = ?", (search_run_id,)
    ).fetchone()
    if row is None:
        raise EvidenceError(f"Search run {search_run_id} does not exist.")
    status = str(row["status"])
    if status == "running":
        raise EvidenceError(
            f"Search run {search_run_id} is still running without a packet."
        )
    if status == "failed":
        raise EvidenceError(f"Search run {search_run_id} failed and has no packet.")
    raise EvidenceError(
        f"Search run {search_run_id} completed with status {status} but has no packet."
    )


def explain_search_coverage(config: Config, search_run_id: int) -> SearchCoverage:
    """Return exact packet coverage or partial persisted coverage for a run."""
    if (
        not isinstance(search_run_id, int)
        or isinstance(search_run_id, bool)
        or search_run_id <= 0
    ):
        raise EvidenceError("search_run_id must be a positive integer")
    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            packet_row = connection.execute(
                """
                SELECT evidence_packets.packet_json,
                       search_runs.status AS run_status
                FROM evidence_packets
                JOIN search_runs ON search_runs.id = evidence_packets.search_run_id
                WHERE evidence_packets.search_run_id = ?
                """,
                (search_run_id,),
            ).fetchone()
            if packet_row is not None:
                try:
                    packet = EvidencePacket.from_dict(
                        json.loads(packet_row["packet_json"])
                    )
                    if packet.search_run_id != search_run_id:
                        raise EvidenceError(
                            f"Stored evidence packet for search run {search_run_id} is attached to the wrong run."
                        )
                    run_status = str(packet_row["run_status"])
                    if run_status not in {"completed", "unsupported"}:
                        raise EvidenceError(
                            f"Stored evidence packet for search run {search_run_id} "
                            f"is attached to a run with invalid status: {run_status}."
                        )
                    if packet.coverage.status != run_status:
                        raise EvidenceError(
                            f"Stored evidence packet for search run {search_run_id} status "
                            f"{packet.coverage.status!r} does not match owning run status "
                            f"{run_status!r}."
                        )
                    return packet.coverage
                except (json.JSONDecodeError, TypeError, EvidenceModelError) as exc:
                    raise EvidenceError(
                        f"Stored evidence packet for search run {search_run_id} is invalid: {exc}"
                    ) from exc
            run_row = connection.execute(
                "SELECT * FROM search_runs WHERE id = ?", (search_run_id,)
            ).fetchone()
            if run_row is None:
                raise EvidenceError(f"Search run {search_run_id} does not exist.")
            return _coverage_from_partial_rows(connection, run_row)
    except EvidenceError:
        raise
    except sqlite3.Error as exc:
        raise StorageError(f"Search coverage could not be loaded: {exc}") from exc


def _coverage_from_partial_rows(
    connection: sqlite3.Connection,
    run_row: sqlite3.Row,
) -> SearchCoverage:
    try:
        plan = _query_plan_from_json(run_row["query_plan_json"])
    except EvidenceError:
        # Failed historical rows may predate strict Feature 09 plan JSON. The
        # persisted run/search fields still support a useful partial report.
        plan = QueryPlan(
            query_type="unknown_or_unsupported",
            candidate_courses=(),
            candidate_indexes=(),
            keyword_terms=(),
            semantic_queries=(),
            needs_file_inspection=False,
            needs_python=False,
            plan_confidence=0.0,
            plan_reason="Persisted query plan was unavailable.",
        )
    result_set_rows = connection.execute(
        """
        SELECT result_set_id, retrieval_method, result_count
        FROM search_result_sets
        WHERE search_run_id = ?
        ORDER BY id
        """,
        (run_row["id"],),
    ).fetchall()
    fused_rows = connection.execute(
        """
        SELECT result_json, chunk_id, file_id
        FROM search_results
        WHERE search_run_id = ? AND retrieval_method = 'hybrid'
        ORDER BY rank
        """,
        (run_row["id"],),
    ).fetchall()
    counts = {"metadata": 0, "keyword": 0, "semantic": 0}
    raw_queries: dict[str, int] = {}
    if result_set_rows:
        for row in result_set_rows:
            method = str(row["retrieval_method"])
            if method in counts:
                counts[method] += int(row["result_count"])
                if method == "semantic":
                    raw_queries[str(row["result_set_id"])] = int(row["result_count"])
    else:
        # Historical Feature 08 rows have no completion envelopes. Preserve a
        # useful best-effort report for them while using envelopes for all new
        # runs, where an absent envelope means the backend was not reached.
        raw_rows = connection.execute(
            """
            SELECT retrieval_method, result_json, chunk_id, file_id
            FROM search_results
            WHERE search_run_id = ? AND retrieval_method <> 'hybrid'
            ORDER BY id
            """,
            (run_row["id"],),
        ).fetchall()
        for row in raw_rows:
            method = str(row["retrieval_method"])
            if method in counts:
                counts[method] += 1
            try:
                payload = json.loads(row["result_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if method == "semantic":
                result_set_id = str(payload.get("result_set_id", ""))
                raw_queries[result_set_id] = raw_queries.get(result_set_id, 0) + 1
    hit_courses: list[str] = []
    hit_source_types: list[str] = []
    selectable = 0
    for row in fused_rows:
        if row["chunk_id"] is None:
            continue
        current = connection.execute(
            """
            SELECT courses.name AS course, chunks.source_type
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            LEFT JOIN courses ON courses.id = files.course_id
            WHERE chunks.id = ? AND chunks.file_id = ?
              AND files.index_status = 'indexed'
              AND COALESCE(chunks.text, '') <> ''
            """,
            (row["chunk_id"], row["file_id"]),
        ).fetchone()
        if current is None:
            continue
        selectable += 1
        try:
            fused_payload = json.loads(row["result_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            fused_payload = {}
        if not _payload_has_content_contribution(fused_payload):
            continue
        if current["course"] and current["course"] not in hit_courses:
            hit_courses.append(str(current["course"]))
        if current["source_type"] not in hit_source_types:
            hit_source_types.append(str(current["source_type"]))
    searched_courses = _json_string_tuple(run_row["searched_courses_json"])
    searched_indexes = _json_string_tuple(run_row["searched_indexes_json"])
    keyword_terms = _json_string_tuple(run_row["keyword_terms_json"])
    semantic_queries = _json_string_tuple(run_row["semantic_queries_json"])
    courses_with = tuple(course for course in searched_courses if course in hit_courses)
    indexes_with = tuple(
        index
        for index in searched_indexes
        if INDEX_TO_SOURCE_TYPE.get(index) in hit_source_types
    )
    source_types_with = tuple(
        INDEX_TO_SOURCE_TYPE[index]
        for index in searched_indexes
        if INDEX_TO_SOURCE_TYPE.get(index) in hit_source_types
    )
    weaknesses = _json_string_tuple(run_row["weaknesses_json"] or "[]")
    if not weaknesses and str(run_row["status"]) in {"failed", "running"}:
        weaknesses = ("Search run has no completed evidence packet.",)
    return SearchCoverage(
        search_run_id=int(run_row["id"]),
        status=str(run_row["status"]),
        searched_courses=searched_courses,
        searched_indexes=searched_indexes,
        keyword_terms=keyword_terms,
        semantic_queries=semantic_queries,
        raw_result_count=sum(counts.values()),
        raw_result_counts_by_method=counts,
        fused_candidate_count=len(fused_rows),
        selectable_candidate_count=selectable,
        evidence_count=0,
        evidence_token_count=0,
        courses_with_chunk_hits=courses_with,
        indexes_with_chunk_hits=indexes_with,
        source_types_with_chunk_hits=source_types_with,
        courses_without_chunk_hits=tuple(
            course for course in searched_courses if course not in courses_with
        ),
        indexes_without_chunk_hits=tuple(
            index for index in searched_indexes if index not in indexes_with
        ),
        semantic_queries_without_hits=tuple(
            query
            for index, query in enumerate(semantic_queries, start=1)
            if raw_queries.get(f"semantic:{index}") == 0
        ),
        missing_capabilities=_missing_capabilities(plan),
        file_only_candidate_count=sum(
            1 for row in fused_rows if row["chunk_id"] is None
        ),
        token_budget_omission_count=0,
        oversized_evidence_omission_count=0,
        unselected_selectable_candidate_count=selectable,
        weaknesses=weaknesses,
    )


def _query_plan_from_json(value: object) -> QueryPlan:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Persisted query plan is invalid: {exc}") from exc
    from .evidence_models import _query_plan_from_dict

    try:
        return _query_plan_from_dict(payload)
    except EvidenceModelError as exc:
        raise EvidenceError(f"Persisted query plan is invalid: {exc}") from exc


def _payload_has_content_contribution(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    contributions = value.get("contributions")
    if not isinstance(contributions, list):
        return False
    return any(
        isinstance(contribution, Mapping)
        and contribution.get("retrieval_method") in {"keyword", "semantic"}
        for contribution in contributions
    )


def _json_string_tuple(value: object) -> tuple[str, ...]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Persisted search metadata is invalid: {exc}") from exc
    if not isinstance(payload, list) or any(
        not isinstance(item, str) for item in payload
    ):
        raise EvidenceError("Persisted search metadata must be JSON string arrays")
    return tuple(payload)


def _mark_packet_build_failed(
    config: Config, search_run_id: int, error: Exception
) -> None:
    if search_run_id <= 0:
        return
    try:
        with closing(connect_sqlite(config)) as connection:
            clear_selection_flags(connection, search_run_id=search_run_id)
            connection.execute(
                """
                UPDATE search_runs
                SET status = ?, finished_at = ?, error = ?
                WHERE id = ?
                """,
                ("failed", _utc_now(), sanitize_error(error), search_run_id),
            )
            connection.commit()
    except sqlite3.Error:
        return


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _optional_row_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
