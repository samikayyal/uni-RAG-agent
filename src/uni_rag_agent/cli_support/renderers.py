"""Human-readable CLI renderers kept separate from command orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import closing

from ..config import Config
from ..extraction import (
    DataSummaryRunResult,
    ExtractionRunResult,
    ExtractionStatus,
)
from ..indexing import KeywordIndexResult, VectorIndexResult
from ..inventory import InventoryRunResult, InventorySummary
from ..retrieval import EvidenceBuildResult, RetrievalResult
from ..retrieval.evidence_models import EvidencePacket
from ..retrieval.models import RetrievalRun
from ..storage import StorageCheckResult, connect_sqlite_read_only


def print_answer_result(answer: object, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(answer.as_safe_dict(), indent=2, sort_keys=True))
    else:
        print(answer.answer_text)


def print_evidence_build_result(
    result: EvidenceBuildResult,
    *,
    debug: bool,
    config: Config,
) -> None:
    packet = result.packet
    coverage = result.coverage
    print(f"status: {coverage.status}")
    print(f"search_run_id: {result.search_run_id}")
    print(f"evidence_packet_id: {result.evidence_packet_id}")
    print(f"query_type: {packet.interpreted_intent}")
    print(f"searched_courses: {', '.join(packet.searched['courses']) or 'none'}")
    print(f"searched_indexes: {', '.join(packet.searched['indexes']) or 'none'}")
    print(f"keyword_terms: {', '.join(packet.searched['keyword_terms']) or 'none'}")
    print(
        f"semantic_queries: {', '.join(packet.searched['semantic_queries']) or 'none'}"
    )
    print(f"raw_result_count: {coverage.raw_result_count}")
    print(f"fused_candidate_count: {coverage.fused_candidate_count}")
    print(f"selectable_candidate_count: {coverage.selectable_candidate_count}")
    print(f"evidence_count: {coverage.evidence_count}")
    print(f"evidence_token_count: {coverage.evidence_token_count}")
    if packet.evidence:
        print("selected_evidence:")
        for item in packet.evidence:
            print(
                f"- rank={item.rank} score={item.score:.6g} course={item.course} "
                f"file={item.file} location={item.location.label} "
                f"tokens={item.token_count} text={table_value(item.text, 160)}"
            )
    else:
        print("selected_evidence: none")
    print_evidence_weaknesses(packet.weaknesses)
    if debug:
        print_persisted_evidence_debug(config, result.search_run_id)


def print_evidence_packet(packet: EvidencePacket) -> None:
    print(f"search_run_id: {packet.search_run_id}")
    print(f"query: {packet.query}")
    print(f"query_type: {packet.interpreted_intent}")
    print(f"status: {packet.coverage.status}")
    print(f"evidence_count: {packet.coverage.evidence_count}")
    print(f"evidence_token_count: {packet.coverage.evidence_token_count}")
    print("searched:")
    for key in ("courses", "indexes", "keyword_terms", "semantic_queries"):
        print(f"- {key}: {', '.join(packet.searched[key]) or 'none'}")
    if packet.evidence:
        print("evidence:")
        for item in packet.evidence:
            print(
                f"- rank={item.rank} score={item.score:.6g} course={item.course} "
                f"file={item.file} location={item.location.label} "
                f"tokens={item.token_count} text={table_value(item.text, 160)}"
            )
    else:
        print("evidence: none")
    print_evidence_weaknesses(packet.weaknesses)


def print_evidence_weaknesses(weaknesses: Sequence[str]) -> None:
    if not weaknesses:
        print("weaknesses: none")
        return
    print("weaknesses:")
    for weakness in weaknesses:
        print(f"- {weakness}")


def print_persisted_evidence_debug(config: Config, search_run_id: int) -> None:
    print("debug_persisted_result_rows:")
    with closing(connect_sqlite_read_only(config)) as connection:
        rows = connection.execute(
            """
            SELECT retrieval_method, rank, score, selected_for_evidence, result_json
            FROM search_results
            WHERE search_run_id = ?
            ORDER BY CASE retrieval_method
                         WHEN 'metadata' THEN 1
                         WHEN 'keyword' THEN 2
                         WHEN 'semantic' THEN 3
                         WHEN 'hybrid' THEN 4
                         ELSE 5
                     END,
                     rank,
                     id
            """,
            (search_run_id,),
        ).fetchall()
    for row in rows:
        print(
            f"- method={row['retrieval_method']} rank={row['rank']} "
            f"score={row['score']} selected={row['selected_for_evidence']} "
            f"json={row['result_json']}"
        )


def print_retrieval_run(run: RetrievalRun, *, debug: bool) -> None:
    plan = run.query_plan
    print(f"status: {run.status}")
    print(f"query: {run.query}")
    print(f"query_type: {plan.query_type}")
    print(f"plan_confidence: {plan.plan_confidence:.6g}")
    print(f"plan_reason: {plan.plan_reason}")
    print(f"embedding_model: {run.embedding_model}")
    print(f"searched_courses: {', '.join(run.searched_courses) or 'none'}")
    print(f"searched_indexes: {', '.join(run.searched_indexes) or 'none'}")
    print(f"keyword_terms: {', '.join(run.keyword_terms) or 'none'}")
    print(f"semantic_queries: {', '.join(run.semantic_queries) or 'none'}")
    if run.results:
        print(
            "rank | rrf_score | course | file | source/location | contributions | snippet"
        )
        print(
            "---- | --------- | ------ | ---- | --------------- | ------------- | -------"
        )
        for result in run.results:
            contribution_summary = ", ".join(
                f"{item.result_set_id}@{item.source_rank}:{item.rrf_contribution:.4f}"
                for item in result.contributions
            )
            print(
                " | ".join(
                    (
                        str(result.rank),
                        f"{result.score:.6g}",
                        table_value(result.course or "", 28),
                        table_value(result.file_path, 48),
                        table_value(format_source_location(result), 24),
                        table_value(contribution_summary, 34),
                        table_value(result.snippet.replace("\n", " "), 80),
                    )
                )
            )
    else:
        print("No fused retrieval results")
    if run.weaknesses:
        print("weaknesses:")
        for weakness in run.weaknesses:
            print(f"- {weakness}")
    if debug:
        print("debug_result_sets:")
        for result_set in run.result_sets:
            print(
                f"- {result_set.result_set_id} ({result_set.retrieval_method}) "
                f"query={result_set.query!r} count={len(result_set.results)}"
            )
            for result in result_set.results:
                print(
                    f"  - rank={result.rank} score={result.score:.6g} "
                    f"chunk_id={result.chunk_id} file_id={result.file_id} "
                    f"path={result.file_path} "
                    f"status={result.file_index_status or '-'} "
                    f"reason={result.reason_not_indexed or '-'} "
                    f"matched={','.join(result.matched_fields) or '-'}"
                )
        print("debug_rrf_contributions:")
        for result in run.results:
            print(
                f"- rank={result.rank} chunk_id={result.chunk_id} file_id={result.file_id}"
            )
            for contribution in result.contributions:
                print(f"  - {json.dumps(contribution.as_safe_dict(), sort_keys=True)}")


def print_storage_result(result: StorageCheckResult) -> None:
    safe = result.as_safe_dict()
    for key in (
        "data_dir",
        "sqlite_path",
        "extracted_dir",
        "chroma_dir",
        "runs_dir",
        "sqlite_exists",
        "fts5_available",
    ):
        print(f"{key}: {safe[key]}")
    print(f"required_tables_present: {len(result.required_tables_present)}")
    if result.missing_tables:
        print(f"missing_tables: {', '.join(result.missing_tables)}")
    else:
        print("missing_tables: none")
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_inventory_run_result(result: InventoryRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"courses_seen: {result.courses_seen}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_pending: {result.files_pending}")
    print(f"files_metadata_only: {result.files_metadata_only}")
    print(f"files_failed: {result.files_failed}")
    print(f"files_missing: {result.files_missing}")
    print(f"bytes_seen: {result.bytes_seen}")
    print_count_mapping("by_status", result.by_status)
    print_count_mapping("by_category", result.by_category)
    print_count_mapping("by_extension", result.by_extension)
    print_count_mapping("by_reason", result.by_reason)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_inventory_summary(summary: InventorySummary) -> None:
    print(f"latest_inventory_run_id: {summary.latest_inventory_run_id}")
    print(f"latest_inventory_started_at: {summary.latest_inventory_started_at}")
    print(f"courses_total: {summary.courses_total}")
    print(f"files_total: {summary.files_total}")
    print(f"files_missing: {summary.files_missing}")
    print(f"bytes_total: {summary.bytes_total}")
    if summary.by_course:
        print("by_course:")
        for course in summary.by_course:
            print(
                f"- {course.name}: files={course.file_count}, bytes={course.total_bytes}"
            )
    print_count_mapping("by_status", summary.by_status)
    print_count_mapping("by_category", summary.by_category)
    print_count_mapping("by_extension", summary.by_extension)
    print_count_mapping("by_reason", summary.by_reason)


def print_extraction_run_result(result: ExtractionRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"category: {result.category or 'all'}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_indexed: {result.files_indexed}")
    print(f"files_failed: {result.files_failed}")
    print(f"chunks_created: {result.chunks_created}")
    print_count_mapping("by_source_type", result.by_source_type)
    if result.failures:
        print("failures:")
        for failure in result.failures:
            print(f"- file_id={failure.file_id} path={failure.path}: {failure.error}")
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_data_summary_run_result(result: DataSummaryRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"file_id: {result.file_id or 'all'}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_indexed: {result.files_indexed}")
    print(f"files_failed: {result.files_failed}")
    print(f"summaries_created: {result.summaries_created}")
    print(f"chunks_created: {result.chunks_created}")
    print_count_mapping("by_format", result.by_format)
    if result.failures:
        print("failures:")
        for failure in result.failures:
            print(f"- file_id={failure.file_id} path={failure.path}: {failure.error}")
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_extraction_status(status: ExtractionStatus) -> None:
    print(f"latest_extraction_run_id: {status.latest_extraction_run_id}")
    print(f"latest_extraction_started_at: {status.latest_extraction_started_at}")
    print(f"pending_text_files: {status.pending_text_files}")
    print(f"indexed_text_files: {status.indexed_text_files}")
    print(f"failed_text_files: {status.failed_text_files}")
    print(f"extracted_documents: {status.extracted_documents}")
    print(f"chunks_total: {status.chunks_total}")
    print_count_mapping("pending_by_category", status.pending_by_category)
    print_count_mapping("chunks_by_source_type", status.chunks_by_source_type)
    if status.recent_failures:
        print("recent_failures:")
        for failure in status.recent_failures:
            print(f"- file_id={failure.file_id} path={failure.path}: {failure.error}")


def print_keyword_index_result(result: KeywordIndexResult) -> None:
    print("mode: rebuild")
    print(f"rows_removed: {result.rows_removed}")
    print(f"chunks_seen: {result.chunks_seen}")
    print(f"rows_indexed: {result.rows_indexed}")
    print_count_mapping("by_source_type", result.by_source_type)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_keyword_search_results(results: Sequence[RetrievalResult]) -> None:
    if not results:
        print("No keyword results")
        return

    print("rank | score | chunk_id | source/location | course | path | snippet")
    print("---- | ----- | -------- | --------------- | ------ | ---- | -------")
    for result in results:
        location = format_source_location(result)
        print(
            " | ".join(
                (
                    str(result.rank),
                    f"{result.score:.6g}",
                    str(result.chunk_id),
                    table_value(location, 24),
                    table_value(result.course or "", 28),
                    table_value(result.file_path, 48),
                    table_value(result.snippet.replace("\n", " "), 80),
                )
            )
        )


def print_vector_index_result(result: VectorIndexResult) -> None:
    print(f"mode: {'rebuild' if result.rebuild else 'incremental'}")
    print(f"model: {result.model}")
    print(f"provider: {result.provider}")
    print(f"embedding_dim: {result.embedding_dim}")
    print(f"collections: {', '.join(result.collections)}")
    print(f"chunks_seen: {result.chunks_seen}")
    print(f"rows_removed: {result.rows_removed}")
    print(f"mappings_removed: {result.mappings_removed}")
    print(f"vectors_removed: {result.vectors_removed}")
    print(f"vectors_indexed: {result.vectors_indexed}")
    print(f"embeddings_total: {result.embeddings_total}")
    print_count_mapping("by_source_type", result.by_source_type)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def print_semantic_search_results(results: Sequence[RetrievalResult]) -> None:
    if not results:
        print("No semantic results")
        return

    print("rank | score | chunk_id | source/location | course | path | snippet")
    print("---- | ----- | -------- | --------------- | ------ | ---- | -------")
    for result in results:
        location = format_source_location(result)
        print(
            " | ".join(
                (
                    str(result.rank),
                    f"{result.score:.6g}",
                    str(result.chunk_id),
                    table_value(location, 24),
                    table_value(result.course or "", 28),
                    table_value(result.file_path, 48),
                    table_value(result.snippet.replace("\n", " "), 80),
                )
            )
        )


def format_source_location(result: RetrievalResult) -> str:
    if result.location_type and result.location_value:
        prefix = f"{result.source_type}:" if result.source_type else ""
        return f"{prefix}{result.location_type} {result.location_value}"
    if result.location_type:
        prefix = f"{result.source_type}:" if result.source_type else ""
        return f"{prefix}{result.location_type}"
    return result.source_type or "file"


def table_value(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def print_count_mapping(label: str, counts: Mapping[str, int]) -> None:
    if not counts:
        print(f"{label}: none")
        return
    print(f"{label}:")
    for key, value in counts.items():
        print(f"- {key}: {value}")


__all__ = [
    "format_source_location",
    "print_answer_result",
    "print_count_mapping",
    "print_data_summary_run_result",
    "print_evidence_build_result",
    "print_evidence_packet",
    "print_evidence_weaknesses",
    "print_extraction_run_result",
    "print_extraction_status",
    "print_inventory_run_result",
    "print_inventory_summary",
    "print_keyword_index_result",
    "print_keyword_search_results",
    "print_persisted_evidence_debug",
    "print_retrieval_run",
    "print_semantic_search_results",
    "print_storage_result",
    "print_vector_index_result",
    "table_value",
]
