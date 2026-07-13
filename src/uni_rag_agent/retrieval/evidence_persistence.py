"""SQLite persistence primitives for Feature 09 evidence builds."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone

from uni_rag_agent.config import Config
from uni_rag_agent.storage import StorageError, connect_sqlite

from .evidence_models import RetrievalSettings, canonical_json
from .models import FusedRetrievalResult, QueryPlan, RetrievalResultSet

MAX_STORED_ERROR_LENGTH = 500
_SENSITIVE_ERROR_RE = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|token|secret|password|authorization)"
    r"\s*[:=]\s*[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password|authorization)=)"
    r"[^&#\s,;]+"
)
_TOKEN_LITERAL_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)\b")


class _SearchRunRecorder:
    """Commit complete result sets independently for a persisted run."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.search_run_id = 0
        self.retrieval_settings: RetrievalSettings | None = None
        self.query_plan: QueryPlan | None = None

    def start(
        self,
        *,
        query: str,
        query_plan: object,
        embedding_model: str,
        conversation_message_count: int,
    ) -> None:
        if not isinstance(query_plan, QueryPlan):
            raise StorageError(
                "Cannot persist a search run without a validated QueryPlan"
            )
        settings = retrieval_settings_for(
            self.config,
            embedding_model=embedding_model,
            conversation_message_count=conversation_message_count,
        )
        self.query_plan = query_plan
        self.retrieval_settings = settings
        try:
            with closing(connect_sqlite(self.config)) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO search_runs (
                        query,
                        query_type,
                        query_plan_json,
                        retrieval_settings_json,
                        searched_courses_json,
                        searched_indexes_json,
                        keyword_terms_json,
                        semantic_queries_json,
                        started_at,
                        finished_at,
                        status,
                        weaknesses_json,
                        error
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
                    """,
                    (
                        query,
                        query_plan.query_type,
                        canonical_json(query_plan),
                        canonical_json(settings),
                        canonical_json(list(query_plan.candidate_courses)),
                        canonical_json(list(query_plan.candidate_indexes)),
                        canonical_json(list(query_plan.keyword_terms)),
                        canonical_json(list(query_plan.semantic_queries)),
                        _utc_now(),
                        "running",
                    ),
                )
                self.search_run_id = int(cursor.lastrowid or 0)
                connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not create persisted search run: {exc}") from exc
        if self.search_run_id <= 0:
            raise StorageError(
                "Could not create persisted search run: missing identifier"
            )

    def record_result_set(self, result_set: RetrievalResultSet) -> None:
        self._require_started()
        envelope = (
            self.search_run_id,
            result_set.result_set_id,
            result_set.retrieval_method,
            result_set.query,
            len(result_set.results),
            _utc_now(),
        )
        rows = [
            (
                self.search_run_id,
                result.chunk_id,
                result.file_id,
                result_set.retrieval_method,
                result.rank,
                result.score,
                0,
                canonical_json(
                    {
                        "result_set_id": result_set.result_set_id,
                        "result_set_query": result_set.query,
                        "result": result.as_safe_dict(),
                    }
                ),
            )
            for result in result_set.results
        ]
        try:
            with closing(connect_sqlite(self.config)) as connection:
                connection.execute(
                    """
                    INSERT INTO search_result_sets (
                        search_run_id,
                        result_set_id,
                        retrieval_method,
                        query,
                        result_count,
                        completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    envelope,
                )
                if rows:
                    connection.executemany(
                        """
                        INSERT INTO search_results (
                            search_run_id,
                            chunk_id,
                            file_id,
                            retrieval_method,
                            rank,
                            score,
                            selected_for_evidence,
                            result_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not persist result set {result_set.result_set_id}: {exc}"
            ) from exc

    def record_fused_results(self, results: Sequence[FusedRetrievalResult]) -> None:
        self._require_started()
        rows = [
            (
                self.search_run_id,
                result.chunk_id,
                result.file_id,
                "hybrid",
                result.rank,
                result.score,
                0,
                canonical_json(result),
            )
            for result in results
        ]
        if not rows:
            return
        try:
            with closing(connect_sqlite(self.config)) as connection:
                connection.executemany(
                    """
                    INSERT INTO search_results (
                        search_run_id,
                        chunk_id,
                        file_id,
                        retrieval_method,
                        rank,
                        score,
                        selected_for_evidence,
                        result_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not persist fused retrieval results: {exc}"
            ) from exc

    def mark_failed(self, error: Exception) -> None:
        if self.search_run_id <= 0:
            return
        try:
            with closing(connect_sqlite(self.config)) as connection:
                connection.execute(
                    """
                    UPDATE search_runs
                    SET status = ?, finished_at = ?, error = ?
                    WHERE id = ?
                    """,
                    ("failed", _utc_now(), sanitize_error(error), self.search_run_id),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not mark search run failed: {exc}") from exc

    def _require_started(self) -> None:
        if self.search_run_id <= 0:
            raise StorageError(
                "Cannot persist retrieval results before creating a search run"
            )


def retrieval_settings_for(
    config: Config,
    *,
    embedding_model: str,
    conversation_message_count: int,
) -> RetrievalSettings:
    """Snapshot effective non-secret settings for one persisted run."""
    if config.llm_provider is None or config.llm_model is None:
        raise StorageError(
            "Cannot snapshot retrieval settings without LLM configuration"
        )
    return RetrievalSettings(
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        embedding_model=embedding_model,
        keyword_top_k=config.keyword_top_k,
        semantic_top_k=config.semantic_top_k,
        metadata_top_k=config.metadata_top_k,
        semantic_query_limit=config.semantic_query_limit,
        query_plan_min_confidence=config.query_plan_min_confidence,
        filename_fuzzy_threshold=config.filename_fuzzy_threshold,
        path_fuzzy_threshold=config.path_fuzzy_threshold,
        rrf_k=config.rrf_k,
        final_top_k=config.final_top_k,
        evidence_max_tokens=config.evidence_max_tokens,
        conversation_context_message_count=conversation_message_count,
    )


def finalize_run(
    connection: sqlite3.Connection,
    *,
    search_run_id: int,
    status: str,
    weaknesses: Sequence[str],
) -> None:
    connection.execute(
        """
        UPDATE search_runs
        SET status = ?, finished_at = ?, weaknesses_json = ?, error = NULL
        WHERE id = ?
        """,
        (status, _utc_now(), canonical_json(list(weaknesses)), search_run_id),
    )


def insert_packet(
    connection: sqlite3.Connection,
    *,
    search_run_id: int,
    packet_json: str,
    evidence_count: int,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO evidence_packets (
            search_run_id,
            packet_json,
            evidence_count,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (search_run_id, packet_json, evidence_count, _utc_now()),
    )
    packet_id = int(cursor.lastrowid or 0)
    if packet_id <= 0:
        raise StorageError("Could not create evidence packet: missing identifier")
    return packet_id


def clear_selection_flags(
    connection: sqlite3.Connection,
    *,
    search_run_id: int,
) -> None:
    connection.execute(
        """
        UPDATE search_results
        SET selected_for_evidence = 0
        WHERE search_run_id = ?
        """,
        (search_run_id,),
    )


def sanitize_error(error: Exception | str) -> str:
    """Keep persisted errors bounded and free of common credential material."""
    text = str(error).splitlines()[0].strip()
    if not text:
        text = type(error).__name__
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _URL_SECRET_RE.sub(r"\1[redacted]", text)
    text = _SENSITIVE_ERROR_RE.sub(r"\1=[redacted]", text)
    text = _TOKEN_LITERAL_RE.sub("[redacted]", text)
    text = text.replace("\x00", "")
    return text[:MAX_STORED_ERROR_LENGTH]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
