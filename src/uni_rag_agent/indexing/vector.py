"""ChromaDB vector indexing and semantic search.

SQLite stays authoritative for chunk text and citation metadata. ChromaDB holds
only vectors plus filter metadata. Embedding mapping rows in the SQLite
``embeddings`` table record the physical collection and stable vector id so the
two stores can be reconciled. Different embedding models persist into distinct
model-namespaced physical collections, enabling side-by-side models.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone

from uni_rag_agent.config import Config
from uni_rag_agent.retrieval import RetrievalResult
from uni_rag_agent.storage import (
    StorageError,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)

from .eligibility import (
    INDEX_TO_SOURCE_TYPE,
    current_chunk_where_sql,
    placeholders,
    source_types_for_indexes,
    validate_logical_index,
)
from .embedding_providers.common import EmbeddingValidationError, validate_vectors
from .embedding_providers.factory import BuiltEmbeddingModel, build_embedding_model
from .models import SemanticSearchError, VectorIndexError, VectorIndexResult
from .profiles import EmbeddingProfile, physical_collection_name

_EMBED_BATCH = 64
_VECTOR_ID_BATCH = 256
_HYDRATE_CANDIDATE_BATCH = 250
_SNIPPET_CHAR_LIMIT = 240
VECTOR_BACKEND = "chroma"

_INSERT_EMBEDDING_SQL = """
INSERT INTO embeddings (
    chunk_id,
    vector_backend,
    vector_collection,
    vector_id,
    embedding_model,
    embedding_dim,
    embedded_at
)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chunk_id, vector_backend, vector_collection) DO UPDATE SET
    vector_id = excluded.vector_id,
    embedding_model = excluded.embedding_model,
    embedding_dim = excluded.embedding_dim,
    embedded_at = excluded.embedded_at
"""


@dataclass(frozen=True)
class _Candidate:
    chunk_id: int
    distance: float
    physical_collection: str
    vector_id: str


@dataclass
class _SemanticContext:
    built: BuiltEmbeddingModel
    collections: dict[str, object]
    counts: dict[str, int]


@dataclass(frozen=True)
class _ReconciliationResult:
    """Counts emitted while bringing one physical collection back in sync."""

    mappings_removed: int = 0
    vectors_removed: int = 0
    metadata_updated: int = 0


def sync_vector_index(
    config: Config,
    collection: str | None = None,
    model: str | None = None,
    rebuild: bool = False,
    *,
    show_progress: bool = False,
) -> VectorIndexResult:
    """Embed eligible chunks into ChromaDB for the selected model.

    The default behavior is incremental: the selected physical collection is
    reconciled with SQLite, then only current eligible chunks missing that
    profile are embedded. ``rebuild`` clears and repopulates only the selected
    model/profile (and optional logical ``collection``).
    """
    built = build_embedding_model(config, model, error=VectorIndexError)
    profile = built.profile
    dimension = built.dimension
    selected = _selected_logical_indexes(collection)

    ensure_data_dirs(config)
    try:
        client = _chroma_client(config, error=VectorIndexError)
        rows_removed = 0
        mappings_removed = 0
        vectors_removed = 0
        metadata_updated = 0
        vectors_indexed = 0
        by_source_type: dict[str, int] = {}
        physical_names: list[str] = []
        with closing(connect_sqlite(config)) as connection:
            initialize_schema(connection)
            for logical_index, source_type in selected:
                physical = _physical_name(logical_index, profile, dimension)
                physical_names.append(physical)
                if rebuild:
                    rows_removed += _clear_collection(
                        client,
                        connection,
                        physical=physical,
                    )
                chroma_collection = client.get_or_create_collection(
                    name=physical,
                    metadata={"hnsw:space": profile.metric},
                )
                reconciliation = _reconcile_collection(
                    connection,
                    chroma_collection=chroma_collection,
                    source_type=source_type,
                    physical=physical,
                    logical_index=logical_index,
                    model_name=profile.model_name,
                    dimension=dimension,
                )
                mappings_removed += reconciliation.mappings_removed
                vectors_removed += reconciliation.vectors_removed
                metadata_updated += reconciliation.metadata_updated
                indexed = _embed_missing_chunks(
                    connection,
                    chroma_collection=chroma_collection,
                    source_type=source_type,
                    logical_index=logical_index,
                    physical=physical,
                    model_name=profile.model_name,
                    dimension=dimension,
                    embeddings=built.embeddings,
                    show_progress=show_progress,
                )
                if indexed:
                    by_source_type[source_type] = indexed
                    vectors_indexed += indexed
            chunks_seen = _eligible_chunk_count(
                connection, [source_type for _, source_type in selected]
            )
            embeddings_total = _embeddings_total(
                connection,
                physical_names=physical_names,
            )
            connection.commit()
    except VectorIndexError:
        raise
    except sqlite3.Error as exc:
        raise VectorIndexError(f"Vector index sync failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface ChromaDB failures clearly
        raise VectorIndexError(f"Vector index sync failed: {exc}") from exc

    diagnostics = _sync_diagnostics(
        chunks_seen=chunks_seen,
        vectors_indexed=vectors_indexed,
        embeddings_total=embeddings_total,
        model_name=profile.model_name,
        mappings_removed=mappings_removed,
        vectors_removed=vectors_removed,
        metadata_updated=metadata_updated,
    )
    return VectorIndexResult(
        rebuild=rebuild,
        model=profile.model_name,
        provider=profile.provider,
        embedding_dim=dimension,
        collections=tuple(logical for logical, _ in selected),
        chunks_seen=chunks_seen,
        rows_removed=rows_removed,
        mappings_removed=mappings_removed,
        vectors_removed=vectors_removed,
        vectors_indexed=vectors_indexed,
        embeddings_total=embeddings_total,
        by_source_type=dict(sorted(by_source_type.items())),
        diagnostics=tuple(diagnostics),
    )


def semantic_search(
    config: Config,
    query: str,
    course: str | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
    model: str | None = None,
    *,
    courses: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """Run one semantic query using the multi-query request seam."""
    return semantic_search_many(
        config,
        (query,),
        course=course,
        indexes=indexes,
        top_k=top_k,
        model=model,
        courses=courses,
    )[0]


def semantic_search_many(
    config: Config,
    queries: Sequence[str],
    course: str | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
    model: str | None = None,
    *,
    courses: Sequence[str] | None = None,
) -> list[list[RetrievalResult]]:
    """Search all semantic queries with one embedding/Chroma request context."""
    limit = top_k if top_k is not None else config.semantic_top_k
    if limit <= 0:
        raise SemanticSearchError("top_k must be greater than zero")

    query_texts = tuple(query.strip() for query in queries)
    if any(not query_text for query_text in query_texts):
        raise SemanticSearchError("Semantic query must not be empty.")
    if not query_texts:
        return []
    if course is not None and courses is not None:
        raise SemanticSearchError("Specify either course or courses, not both")
    if courses is not None and not courses:
        return [[] for _ in query_texts]
    if courses is None and course is not None:
        courses = (course,)

    source_types = source_types_for_indexes(indexes, error=SemanticSearchError)
    if source_types == ():
        return [[] for _ in query_texts]
    selected = _selected_logical_indexes_for_search(source_types)

    try:
        canonical_courses = _canonical_course_names(config, courses)
        if courses is not None and not canonical_courses:
            return [[] for _ in query_texts]

        context = _build_semantic_context(
            config,
            model=model,
            selected=selected,
        )
        try:
            query_vectors = validate_vectors(
                _embed_query_batch(context.built.embeddings, query_texts),
                expected_count=len(query_texts),
                expected_dimension=context.built.dimension,
                context="embedding provider query response",
            )
        except EmbeddingValidationError as exc:
            raise SemanticSearchError(str(exc)) from exc

        candidate_sets = _query_candidates_many(
            context,
            selected=selected,
            query_vectors=query_vectors,
            limit=limit,
            courses=canonical_courses,
        )
        if not any(candidate_sets):
            return [[] for _ in query_texts]

        results: list[list[RetrievalResult]] = []
        for candidates in candidate_sets:
            if not candidates:
                results.append([])
                continue
            rows = _hydrate_candidates(
                config,
                candidates=candidates,
                source_types=[source_type for _, source_type in selected],
                courses=canonical_courses,
            )
            results.append(_semantic_results(candidates, rows, limit=limit))
        return results
    except SemanticSearchError:
        raise
    except sqlite3.OperationalError as exc:
        raise SemanticSearchError(
            f"Semantic query could not be executed: {exc}"
        ) from exc
    except sqlite3.Error as exc:
        raise StorageError(f"Semantic search could not inspect SQLite: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface ChromaDB failures clearly
        raise SemanticSearchError(f"Semantic search failed: {exc}") from exc


def _build_semantic_context(
    config: Config,
    *,
    model: str | None,
    selected: Sequence[tuple[str, str]],
) -> _SemanticContext:
    built = build_embedding_model(config, model, error=SemanticSearchError)
    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise SemanticSearchError(f"Semantic search storage check failed: {details}")

    client = _chroma_client(config, error=SemanticSearchError)
    existing = _existing_collection_names(client)
    collections: dict[str, object] = {}
    counts: dict[str, int] = {}
    for logical_index, _source_type in selected:
        physical = _physical_name(logical_index, built.profile, built.dimension)
        if physical not in existing:
            continue
        collection = client.get_collection(name=physical)  # type: ignore[attr-defined]
        count = int(collection.count())  # type: ignore[attr-defined]
        if count > 0:
            collections[physical] = collection
            counts[physical] = count
    return _SemanticContext(built=built, collections=collections, counts=counts)


def _embed_query_batch(embeddings: object, queries: Sequence[str]) -> object:
    batch_method = getattr(embeddings, "embed_queries", None)
    if callable(batch_method):
        return batch_method(list(queries))
    embed_query = getattr(embeddings, "embed_query", None)
    if not callable(embed_query):
        raise SemanticSearchError("Embedding provider has no query method")
    return [embed_query(query) for query in queries]


def _semantic_results(
    candidates: dict[int, _Candidate],
    rows: Sequence[sqlite3.Row],
    *,
    limit: int,
) -> list[RetrievalResult]:
    ranked = sorted(
        rows,
        key=lambda row: (
            candidates[int(row["chunk_id"])].distance,
            int(row["chunk_id"]),
        ),
    )
    results: list[RetrievalResult] = []
    for rank, row in enumerate(ranked[:limit], start=1):
        candidate = candidates[int(row["chunk_id"])]
        results.append(
            RetrievalResult(
                chunk_id=int(row["chunk_id"]),
                file_id=int(row["file_id"]),
                course=row["course"],
                file_path=row["file_path"],
                source_type=row["source_type"],
                location_type=row["location_type"],
                location_value=row["location_value"],
                rank=rank,
                score=1.0 - float(candidate.distance),
                snippet=_snippet(row["text"]),
                retrieval_method="semantic",
                vector_collection=candidate.physical_collection,
                vector_id=candidate.vector_id,
            )
        )
    return results


def _selected_logical_indexes(collection: str | None) -> list[tuple[str, str]]:
    if collection is None:
        return list(INDEX_TO_SOURCE_TYPE.items())
    source_type = validate_logical_index(collection, error=VectorIndexError)
    return [(collection, source_type)]


def _selected_logical_indexes_for_search(
    source_types: tuple[str, ...] | None,
) -> list[tuple[str, str]]:
    if source_types is None:
        return list(INDEX_TO_SOURCE_TYPE.items())
    wanted = set(source_types)
    return [
        (logical_index, source_type)
        for logical_index, source_type in INDEX_TO_SOURCE_TYPE.items()
        if source_type in wanted
    ]


def _physical_name(
    logical_index: str,
    profile: EmbeddingProfile,
    dimension: int,
) -> str:
    return physical_collection_name(
        logical_index,
        provider=profile.provider,
        model_name=profile.model_name,
        dimension=dimension,
        metric=profile.metric,
    )


def _clear_collection(
    client: object,
    connection: sqlite3.Connection,
    *,
    physical: str,
) -> int:
    if physical in _existing_collection_names(client):
        client.delete_collection(name=physical)  # type: ignore[attr-defined]
    cursor = connection.execute(
        """
        DELETE FROM embeddings
        WHERE vector_backend = ?
          AND vector_collection = ?
        """,
        (VECTOR_BACKEND, physical),
    )
    connection.commit()
    return max(cursor.rowcount, 0)


def _embed_missing_chunks(
    connection: sqlite3.Connection,
    *,
    chroma_collection: object,
    source_type: str,
    logical_index: str,
    physical: str,
    model_name: str,
    dimension: int,
    embeddings: object,
    show_progress: bool,
) -> int:
    rows = _missing_chunk_rows(
        connection,
        source_type=source_type,
        physical=physical,
    )
    if not rows:
        return 0

    indexed = 0
    embedded_at = _utc_now()
    if show_progress:
        print(f"Embedding {logical_index}: 0/{len(rows)}", flush=True)
    for batch in _batches(rows, _EMBED_BATCH):
        texts = [str(row["text"]) for row in batch]
        try:
            vectors = validate_vectors(
                embeddings.embed_documents(texts),  # type: ignore[attr-defined]
                expected_count=len(batch),
                expected_dimension=dimension,
                context="embedding provider response",
            )
        except EmbeddingValidationError as exc:
            raise VectorIndexError(str(exc)) from exc
        ids = [f"chunk:{int(row['chunk_id'])}" for row in batch]
        metadatas = [
            _chunk_metadata(row, logical_index, model_name, dimension) for row in batch
        ]
        chroma_collection.upsert(  # type: ignore[attr-defined]
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
        )
        for row, vector_id in zip(batch, ids):
            connection.execute(
                _INSERT_EMBEDDING_SQL,
                (
                    int(row["chunk_id"]),
                    VECTOR_BACKEND,
                    physical,
                    vector_id,
                    model_name,
                    dimension,
                    embedded_at,
                ),
            )
        connection.commit()
        indexed += len(batch)
        if show_progress:
            print(f"Embedding {logical_index}: {indexed}/{len(rows)}", flush=True)
    return indexed


def _missing_chunk_rows(
    connection: sqlite3.Connection,
    *,
    source_type: str,
    physical: str,
) -> list[sqlite3.Row]:
    where_sql = current_chunk_where_sql((source_type,), require_non_empty_text=True)
    return connection.execute(
        f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.text AS text,
            chunks.file_id AS file_id,
            chunks.source_type AS source_type,
            files.relative_path AS file_path,
            courses.name AS course
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE {where_sql}
          AND NOT EXISTS (
              SELECT 1
              FROM embeddings
              WHERE embeddings.chunk_id = chunks.id
                AND embeddings.vector_backend = ?
                AND embeddings.vector_collection = ?
          )
        ORDER BY chunks.id
        """,
        (source_type, VECTOR_BACKEND, physical),
    ).fetchall()


def _reconcile_collection(
    connection: sqlite3.Connection,
    *,
    chroma_collection: object,
    source_type: str,
    physical: str,
    logical_index: str,
    model_name: str,
    dimension: int,
) -> _ReconciliationResult:
    """Remove stale state and make missing vectors eligible for re-embedding.

    SQLite is authoritative: rows must describe current, non-empty chunks in
    this logical collection. Chroma-only ids are deleted; SQLite mappings whose
    vector disappeared are removed so the normal missing-chunk path restores
    them in the same sync. Vectors whose stored course/path filter metadata
    drifted from SQLite (e.g. a file reassigned to another course by a later
    inventory run) get their metadata updated in place without re-embedding.
    """
    all_rows = connection.execute(
        """
        SELECT id, vector_id
        FROM embeddings
        WHERE vector_backend = ?
          AND vector_collection = ?
        """,
        (VECTOR_BACKEND, physical),
    ).fetchall()
    current_where = current_chunk_where_sql((source_type,), require_non_empty_text=True)
    current_rows = connection.execute(
        f"""
        SELECT embeddings.id, embeddings.vector_id
        FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN files ON files.id = chunks.file_id
        WHERE embeddings.vector_backend = ?
          AND embeddings.vector_collection = ?
          AND {current_where}
        """,
        (VECTOR_BACKEND, physical, source_type),
    ).fetchall()

    current_ids = {int(row["id"]) for row in current_rows}
    stale_mapping_ids = [
        int(row["id"]) for row in all_rows if int(row["id"]) not in current_ids
    ]
    if stale_mapping_ids:
        connection.execute(
            f"DELETE FROM embeddings WHERE id IN ({placeholders(stale_mapping_ids)})",
            stale_mapping_ids,
        )

    expected_vector_ids = {str(row["vector_id"]) for row in current_rows}
    actual_vector_ids = _collection_vector_ids(chroma_collection)
    stale_vector_ids = actual_vector_ids - expected_vector_ids
    if stale_vector_ids:
        _delete_vectors(chroma_collection, stale_vector_ids)

    missing_vector_ids = expected_vector_ids - actual_vector_ids
    if missing_vector_ids:
        connection.execute(
            f"""
            DELETE FROM embeddings
            WHERE vector_backend = ?
              AND vector_collection = ?
              AND vector_id IN ({placeholders(tuple(missing_vector_ids))})
            """,
            (VECTOR_BACKEND, physical, *missing_vector_ids),
        )

    metadata_updated = _reconcile_vector_metadata(
        connection,
        chroma_collection=chroma_collection,
        source_type=source_type,
        physical=physical,
        logical_index=logical_index,
        model_name=model_name,
        dimension=dimension,
        present_vector_ids=expected_vector_ids & actual_vector_ids,
    )

    connection.commit()
    return _ReconciliationResult(
        mappings_removed=len(stale_mapping_ids) + len(missing_vector_ids),
        vectors_removed=len(stale_vector_ids),
        metadata_updated=metadata_updated,
    )


def _reconcile_vector_metadata(
    connection: sqlite3.Connection,
    *,
    chroma_collection: object,
    source_type: str,
    physical: str,
    logical_index: str,
    model_name: str,
    dimension: int,
    present_vector_ids: set[str],
) -> int:
    """Upsert Chroma filter metadata that drifted from authoritative SQLite."""
    if not present_vector_ids:
        return 0
    current_where = current_chunk_where_sql((source_type,), require_non_empty_text=True)
    rows = connection.execute(
        f"""
        SELECT
            embeddings.vector_id AS vector_id,
            chunks.id AS chunk_id,
            chunks.file_id AS file_id,
            chunks.source_type AS source_type,
            files.relative_path AS file_path,
            courses.name AS course
        FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE embeddings.vector_backend = ?
          AND embeddings.vector_collection = ?
          AND {current_where}
        """,
        (VECTOR_BACKEND, physical, source_type),
    ).fetchall()
    expected_by_vector_id = {
        str(row["vector_id"]): _chunk_metadata(
            row, logical_index, model_name, dimension
        )
        for row in rows
        if str(row["vector_id"]) in present_vector_ids
    }
    if not expected_by_vector_id:
        return 0

    drifted_ids: list[str] = []
    drifted_metadatas: list[dict[str, object]] = []
    ordered_ids = sorted(expected_by_vector_id)
    for start in range(0, len(ordered_ids), _VECTOR_ID_BATCH):
        batch_ids = ordered_ids[start : start + _VECTOR_ID_BATCH]
        result = chroma_collection.get(  # type: ignore[attr-defined]
            ids=batch_ids, include=["metadatas"]
        )
        ids = result.get("ids", []) if isinstance(result, dict) else []
        metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
        for vector_id, actual in zip(ids, metadatas or []):
            expected = expected_by_vector_id.get(str(vector_id))
            if expected is None:
                continue
            actual_map = actual if isinstance(actual, dict) else {}
            if (
                actual_map.get("course") != expected["course"]
                or actual_map.get("file_path") != expected["file_path"]
            ):
                drifted_ids.append(str(vector_id))
                drifted_metadatas.append(expected)
    for start in range(0, len(drifted_ids), _VECTOR_ID_BATCH):
        chroma_collection.update(  # type: ignore[attr-defined]
            ids=drifted_ids[start : start + _VECTOR_ID_BATCH],
            metadatas=drifted_metadatas[start : start + _VECTOR_ID_BATCH],
        )
    return len(drifted_ids)


def _chunk_metadata(
    row: sqlite3.Row,
    logical_index: str,
    model_name: str,
    dimension: int,
) -> dict[str, object]:
    return {
        "chunk_id": int(row["chunk_id"]),
        "file_id": int(row["file_id"]),
        "logical_index": logical_index,
        "source_type": str(row["source_type"]),
        "course": row["course"] or "",
        "file_path": str(row["file_path"]),
        "embedding_model": model_name,
        "embedding_dim": dimension,
    }


def _eligible_chunk_count(
    connection: sqlite3.Connection,
    source_types: Sequence[str],
) -> int:
    unique_source_types = tuple(dict.fromkeys(source_types))
    if not unique_source_types:
        return 0
    where_sql = current_chunk_where_sql(
        unique_source_types, require_non_empty_text=True
    )
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        WHERE {where_sql}
        """,
        unique_source_types,
    ).fetchone()
    return int(row[0])


def _embeddings_total(
    connection: sqlite3.Connection,
    *,
    physical_names: Sequence[str],
) -> int:
    if not physical_names:
        return 0
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM embeddings
        WHERE vector_backend = ?
          AND vector_collection IN ({placeholders(physical_names)})
        """,
        (VECTOR_BACKEND, *physical_names),
    ).fetchone()
    return int(row[0])


def _query_candidates_many(
    context: _SemanticContext,
    *,
    selected: Sequence[tuple[str, str]],
    query_vectors: Sequence[Sequence[float]],
    limit: int,
    courses: Sequence[str] | None,
) -> list[dict[int, _Candidate]]:
    candidates_by_query: list[dict[int, _Candidate]] = [{} for _ in query_vectors]
    for logical_index, _source_type in selected:
        physical = _physical_name(
            logical_index,
            context.built.profile,
            context.built.dimension,
        )
        collection = context.collections.get(physical)
        if collection is None:
            continue
        count = context.counts[physical]
        # The collection stores the exact canonical course name as metadata.
        # Restricting Chroma first keeps the final top-K meaningful for course
        # queries; SQLite reapplies the same filter while hydrating.
        fetch_k = min(count, max(limit * 4, limit))
        query_kwargs: dict[str, object] = {
            "query_embeddings": [list(vector) for vector in query_vectors],
            "n_results": fetch_k,
            "include": ["distances", "metadatas"],
        }
        if courses is not None:
            query_kwargs["where"] = {"course": {"$in": list(courses)}}
        result = collection.query(  # type: ignore[attr-defined]
            **query_kwargs,
        )
        for query_index, query_vector in enumerate(query_vectors):
            hits = list(_iter_query_hits(result, query_index=query_index))
            if courses is not None and not hits:
                # Chroma's filtered HNSW query can occasionally exhaust its
                # search frontier before reaching a sparse metadata partition.
                # Fall back to exact local scoring so the course filter is
                # applied before final top-K truncation.
                hits = _course_filter_fallback_hits(
                    collection,
                    courses=courses,
                    query_vector=query_vector,
                    limit=limit,
                )
            candidates = candidates_by_query[query_index]
            for chunk_id, distance, vector_id in hits:
                current = candidates.get(chunk_id)
                if current is None or distance < current.distance:
                    candidates[chunk_id] = _Candidate(
                        chunk_id=chunk_id,
                        distance=distance,
                        physical_collection=physical,
                        vector_id=vector_id,
                    )
    return candidates_by_query


def _course_filter_fallback_hits(
    collection: object,
    *,
    courses: Sequence[str],
    query_vector: Sequence[float],
    limit: int,
) -> list[tuple[int, float, str]]:
    """Return exact top results from a course partition missed by HNSW."""
    result = collection.get(  # type: ignore[attr-defined]
        where={"course": {"$in": list(courses)}},
        include=["embeddings"],
    )
    ids = result.get("ids") if isinstance(result, dict) else None
    embeddings = result.get("embeddings") if isinstance(result, dict) else None
    if not ids or embeddings is None:
        return []

    scored: list[tuple[int, float, str]] = []
    for vector_id, embedding in zip(ids, embeddings):
        if not isinstance(vector_id, str):
            continue
        chunk_id = _chunk_id_from_vector_id(vector_id)
        if chunk_id is None:
            continue
        distance = _cosine_distance(query_vector, embedding)
        if distance is not None:
            scored.append((chunk_id, distance, vector_id))
    return sorted(scored, key=lambda hit: (hit[1], hit[0]))[:limit]


def _chunk_id_from_vector_id(vector_id: str) -> int | None:
    if ":" not in vector_id:
        return None
    try:
        return int(vector_id.split(":", 1)[1])
    except ValueError:
        return None


def _cosine_distance(
    query_vector: Sequence[float],
    embedding: Sequence[float],
) -> float | None:
    if len(query_vector) != len(embedding):
        return None
    dot_product = sum(left * right for left, right in zip(query_vector, embedding))
    query_norm = sum(value * value for value in query_vector) ** 0.5
    embedding_norm = sum(value * value for value in embedding) ** 0.5
    if query_norm == 0.0 or embedding_norm == 0.0:
        return None
    return 1.0 - (dot_product / (query_norm * embedding_norm))


def _iter_query_hits(
    result: object,
    *,
    query_index: int = 0,
) -> Iterator[tuple[int, float, str]]:
    ids = _query_values(result, "ids", query_index)
    distances = _query_values(result, "distances", query_index)
    metadatas = _query_values(result, "metadatas", query_index)
    for vector_id, distance, metadata in zip(ids, distances, metadatas):
        md = metadata or {}
        chunk_id = md.get("chunk_id")
        if chunk_id is None and isinstance(vector_id, str) and ":" in vector_id:
            chunk_id = vector_id.split(":", 1)[1]
        if chunk_id is None:
            continue
        try:
            yield int(chunk_id), float(distance), str(vector_id)
        except (TypeError, ValueError):
            continue


def _query_values(result: object, key: str, query_index: int) -> list[object]:
    value = result.get(key) if isinstance(result, dict) else None
    if not value:
        return []
    try:
        selected = value[query_index]
    except (IndexError, TypeError):
        return []
    return list(selected) if selected else []


def _hydrate_candidates(
    config: Config,
    *,
    candidates: dict[int, _Candidate],
    source_types: Sequence[str],
    courses: Sequence[str] | None,
) -> list[sqlite3.Row]:
    unique_source_types = tuple(dict.fromkeys(source_types))
    if not unique_source_types or not candidates:
        return []
    where_sql = current_chunk_where_sql(
        unique_source_types, require_non_empty_text=False
    )
    hydrated: list[sqlite3.Row] = []
    with closing(connect_sqlite_read_only(config)) as connection:
        candidate_values = tuple(candidates.values())
        for start in range(0, len(candidate_values), _HYDRATE_CANDIDATE_BATCH):
            batch = candidate_values[start : start + _HYDRATE_CANDIDATE_BATCH]
            hit_sql = ", ".join("(?, ?, ?)" for _ in batch)
            sql = f"""
                WITH candidate_hits(chunk_id, vector_collection, vector_id) AS (
                    VALUES {hit_sql}
                )
                SELECT
                    chunks.id AS chunk_id,
                    files.id AS file_id,
                    courses.name AS course,
                    files.relative_path AS file_path,
                    chunks.source_type AS source_type,
                    chunks.location_type AS location_type,
                    chunks.location_value AS location_value,
                    chunks.text AS text
                FROM candidate_hits
                JOIN embeddings
                  ON embeddings.chunk_id = candidate_hits.chunk_id
                 AND embeddings.vector_backend = ?
                 AND embeddings.vector_collection = candidate_hits.vector_collection
                 AND embeddings.vector_id = candidate_hits.vector_id
                JOIN chunks ON chunks.id = embeddings.chunk_id
                JOIN files ON files.id = chunks.file_id
                LEFT JOIN courses ON courses.id = files.course_id
                WHERE {where_sql}
            """
            params: list[object] = [
                *(
                    value
                    for candidate in batch
                    for value in (
                        candidate.chunk_id,
                        candidate.physical_collection,
                        candidate.vector_id,
                    )
                ),
                VECTOR_BACKEND,
                *unique_source_types,
            ]
            if courses is not None:
                sql += f" AND courses.name IN ({', '.join('?' for _ in courses)})"
                params.extend(courses)
            hydrated.extend(connection.execute(sql, params).fetchall())
    return hydrated


def _canonical_course_names(
    config: Config,
    courses: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Resolve course filters to canonical SQLite spelling."""
    if courses is None:
        return None
    requested: list[str] = []
    seen: set[str] = set()
    for course in courses:
        normalized = course.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            requested.append(course.strip())
    if not requested:
        return ()
    placeholders = ", ".join("?" for _ in requested)
    with closing(connect_sqlite_read_only(config)) as connection:
        rows = connection.execute(
            f"SELECT name FROM courses WHERE LOWER(name) IN ({placeholders})",
            [value.casefold() for value in requested],
        ).fetchall()
    canonical_by_key = {str(row["name"]).casefold(): str(row["name"]) for row in rows}
    return tuple(
        canonical_by_key[value.casefold()]
        for value in requested
        if value.casefold() in canonical_by_key
    )


def _sync_diagnostics(
    *,
    chunks_seen: int,
    vectors_indexed: int,
    embeddings_total: int,
    model_name: str,
    mappings_removed: int,
    vectors_removed: int,
    metadata_updated: int = 0,
) -> list[str]:
    diagnostics: list[str] = []
    if mappings_removed or vectors_removed:
        diagnostics.append(
            "Reconciled vector storage: removed "
            f"{vectors_removed} stale Chroma vector(s) and "
            f"{mappings_removed} SQLite mapping row(s)."
        )
    if metadata_updated:
        diagnostics.append(
            f"Updated drifted course/path filter metadata on {metadata_updated} "
            "existing Chroma vector(s)."
        )
    if chunks_seen == 0:
        diagnostics.append("No eligible indexed chunks found for vector indexing.")
    elif vectors_indexed == 0:
        diagnostics.append(
            f"No new chunks to embed; {embeddings_total} eligible chunk(s) "
            f"already embedded for model '{model_name}'."
        )
    return diagnostics


def _chroma_client(config: Config, *, error: type[Exception]) -> object:
    chromadb = _require_chromadb(error=error)
    from chromadb.config import Settings

    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _require_chromadb(*, error: type[Exception]) -> object:
    try:
        import chromadb
    except ImportError as exc:
        raise error(
            "ChromaDB is required for vector features. Install dependencies with: "
            "uv sync"
        ) from exc
    return chromadb


def _existing_collection_names(client: object) -> set[str]:
    names: set[str] = set()
    for collection in client.list_collections():  # type: ignore[attr-defined]
        names.add(collection if isinstance(collection, str) else collection.name)
    return names


def _collection_vector_ids(chroma_collection: object) -> set[str]:
    result = chroma_collection.get(include=[])  # type: ignore[attr-defined]
    ids = result.get("ids", []) if isinstance(result, dict) else []
    return {str(vector_id) for vector_id in ids}


def _delete_vectors(chroma_collection: object, vector_ids: set[str]) -> None:
    ordered_ids = sorted(vector_ids)
    for start in range(0, len(ordered_ids), _VECTOR_ID_BATCH):
        chroma_collection.delete(  # type: ignore[attr-defined]
            ids=ordered_ids[start : start + _VECTOR_ID_BATCH]
        )


def _snippet(text: object) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if not normalized:
        return ""
    if len(normalized) <= _SNIPPET_CHAR_LIMIT:
        return normalized
    return normalized[: _SNIPPET_CHAR_LIMIT - 3].rstrip() + "..."


def _batches(rows: Sequence[sqlite3.Row], size: int) -> Iterable[list[sqlite3.Row]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
