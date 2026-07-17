# Context entrypoint

This directory documents the behavior that is live in `src/uni_rag_agent/`.
Read this file first, then open only the task-relevant capability page and the
binding decisions it links. Source and tests are authoritative for implemented
behavior; this layer records the public contract, safety boundaries, and where
to verify details.

## Task router

| Concern | Start here | Source and tests | Entry points / artifacts |
| --- | --- | --- | --- |
| Configuration, logging, storage | [capabilities/configuration-and-storage.md](capabilities/configuration-and-storage.md) | `src/uni_rag_agent/config.py`, `logging_config.py`, `storage/core.py`; `tests/test_config.py`, `test_logging_config.py`, `test_storage.py`, `test_cli.py` | `config check`, `storage init/check`; `data/uni_rag.sqlite`, `data/extracted/`, `data/indexes/vector/`, `data/runs/` |
| Source inventory and admission | [capabilities/inventory-and-source-admission.md](capabilities/inventory-and-source-admission.md) | `src/uni_rag_agent/inventory/`; `tests/test_inventory.py` | `inventory run/summary`; SQLite `courses`, `files`, `extraction_runs`; `data/runs/*.jsonl` |
| Text extraction and data summaries | [capabilities/extraction-and-data-summaries.md](capabilities/extraction-and-data-summaries.md) | `src/uni_rag_agent/extraction/`; `tests/test_extraction.py`, `tests/test_data_summaries.py` | `extract run/status`, `extract data-summaries`; SQLite `extracted_documents`, `chunks`, `data_summaries`; extraction run JSONL |
| Keyword and vector indexing | [capabilities/keyword-and-vector-indexing.md](capabilities/keyword-and-vector-indexing.md) | `src/uni_rag_agent/indexing/`; `tests/test_keyword_indexing.py`, `test_vector_indexing.py`, `test_embedding_providers.py` | `index keyword/vector`, `search keyword/semantic`; SQLite `chunk_fts`, `embeddings`; Chroma under `data/indexes/vector/`; index JSONL |
| Planning and hybrid retrieval | [capabilities/query-planning-and-retrieval.md](capabilities/query-planning-and-retrieval.md) | `src/uni_rag_agent/retrieval/{planner,metadata,rrf,core}.py`; `tests/test_query_planning.py`, `test_hybrid_retrieval.py` | `retrieve`; safe result JSON; non-persisting for SQLite search/evidence rows, Chroma, and `Courses/`; CLI JSONL run telemetry under `data/runs/` |
| Evidence, coverage, and answers | [capabilities/evidence-and-answering.md](capabilities/evidence-and-answering.md) | `src/uni_rag_agent/retrieval/{evidence,evidence_persistence,evidence_models}.py`, `src/uni_rag_agent/answering/`; `tests/test_evidence_packets.py`, `test_answering.py` | `evidence build/show`, `answer`, `ask`; SQLite `search_runs`, `search_result_sets`, `search_results`, `evidence_packets`, append-only `answers`; retrieval/answering notebooks |
| Web application | [capabilities/web-application.md](capabilities/web-application.md) | `src/uni_rag_agent/app/{api,service}.py`, `app/static/`; `tests/test_app.py` | `app serve`; local FastAPI routes and package-owned UI |
| Evaluation | [capabilities/evaluation.md](capabilities/evaluation.md) | `src/uni_rag_agent/evaluation/`; `tests/test_evaluation.py` | `eval prepare-fixtures`, `eval run`; `evals/fixtures.json`, `evals/sources/`, `data/runs/eval/`, evaluation EDA notebook |

Shared test assets: [`tests/fixtures/`](../tests/fixtures/),
[`tests/support.py`](../tests/support.py), [`tests/sqlite_helpers.py`](../tests/sqlite_helpers.py),
[`tests/embedding_doubles.py`](../tests/embedding_doubles.py), and
[`tests/subprocess_shim/`](../tests/subprocess_shim/).

## Cross-cutting work

For a change spanning stages, begin with [architecture.md](architecture.md),
then read each affected capability page and [decisions.md](decisions.md).
Trace the call path from the CLI in `src/uni_rag_agent/cli.py` or the API in
`src/uni_rag_agent/app/api.py` to the owning package and its focused tests.
Document a new externally observable invariant in `decisions.md`; update
[progress_tracker.md](progress_tracker.md) only when implementation or status
meaningfully changes. Keep generated state outside `Courses/` and use the live
DDL in [`storage/core.py`](../src/uni_rag_agent/storage/core.py) rather than
copying the schema into docs.

## Other anchors

- Product boundary and stack: [project_overview.md](project_overview.md).
- Domain vocabulary: [glossary.md](glossary.md).
- Setup, reset, generated state, and evaluation modes: [operations.md](operations.md).
- Root [README.md](../README.md) is the short user/developer quickstart.
