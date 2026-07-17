# Architecture

The application is a local, staged pipeline. `Courses/` is source data and is
never modified by normal commands. SQLite is authoritative for file identity,
chunk text, search/evidence history, and answer traces; Chroma stores vectors
that are reconciled through SQLite mappings. The complete executable schema is
[`src/uni_rag_agent/storage/core.py`](../src/uni_rag_agent/storage/core.py).

## Boundaries and flows

### Ingestion and indexing

1. `inventory` walks the configured courses root, pruning any path component
   named `.ipynb_checkpoints`, classifies by extension, hashes changed files,
   and upserts `courses`/`files`. Missing files are soft-marked; source files
   are not deleted.
2. `extract` handles supported text-like files and stores an
   `extracted_documents` row plus source-aware `chunks`. A file failure is
   isolated and recorded in the extraction run. `extract data-summaries` reads
   CSV/XLSX/JSON/JSONL/SQLite/DB schemas and samples, stores `data_summaries`,
   and emits `data_schema` chunks.
3. `index keyword` rebuilds the SQLite FTS5 `chunk_fts` projection from current
   indexed chunks. `index vector` embeds the same eligible chunk set into one
   model-namespaced Chroma collection per logical index and records `embeddings`
   mappings. Both indexers share current-file eligibility and the canonical
   logical-index/source-type mapping in `src/uni_rag_agent/search_contracts.py`.

### Planning, retrieval, evidence, and answering

1. `retrieve` calls the configured chat model once through
   `retrieval/planner.py` and validates a `QueryPlan` (or a valid unsupported
   plan). The deterministic retriever runs metadata, keyword, and semantic
   backends with planned filters, then merges ranked results with RRF. It is
   non-persisting with respect to SQLite search/evidence rows, Chroma, and
   `Courses/` source files; the CLI still writes JSONL run telemetry under
   `data/runs/`.
2. `evidence build` uses the same planner/retriever behind a persistence
   recorder. It stores the plan/settings, every complete raw result set (even a
   successful empty set), fused rows, coverage, and one canonical packet of
   whole current chunks. `evidence show` and the API load that packet without
   re-running retrieval.
3. `answer` loads a packet and asks a separate configured answer model for one
   strict JSON object. The application validates packet-relative citations,
   renders stable `[E1]` markers and references, and appends an `answers` row.
   Empty evidence and budget exhaustion produce deterministic no-provider
   responses. `ask` composes build and answer; a failed answer does not erase a
   packet already persisted.

## Module map

| Layer | Modules | Responsibility |
| --- | --- | --- |
| CLI composition | `src/uni_rag_agent/__main__.py`, `cli.py`, `cli_commands/`, `cli_support/` | Thin parser/dispatcher, cohesive command-family handlers, shared renderers, and telemetry adapters |
| Configuration | `config.py`, `logging_config.py` | Typed environment loading and safe JSONL telemetry |
| Storage | `storage/core.py` | Paths, SQLite connections, schema initialization/migrations, health checks |
| Admission | `inventory/{core,file_io,classification,models}.py` | Crawl, checkpoint pruning, classification, idempotent inventory and soft-missing state |
| Extraction | `extraction/{core,chunking,persistence,extractors,data_summaries}/` | Format adapters, bounded chunking, per-file lifecycle, schema/sample summaries |
| Search contracts | `search_contracts.py` | Canonical logical-index/source-type mapping and derived collections/inverse lookups |
| Indexing | `indexing/{eligibility,keyword,vector,profiles,embedding_providers}/` | FTS5 projection/search, reviewed embedding construction, Chroma reconciliation/search |
| Retrieval | `retrieval/{planner,metadata,rrf,core,evidence,evidence_persistence,evidence_models}.py` | Query-plan validation, backend orchestration, RRF provenance, persistence and coverage |
| Answering | `answering/{core,persistence,session,audit,providers}.py` | Packet-only generation, citation/rendering validation, append-only traces, planner-only memory |
| Web | `app/{api,service}.py`, `app/static/` | Provider-lazy FastAPI routes, bounded sessions, timeout-safe ask orchestration, UI |
| Evaluation | `evaluation/{core,models}.py` | Fixture state preparation/validation, deterministic scoring, safe JSON/Markdown reports |

Focused tests mirror these boundaries under `tests/`; use
[context/README.md](README.md) to route a task to exact files.

## Authoritative stores and compatibility invariants

- `courses.name` and `courses.path`, and `files.path`/`relative_path`, preserve
  the archive's exact spelling. A file is associated with at most one course;
  inventory updates existing identities rather than creating path aliases.
- `files.index_status` is the source-admission state. Normal indexes and all
  answer-time retrieval join only `indexed` files. Missing, failed, skipped,
  metadata-only, and stale chunks stay inspectable but cannot leak into normal
  answers. `.ipynb_checkpoints` has no inventory row at all.
- `chunks.file_id` anchors text to a source file. `chunk_uid` is unique;
  `source_type`, `location_type`, and `location_value` carry citation context.
  Re-extraction replaces a file's chunks. `search_results.chunk_id` uses
  `ON DELETE SET NULL` so historical rows remain auditable; `embeddings.chunk_id`
  uses `ON DELETE CASCADE` so vector mappings cannot outlive chunks.
- Logical indexes map one-to-one to chunk source types: `document_index`,
  `slides_index`, `notebook_index`, `code_index`, `data_schema_index`, and
  `transcript_index`. FTS5 and Chroma must use the same current eligible set.
- A vector mapping is unique per chunk/backend/physical collection. Physical
  Chroma names include provider, canonical model, dimension, metric, and a
  stable model slug; the accepted Gemini alias is canonicalized before any
  storage or telemetry.
- A `search_runs` row owns `search_result_sets`, `search_results`, and at most
  one `evidence_packets` row. Result-set completion envelopes and packet JSON
  are immutable audit material once written. `answers` references a packet and
  is append-only.
- Evidence packet locations and citations must match current authoritative chunk
  identity and nonblank text. Answer citations must resolve to packet positions
  (or the compatibility `chunk:<id>` alias) and are revalidated at persistence.

Schema changes must preserve these relationships and deletion behaviors. Add a
binding decision before changing a public field, table relationship, or
generated-artifact boundary; do not duplicate the full DDL in documentation.
