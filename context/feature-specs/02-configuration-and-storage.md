# Feature Spec 02: Configuration and Storage

## Purpose

Define configuration loading and initialize the generated local storage layout. This spec makes paths, optional model settings, retrieval limits, SQLite, ChromaDB persistence, and provider-inferred embedding profiles explicit before feature modules write data.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- `context/architecture.md`
- DEC-009, DEC-011, DEC-021, DEC-039

## In Scope

- Load settings from environment variables and `.env`.
- Provide a typed config object for all modules.
- Create and validate the `data/` directory layout.
- Initialize `data/uni_rag.sqlite` with the MVP schema from `context/architecture.md`.
- Define ChromaDB persistence under `data/indexes/vector/`.
- Define optional LLM provider/model settings without requiring a specific paid/cloud provider.
- Define the optional reviewed embedding model setting used by Feature 07 and
  resolve its provider from the explicit profile registry.
- Document the provider-specific embedding extras and credential names without
  loading any SDK during configuration.

## Out of Scope

- Running inventory or extraction.
- Creating Chroma collections with real embeddings.
- Calling any LLM or embedding API.
- Implementing schema migrations beyond MVP initialization.

## Public Interfaces

Configuration object fields:

```text
courses_root
data_dir
sqlite_path
chroma_dir
runs_dir
log_level
keyword_top_k
semantic_top_k
final_top_k
rrf_k
metadata_top_k
semantic_query_limit
query_plan_min_confidence
filename_fuzzy_threshold
path_fuzzy_threshold
evidence_max_tokens
llm_provider: str | None
llm_model: str | None
embedding_model: str | None
ocr_enabled
google_api_key: str | None  # private/repr-suppressed; omitted from safe output
nebius_api_key: str | None  # private/repr-suppressed; omitted from safe output
```

Environment variables:

```text
UNI_RAG_COURSES_ROOT
UNI_RAG_DATA_DIR
UNI_RAG_SQLITE_PATH
UNI_RAG_CHROMA_DIR
UNI_RAG_RUNS_DIR
UNI_RAG_LOG_LEVEL
UNI_RAG_KEYWORD_TOP_K
UNI_RAG_SEMANTIC_TOP_K
UNI_RAG_FINAL_TOP_K
UNI_RAG_RRF_K
UNI_RAG_METADATA_TOP_K
UNI_RAG_SEMANTIC_QUERY_LIMIT
UNI_RAG_QUERY_PLAN_MIN_CONFIDENCE
UNI_RAG_FILENAME_FUZZY_THRESHOLD
UNI_RAG_PATH_FUZZY_THRESHOLD
UNI_RAG_EVIDENCE_MAX_TOKENS
UNI_RAG_LLM_PROVIDER
UNI_RAG_LLM_MODEL
UNI_RAG_EMBEDDING_MODEL
UNI_RAG_OCR_ENABLED
```

The merged environment may also contain the provider credentials
`GOOGLE_API_KEY` and `NEBIUS_API_KEY`. They are construction inputs, not model
selection fields, and must never appear in `Config.as_safe_dict()`, SQLite,
telemetry, or diagnostics.

`UNI_RAG_EMBEDDING_MODEL` is the only embedding-selection environment variable.
The selected identifier is resolved through the reviewed profile registry, which
infers whether construction is local Hugging Face, hosted Google Gemini, or
hosted Nebius Token Factory. `UNI_RAG_EMBEDDING_PROVIDER` does not exist and is
not part of `Config`.

Provider-specific optional installation paths are selected by the profile:

```powershell
# Local Hugging Face profiles.
uv sync --extra embeddings

# Hosted Google Gemini and Nebius Token Factory profiles.
uv sync --extra embeddings-cloud

# Query-planner/answer LLM integrations retain their existing semantics.
uv sync --extra llm
```

The embedding SDKs are loaded lazily by Feature 07, not during `config check`.
Hosted credentials are read by the selected provider construction path from
`GOOGLE_API_KEY` or `NEBIUS_API_KEY`. If the implementation retains them in
private, representation-suppressed config fields for construction, they must be
omitted from safe configuration output, SQLite, telemetry, and user-facing
diagnostics.

Commands:

```powershell
uv run -m uni_rag_agent config check
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent storage check
```

Internal interfaces:

```python
load_config() -> Config
ensure_data_dirs(config: Config) -> None
connect_sqlite(config: Config) -> sqlite3.Connection
initialize_schema(connection: sqlite3.Connection) -> None
check_storage(config: Config) -> StorageCheckResult
```

## Storage and Schema Impact

Create these paths:

```text
data/
|-- uni_rag.sqlite
|-- extracted/
|-- indexes/
|   `-- vector/
`-- runs/
```

Initialize the schema tables from `context/architecture.md`:

- `courses`
- `files`
- `extraction_runs`
- `extracted_documents`
- `chunks`
- `chunk_fts`
- `embeddings`
- `data_summaries`
- `search_runs`
- `search_result_sets`
- `search_results`
- `evidence_packets`
- `answers`

The implementation may add a lightweight schema version table if needed, but it must not remove or rename the contracted tables without updating `context/architecture.md`.

Feature 09 initialization also performs targeted, data-preserving migrations:
legacy `search_runs.router_output_json` becomes `query_plan_json`,
`retrieval_settings_json` is added with a safe empty-object default when absent,
the composite result lookup index is created, and a unique packet-per-run index
is created only after duplicate packet rows have been rejected explicitly.
Each completed raw backend call also receives a `search_result_sets` completion
envelope, including successful zero-row calls; the envelope is committed in the
same transaction as its non-empty `search_results` rows.

## Workflow

1. Load `.env` if present.
2. Resolve defaults relative to the repo root.
3. Validate `Courses` exists but do not traverse it.
4. Create `data/`, `data/extracted/`, `data/indexes/vector/`, and `data/runs/`.
5. Open SQLite and initialize the MVP schema.
6. Check that SQLite FTS5 is available.
7. Report config and storage health without printing secrets.

## Failure and Safety Rules

- Missing `Courses` root should fail `config check` with a clear path-specific error.
- Unset optional model/provider values must not fail config checks. In particular,
  an unset `UNI_RAG_EMBEDDING_MODEL` is allowed until a vector/retrieval command
  requires an explicit model selection.
- `llm_provider` and `llm_model` are an atomic pair: both unset or both
  nonblank. Providers are exactly `openai`, `anthropic`, `gemini`, or `ollama`.
- Retrieval tuning values use Feature 08 defaults: metadata top-K 20, semantic
  query limit 3, query-plan confidence 0.60, and metadata fuzzy thresholds 85/90.
- Evidence token budget defaults to 12,000 whitespace-estimated tokens and must
  be a positive integer; blank, non-integer, zero, and negative values fail
  clearly.
- Storage initialization must be idempotent.
- The implementation must never create files under `Courses`.
- `.env` values must not be logged verbatim if they look like secrets.
- Configuration must not infer a provider from a free-form environment variable;
  provider inference is registry-based and the supported profile list is
  explicit.

## Tests

- Automated tests use temporary directories, not the real `Courses`.
- Verify defaults resolve correctly from a temporary repo root.
- Verify `.env` overrides are loaded.
- Verify `storage init` creates the expected directories and all required tables.
- Verify FTS5 table creation works or returns a clear diagnostic if unavailable.
- Verify unset optional model/provider values are reported as `null`/`None` without invented defaults.
- Optional smoke: run `uv run -m uni_rag_agent storage check` against the real repo without traversing `Courses`.

## Acceptance Criteria

- `uv run -m uni_rag_agent config check` reports paths and optional model/provider values without secrets.
- `uv run -m uni_rag_agent storage init` creates `data/uni_rag.sqlite` and required directories.
- The SQLite schema matches the architecture contract.
- Chroma persistence path is configured under `data/indexes/vector/`.
- Tests do not require real LLM or embedding providers.
