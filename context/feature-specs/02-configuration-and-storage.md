# Feature Spec 02: Configuration and Storage

## Purpose

Define configuration loading and initialize the generated local storage layout. This spec makes paths, provider settings, retrieval limits, SQLite, and ChromaDB persistence explicit before feature modules write data.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- `context/architecture.md`
- DEC-009, DEC-011, DEC-021

## In Scope

- Load settings from environment variables and `.env`.
- Provide a typed config object for all modules.
- Create and validate the `data/` directory layout.
- Initialize `data/uni_rag.sqlite` with the MVP schema from `context/architecture.md`.
- Define ChromaDB persistence under `data/indexes/vector/`.
- Define provider/model config keys without requiring a specific paid/cloud provider.
- Provide deterministic fake provider settings for tests.

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
llm_provider
llm_model
embedding_provider
embedding_model
embedding_dim
use_fake_llm
use_fake_embeddings
ocr_enabled
```

Environment variables:

```text
UNI_RAG_COURSES_ROOT
UNI_RAG_DATA_DIR
UNI_RAG_SQLITE_PATH
UNI_RAG_CHROMA_DIR
UNI_RAG_LOG_LEVEL
UNI_RAG_KEYWORD_TOP_K
UNI_RAG_SEMANTIC_TOP_K
UNI_RAG_FINAL_TOP_K
UNI_RAG_RRF_K
UNI_RAG_LLM_PROVIDER
UNI_RAG_LLM_MODEL
UNI_RAG_EMBEDDING_PROVIDER
UNI_RAG_EMBEDDING_MODEL
UNI_RAG_EMBEDDING_DIM
UNI_RAG_USE_FAKE_LLM
UNI_RAG_USE_FAKE_EMBEDDINGS
UNI_RAG_OCR_ENABLED
```

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
- `search_results`
- `evidence_packets`
- `answers`

The implementation may add a lightweight schema version table if needed, but it must not remove or rename the contracted tables without updating `context/architecture.md`.

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
- Missing API keys must not fail config checks when fake providers are enabled.
- Storage initialization must be idempotent.
- The implementation must never create files under `Courses`.
- `.env` values must not be logged verbatim if they look like secrets.

## Tests

- Automated tests use temporary directories, not the real `Courses`.
- Verify defaults resolve correctly from a temporary repo root.
- Verify `.env` overrides are loaded.
- Verify `storage init` creates the expected directories and all required tables.
- Verify FTS5 table creation works or returns a clear diagnostic if unavailable.
- Verify fake provider config requires no API keys.
- Optional smoke: run `uv run -m uni_rag_agent storage check` against the real repo without traversing `Courses`.

## Acceptance Criteria

- `uv run -m uni_rag_agent config check` reports paths and provider mode without secrets.
- `uv run -m uni_rag_agent storage init` creates `data/uni_rag.sqlite` and required directories.
- The SQLite schema matches the architecture contract.
- Chroma persistence path is configured under `data/indexes/vector/`.
- Tests do not require real LLM or embedding providers.
