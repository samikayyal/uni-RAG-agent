# Configuration and storage

## Current behavior

`load_config()` in `src/uni_rag_agent/config.py` loads `.env` plus process
environment into a typed `Config`; `validate_config()` enforces numeric bounds,
allowed providers, and atomic planner/answer provider-model pairs. Model values
remain nullable during general setup. `logging_config.py` emits console output
or sanitized JSONL run events. `storage/core.py` creates configured generated
directories, opens SQLite with foreign keys enabled, initializes/migrates the
schema, and reports table/FTS5/path health.

Default paths are `Courses/`, `data/uni_rag.sqlite`,
`data/indexes/vector/`, and `data/runs/`; all can be overridden by the
corresponding `UNI_RAG_*` path variables. Secrets (`GOOGLE_API_KEY`,
`NEBIUS_API_KEY`) are construction inputs only and are suppressed from safe
output and telemetry.

## Public entry points

- `uv run -m uni_rag_agent config check`
- `uv run -m uni_rag_agent storage init`
- `uv run -m uni_rag_agent storage check`
- Python: `load_config`, `validate_config`, `ensure_data_dirs`,
  `connect_sqlite`, `initialize_schema`, and `check_storage`.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/{config.py,logging_config.py,storage/core.py}`.
- Tests: `tests/test_config.py`, `tests/test_logging_config.py`,
  `tests/test_storage.py`, `tests/test_cli.py`.
- Generated: SQLite database, extracted-text directory, Chroma directory, and
  command JSONL logs under `data/`.
- Schema authority: [`storage/core.py`](../../src/uni_rag_agent/storage/core.py);
  this page intentionally lists no duplicate DDL.

## Invariants and boundaries

- `UNI_RAG_LLM_PROVIDER`/`UNI_RAG_LLM_MODEL` and answer equivalents are both
  set or both unset; retrieval/answer consumers enforce their own requirement.
- There is no embedding-provider environment variable and no default embedding
  model. Profile selection and provider construction belong to
  [keyword-and-vector-indexing.md](keyword-and-vector-indexing.md).
- Initialization is additive/migrating. Foreign-key relationships and the
  compatibility-sensitive deletion behavior in [architecture.md](../architecture.md)
  must remain intact.
- Safe config output reports operational values, not API keys; public `/config`
  additionally omits absolute paths.

Binding decisions: [DEC-009/021](../decisions.md#dec-009021--uv-and-environment-configuration),
[DEC-011](../decisions.md#dec-011--sqlite-authority-with-chroma-logical-indexes),
and [DEC-031/039](../decisions.md#dec-031039--explicit-reviewed-embedding-profiles).
