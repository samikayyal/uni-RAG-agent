# Extraction and data summaries

## Current behavior

`extract_pending_files()` processes pending text categories (`document`,
`slides`, `notebook`, `code`, `transcript`) with grouped format extractors.
PDF, PPTX, DOCX, text/Markdown, notebooks, Python/R/C++/MATLAB, and VTT
inputs produce source-aware chunks. Python code is parsed structurally where
possible; notebook outputs are bounded. Chunks are split at the shared
whitespace-token limit and carry `source_type`, title, token count, and a
location. Legacy `.doc`/`.ppt` and no-text files fail per file; other failures
are recorded without aborting unrelated files. Re-extraction replaces prior
chunks for the file.

`summarize_data_files()` handles CSV, XLSX, JSON, JSONL, SQLite, and DB files.
Persisted summaries and retrieval chunks are bounded: CSV and JSONL are iterated;
XLSX uses read-only workbook mode; SQLite/DB uses a read-only URI. JSON at or
below `MAX_JSON_FULL_LOAD_BYTES` (2,000,000 bytes) is fully parsed, while larger
JSON uses a bounded preview. See [`formats.py`](../../src/uni_rag_agent/extraction/data_summaries/formats.py)
and [`builders.py`](../../src/uni_rag_agent/extraction/data_summaries/builders.py).

## Public entry points

- `uv run -m uni_rag_agent extract run [--category <category>]`
- `uv run -m uni_rag_agent extract status`
- `uv run -m uni_rag_agent extract data-summaries [--file-id <id>]`
- Python: `extract_pending_files`, `load_extraction_status`,
  `summarize_data_files`, and the format-level helpers under
  `src/uni_rag_agent/extraction/`.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/extraction/{core,chunking,persistence,extractors}/`
  and `extraction/data_summaries/`.
- Tests: `tests/test_extraction.py`, `tests/test_data_summaries.py`.
- Notebooks: `notebooks/extraction_eda.ipynb` and
  `notebooks/data_schema_eda.ipynb` (read-only extraction/data-summary EDA).
- SQLite: `extraction_runs`, `extracted_documents`, `chunks`, and
  `data_summaries`; extraction/data-summary lifecycle JSONL is under
  `data/runs/`. `data/extracted/` is created as a reserved generated directory,
  but SQLite is the current extraction-output store.

## Invariants and failure boundaries

- Only inventory-pending files are processed; source files remain untouched.
- One file's extraction or summary failure produces a failed row/diagnostic and
  does not roll back successful files in the same run. A run-level storage
  failure is fatal.
- `chunks.file_id` and source-location fields are authoritative for citations.
  Replacing chunks must preserve the deletion semantics in
  [architecture.md](../architecture.md): historical result references are
  nulled and embedding mappings cascade.
- Data summaries are intentionally bounded metadata. Unsafe serialized/model
  artifacts are never sent to a parser.

Binding decisions: [DEC-012/019](../decisions.md#dec-012019--source-aware-bounded-chunks),
[DEC-022/025](../decisions.md#dec-022025--isolated-failures-and-safe-summaries),
and [DEC-002/003/006/007](../decisions.md#dec-002003006007--selective-non-destructive-source-admission).
