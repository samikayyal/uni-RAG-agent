# Feature Spec 05: Data Schema Summaries

## Purpose

Summarize structured and semi-structured data files without embedding entire datasets. The output should help users find datasets by schema, columns, tables, row counts, and small samples while avoiding noisy or expensive full-data ingestion.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- [03-inventory-and-file-classification.md](03-inventory-and-file-classification.md)
- `context/architecture.md` tables: `data_summaries`, `extracted_documents`, `chunks`
- DEC-003, DEC-008, DEC-022, DEC-025

## In Scope

- Summarize `.csv`, `.xlsx`, `.json`, `.jsonl`, `.sqlite`, and `.db`.
- Capture column names, inferred types, row counts when cheap, sheet/table counts, table names, and first 5 rows or records.
- Generate deterministic `summary_text` with no LLM call.
- Store one summary chunk per data file, or one chunk per table/sheet when that is more useful.
- Avoid loading entire large datasets into memory.
- Add or update the stage EDA notebook for data-summary output once this feature is implemented.

## Out of Scope

- Full dataset embedding.
- Data cleaning, profiling, visualization, or statistical analysis beyond schema/sample metadata.
- Unsafe deserialization formats such as pickle/joblib.
- Running arbitrary SQL from user-provided files beyond safe metadata/sample inspection.
- Connecting to external databases.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent extract data-summaries
uv run -m uni_rag_agent extract data-summaries --file-id 123
```

Notebook:

```text
notebooks/data_schema_eda.ipynb
```

Create this notebook when data summaries are implemented. It should inspect `data_summaries`, data-schema chunks, and joined `files`/`courses` metadata for summary coverage, row/column/table/sheet counts, sample availability, large files, and failed data-summary rows.

Internal interfaces:

```python
summarize_data_files(config: Config) -> DataSummaryRunResult
summarize_csv(path: Path) -> DataSummary
summarize_xlsx(path: Path) -> DataSummary
summarize_json(path: Path) -> DataSummary
summarize_jsonl(path: Path) -> DataSummary
summarize_sqlite(path: Path) -> DataSummary
data_summary_to_chunk(summary: DataSummary) -> ChunkRecord
```

Summary fields:

```text
file_id
format
row_count
column_count
table_count
sheet_count
schema_json
sample_json
summary_text
```

## Storage and Schema Impact

Populate:

- `data_summaries`
- `extracted_documents`
- `chunks`

Update:

- `files.index_status` to `indexed` when a data summary and chunk are created.
- `files.index_status` to `failed` when a data file cannot be safely summarized.

The summary chunk should use:

```text
source_type=data_schema
location_type=schema
location_value=<sheet/table/file>
```

## Workflow

1. Select pending files with category `data_schema`.
2. Choose summarizer by extension.
3. Read only enough content for schema, row count when cheap, and first 5 rows/records.
4. For XLSX, summarize sheets separately when multiple sheets exist.
5. For SQLite/DB, inspect table names and safe schema metadata, then sample up to 5 rows per table.
6. Write `data_summaries`.
7. Convert summaries into chunks for keyword and vector indexing.
8. Emit per-file and aggregate summary counts.
9. Keep `notebooks/data_schema_eda.ipynb` aligned with summary fields, data-schema chunk behavior, command behavior, and failure/status semantics.

## Failure and Safety Rules

- Do not load very large files fully into memory.
- Do not execute user-defined code or extensions from databases.
- Do not run destructive SQL.
- Corrupted or unsupported files fail per file and do not stop the run.
- Sample values should be truncated to keep chunks bounded.
- Do not infer private secrets or credentials from sample rows; include only literal schema/sample text needed for retrieval.
- The EDA notebook must read generated app data only, must not mutate SQLite or `Courses`, and must not run arbitrary SQL from course database files.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated fixtures for CSV, XLSX with multiple sheets, JSON array, JSON object, JSONL, and SQLite with two tables.
- Verify first 5 rows/records are included and later rows are not.
- Verify row count behavior is correct when cheap.
- Verify schema JSON contains column names and inferred types.
- Verify large fixture simulation uses streaming/chunked reads.
- Verify no LLM provider is called.
- Verify `notebooks/data_schema_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional smoke: summarize a copied subset of real course data files in a temp directory.

## Acceptance Criteria

- `uv run -m uni_rag_agent extract data-summaries` creates deterministic summaries and chunks for supported data files.
- Data summaries are searchable by column, table, sheet, and sample terms.
- Full datasets are not embedded.
- Failures are per-file and explainable.
- `notebooks/data_schema_eda.ipynb` exists once this feature lands and can inspect data-summary coverage without mutating app data or source files.
