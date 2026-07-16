# Feature Spec 04: Text Extraction and Chunking

## Purpose

Extract text-like course knowledge from pending inventory files and store retrieval-ready chunks with source locations. This spec covers documents, slides, notebooks, code, plain text, Markdown, and existing transcripts.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- [03-inventory-and-file-classification.md](03-inventory-and-file-classification.md)
- `context/architecture.md` tables: `extraction_runs`, `extracted_documents`, `chunks`
- DEC-012, DEC-015, DEC-016, DEC-019, DEC-022, DEC-024

## In Scope

- Extract PDFs with PyMuPDF.
- Optionally use Tesseract OCR for scanned PDFs only when configured and installed.
- Extract PPTX slide text and speaker notes.
- Extract DOCX paragraph/table text.
- Extract TXT and Markdown as plain text chunks.
- Parse notebooks into one chunk per markdown/code cell with truncated text outputs.
- Extract Python code with `ast` into imports, functions, classes, and module-level text.
- Extract R, C++, header, and MATLAB files with regex/whole-file fallback.
- Parse existing VTT transcript files with timestamp locations.
- Apply natural-boundary chunking with sub-chunking for overlarge units.
- Persist extractor status and per-file failures.
- Add or update the stage EDA notebook for extraction output once this feature is implemented.

## Out of Scope

- Legacy `.doc` and `.ppt` conversion.
- Standalone image OCR or captioning.
- Full video/audio transcription.
- Executing notebooks or old course scripts.
- Loading pickle/joblib/model artifacts.
- Data schema summaries for CSV/XLSX/JSON/SQLite; see spec 05.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract run --category document
uv run -m uni_rag_agent extract status
```

Notebook:

```text
notebooks/extraction_eda.ipynb
```

Create this notebook when extraction is implemented. It should inspect `extraction_runs`, `extracted_documents`, `chunks`, and joined `files`/`courses` metadata for extraction yield, failures, text length, chunk counts, source-type coverage, source-location coverage, and matplotlib-backed diagnostic plots for counts, distributions, and failure hotspots.

Internal interfaces:

```python
extract_pending_files(config: Config, category: str | None = None) -> ExtractionRunResult
extract_file(file_record: FileRecord) -> ExtractedDocument
chunk_extracted_document(file_record: FileRecord, extracted: ExtractedDocument) -> list[ChunkRecord]
```

Extractor result shape:

```text
file_id
extractor_name
extractor_version
status
text_length
chunk_count
metadata
error
chunks[]
```

Chunk record shape:

```text
file_id
extracted_document_id
chunk_uid
source_type
chunk_index
title
text
token_count
location_type
location_value
metadata_json
```

`source_type` must use the logical chunk vocabulary from `context/architecture.md`: `document`, `slides`, `notebook`, `code`, `data_schema`, or `transcript`. Do not store file extensions such as `.pdf` or `.pptx` in `source_type`; those remain in `files.extension`.

## Storage and Schema Impact

Populate:

- `extraction_runs`
- `extracted_documents`
- `chunks`

Update:

- `files.index_status` to `indexed` for successful extraction.
- `files.index_status` to `failed` for extractable files that fail.
- `files.reason_not_indexed` for unsupported legacy formats and failed scanned PDFs without OCR.

Chunk `location_type` values:

```text
page
slide
docx_section
text_section
markdown_section
notebook_cell
function
class
module
timestamp
subchunk
```

## Workflow

1. Select files with `index_status=pending` and categories handled by this spec.
2. Open each file with the appropriate extractor.
3. Produce natural chunks:
   - PDF: one chunk per page.
   - PPTX: one chunk per slide, including speaker notes when available.
   - DOCX: paragraph/table groups preserving document order.
   - TXT/MD: section or size-bounded chunks.
   - Notebook: one chunk per markdown or code cell; append text outputs truncated to about 500 characters.
   - Python: imports plus functions/classes using `ast`.
   - Other code: regex function split or whole-file fallback.
   - VTT: timestamp blocks.
4. Sub-chunk any unit over the configured max token limit.
5. Persist `extracted_documents` and `chunks` transactionally per file.
6. Continue after per-file failures and report summary counts.
7. Keep `notebooks/extraction_eda.ipynb` aligned with extraction status fields, chunk fields, source types, location types, diagnostic plots, and command behavior.

## Failure and Safety Rules

- Do not execute course code, notebooks, installers, archives, or media files.
- Do not mutate source files under `Courses`.
- Do not process files under `.ipynb_checkpoints`; inventory excludes them and
  inventory is the source admission boundary for extraction.
- Password-protected, corrupted, or unsupported files fail per file with a detailed error.
- OCR is disabled unless explicitly configured. If a scanned PDF needs OCR and OCR is disabled or unavailable, mark it failed with reason `scanned PDF, OCR not available`.
- Do not include notebook image/binary outputs.
- Truncate long text outputs and long error tracebacks.
- The EDA notebook must read generated app data only, must not mutate SQLite or `Courses`, and must not execute course files or course notebooks.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated fixture tests for PDF, PPTX, DOCX, TXT, MD, IPYNB, Python, R/C++/MATLAB fallback, and VTT.
- Verify source locations are preserved for page, slide, notebook cell, function/class, and timestamp chunks.
- Verify per-file failure does not abort the extraction run.
- Verify unsupported `.doc` and `.ppt` are marked failed with the expected reason.
- Verify no extractor executes fixture code.
- Verify overlarge fixture content is sub-chunked.
- Verify `notebooks/extraction_eda.ipynb`, once created, is valid notebook JSON, imports pandas/matplotlib successfully, documents its read-only safety boundary, and includes plots for extraction outcomes, chunk coverage, and failure reasons.
- Optional smoke: extract a tiny copied subset of representative real course files into a temp database.

## Acceptance Criteria

- `uv run -m uni_rag_agent extract run` processes pending supported text-like files and writes chunks.
- Every successful extraction has an `extracted_documents` row and one or more `chunks`.
- Every failed extraction has a clear status and error.
- Chunks contain enough source location metadata for citations.
- `notebooks/extraction_eda.ipynb` exists once this feature lands and can inspect extraction/chunk coverage with tabular views and plots without mutating app data or source files.
- Automated tests do not require the full `Courses` archive.
