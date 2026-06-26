# Progress Tracker

## Current Status

* **Current Phase**: Phase 2: Text Extraction
* **Current Goal**: Implement data schema summaries for pending structured files.

The project now has the Feature 01 foundation package, CLI dispatcher, typed configuration loader, JSONL logging helper, fixture convention, `.env.example`, README developer commands, Feature 02 storage initialization, SQLite MVP schema creation, storage health checks, focused foundation/storage tests, negative-path config/storage diagnostics coverage, pytest discovery constrained to this project's `tests/` directory, Feature 03 inventory/file classification with CLI commands, SQLite upserts, idempotent reruns, timestamp-first/hash-on-change behavior, metadata-only skip reasons, missing-file soft marking, current course-total resets, explicit SQLite connection closing, accurate inventory run metrics, inventory summaries, a pandas-based read-only inventory EDA notebook under `notebooks/`, and Feature 04 text extraction/chunking with CLI commands, per-file extraction failures, source-location chunks, stale chunk/index cleanup on re-extraction, optional scanned-PDF OCR gating, legacy `.doc`/`.ppt` failure reasons, and a pandas-based read-only extraction EDA notebook. The project has not yet implemented data schema summaries, indexing, retrieval, or answering behavior.

## Project Roadmap

- [x] **Phase 0: Design and Context Setup**
  - [x] Profile `Courses` folder file types, size, and major data characteristics.
  - [x] Record core project overview and product direction.
  - [x] Decide to ignore images for RAG and keep them metadata-only.
  - [x] Define researcher/evidence-packet/answerer workflow.
  - [x] Fill context templates.
  - [x] Add technical architecture document.
  - [x] Complete design review: resolve LLM framework, vector store, chunking, routing, interface, and 20+ implementation decisions.
  - [x] Align context docs around accepted stack and MVP boundaries.
  - [x] Add module-level feature specifications under `context/feature-specs/`.
  - [x] Clarify chunk `source_type` as a logical category and map source types to logical indexes.
  - [x] Clarify OCR scope: standalone images stay metadata-only, scanned PDFs may use configured Tesseract fallback.
  - [x] Align Feature 01 foundation package layout and fixture contracts with architecture.

- [x] **Phase 1: Inventory Foundation**
  - [x] Create project package structure.
  - [x] Add dependencies with `uv`.
  - [x] Implement Feature 02 configuration and storage initialization.
  - [x] Create SQLite metadata database.
  - [x] Implement filesystem crawler for `Courses`.
  - [x] Classify files into indexed vs metadata-only categories.
  - [x] Store skip reasons for images, binaries, archives, media, and unsafe artifacts.
  - [x] Add a pandas-based read-only EDA notebook for analyzing SQLite inventory output after `inventory run`.

- [ ] **Phase 2: Text Extraction**
  - [x] Extract text from PDFs (PyMuPDF + optional Tesseract OCR fallback).
  - [x] Extract text from PPTX slides (one chunk per slide with speaker notes).
  - [x] Extract text from DOCX files.
  - [x] Parse TXT and Markdown files.
  - [x] Parse notebooks into one chunk per cell with truncated text outputs.
  - [x] Parse Python code via AST into functions/classes/imports; regex fallback for R/C++/MATLAB.
  - [x] Parse existing VTT transcripts.
  - [ ] Generate schema summaries for CSV/XLSX/JSON/JSONL/SQLite/DB files.
  - [x] Add `notebooks/extraction_eda.ipynb` when text extraction lands.
  - [ ] Add `notebooks/data_schema_eda.ipynb` when data summaries land.

- [ ] **Phase 3: Indexing and Search**
  - [ ] Build chunk table and content metadata.
  - [ ] Add keyword search, preferably SQLite FTS5 for MVP.
  - [ ] Add vector embedding pipeline with ChromaDB (separate collections per logical index).
  - [ ] Keep separate logical indexes for documents, slides, notebooks, code, data schemas, and transcripts.
  - [ ] Add search run logging for searched/found/missing reporting.
  - [ ] Add `notebooks/keyword_index_eda.ipynb` when keyword indexing lands.
  - [ ] Add `notebooks/vector_index_eda.ipynb` when vector indexing lands.

- [ ] **Phase 4: Retrieval and Evidence Packets**
  - [ ] Implement two-stage query router (rule-based pre-filter + LLM fallback).
  - [ ] Implement hybrid retrieval over metadata, keyword, and semantic search.
  - [ ] Implement Reciprocal Rank Fusion for result merging.
  - [ ] Implement evidence packet schema.
  - [ ] Add coverage/weakness reporting.
  - [ ] Add `notebooks/retrieval_eda.ipynb` when retrieval/evidence traces are persisted.

- [ ] **Phase 5: Answering Interface**
  - [ ] Implement answer generator that uses only evidence packets.
  - [ ] Add citation rendering.
  - [ ] Add refusal/insufficient-evidence behavior.
  - [ ] Add `notebooks/answering_eda.ipynb` when answer traces are persisted.
  - [ ] Build FastAPI backend with HTML/JS frontend.

- [ ] **Phase 6: Evaluation and Hardening**
  - [ ] Create a hand-curated eval set of 15-20 questions covering each query type (start early, run incrementally).
  - [ ] Add `notebooks/evaluation_eda.ipynb` when evaluation reports are written.
  - [ ] Maintain existing read-only EDA notebooks whenever their source commands, tables, JSON artifacts, status vocabulary, or interpretation rules change.
  - [ ] Measure retrieval quality and citation quality.
  - [ ] Add regression tests for ingestion and evidence packet generation.
  - [ ] Optimize slow filesystem scans.
  - [ ] Keep feature specs updated as implementation decisions change.

- [ ] **Later / Optional**
  - [ ] Opt-in video/audio transcription.
  - [ ] Opt-in standalone image OCR/captioning for selected non-dataset folders.
  - [ ] Knowledge graph over courses, topics, assignments, projects, datasets, and code.
  - [ ] Portfolio/resume mode.
  - [ ] Study/quiz mode.

## Active Task List

- `[ ]` Pending task
- `[/]` In-progress task
- `[x]` Completed task

### Active Tasks

* [x] Fill `context/project_overview.md` from the root project overview.
* [x] Fill `context/decisions.md` with accepted design decisions.
* [x] Fill `context/progress_tracker.md` with roadmap and current status.
* [x] Add `context/architecture.md` with technical storage, schema, and pipeline details.
* [x] Complete design review session resolving framework, vector store, chunking, routing, interface, and implementation decisions.
* [x] Review and patch planning-doc drift before module spec creation.
* [x] Create `context/feature-specs/` with MVP module implementation contracts.
* [x] Clarify logical `source_type` values across architecture and MVP specs.
* [x] Clarify OCR wording across overview, architecture, decisions, and feature specs.
* [x] Align Feature 01 package layout, fixture contract, and logging helper naming with `context/architecture.md`.
* [x] Implement Feature 01 project foundation: package skeleton, CLI stubs, config loader, JSONL logging helper, fixtures, `.env.example`, README, and tests.
* [x] Implement Feature 02 configuration and storage: typed config contract, safe `.env` reporting, generated data directories, SQLite MVP schema, FTS5 check, storage CLI, pytest discovery scoped to `tests/`, positive-path tests, and negative-path diagnostics tests.
* [x] Implement Feature 03 inventory and file classification: course discovery, streaming file crawl, spec category mapping, SQLite course/file upserts, inventory run records, metadata-only reasons, idempotent unchanged-file reruns, hash-on-change behavior, missing-file soft marking, summary CLI, and focused tests.
* [x] Address Feature 03 review findings: close SQLite connections explicitly, reset stale course totals when course folders disappear, keep inventory run `files_indexed` metrics accurate, and add regression tests.
* [x] Add Feature 03 inventory failure-path regression tests for file stat failures, hash failures, nested directory listing diagnostics, and failed root-listing run records.
* [x] Run Brooks full sweep on Features 01-03: fix transient hash-failure recovery, align the Feature 02 runs-dir environment variable contract, and verify the current safety net.
* [x] Add `notebooks/inventory_eda.ipynb` for pandas-based read-only EDA over `data/uni_rag.sqlite` after `inventory run`, and document the notebook strategy across `context/`.
* [x] Expand the notebook roadmap across applicable stages and record that existing notebooks must be updated when their source artifact contracts change.
* [x] Implement Feature 04 text extraction and chunking.
* [ ] Implement Feature 05 data schema summaries.
