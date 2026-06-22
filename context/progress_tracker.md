# Progress Tracker

## Current Status

* **Current Phase**: Phase 0: Design and Context Setup
* **Current Goal**: Convert the project idea into implementation-ready context documents and architecture notes.

The project has not yet implemented ingestion, indexing, retrieval, or answering code. The current work is design scaffolding.

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

- [ ] **Phase 1: Inventory Foundation**
  - [ ] Create project package structure.
  - [ ] Add dependencies with `uv`.
  - [ ] Implement filesystem crawler for `Courses`.
  - [ ] Create SQLite metadata database.
  - [ ] Classify files into indexed vs metadata-only categories.
  - [ ] Store skip reasons for images, binaries, archives, media, and unsafe artifacts.

- [ ] **Phase 2: Text Extraction**
  - [ ] Extract text from PDFs (PyMuPDF + optional Tesseract OCR fallback).
  - [ ] Extract text from PPTX slides (one chunk per slide with speaker notes).
  - [ ] Extract text from DOCX files.
  - [ ] Parse TXT and Markdown files.
  - [ ] Parse notebooks into one chunk per cell with truncated text outputs.
  - [ ] Parse Python code via AST into functions/classes/imports; regex fallback for R/C++/MATLAB.
  - [ ] Parse existing VTT transcripts.
  - [ ] Generate schema summaries for CSV/XLSX/JSON/JSONL/SQLite/DB files.

- [ ] **Phase 3: Indexing and Search**
  - [ ] Build chunk table and content metadata.
  - [ ] Add keyword search, preferably SQLite FTS5 for MVP.
  - [ ] Add vector embedding pipeline with ChromaDB (separate collections per logical index).
  - [ ] Keep separate logical indexes for documents, slides, notebooks, code, data schemas, and transcripts.
  - [ ] Add search run logging for searched/found/missing reporting.

- [ ] **Phase 4: Retrieval and Evidence Packets**
  - [ ] Implement two-stage query router (rule-based pre-filter + LLM fallback).
  - [ ] Implement hybrid retrieval over metadata, keyword, and semantic search.
  - [ ] Implement Reciprocal Rank Fusion for result merging.
  - [ ] Implement evidence packet schema.
  - [ ] Add coverage/weakness reporting.

- [ ] **Phase 5: Answering Interface**
  - [ ] Implement answer generator that uses only evidence packets.
  - [ ] Add citation rendering.
  - [ ] Add refusal/insufficient-evidence behavior.
  - [ ] Build FastAPI backend with HTML/JS frontend.

- [ ] **Phase 6: Evaluation and Hardening**
  - [ ] Create a hand-curated eval set of 15-20 questions covering each query type (start early, run incrementally).
  - [ ] Measure retrieval quality and citation quality.
  - [ ] Add regression tests for ingestion and evidence packet generation.
  - [ ] Optimize slow filesystem scans.
  - [ ] Keep feature specs updated as implementation decisions change.

- [ ] **Later / Optional**
  - [ ] Opt-in video/audio transcription.
  - [ ] Opt-in OCR for selected non-dataset image folders.
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
* [ ] Implement Phase 1 inventory foundation.
