# Project overview

Uni RAG Agent is a local course-archive intelligence system. It inventories a
mixed `Courses/` archive, admits only useful source types, extracts searchable
text or safe data summaries, combines metadata/keyword/vector retrieval, and
answers from an immutable evidence packet with source locations and explicit
coverage. The project favors a defensible “not found in indexed materials”
result over an unsupported guess.

## What is in scope

- Inventory every discovered file while preserving exact course names and
  relative paths. Standalone images, media, archives, installers, binaries, and
  serialized/model artifacts remain metadata-only; checkpoint subtrees are
  excluded at source admission.
- Extract supported documents, slides, notebooks, code, transcripts, and data
  schemas into bounded chunks. Data files contribute schema/sample summaries,
  not full-dataset embeddings.
- Maintain SQLite metadata plus FTS5 keyword search and model-namespaced Chroma
  vector collections. Retrieval is hybrid and fuses backend ranks with RRF.
- Persist search runs, complete result sets, coverage, and one canonical packet
  when `evidence build` is requested. Generate strict packet-only answers with
  deterministic citations and append-only answer traces.
- Offer a local FastAPI/HTML question surface and a fixture-isolated evaluation
  harness. Ingestion, indexing, reset, and evaluation remain CLI-first.

Not in scope today: executing course code or notebooks, mutating `Courses/`,
automatic audio/video transcription, default standalone-image OCR/captioning,
knowledge-graph construction, reranking, or a cloud-hosted application.

## How it works

`inventory` admits files and records classification. `extract` (and
`extract data-summaries`) writes source-aware chunks. `index keyword` rebuilds
the current FTS5 projection; `index vector` synchronizes eligible chunks to the
selected reviewed embedding profile. `retrieve` asks the configured planner
for one validated query plan, runs deterministic metadata/keyword/semantic
search, and returns a non-persisting run. It does not write SQLite
search/evidence rows, Chroma, or `Courses/` source files; the CLI still writes
JSONL run telemetry under `data/runs/`. `evidence build` repeats that
orchestration with persistence and selects whole authoritative chunks. `answer`
consumes a stored packet; `ask` performs build plus answer in one command. See
[architecture.md](architecture.md) for boundaries and
[context/README.md](README.md) for source/test routing and the
[capabilities](capabilities/configuration-and-storage.md) pages for current
behavior.

## Stack

- Python 3.12+; `uv` is the package and command workflow.
- `python-dotenv` and a typed config in `src/uni_rag_agent/config.py`.
- SQLite (metadata, lifecycle records, FTS5) and ChromaDB (vectors).
- PyMuPDF, `python-pptx`, `python-docx`, `nbformat`, pandas/openpyxl, and
  optional Tesseract for scanned PDFs when enabled.
- LangChain interfaces for configured planner/answer chat models and reviewed
  embedding profiles. Local Hugging Face profiles use `embeddings`; hosted
  Gemini/Nebius profiles use `embeddings-cloud`; planner/answer integrations use
  `llm`.
- FastAPI with package-owned static HTML/JavaScript and a pytest suite.

Generated state belongs under `data/` (normally ignored by git); source files
under `Courses/` are read-only from the application's perspective. Binding
runtime and schema constraints are in [decisions.md](decisions.md).
