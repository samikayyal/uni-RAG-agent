# Uni RAG Agent

Uni RAG Agent is a local course archive intelligence system for `Courses/`.
It is designed to inventory a mixed university archive, selectively extract useful
course knowledge, retrieve source-grounded evidence, and answer only from that
evidence.

The implementation contract lives in `context/`. Start with:

1. `context/project_overview.md`
2. `context/architecture.md`
3. `context/decisions.md`
4. `context/feature-specs/`
5. `context/progress_tracker.md`

## Setup

Use `uv` for all Python workflows.

```powershell
uv sync
uv run -m uni_rag_agent --help
uv run -m uni_rag_agent config check
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent storage check
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent inventory summary
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract status
```

Runtime configuration is loaded from `.env` with non-secret defaults documented
in `.env.example`. The `.env` file, `Courses/`, and generated `data/` directory
are ignored by git.

## Developer Commands

```powershell
uv run -m uni_rag_agent --help
uv run -m uni_rag_agent config check
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent storage check
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent inventory summary
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract run --category document
uv run -m uni_rag_agent extract status
uv run -m pytest tests/test_cli.py tests/test_config.py tests/test_storage.py tests/test_logging_config.py tests/test_inventory.py tests/test_extraction.py
```

Feature 02 storage commands create the generated local data layout:

```text
data/
|-- uni_rag.sqlite
|-- extracted/
|-- indexes/
|   `-- vector/
`-- runs/
```

Inventory commands crawl `Courses/`, classify every discovered file, and write
course/file metadata into SQLite without extracting content or mutating source
files:

```powershell
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent inventory summary
```

Extraction commands process pending text-like files from inventory, write
`extracted_documents` and `chunks`, preserve source locations, fail per file, and
leave data-schema files pending for the Feature 05 summary pipeline:

```powershell
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract run --category document
uv run -m uni_rag_agent extract status
```

Inventory and extraction CLI runs write lifecycle JSONL logs under `data/runs/`.

Remaining MVP command shapes are registered for later specs:

```powershell
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent index vector
uv run -m uni_rag_agent retrieve "query text"
uv run -m uni_rag_agent eval run
uv run -m uni_rag_agent app serve
```

The remaining command groups (`index`, `retrieve`, `eval`, and `app`) are stubs
until their feature specs are implemented. They should fail clearly and must not
scan or mutate `Courses/`.

## MVP Module Order

1. Project foundation
2. Configuration and storage
3. Inventory and file classification
4. Text extraction and chunking
5. Data schema summaries
6. Keyword indexing
7. Vector indexing
8. Query routing and hybrid retrieval
9. Evidence packets and coverage
10. Answering and citations
11. FastAPI HTML UI
12. Evaluation and hardening

## EDA Notebooks

Project-owned notebooks live under `notebooks/`. They are pandas-based,
read-only companions for generated app data, not pipeline implementation code.
They must not mutate `Courses/`, write to SQLite, rewrite indexes, execute
course scripts, or execute course notebooks.

Notebook outputs and execution counts should be cleared before commit. When a
stage changes the command, tables, JSON artifacts, status vocabulary, or
interpretation rules a notebook reads, update that notebook in the same change.

Create stage notebooks when the producing feature lands:

| Stage | Notebook |
| :--- | :--- |
| Inventory | `notebooks/inventory_eda.ipynb` |
| Text extraction | `notebooks/extraction_eda.ipynb` |
| Data schema summaries | `notebooks/data_schema_eda.ipynb` |
| Keyword indexing | `notebooks/keyword_index_eda.ipynb` |
| Vector indexing | `notebooks/vector_index_eda.ipynb` |
| Retrieval and evidence packets | `notebooks/retrieval_eda.ipynb` |
| Answering and citations | `notebooks/answering_eda.ipynb` |
| Evaluation and hardening | `notebooks/evaluation_eda.ipynb` |

## Fixtures

`tests/fixtures/courses_small/` is a tiny synthetic course archive for later
inventory and ingestion tests. It intentionally includes the misspelled course
folder `High Preformance Computing for Big Data` so path-preservation behavior
can be tested exactly.

`tests/fixtures/extracted_samples/` contains expected-output examples that do
not pretend to be source files from `Courses/`.
