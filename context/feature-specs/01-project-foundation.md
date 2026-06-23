# Feature Spec 01: Project Foundation

## Purpose

Create the Python project foundation that every later module uses: package layout, command entrypoint, logging conventions, test layout, fixtures, and `uv` workflows.

This spec does not implement ingestion or retrieval behavior. It creates the structure that makes later specs consistent and testable.

## Depends On

- `context/project_overview.md`
- `context/architecture.md`
- `context/decisions.md`
- `pyproject.toml`

No feature spec depends on this one being fully implemented at runtime, but all later implementation work should follow its layout and command conventions.

## In Scope

- Create `src/uni_rag_agent/` with `config.py`, `logging_config.py`, and module folders for storage, inventory, extraction, indexing, retrieval, answering, tools, app, and evaluation.
- Create `tests/` with small fixture data under `tests/fixtures/`.
- Add a module entrypoint so commands use `uv run -m uni_rag_agent ...`.
- Define a small command registry or CLI dispatcher for future commands.
- Establish JSON-lines logging conventions for long-running operations.
- Add `.env.example` with non-secret placeholders and comments.
- Document developer commands in `README.md`.

## Out of Scope

- Real file inventory.
- Database migrations beyond invoking storage initialization hooks from spec 02.
- Extractors, indexers, retrievers, answer generation, or web routes.
- Adding dependencies not needed by the foundation itself.

## Public Interfaces

Command shape:

```powershell
uv run -m uni_rag_agent --help
uv run -m uni_rag_agent config check
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent index vector
uv run -m uni_rag_agent retrieve "query text"
uv run -m uni_rag_agent eval run
uv run -m uni_rag_agent app serve
```

Package layout:

```text
src/uni_rag_agent/
|-- __init__.py
|-- __main__.py
|-- cli.py
|-- config.py
|-- logging_config.py
|-- storage/
|-- inventory/
|-- extraction/
|-- indexing/
|-- retrieval/
|-- answering/
|-- tools/
|-- app/
`-- evaluation/
```

Test layout:

```text
tests/
|-- fixtures/
|   |-- courses_small/
|   |   |-- High Preformance Computing for Big Data/
|   |   |   |-- lecture_notes.md
|   |   |   |-- assignment.py
|   |   |   |-- dataset.csv
|   |   |   |-- diagram.png
|   |   |   |-- archive.zip
|   |   |   `-- setup.exe
|   |   `-- Information Retrieval/
|   |       |-- syllabus.txt
|   |       |-- search_demo.ipynb
|   |       |-- transcript.vtt
|   |       |-- model.pkl
|   |       |-- lecture.mp4
|   |       `-- vectors.bin
|   `-- extracted_samples/
|       |-- README.md
|       `-- sample_chunk.json
|-- test_cli.py
`-- test_logging_config.py
```

`tests/fixtures/courses_small/` is the committed synthetic course archive used by inventory and later ingestion tests. It must stay tiny, use exact folder names as shown above, and include both extractable files (`.md`, `.py`, `.csv`, `.txt`, `.ipynb`, `.vtt`) and metadata-only files (`.png`, `.zip`, `.exe`, `.pkl`, `.mp4`, `.bin`). The misspelled `High Preformance Computing for Big Data` folder is intentional and tests exact path preservation.

`tests/fixtures/extracted_samples/` is separate from the course archive. It stores small expected-output examples that later extraction, indexing, and evidence-packet tests can extend without pretending those files came directly from `Courses`.

The CLI should dispatch to module functions and return process exit codes. Long-running commands should log progress to console and JSONL under `data/runs/`.

## Storage and Schema Impact

No direct schema changes. This spec only creates the project structure that later storage modules fill.

It should ensure `data/` is never committed and `.env.example` is committed while `.env` remains ignored.

## Workflow

1. Create the package and test directories.
2. Add a minimal `__main__.py` that delegates to `cli.py`.
3. Add a CLI skeleton with help text and stub command groups.
4. Add `logging_config.py` helpers for console output and JSONL run logs.
5. Add fixture folders with tiny representative files for later specs.
6. Update `README.md` with `uv` commands and the MVP module order.
7. Add `.env.example` with expected variables but no secrets.

## Failure and Safety Rules

- CLI stubs must fail clearly when a command is not implemented yet.
- No command may write under `Courses`.
- Fixtures must be tiny and committed intentionally.
- Logs should not include API keys or full environment dumps.
- Commands should create `data/` lazily only when needed.

## Tests

- Automated: `uv run -m pytest tests/test_cli.py tests/test_logging_config.py`.
- Verify `uv run -m uni_rag_agent --help` exits successfully.
- Verify unknown commands return a non-zero exit code and useful message.
- Verify JSONL logging writes valid JSON objects to a temporary run directory.
- Verify `.env.example` exists and `.env` is ignored.

## Acceptance Criteria

- The package can be imported as `uni_rag_agent`.
- `uv run -m uni_rag_agent --help` works from the repo root.
- The repo has a clear test fixture convention.
- `README.md` documents only `uv` workflows.
- No foundation command scans or mutates `Courses`.
