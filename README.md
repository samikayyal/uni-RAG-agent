# Uni RAG Agent

Uni RAG Agent is a local course-archive intelligence system. It inventories
`Courses/`, extracts searchable text and bounded data summaries, maintains FTS5
and Chroma indexes, and answers from persisted source-grounded evidence with
coverage and citations.

## Quickstart

Use `uv` for every Python workflow:

```powershell
uv sync
uv run -m uni_rag_agent --help
uv run -m uni_rag_agent config check
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract data-summaries
uv run -m uni_rag_agent index keyword
```

For semantic retrieval, install the extra matching an explicit reviewed profile
and configure `UNI_RAG_EMBEDDING_MODEL` (or pass `--model`):

```powershell
# Local Hugging Face embeddings plus planner/answer integrations.
uv sync --extra embeddings --extra llm
uv run -m uni_rag_agent index vector --model BAAI/bge-m3

# Hosted Gemini/Nebius embeddings plus planner/answer integrations (shared install).
uv sync --extra embeddings-cloud --extra llm
# Gemini: set GOOGLE_API_KEY in the .env or current shell.
uv run -m uni_rag_agent index vector --model google/gemini-embedding-001
# Nebius: set NEBIUS_API_KEY in the .env or current shell.
uv run -m uni_rag_agent index vector --model Qwen/Qwen3-Embedding-8B
```

Planner settings (`UNI_RAG_LLM_PROVIDER` and `UNI_RAG_LLM_MODEL`) and, for a
non-empty packet, answer settings (`UNI_RAG_ANSWER_LLM_PROVIDER` and
`UNI_RAG_ANSWER_LLM_MODEL`) are separate pairs. There is no default embedding
model or provider variable.

## Ask, app, and evaluation

```powershell
# Non-persisting planner + hybrid retrieval.
uv run -m uni_rag_agent retrieve "Explain MapReduce" --model BAAI/bge-m3 --json

# Persist evidence, then answer it; or do both in one command.
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3
uv run -m uni_rag_agent evidence show --search-run-id 1
uv run -m uni_rag_agent answer --evidence-packet-id 1 --json
uv run -m uni_rag_agent ask "Explain MapReduce" --model BAAI/bge-m3 --json

# Local FastAPI/HTML question surface.
uv run -m uni_rag_agent app serve

# Fixture evaluation (prepare once; real archive is explicit).
uv run -m uni_rag_agent eval prepare-fixtures
uv run -m uni_rag_agent eval run
uv run -m uni_rag_agent eval run --smoke-real-archive
```

`retrieve` is read-only with respect to SQLite search/evidence rows, Chroma, and
`Courses/` source files; the CLI still writes JSONL run telemetry under
`data/runs/`. Use `evidence build` when persisted search and evidence are
required.

Generated state is under `data/` and source files under `Courses/` are
read-only from the application's perspective. Do not execute course code or
notebooks, commit credentials, or send course text to a hosted embedding
provider without considering privacy and cost. The web app does not ingest,
index, evaluate, upload, or mutate source files.

## Documentation and development

- [Context entrypoint](context/README.md) — task router and exact source/test paths.
- [Project overview](context/project_overview.md) — goal, scope, and stack.
- [Architecture](context/architecture.md) — boundaries, flows, and schema invariants.
- [Operations](context/operations.md) — setup, generated state, reset, and eval modes.
- [Binding decisions](context/decisions.md) and [glossary](context/glossary.md).
- [Current capabilities](context/capabilities/) — implemented behavior by concern.

Run focused tests with `uv run pytest tests/test_<area>.py`; the relevant
capability page names the source and test files. The live CLI help in
`src/uni_rag_agent/cli.py` is authoritative for arguments and exit behavior.
