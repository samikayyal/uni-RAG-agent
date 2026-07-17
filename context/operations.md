# Operations

Use PowerShell from the repository root and `uv` for every Python install or
run. Keep secrets in the ignored `.env` or the current shell; never put keys in
source, logs, reports, or committed fixtures.

## Setup and bootstrap

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

Vector indexing requires an explicit reviewed profile and matching extra:

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

Set `UNI_RAG_EMBEDDING_MODEL` or pass `--model`; there is no default embedding
model or provider variable. Configure `UNI_RAG_LLM_PROVIDER` and
`UNI_RAG_LLM_MODEL` together before `retrieve`, `evidence build`, `ask`, or
the API. Configure the separate `UNI_RAG_ANSWER_LLM_PROVIDER` and
`UNI_RAG_ANSWER_LLM_MODEL` pair when answering non-empty packets. See
[`config.py`](../src/uni_rag_agent/config.py) and [decisions.md](decisions.md)
for the complete environment contract.

## Daily surfaces

- Inspect inventory with `inventory summary`; extraction with `extract status`.
- Rebuild/search keyword data with `index keyword` and
  `search keyword "terms" --json`.
- Synchronize/search vectors with `index vector` and
  `search semantic "question" --model <profile> --json`.
- Use `retrieve "question" --model <profile> --json` for a non-persisting
  diagnostic. It does not write SQLite search/evidence rows, Chroma, or
  `Courses/` source files; the CLI still writes JSONL run telemetry under
  `data/runs/`.
- Use `evidence build "question" --model <profile>` then
  `evidence show --search-run-id <id>` for persisted coverage and evidence.
- Use `answer --evidence-packet-id <id>` or `ask "question" --model <profile>`
  for a stored answer and deterministic references. Add `--json` for one safe
  object.

Start the local web surface with `uv run -m uni_rag_agent app serve` (or pass
`--host`/`--port`). The API exposes `GET /health`, `GET /config`, `POST
/api/ask`, and read-only coverage, evidence-packet, and answer lookups. It does
not expose ingestion, indexing, evaluation, uploads, or source mutation.

## Generated state and safe rebuild

Normal generated state is under `data/`: `uni_rag.sqlite`, `extracted/`,
`indexes/vector/`, and JSONL run logs under `runs/`. Evaluation state and paired
reports are under `data/runs/eval/`; notebooks under `notebooks/` read generated
state and must remain read-only. `Courses/` is never a generated target.

For a clean rebuild, first confirm that the resolved `UNI_RAG_DATA_DIR` is the
repository’s intended generated directory, stop running app processes, and
remove only that generated directory. Then run, in order:

```text
storage init → inventory run → extract run → extract data-summaries →
index keyword → index vector --model <reviewed-profile>
```

Rebuild a single vector profile with `index vector --rebuild --model <profile>`;
this does not remove other profile collections. Do not delete or rewrite
`Courses/`, `evals/`, or source notebooks as part of a reset.

## Evaluation modes

`eval prepare-fixtures` builds and atomically activates isolated fixture state;
it requires the configured production embedding profile and does not touch the
normal archive database. Bare `eval run` and `eval run --fixtures` use the
committed `evals/fixtures.json` set and require prepared state. Reports are
paired timestamped JSON and Markdown files under `data/runs/eval/` and contain
safe scores, trace ids, failures, and p50/p95 timings—not raw queries, evidence,
model output, or credentials. `eval run --smoke-real-archive` is explicit and
reads `data/runs/eval/real-archive.json` against normal configured state; it
never traverses or prepares the archive implicitly.

For command arguments and exit behavior, use `uv run -m uni_rag_agent --help`;
the live parser in [`cli.py`](../src/uni_rag_agent/cli.py) is authoritative.
