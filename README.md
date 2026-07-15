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
uv run -m uni_rag_agent extract data-summaries
uv run -m uni_rag_agent extract status
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent search keyword "mapreduce"
uv run -m uni_rag_agent search keyword "mapreduce" --json
uv run -m uni_rag_agent retrieve "Explain MapReduce from Information Retrieval" --model BAAI/bge-m3
uv run -m uni_rag_agent retrieve "Find the Information Retrieval syllabus" --model BAAI/bge-m3 --debug
uv run -m uni_rag_agent retrieve "query text" --model BAAI/bge-m3 --json
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3
uv run -m uni_rag_agent evidence show --search-run-id 1
uv run -m uni_rag_agent answer --evidence-packet-id 1
uv run -m uni_rag_agent ask "Explain MapReduce from my courses" --model BAAI/bge-m3
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
uv run -m uni_rag_agent extract data-summaries
uv run -m uni_rag_agent extract data-summaries --file-id 123
uv run -m uni_rag_agent extract status
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent index keyword --rebuild
uv run -m uni_rag_agent index vector --model BAAI/bge-m3 --collection document_index
uv run -m uni_rag_agent index vector --model BAAI/bge-m3 --rebuild
uv run -m uni_rag_agent search keyword "mapreduce"
uv run -m uni_rag_agent search keyword "mapreduce" --course "Information Retrieval"
uv run -m uni_rag_agent search keyword "mapreduce" --index slides_index --top-k 10
uv run -m uni_rag_agent search keyword "mapreduce" --json
uv run -m uni_rag_agent answer --evidence-packet-id 1 --json
uv run -m uni_rag_agent ask "Explain MapReduce from my courses" --model BAAI/bge-m3 --json
uv run -m uni_rag_agent search semantic "distributed computation" --model BAAI/bge-m3
uv run -m uni_rag_agent search semantic "distributed computation" --model BAAI/bge-m3 --index slides_index --course "Information Retrieval" --top-k 10 --json
uv run -m pytest tests/test_cli.py tests/test_config.py tests/test_storage.py tests/test_logging_config.py tests/test_inventory.py tests/test_extraction.py tests/test_data_summaries.py tests/test_keyword_indexing.py tests/test_vector_indexing.py
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
`extracted_documents` and `chunks`, preserve source locations, and fail per file:

```powershell
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent extract run --category document
uv run -m uni_rag_agent extract status
```

Data-summary extraction processes pending `data_schema` files, writes
`data_summaries`, creates `data_schema` chunks for later keyword/vector indexing,
and samples schemas without embedding full datasets:

```powershell
uv run -m uni_rag_agent extract data-summaries
uv run -m uni_rag_agent extract data-summaries --file-id 123
```

Inventory, extraction, and data-summary CLI runs write lifecycle JSONL logs under
`data/runs/`.

Keyword indexing rebuilds the SQLite FTS5 `chunk_fts` projection from current
indexed chunks only. It searches chunk text, titles, course names, and file
paths, and it supports exact course filters plus logical index filters such as
`slides_index`:

```powershell
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent search keyword "mapreduce"
uv run -m uni_rag_agent search keyword "mapreduce" --course "Information Retrieval"
uv run -m uni_rag_agent search keyword "mapreduce" --index slides_index --top-k 10
uv run -m uni_rag_agent search keyword "mapreduce" --json
```

`index keyword` writes lifecycle JSONL logs under `data/runs/`. Direct keyword
search does not write `search_runs` or `search_results`; persistent retrieval
traces belong to later evidence/retrieval specs.

Vector indexing embeds current eligible chunks into ChromaDB (one
model-namespaced collection per logical index, cosine distance) and records
`embeddings` mapping rows in SQLite. There is no default embedding adapter or
model. An unqualified `index vector` or `search semantic` command fails clearly
until `UNI_RAG_EMBEDDING_MODEL` is configured or `--model` is supplied. The
provider is inferred from the selected registry profile; there is no
`UNI_RAG_EMBEDDING_PROVIDER`. The default run is incremental; `--rebuild` clears
and repopulates only the selected model/profile and optional `--collection`.
Semantic search queries those collections and joins ids back to SQLite for chunk
text and citations.

The lightweight base installation is enough for Features 01-06, including
configuration, storage, inventory, extraction, data summaries, and keyword
search. Reviewed embedding profiles are:

| Identifier | Provider / execution | Declared dimension |
| :--- | :--- | :---: |
| `BAAI/bge-m3` | Hugging Face / local | 1024 |
| `jinaai/jina-embeddings-v3` | Hugging Face / local | 1024 |
| `jinaai/jina-embeddings-v5-text-small` | Hugging Face / local | 1024 |
| `google/embeddinggemma-300m` | Hugging Face / local | 768 |
| `google/gemini-embedding-001` | Direct Google Gemini API / hosted | 3072 |
| `Qwen/Qwen3-Embedding-8B` | Nebius Token Factory / hosted | 4096 |

`gemini-embedding-001` is an accepted alias for the canonical
`google/gemini-embedding-001` identity. The canonical identifier is used in
Chroma collection identity, SQLite, retrieval, evidence, and telemetry.

Install the extra that matches the selected profile:

```powershell
# Local Hugging Face profiles.
uv sync --extra embeddings

# Hosted Google Gemini or Nebius Token Factory profiles.
uv sync --extra embeddings-cloud

# Query-planner and answer-generation LLM integrations remain separate.
uv sync --extra llm
```

Local construction loads Hugging Face SDKs lazily and probes the runtime vector
dimension. Hosted construction loads provider SDKs lazily, uses the declared
dimension, makes no dedicated hosted probe, and validates actual vectors during
normal embedding batches. Both paths use shared validation/retry rules, batches
of up to 64 chunks, and per-batch commits. If hosted retries are exhausted,
already committed batches remain durable and a later incremental run resumes the
missing chunks. Shared retries allow three total attempts for network failures,
HTTP 408/429, and HTTP 5xx responses; malformed/dimension-invalid responses,
credential or permission failures, model failures, and other HTTP 4xx responses
fail without retry.

For a manual optional local smoke run against the fixture or a small local corpus:

```powershell
uv sync --extra embeddings
$env:UNI_RAG_EMBEDDING_MODEL = "BAAI/bge-m3"
uv run -m uni_rag_agent storage init
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent extract run
uv run -m uni_rag_agent index vector --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --json
```

For an optional manual credentialed Gemini smoke:

```powershell
uv sync --extra embeddings-cloud
$env:GOOGLE_API_KEY = "<set locally; never commit>"
uv run -m uni_rag_agent index vector --model gemini-embedding-001 --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --model gemini-embedding-001 --json
```

For an optional manual credentialed Nebius smoke:

```powershell
uv sync --extra embeddings-cloud
$env:NEBIUS_API_KEY = "<set locally; never commit>"
uv run -m uni_rag_agent index vector --model Qwen/Qwen3-Embedding-8B --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --model Qwen/Qwen3-Embedding-8B --json
```

The hosted Google path uses the direct Gemini API with `GOOGLE_API_KEY`; Vertex
AI is not supported. The Nebius path uses the fixed endpoint
`https://api.tokenfactory.nebius.com/v1/` with `NEBIUS_API_KEY`. Its semantic
query input is exactly:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

The first local model run may download weights. Review local model-specific
remote-code, license, gating, token, or authentication requirements before
running it. Missing extras and credentials produce sanitized setup errors.

Hosted profiles send eligible course text and semantic queries to external
providers and may incur charges. Local profiles keep model execution local apart
from model downloads as applicable. `index vector` writes lifecycle JSONL logs
under `data/runs/`; direct semantic search does not write `search_runs` or
`search_results`.

Embedding credentials belong only in the ignored `.env` or the current shell:

- Direct Google Gemini embedding construction uses `GOOGLE_API_KEY`.
- Nebius Token Factory embedding construction uses `NEBIUS_API_KEY` and the
  fixed endpoint documented above.
- Vertex AI credentials and settings are not supported for embeddings.

Credentialed hosted smokes are optional. Automated tests and the normal local
workflow do not require either hosted key. Do not print or commit key values;
missing-extra and missing-credential failures are sanitized.

The `retrieve` command is read-only: a configured LLM first produces a
validated query plan, then it runs metadata/keyword/semantic search and merges
ranked results with RRF. It does not write `search_runs`, `search_results`,
evidence packets, or files under `Courses`; use Feature 09's `evidence build`
workflow when a persisted search and packet are required.
Retrieval requires an explicit reviewed embedding model, the embedding extra
matching that profile, and the optional `llm` extra with a configured LLM
provider/model. Other commands do not require the `llm` extra:

```powershell
# Local embedding + query-planner LLM.
uv sync --extra embeddings
uv sync --extra llm

# Hosted embedding + query-planner LLM.
uv sync --extra embeddings-cloud
uv sync --extra llm

# Both local and hosted embedding constructors plus the LLM integrations.
uv sync --extra embeddings --extra llm
uv sync --extra embeddings-cloud --extra llm
```

Evaluation and hardening uses the committed fixture set by default. Preparation
builds isolated inventory, extraction, keyword, and vector state under
`data/runs/eval/fixture-state`; it requires a configured reviewed production
embedding model and never touches the normal archive database or index:

```powershell
uv run -m uni_rag_agent eval run
uv run -m uni_rag_agent eval run --fixtures
uv run -m uni_rag_agent eval prepare-fixtures
uv run -m uni_rag_agent eval run --smoke-real-archive
```

Bare `eval run` and `eval run --fixtures` are equivalent. The real-archive mode
is explicit and reads only `data/runs/eval/real-archive.json` against the normal
configured archive state; it never prepares or traverses the archive
implicitly. Every run writes paired timestamped JSON and Markdown reports under
`data/runs/eval/`, with safe scores, trace ids, failures, and p50/p95 timings
but no raw evidence, model output, environment values, or secrets. Inspect
report trends with `notebooks/evaluation_eda.ipynb` (read-only).

Feature 09 adds the persisted evidence workflow. `retrieve` remains read-only;
`evidence build` invokes the same mandatory planner/retriever, records raw and
complete fused results, and stores one canonical evidence packet. Packet
selection keeps whole authoritative chunks, defaults to a 12,000-token
whitespace-estimated budget, skips individually oversized chunks, and reports
coverage/weaknesses. Configure the budget with
`UNI_RAG_EVIDENCE_MAX_TOKENS` or inspect it through `config check`.

Representative commands:

```powershell
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3 --json
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3 --debug
uv run -m uni_rag_agent evidence show --search-run-id 1 --json
```

Feature 10 answers a stored packet without rerunning retrieval. Configure a
separate answer provider/model pair in `.env` (the pair is optional for general
commands and required only when a non-empty packet reaches answer generation):

```powershell
$env:UNI_RAG_ANSWER_LLM_PROVIDER = "ollama"
$env:UNI_RAG_ANSWER_LLM_MODEL = "llama3.2"
$env:UNI_RAG_ANSWER_MAX_RETRIES = "1"
$env:UNI_RAG_ANSWER_SESSION_MESSAGE_LIMIT = "20"
$env:UNI_RAG_ANSWER_PROMPT_MAX_TOKENS = "16000"
uv sync --extra llm
uv run -m uni_rag_agent answer --evidence-packet-id 1
```

The answer model returns strict JSON paragraphs with positional ids (`E1`,
`E2`, ...) or the unambiguous `chunk:<id>` compatibility alias; aliases are
canonicalized to positional ids before rendering or storage. The application
adds inline markers and a deterministic References section, stores structured
citations and limitations in the append-only `answers` table, and never stores
prompts or conversation context. The complete prompt is bounded by
`UNI_RAG_ANSWER_PROMPT_MAX_TOKENS`; complete evidence items are selected in
packet rank order, retain their original positional ids, and any omissions are
reported as a deterministic limitation. Empty evidence, or
a budget too small for any complete item, produces a deterministic
insufficient-evidence answer without calling the answer model. `ask` runs
planner/retrieval and answer generation in one shot; the evidence packet remains
persisted if answer-provider construction or invocation fails. Answer failures
use exit code `9`.

Feature 11 exposes the same persisted ask workflow through a local FastAPI
application and package-owned HTML/JavaScript interface:

```powershell
$env:UNI_RAG_ASK_TIMEOUT_SECONDS = "120"
uv run -m uni_rag_agent app serve
uv run -m uni_rag_agent app serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` to ask questions and inspect the answer,
structured citations/references, limitations, searched/found/missing coverage,
and persisted evidence details. The web surface does not expose ingestion,
indexing, evaluation, uploads, or source-file mutation; those workflows remain
CLI-first.

An omitted API `session_id` is stateless. A valid supplied id keeps bounded
planner-only context in process: at most 20 least-recently-used sessions, each
expiring after two hours of inactivity. Sessions disappear when the server
restarts. Ask requests time out after `UNI_RAG_ASK_TIMEOUT_SECONDS` (120 by
default); an evidence packet created before a later timeout or answer failure
remains inspectable, while the timed-out request cannot append a late answer.

The local API includes:

- `GET /health`
- `GET /config`
- `POST /api/ask`
- `GET /api/search-runs/{search_run_id}/coverage`
- `GET /api/evidence-packets/{evidence_packet_id}`
- `GET /api/answers/{answer_id}`

`/health` is provider- and storage-independent liveness. `/config` reports only
operational non-secret settings and path-existence flags, never credentials or
absolute local paths.

## MVP Module Order

1. Project foundation
2. Configuration and storage
3. Inventory and file classification
4. Text extraction and chunking
5. Data schema summaries
6. Keyword indexing
7. Vector indexing
8. LLM query planning and hybrid retrieval
9. Evidence packets and coverage
10. Answering and citations
11. FastAPI HTML UI
12. Evaluation and hardening

## EDA Notebooks

Project-owned notebooks live under `notebooks/`. They are pandas-based,
matplotlib-backed, read-only companions for generated app data, not pipeline
implementation code. Use plots for useful count, distribution, coverage, and
failure diagnostics. Notebooks must not mutate `Courses/`, write to SQLite,
rewrite indexes, execute course scripts, or execute course notebooks.

Notebook outputs and execution counts should be cleared before commit. When a
stage changes the command, tables, JSON artifacts, status vocabulary, plots, or
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
