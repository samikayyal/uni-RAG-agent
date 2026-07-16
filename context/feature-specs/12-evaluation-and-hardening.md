# Feature Spec 12: Evaluation and Hardening

## Purpose

Create a small, repeatable evaluation harness that checks retrieval quality, citation quality, weak-retrieval reporting, and safety boundaries as the MVP is implemented.

## Depends On

- [03-inventory-and-file-classification.md](03-inventory-and-file-classification.md)
- [04-text-extraction-and-chunking.md](04-text-extraction-and-chunking.md)
- [08-query-routing-and-hybrid-retrieval.md](08-query-routing-and-hybrid-retrieval.md)
- [09-evidence-packets-and-coverage.md](09-evidence-packets-and-coverage.md)
- [10-answering-and-citations.md](10-answering-and-citations.md)
- DEC-022, DEC-026, DEC-039

## In Scope

- Create 15-20 hand-curated evaluation questions.
- Cover each query type and major source type.
- Store expected courses/files or expected absence for each eval item.
- Run evals against fixture data automatically.
- Support optional smoke evals against the real `Courses` archive.
- Check citation validity, evidence packet completeness, and weak retrieval reporting.
- Record performance and failure summaries.
- Keep evaluation provider-neutral while preserving canonical embedding model
  identity and declared dimension in safe traces; allow optional manual smokes
  for the reviewed Gemini and Nebius hosted profiles.
- Add or update the evaluation EDA notebook for eval reports and quality trends.

## Out of Scope

- Large-scale benchmark infrastructure.
- Model-vs-model leaderboard.
- Automatic grading that trusts an LLM as the only judge.
- Full archive traversal during normal automated tests.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent eval run
uv run -m uni_rag_agent eval run --fixtures
uv run -m uni_rag_agent eval run --smoke-real-archive
```

Notebook:

```text
notebooks/evaluation_eda.ipynb
```

Create this notebook when evaluation reporting is implemented. It should inspect `data/runs/eval/` JSON/Markdown reports, retrieval/citation scores, expected-vs-found source coverage, weak-retrieval cases, failures, and runtime summaries.

Eval item shape:

```text
id
query
query_type
expected_courses
expected_files
expected_indexes
must_include_terms
expected_weaknesses
notes
```

Internal interfaces:

```python
load_eval_set(path: Path) -> list[EvalItem]
run_eval_item(item: EvalItem, config: Config) -> EvalResult
score_retrieval(item: EvalItem, packet: EvidencePacket) -> RetrievalScore
score_citations(packet: EvidencePacket, answer: AnswerResult) -> CitationScore
write_eval_report(results: list[EvalResult]) -> Path
```

## Storage and Schema Impact

No required new tables for MVP. Eval reports should be written under:

```text
data/runs/eval/
```

If later persistence is useful, add an explicit architecture update before introducing eval tables.

Locked implementation rules for this slice:

- `evals/fixtures.json` is strict UTF-8 JSON with exactly the item fields shown
  above. Arrays are explicit JSON arrays; unknown or missing fields are errors.
- Bare `eval run` means fixture mode and is equivalent to `--fixtures`.
  `--smoke-real-archive` is explicit and mutually exclusive. Fixture state is
  prepared only by `eval prepare-fixtures` under
  `data/runs/eval/fixture-state`, using production embedding/model providers;
  deterministic doubles remain pytest-only.
- Fixture preparation inventories, extracts (including data-schema summaries),
  rebuilds keyword/vector indexes, and records a manifest. Fixture runs fail
  with setup guidance when the manifest or generated state is absent or stale;
  they never rebuild implicitly or touch normal archive state.
- Fixture preparation selects a reviewed embedding profile by canonical model
  identifier. Local Hugging Face preparation uses `uv sync --extra embeddings`;
  hosted Google/Nebius preparation uses `uv sync --extra embeddings-cloud`.
  Provider inference comes from the registry and there is no
  `UNI_RAG_EMBEDDING_PROVIDER` setting. The existing `uv sync --extra llm`
  semantics remain independent.
- Preparation builds below a guarded temporary sibling, validates deterministic
  file/chunk/FTS/embedding identities plus vector-collection/Chroma state, and
  activates the result only after validation. A failed preparation preserves a
  previously valid active fixture state.
- Retrieval scoring matches canonical courses/indexes and fixture-root-relative
  files exactly. Required terms must occur in both selected evidence and the
  final answer; expected weakness substrings must occur in both packet
  weaknesses and answer limitations. Empty expected sources with nonempty
  weaknesses is an explicit zero-evidence case.
- Every item uses the existing `build_evidence` and `store_answer` boundaries.
  Paired timestamped JSON/Markdown reports under `data/runs/eval` include only
  safe per-item field results, trace IDs, failures, and evidence/answer/total
  timings plus p50/p95 aggregates. They never include raw evidence, model
  output, raw query text, full environment values, or secrets.

## Workflow

1. Maintain a committed eval set for fixture data.
2. Optionally maintain a local-only eval set for the real course archive if paths are personal or too large to guarantee.
3. For each eval item, run route, retrieval, evidence packet, and answer generation.
4. Score whether expected courses/files/indexes appear in searched and evidence fields.
5. Validate citations map to packet evidence.
6. Check weak-retrieval explanations when expected evidence is missing.
7. Write a JSON and Markdown report under `data/runs/eval/`.
8. Keep `notebooks/evaluation_eda.ipynb` aligned with eval item fields, report JSON shape, scoring fields, and runtime summary semantics.

## Failure and Safety Rules

- Normal automated evals must use fixtures only.
- Real archive smoke evals require an explicit `--smoke-real-archive` flag.
- Eval runs must not mutate `Courses`.
- Jupyter `.ipynb_checkpoints` trees are excluded from fixture source digests and
  cannot be expected evaluation files.
- Do not use LLM judging as the sole pass/fail mechanism.
- Test-only injected model/chat dependencies should keep fixture evals independent of provider credentials.
- Reports should avoid storing secrets or full environment values.
- Hosted evaluation sends eligible course text and semantic queries to an
  external provider and may incur charges; local profiles keep model execution
  local apart from model downloads as applicable. Do not run hosted evaluation
  in the automated fixture path merely because credentials are present.
- The EDA notebook must read generated eval reports and app traces only; it must not mutate reports, SQLite, indexes, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated tests for eval item loading and validation.
- Verify fixture evals can run without real API keys.
- Verify retrieval scoring catches missing expected files/courses.
- Verify citation scoring catches citations not present in packets.
- Verify weak-retrieval expected cases pass only when limitations are reported.
- Verify reports are written under a temporary runs directory in tests.
- Verify `notebooks/evaluation_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional manual credentialed smokes: after installing the matching embedding
  extra, run the small Gemini or Nebius vector/retrieval smoke commands from the
  root README. These are optional and are not part of pytest or the default eval
  path.

## Acceptance Criteria

- The repo contains a small committed fixture eval set.
- `uv run -m uni_rag_agent eval run --fixtures` produces a useful report.
- Eval results cover retrieval, evidence packets, citations, and safety boundaries.
- Real archive eval is explicit and never part of the default automated test path.
- `notebooks/evaluation_eda.ipynb` exists once this feature lands and can inspect evaluation reports without mutating generated or source data.
