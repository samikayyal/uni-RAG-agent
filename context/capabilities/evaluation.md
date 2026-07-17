# Evaluation

## Current behavior

`evaluation/core.py` loads the strict UTF-8 fixture set from
`evals/fixtures.json`, validates exact fields and fixture-root-relative paths,
and scores retrieval coverage, required terms, packet-relative citations,
answer limitations, and explicit absence cases deterministically. Fixture
preparation runs the production inventory/extraction/keyword/vector pipeline in
an isolated state directory, validates an identity/count manifest and Chroma
digest, then atomically activates the state. It never mutates the normal
archive database or index.

Bare `eval run` and `eval run --fixtures` use the prepared committed fixture
state. `--smoke-real-archive` is the explicit real-archive mode and reads the
local `data/runs/eval/real-archive.json`; it never traverses or prepares the
archive implicitly. Each run writes paired timestamped JSON and Markdown
reports with safe scores, trace ids, failures, and p50/p95 stage timings.

## Public entry points

- `uv run -m uni_rag_agent eval prepare-fixtures`
- `uv run -m uni_rag_agent eval run [--fixtures | --smoke-real-archive]`
- Python: fixture loading/validation, `prepare_fixture_state`,
  `validate_fixture_state`, `run_eval_set`, `run_eval_item`, scoring helpers,
  and report writers.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/evaluation/{core,models}.py`.
- Tests: `tests/test_evaluation.py`.
- Inputs: `evals/fixtures.json`, `evals/sources/`, and optional local
  `data/runs/eval/real-archive.json`.
- Generated: `data/runs/eval/fixture-state/` plus timestamped
  `data/runs/eval/*.json` and `*.md`; `notebooks/evaluation_eda.ipynb` reads
  those reports without modifying them.

## Invariants and failure boundaries

- Fixture state is generated below a guarded temporary sibling and activated
  only after complete identity/count validation. A failed preparation leaves a
  prior active state untouched; stale/missing state fails with setup guidance.
- Expected source paths match exactly relative to the configured courses root;
  checkpoint files are excluded from source digests. Same-count file/chunk/FTS/
  vector/Chroma drift is detected by manifest identities and digests.
- Public eval commands use configured production providers/models. Deterministic
  doubles belong only to pytest seams. Hosted profiles may incur cost and are
  not silently selected.
- Reports omit raw queries, evidence, model output, authorization values, and
  secrets. Failures are sanitized and linked to persisted search/evidence trace
  ids when available.
- Expected logical indexes and source-type-to-index scoring use the shared
  taxonomy from `src/uni_rag_agent/search_contracts.py`.

Binding decisions: [DEC-037/038](../decisions.md#dec-037038--isolated-safe-evaluation),
[DEC-002/003/006/007](../decisions.md#dec-002003006007--selective-non-destructive-source-admission),
and [DEC-031/039](../decisions.md#dec-031039--explicit-reviewed-embedding-profiles).
