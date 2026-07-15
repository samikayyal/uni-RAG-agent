"""Evaluation and hardening services for Feature 12."""

from .core import (
    default_eval_set_path,
    fixture_source_root,
    fixture_state_config,
    fixture_state_dir,
    load_eval_set,
    prepare_fixture_state,
    real_archive_eval_path,
    run_eval_item,
    run_eval_set,
    score_citations,
    score_retrieval,
    validate_fixture_state,
    write_eval_report,
)
from .models import (
    CitationScore,
    EvalItem,
    EvalResult,
    EvalSetError,
    EvaluationError,
    RetrievalScore,
)

__all__ = [
    "CitationScore",
    "EvalItem",
    "EvalResult",
    "EvalSetError",
    "EvaluationError",
    "RetrievalScore",
    "default_eval_set_path",
    "fixture_source_root",
    "fixture_state_config",
    "fixture_state_dir",
    "load_eval_set",
    "prepare_fixture_state",
    "real_archive_eval_path",
    "run_eval_item",
    "run_eval_set",
    "score_citations",
    "score_retrieval",
    "validate_fixture_state",
    "write_eval_report",
]
