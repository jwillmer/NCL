"""Pytest regression gate against a baseline eval run.

Configure with env vars:
    MTSS_EVAL_BASELINE       — path to baseline run dir (default: tests/eval/runs/baseline)
    MTSS_EVAL_CANDIDATE      — path to candidate run dir (default: tests/eval/runs/latest)
    MTSS_EVAL_TOLERANCE      — allowed drop in overall_mean (default: 0.05)
    MTSS_EVAL_PER_Q_TOLERANCE — allowed drop per question (default: 0.10)

Both dirs must contain summary.json + scores.jsonl produced by `mtss eval score`.
Tests are skipped (not failed) when either dir is absent so a fresh checkout
or a no-baseline state passes CI.

Wire baseline by symlinking a known-good run:
    ln -s <run-id> tests/eval/runs/baseline
or set MTSS_EVAL_BASELINE explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pytest

from .scoring import load_scores
from .types import ScoreResult, ScoreSummary

BASELINE = Path(os.environ.get("MTSS_EVAL_BASELINE", "tests/eval/runs/baseline"))
CANDIDATE = Path(os.environ.get("MTSS_EVAL_CANDIDATE", "tests/eval/runs/latest"))
TOLERANCE = float(os.environ.get("MTSS_EVAL_TOLERANCE", "0.05"))
PER_Q_TOLERANCE = float(os.environ.get("MTSS_EVAL_PER_Q_TOLERANCE", "0.10"))


def _have_run(d: Path) -> bool:
    return (d / "summary.json").exists() and (d / "scores.jsonl").exists()


pytestmark = pytest.mark.skipif(
    not (_have_run(BASELINE) and _have_run(CANDIDATE)),
    reason=(
        f"Skipping regression gate — need both BASELINE ({BASELINE}) and "
        f"CANDIDATE ({CANDIDATE}) to have summary.json + scores.jsonl. "
        "Run `mtss eval run` + `mtss eval score` and set MTSS_EVAL_BASELINE / "
        "MTSS_EVAL_CANDIDATE."
    ),
)


@pytest.fixture(scope="module")
def base_summary() -> ScoreSummary:
    return ScoreSummary.model_validate_json((BASELINE / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cand_summary() -> ScoreSummary:
    return ScoreSummary.model_validate_json((CANDIDATE / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_scores() -> Dict[str, ScoreResult]:
    return {s.question_id: s for s in load_scores(BASELINE / "scores.jsonl")}


@pytest.fixture(scope="module")
def cand_scores() -> Dict[str, ScoreResult]:
    return {s.question_id: s for s in load_scores(CANDIDATE / "scores.jsonl")}


def test_overall_mean_no_regression(base_summary, cand_summary):
    """Aggregate overall_mean should not drop more than TOLERANCE."""
    delta = cand_summary.overall_mean - base_summary.overall_mean
    assert delta >= -TOLERANCE, (
        f"overall_mean regressed by {-delta:.4f} (tolerance {TOLERANCE}): "
        f"baseline={base_summary.overall_mean}, candidate={cand_summary.overall_mean}"
    )


def test_no_question_collapsed(base_scores, cand_scores):
    """Per-question regression check — no individual Q drops more than PER_Q_TOLERANCE."""
    failures = []
    for qid, b in base_scores.items():
        c = cand_scores.get(qid)
        if c is None:
            failures.append(f"{qid}: missing in candidate")
            continue
        delta = c.overall - b.overall
        if delta < -PER_Q_TOLERANCE:
            failures.append(
                f"{qid}: dropped {-delta:.3f} (baseline={b.overall:.3f}, candidate={c.overall:.3f})"
            )
    assert not failures, "Per-question regressions:\n  " + "\n  ".join(failures)


def test_no_new_failures(base_summary, cand_summary):
    """Failed-question count should not grow."""
    new_failures = set(cand_summary.failures) - set(base_summary.failures)
    assert not new_failures, f"New question failures vs baseline: {sorted(new_failures)}"


def test_format_compliance_floor(cand_summary):
    """follows_response_format_pct should stay above the baseline minus a tolerance."""
    pct = cand_summary.auto_aggregates.get("follows_response_format_pct", 0.0)
    assert pct >= 0.90, f"follows_response_format_pct dropped below 0.90: {pct}"
