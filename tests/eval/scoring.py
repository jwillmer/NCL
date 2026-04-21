"""Phase 2 orchestrator: read results.jsonl, apply graders, write scores.jsonl.

Auto-graders always run (cheap). LLM judge runs only when --judge llm|both.
Re-running judge over the same results is free thanks to content-hash caching
in metrics/answer.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml

from .metrics.answer import judge as llm_judge
from .metrics.citation import score_citations
from .metrics.format import score_format
from .metrics.retrieval import score_retrieval
from .runner import load_goldens, load_results
from .types import (
    AutoScores,
    GoldenQuestion,
    JudgeScores,
    RunResult,
    ScoreResult,
    ScoreSummary,
)

logger = logging.getLogger(__name__)

# Weights for the per-question overall score (0-1)
W_FAITHFULNESS = 0.30
W_COMPLETENESS = 0.25
W_RELEVANCE = 0.20
W_ACTIONABILITY = 0.10
W_AUTO_CITATION = 0.10  # citations_valid_pct
W_AUTO_GROUNDING = 0.05  # citation_grounding_score


def _auto_score_one(run: RunResult, golden: GoldenQuestion) -> AutoScores:
    cit = score_citations(run)
    fmt = score_format(run)
    ret = score_retrieval(run, golden)
    return AutoScores(
        citations_count=cit["citations_count"],
        citations_valid_count=cit["citations_valid_count"],
        citations_valid_pct=cit["citations_valid_pct"],
        citation_grounding_score=cit["citation_grounding_score"],
        response_length_chars=fmt["response_length_chars"],
        has_no_results_disclaimer=fmt["has_no_results_disclaimer"],
        recall_at_5=ret.get("recall_at_5"),
        recall_at_10=ret.get("recall_at_10"),
        recall_at_20=ret.get("recall_at_20"),
        mrr=ret.get("mrr"),
        ndcg_at_10=ret.get("ndcg_at_10"),
        follows_response_format=fmt["follows_response_format"],
        format_violations=fmt["format_violations"],
    )


def _overall_score(auto: AutoScores, judge: Optional[JudgeScores]) -> float:
    """Weighted aggregate, 0-1. Auto-only baseline if judge missing."""
    auto_part = (
        W_AUTO_CITATION * auto.citations_valid_pct
        + W_AUTO_GROUNDING * auto.citation_grounding_score
    )
    if judge is None:
        # Rescale auto-only part to fill 0-1 (otherwise capped at 0.15)
        denom = W_AUTO_CITATION + W_AUTO_GROUNDING
        return round(auto_part / denom, 4) if denom else 0.0

    judge_part = (
        W_FAITHFULNESS * (judge.faithfulness / 5)
        + W_COMPLETENESS * (judge.completeness / 5)
        + W_RELEVANCE * (judge.relevance / 5)
        + W_ACTIONABILITY * (judge.actionability / 5)
    )
    return round(auto_part + judge_part, 4)


async def execute_judge(
    *,
    run_dir: Path,
    questions_path: Path,
    judge_mode: Literal["auto", "llm", "both"] = "both",
    judge_model: Optional[str] = None,
    use_cache: bool = True,
    concurrency: int = 4,
) -> ScoreSummary:
    """Score every result in `run_dir/results.jsonl`. Writes scores.jsonl + summary.json."""
    results_path = run_dir / "results.jsonl"
    scores_path = run_dir / "scores.jsonl"
    summary_path = run_dir / "summary.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl in {run_dir}")

    runs = load_results(results_path)
    goldens_by_id: Dict[str, GoldenQuestion] = {
        g.id: g for g in load_goldens(questions_path)
    }

    use_judge = judge_mode in ("llm", "both")
    sem = asyncio.Semaphore(concurrency)

    async def _score_one(run: RunResult) -> ScoreResult:
        golden = goldens_by_id.get(run.question_id)
        if golden is None:
            logger.warning("No golden for %s — auto-scoring with empty reference", run.question_id)
            golden = GoldenQuestion(
                id=run.question_id, category="unknown",
                question=run.question, reference_answer="",
            )
        auto = _auto_score_one(run, golden)
        judge_scores: Optional[JudgeScores] = None
        if use_judge:
            async with sem:
                judge_scores = await llm_judge(
                    golden, run, judge_model=judge_model, use_cache=use_cache,
                )
        overall = _overall_score(auto, judge_scores)
        return ScoreResult(
            question_id=run.question_id,
            auto=auto,
            judge=judge_scores,
            overall=overall,
        )

    score_results = await asyncio.gather(*(_score_one(r) for r in runs))

    # Write scores.jsonl
    with scores_path.open("w", encoding="utf-8") as f:
        for s in score_results:
            f.write(s.model_dump_json() + "\n")

    summary = _build_summary(run_id=run_dir.name, runs=runs, scores=score_results)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    # Also write a human-readable markdown summary
    md_path = run_dir / "summary.md"
    md_path.write_text(_render_summary_md(summary, runs, score_results, goldens_by_id),
                       encoding="utf-8")
    return summary


def _build_summary(
    *,
    run_id: str,
    runs: List[RunResult],
    scores: List[ScoreResult],
) -> ScoreSummary:
    overall_vals = [s.overall for s in scores]
    overall_mean = statistics.mean(overall_vals) if overall_vals else 0.0
    p50 = statistics.median(overall_vals) if overall_vals else 0.0
    p10 = statistics.quantiles(overall_vals, n=10)[0] if len(overall_vals) >= 10 else min(overall_vals or [0])

    auto_aggregates = {
        "citations_valid_pct_mean": _mean(s.auto.citations_valid_pct for s in scores),
        "citation_grounding_score_mean": _mean(s.auto.citation_grounding_score for s in scores),
        "follows_response_format_pct": _mean(float(s.auto.follows_response_format) for s in scores),
    }
    judged = [s for s in scores if s.judge is not None]
    judge_aggregates = None
    if judged:
        judge_aggregates = {
            "faithfulness_mean": _mean(s.judge.faithfulness for s in judged),
            "completeness_mean": _mean(s.judge.completeness for s in judged),
            "relevance_mean": _mean(s.judge.relevance for s in judged),
            "actionability_mean": _mean(s.judge.actionability for s in judged),
            "judge_cost_usd_total": sum(s.judge.judge_cost_usd for s in judged),
        }

    failures = [r.question_id for r in runs if r.error]
    return ScoreSummary(
        run_id=run_id,
        question_count=len(runs),
        scored_count=len(scores),
        auto_aggregates=auto_aggregates,
        judge_aggregates=judge_aggregates,
        overall_mean=round(overall_mean, 4),
        overall_p50=round(p50, 4),
        overall_p10=round(p10, 4),
        total_cost_usd=sum(r.metrics.cost_usd for r in runs),
        total_latency_ms=sum(r.metrics.latency_ms for r in runs),
        failures=failures,
    )


def _mean(values) -> float:
    vs = list(values)
    return round(statistics.mean(vs), 4) if vs else 0.0


def _render_summary_md(
    summary: ScoreSummary,
    runs: List[RunResult],
    scores: List[ScoreResult],
    goldens: Dict[str, GoldenQuestion],
) -> str:
    lines: List[str] = [
        f"# Eval run `{summary.run_id}`",
        "",
        f"- Questions: **{summary.question_count}** ({summary.scored_count} scored, {len(summary.failures)} failed)",
        f"- Overall mean: **{summary.overall_mean}** (p50 {summary.overall_p50}, p10 {summary.overall_p10})",
        f"- Total agent cost: ${summary.total_cost_usd:.4f}",
        f"- Total agent latency: {summary.total_latency_ms / 1000:.1f}s",
        "",
        "## Auto aggregates",
        "",
    ]
    for k, v in summary.auto_aggregates.items():
        lines.append(f"- {k}: {v}")
    if summary.judge_aggregates:
        lines += ["", "## Judge aggregates", ""]
        for k, v in summary.judge_aggregates.items():
            lines.append(f"- {k}: {v}")

    lines += ["", "## Per question", "",
              "| ID | Overall | Faith | Compl | Rel | Act | Citations | Latency |",
              "|----|---------|-------|-------|-----|-----|-----------|---------|"]
    score_by_qid = {s.question_id: s for s in scores}
    run_by_qid = {r.question_id: r for r in runs}
    for qid, s in sorted(score_by_qid.items()):
        r = run_by_qid.get(qid)
        j = s.judge
        lines.append(
            f"| {qid} | {s.overall:.2f} | "
            f"{j.faithfulness if j else '-'} | "
            f"{j.completeness if j else '-'} | "
            f"{j.relevance if j else '-'} | "
            f"{j.actionability if j else '-'} | "
            f"{s.auto.citations_valid_count}/{s.auto.citations_count} | "
            f"{r.metrics.latency_ms if r else '-'}ms |"
        )
    if summary.failures:
        lines += ["", "## Failures", ""]
        for qid in summary.failures:
            r = run_by_qid.get(qid)
            lines.append(f"- {qid}: {r.error if r else '?'}")
    return "\n".join(lines) + "\n"


def load_scores(scores_path: Path) -> List[ScoreResult]:
    out: List[ScoreResult] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ScoreResult.model_validate_json(line))
    return out
