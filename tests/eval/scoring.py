"""Phase 2: read results.jsonl, apply auto-graders, write scores.jsonl.

Auto-graders only — LLM judge was removed by design. Humans review the
responses directly against goldens. The auto-grader layer is retained
because it is cheap and deterministic: citations coverage, response
format adherence, and (when goldens carry labeled chunk_ids)
retrieval recall/MRR/nDCG.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from pathlib import Path
from typing import Dict, List

from .metrics.citation import score_citations, score_citations_async
from .metrics.format import score_format
from .metrics.retrieval import score_retrieval
from .runner import load_goldens, load_results
from .types import (
    AutoScores,
    GoldenQuestion,
    RunResult,
    ScoreResult,
    ScoreSummary,
)

logger = logging.getLogger(__name__)

# Weights for the per-question overall score (0-1). Auto-grader only —
# no judge. Kept simple so the number is interpretable as "this fraction
# of citations were valid, weighted by grounding".
W_AUTO_CITATION = 0.65
W_AUTO_GROUNDING = 0.35


_REFUSAL_CATEGORY = "No Results Expected"


async def _auto_score_one(run: RunResult, golden: GoldenQuestion) -> AutoScores:
    # Async grounding uses the semantic embedding path (text-embedding-3-
    # small via LiteLLM). Falls back to TF-IDF inside the function if the
    # embedding call errors, so offline runs still score.
    cit = await score_citations_async(run)
    fmt = score_format(run, golden)
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


def _overall_score(auto: AutoScores, golden: GoldenQuestion | None = None) -> float:
    """Auto-only weighted aggregate, 0-1.

    "No Results Expected" goldens (off-topic queries, implausible vessels)
    should score 1.0 when the agent correctly refuses or acknowledges no
    results. These responses legitimately carry zero citations, which the
    citations_valid_pct formula punishes as 0.0 — unfair to an otherwise
    correct answer. Detection uses the format grader's disclaimer flag +
    follows_response_format gate so a refusal that violates format still
    falls back to the weighted formula.
    """
    if (
        golden is not None
        and golden.category == _REFUSAL_CATEGORY
        and auto.has_no_results_disclaimer
        and auto.follows_response_format
    ):
        return 1.0
    return round(
        W_AUTO_CITATION * auto.citations_valid_pct
        + W_AUTO_GROUNDING * auto.citation_grounding_score,
        4,
    )


async def execute_judge(
    *,
    run_dir: Path,
    questions_path: Path,
    concurrency: int = 4,
) -> ScoreSummary:
    """Score every result in ``run_dir/results.jsonl``. Writes
    scores.jsonl + summary.json + summary.md.

    (Name kept as ``execute_judge`` for CLI compatibility even though no
    LLM judging happens anymore — the CLI command is ``mtss eval judge``.)
    """
    results_path = run_dir / "results.jsonl"
    scores_path = run_dir / "scores.jsonl"
    summary_path = run_dir / "summary.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl in {run_dir}")

    runs = load_results(results_path)
    goldens_by_id: Dict[str, GoldenQuestion] = {
        g.id: g for g in load_goldens(questions_path)
    }

    sem = asyncio.Semaphore(concurrency)

    async def _score_one(run: RunResult) -> ScoreResult:
        async with sem:
            golden = goldens_by_id.get(run.question_id)
            if golden is None:
                logger.warning(
                    "No golden for %s — auto-scoring with empty reference",
                    run.question_id,
                )
                golden = GoldenQuestion(
                    id=run.question_id,
                    category="unknown",
                    question=run.question,
                    reference_answer="",
                )
            auto = await _auto_score_one(run, golden)
            return ScoreResult(
                question_id=run.question_id,
                auto=auto,
                overall=_overall_score(auto, golden),
            )

    score_results = await asyncio.gather(*(_score_one(r) for r in runs))

    with scores_path.open("w", encoding="utf-8") as f:
        for s in score_results:
            f.write(s.model_dump_json() + "\n")

    summary = _build_summary(run_id=run_dir.name, runs=runs, scores=score_results)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    md_path = run_dir / "summary.md"
    md_path.write_text(
        _render_summary_md(summary, runs, score_results, goldens_by_id),
        encoding="utf-8",
    )
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
    p10 = (
        statistics.quantiles(overall_vals, n=10)[0]
        if len(overall_vals) >= 10
        else min(overall_vals or [0])
    )

    auto_aggregates = {
        "citations_valid_pct_mean": _mean(
            s.auto.citations_valid_pct for s in scores
        ),
        "citation_grounding_score_mean": _mean(
            s.auto.citation_grounding_score for s in scores
        ),
        "follows_response_format_pct": _mean(
            float(s.auto.follows_response_format) for s in scores
        ),
    }

    failures = [r.question_id for r in runs if r.error]
    return ScoreSummary(
        run_id=run_id,
        question_count=len(runs),
        scored_count=len(scores),
        auto_aggregates=auto_aggregates,
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
        f"- Questions: **{summary.question_count}** "
        f"({summary.scored_count} scored, {len(summary.failures)} failed)",
        f"- Overall mean: **{summary.overall_mean}** "
        f"(p50 {summary.overall_p50}, p10 {summary.overall_p10})",
        f"- Total agent cost: ${summary.total_cost_usd:.4f}",
        f"- Total agent latency: {summary.total_latency_ms / 1000:.1f}s",
        "",
        "## Auto aggregates",
        "",
    ]
    for k, v in summary.auto_aggregates.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Per question",
        "",
        "| ID | Overall | Citations | Format | Chars | Latency |",
        "|----|---------|-----------|--------|-------|---------|",
    ]
    score_by_qid = {s.question_id: s for s in scores}
    run_by_qid = {r.question_id: r for r in runs}
    for qid, s in sorted(score_by_qid.items()):
        r = run_by_qid.get(qid)
        lines.append(
            f"| {qid} | {s.overall:.2f} | "
            f"{s.auto.citations_valid_count}/{s.auto.citations_count} | "
            f"{'ok' if s.auto.follows_response_format else 'no'} | "
            f"{s.auto.response_length_chars} | "
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
