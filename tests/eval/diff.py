"""Phase 3: side-by-side comparison of two runs.

Reads `runs/<a>/{summary.json,scores.jsonl,manifest.json}` and the same for
`<b>`, then writes `runs/<b>/diff_vs_<a>.md` highlighting per-question deltas
and which Settings changed between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .scoring import load_scores
from .types import RunManifest, ScoreResult, ScoreSummary


def diff_runs(base_dir: Path, candidate_dir: Path) -> str:
    """Build a markdown diff report. Returns the rendered text."""
    base_manifest = RunManifest.model_validate_json(
        (base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cand_manifest = RunManifest.model_validate_json(
        (candidate_dir / "manifest.json").read_text(encoding="utf-8")
    )
    base_summary = ScoreSummary.model_validate_json(
        (base_dir / "summary.json").read_text(encoding="utf-8")
    )
    cand_summary = ScoreSummary.model_validate_json(
        (candidate_dir / "summary.json").read_text(encoding="utf-8")
    )

    base_scores = {s.question_id: s for s in load_scores(base_dir / "scores.jsonl")}
    cand_scores = {s.question_id: s for s in load_scores(candidate_dir / "scores.jsonl")}

    return _render_diff_md(
        base_dir, candidate_dir,
        base_manifest, cand_manifest,
        base_summary, cand_summary,
        base_scores, cand_scores,
    )


def _render_diff_md(
    base_dir: Path, cand_dir: Path,
    base_m: RunManifest, cand_m: RunManifest,
    base_s: ScoreSummary, cand_s: ScoreSummary,
    base_scores: Dict[str, ScoreResult],
    cand_scores: Dict[str, ScoreResult],
) -> str:
    lines: List[str] = [
        f"# Diff: `{cand_m.run_id}` vs `{base_m.run_id}`",
        "",
        f"- Base: `{base_dir}` (git {base_m.git_sha or '?'})",
        f"- Candidate: `{cand_dir}` (git {cand_m.git_sha or '?'})",
        "",
        "## Aggregate deltas",
        "",
        f"| Metric | Base | Candidate | Δ |",
        f"|---|---|---|---|",
        f"| Overall mean | {base_s.overall_mean} | {cand_s.overall_mean} | {_delta(cand_s.overall_mean, base_s.overall_mean)} |",
        f"| Overall p50  | {base_s.overall_p50}  | {cand_s.overall_p50}  | {_delta(cand_s.overall_p50, base_s.overall_p50)} |",
        f"| Overall p10  | {base_s.overall_p10}  | {cand_s.overall_p10}  | {_delta(cand_s.overall_p10, base_s.overall_p10)} |",
        f"| Cost (USD)   | {base_s.total_cost_usd:.4f} | {cand_s.total_cost_usd:.4f} | {_delta(cand_s.total_cost_usd, base_s.total_cost_usd)} |",
        f"| Latency (ms) | {base_s.total_latency_ms} | {cand_s.total_latency_ms} | {_delta(cand_s.total_latency_ms, base_s.total_latency_ms)} |",
    ]

    if base_s.judge_aggregates and cand_s.judge_aggregates:
        lines += ["", "## Judge deltas", "", "| Metric | Base | Candidate | Δ |", "|---|---|---|---|"]
        for k in sorted(base_s.judge_aggregates.keys() | cand_s.judge_aggregates.keys()):
            b = base_s.judge_aggregates.get(k, 0)
            c = cand_s.judge_aggregates.get(k, 0)
            lines.append(f"| {k} | {b} | {c} | {_delta(c, b)} |")

    # Settings diff
    settings_diff = _diff_dict(base_m.settings_snapshot, cand_m.settings_snapshot)
    if settings_diff:
        lines += ["", "## Settings changed", "", "| Key | Base | Candidate |", "|---|---|---|"]
        for k, (bv, cv) in settings_diff.items():
            lines.append(f"| {k} | `{bv}` | `{cv}` |")

    # Per question regressions / improvements
    common_ids = sorted(set(base_scores) & set(cand_scores))
    rows: List[tuple[str, float, float, float]] = []
    for qid in common_ids:
        b = base_scores[qid].overall
        c = cand_scores[qid].overall
        rows.append((qid, b, c, c - b))
    rows.sort(key=lambda r: r[3])  # worst regressions first

    regressions = [r for r in rows if r[3] < -0.001]
    improvements = [r for r in rows if r[3] > 0.001]

    if regressions:
        lines += ["", "## Regressions (worst first)", "",
                  "| ID | Base | Candidate | Δ |", "|---|---|---|---|"]
        for qid, b, c, d in regressions[:20]:
            lines.append(f"| {qid} | {b:.2f} | {c:.2f} | {d:+.2f} |")
    if improvements:
        lines += ["", "## Improvements (best first)", "",
                  "| ID | Base | Candidate | Δ |", "|---|---|---|---|"]
        for qid, b, c, d in sorted(improvements, key=lambda r: -r[3])[:20]:
            lines.append(f"| {qid} | {b:.2f} | {c:.2f} | {d:+.2f} |")

    only_base = sorted(set(base_scores) - set(cand_scores))
    only_cand = sorted(set(cand_scores) - set(base_scores))
    if only_base or only_cand:
        lines += ["", "## Question set differs", ""]
        if only_base:
            lines.append(f"- Only in base: {', '.join(only_base)}")
        if only_cand:
            lines.append(f"- Only in candidate: {', '.join(only_cand)}")

    return "\n".join(lines) + "\n"


def _delta(c: float, b: float) -> str:
    d = c - b
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.4f}"


def _diff_dict(a: dict, b: dict) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out[k] = (a.get(k), b.get(k))
    return out
