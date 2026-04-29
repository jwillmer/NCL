"""Phase-1 measurement harness for the context-summary swap.

Compares the current ``CitationProcessor.build_context`` against the
proposed ``build_context_hybrid`` (which skips full text for chunks whose
``embedding_mode == "summary"``). For each golden question:

  1. Run the production retrieval pipeline (``Retriever``).
  2. Hydrate ``embedding_mode`` for each retrieved chunk by joining
     against ``chunks.embedding_mode`` in Supabase. The pgvector
     ``match_chunks`` function does not yet RETURN the column; this
     script side-loads it so we can measure now without bumping the SQL
     migration.
  3. Build context with both builders, count tokens with tiktoken
     (``cl100k_base``), and record per-question + aggregate stats.
  4. Emit a markdown report under ``reports/context-summary-measurement/``.

NO synthesis LLM call is made. This harness only measures context shape
and token counts; the actual A/B answer-quality comparison is a
follow-up.

Tokeniser choice — ``cl100k_base`` is an approximation. The production
synthesizer is OpenRouter-routed and the actual tokeniser may vary per
model; ``cl100k_base`` is used here for relative-comparison purposes
only.

Usage::

    set -a; source .env.test; set +a
    uv run python scripts/measure_context_summary_swap.py \\
        --questions tests/eval/goldens/questions.yaml \\
        --output reports/context-summary-measurement/ \\
        --top-k 20 --rerank-top-n 8 --limit 5

Question file shapes accepted:

  * YAML with top-level ``questions:`` list (e.g.
    ``tests/eval/goldens/questions.yaml`` — uses ``id`` + ``question``).
  * JSON list of objects.
  * JSONL — one object per line.

Each record needs at minimum ``id`` (or ``question_id``) and
``question``. Extra fields (``expected_chunk_ids``, ``vessel_filter``,
etc.) are tolerated but not required.

If no golden file is provided AND ``--smoke`` is passed, the script
runs against a hand-crafted 3-chunk fixture (no DB, no LLM) so you can
exercise the report-generation path. Real goldens MUST replace the
fixture before publishing the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("measure_context_summary_swap")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _get_token_counter():
    """Return a callable that counts tokens via tiktoken cl100k_base."""
    import tiktoken  # local import so --help works without the dep

    enc = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(enc.encode(text or ""))

    return count


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


@dataclass
class _Question:
    id: str
    question: str
    vessel_filter: Optional[Dict[str, Any]] = None
    expected_chunk_ids: List[str] = field(default_factory=list)


def _load_questions(path: Path) -> List[_Question]:
    """Best-effort loader for YAML / JSON / JSONL question files."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    raw_records: List[Dict[str, Any]]

    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict) and "questions" in data:
            raw_records = list(data["questions"])
        elif isinstance(data, list):
            raw_records = data
        else:
            raise ValueError(f"{path}: expected list or 'questions:' key")
    elif suffix == ".jsonl":
        raw_records = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    else:  # default: JSON
        data = json.loads(text)
        if isinstance(data, dict) and "questions" in data:
            raw_records = list(data["questions"])
        elif isinstance(data, list):
            raw_records = data
        else:
            raise ValueError(f"{path}: expected list or 'questions' key")

    questions: List[_Question] = []
    for rec in raw_records:
        qid = rec.get("id") or rec.get("question_id")
        qtext = rec.get("question")
        if not qid or not qtext:
            logger.warning("Skipping record without id/question: %r", rec)
            continue
        questions.append(
            _Question(
                id=str(qid),
                question=str(qtext),
                vessel_filter=rec.get("vessel_filter"),
                expected_chunk_ids=list(rec.get("expected_chunk_ids") or []),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Embedding-mode hydration (side-load until match_chunks returns it)
# ---------------------------------------------------------------------------


async def _hydrate_embedding_modes(
    db: Any,
    results: Sequence[Any],
) -> None:
    """Mutate each ``RetrievalResult.embedding_mode`` in-place.

    Joins on ``chunks.chunk_id`` because that is the citation primary
    key the retriever returns. Any chunk_id missing from the join stays
    None (treated as ``full`` by the hybrid builder).
    """
    chunk_ids = [r.chunk_id for r in results if r.chunk_id]
    if not chunk_ids:
        return
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_id, embedding_mode FROM chunks "
            "WHERE chunk_id = ANY($1::text[])",
            chunk_ids,
        )
    mode_by_id = {row["chunk_id"]: row["embedding_mode"] for row in rows}
    for r in results:
        if r.chunk_id in mode_by_id:
            r.embedding_mode = mode_by_id[r.chunk_id]


# ---------------------------------------------------------------------------
# Per-question measurement
# ---------------------------------------------------------------------------


@dataclass
class QuestionMetric:
    """One row of the comparison table — pure data."""

    question_id: str
    question: str
    chunks_total: int
    chunks_summary_mode_count: int
    chunks_full_mode_count: int
    chunks_metadata_only_count: int
    chunks_unknown_mode_count: int
    tokens_full: int
    tokens_hybrid: int
    delta_tokens: int  # full - hybrid (positive = saved)
    pct_saved: float  # 0..1
    chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _measure_question(
    q: _Question,
    results: Sequence[Any],
    *,
    citation_processor: Any,
    count_tokens,
) -> QuestionMetric:
    """Build both contexts, count tokens, return the row dict."""
    ctx_full = citation_processor.build_context(list(results))
    ctx_hybrid = citation_processor.build_context_hybrid(list(results))

    t_full = count_tokens(ctx_full)
    t_hybrid = count_tokens(ctx_hybrid)
    delta = t_full - t_hybrid
    pct = (delta / t_full) if t_full > 0 else 0.0

    summary_count = sum(1 for r in results if r.embedding_mode == "summary")
    full_count = sum(1 for r in results if r.embedding_mode == "full")
    meta_count = sum(1 for r in results if r.embedding_mode == "metadata_only")
    unknown_count = sum(1 for r in results if r.embedding_mode is None)

    return QuestionMetric(
        question_id=q.id,
        question=q.question,
        chunks_total=len(results),
        chunks_summary_mode_count=summary_count,
        chunks_full_mode_count=full_count,
        chunks_metadata_only_count=meta_count,
        chunks_unknown_mode_count=unknown_count,
        tokens_full=t_full,
        tokens_hybrid=t_hybrid,
        delta_tokens=delta,
        pct_saved=pct,
        chunk_ids=[r.chunk_id for r in results],
    )


# ---------------------------------------------------------------------------
# Aggregate / report
# ---------------------------------------------------------------------------


def _ascii_histogram(values: Sequence[float], *, bins: int = 10, width: int = 40) -> str:
    """Cheap ASCII histogram of floats in [0, 1]."""
    if not values:
        return "(no data)"
    counts = [0] * bins
    for v in values:
        # Clamp to [0, 1) then bucket
        v = max(0.0, min(0.999999, float(v)))
        idx = int(v * bins)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo = i / bins
        hi = (i + 1) / bins
        bar = "#" * int(round(c / peak * width))
        lines.append(f"  {lo:>5.2f} – {hi:>5.2f} | {bar} {c}")
    return "\n".join(lines)


def _summarise(metrics: List[QuestionMetric]) -> Dict[str, Any]:
    """Aggregate roll-up across all measured questions."""
    if not metrics:
        return {
            "question_count": 0,
            "tokens_full_total": 0,
            "tokens_hybrid_total": 0,
            "delta_total": 0,
            "pct_saved_total": 0.0,
            "questions_with_savings": 0,
            "pct_saved_mean": 0.0,
            "pct_saved_median": 0.0,
            "pct_saved_p95": 0.0,
            "delta_mean": 0.0,
            "delta_median": 0.0,
            "delta_p95": 0,
            "summary_chunks_share": 0.0,
        }
    tokens_full_total = sum(m.tokens_full for m in metrics)
    tokens_hybrid_total = sum(m.tokens_hybrid for m in metrics)
    delta_total = tokens_full_total - tokens_hybrid_total
    pct_saved_total = (
        delta_total / tokens_full_total if tokens_full_total > 0 else 0.0
    )
    pcts = [m.pct_saved for m in metrics]
    deltas = [m.delta_tokens for m in metrics]
    chunks_total = sum(m.chunks_total for m in metrics) or 1
    summary_chunks = sum(m.chunks_summary_mode_count for m in metrics)
    return {
        "question_count": len(metrics),
        "tokens_full_total": tokens_full_total,
        "tokens_hybrid_total": tokens_hybrid_total,
        "delta_total": delta_total,
        "pct_saved_total": pct_saved_total,
        "questions_with_savings": sum(1 for m in metrics if m.delta_tokens > 0),
        "pct_saved_mean": statistics.fmean(pcts),
        "pct_saved_median": statistics.median(pcts),
        "pct_saved_p95": _percentile(pcts, 0.95),
        "delta_mean": statistics.fmean(deltas),
        "delta_median": statistics.median(deltas),
        "delta_p95": _percentile(deltas, 0.95),
        "summary_chunks_share": summary_chunks / chunks_total,
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    """Inclusive nearest-rank percentile. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[idx]


def _render_report(
    *,
    metrics: List[QuestionMetric],
    summary: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Build the markdown report text."""
    lines: List[str] = []
    lines.append("# Context-summary swap — Phase-1 measurement")
    lines.append("")
    lines.append(f"Generated: {config['timestamp']}")
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append(f"- Questions file: `{config.get('questions_path', 'n/a')}`")
    lines.append(f"- Question count: {summary['question_count']}")
    lines.append(f"- Retrieval top_k: {config.get('top_k')}")
    lines.append(f"- Rerank top_n: {config.get('rerank_top_n')}")
    lines.append(f"- Use rerank: {config.get('use_rerank')}")
    lines.append(f"- Mode source: {config.get('mode_source')}")
    lines.append(
        "- Tokeniser: tiktoken `cl100k_base` "
        "(approximation; production tokeniser is OpenRouter-routed and may vary. "
        "Used here for relative-comparison purposes.)"
    )
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total tokens (current `build_context`) | {summary['tokens_full_total']:,} |")
    lines.append(f"| Total tokens (proposed `build_context_hybrid`) | {summary['tokens_hybrid_total']:,} |")
    lines.append(f"| Tokens saved (sum) | {summary['delta_total']:,} |")
    lines.append(f"| % saved (overall) | {summary['pct_saved_total'] * 100:.2f}% |")
    lines.append(f"| Questions with savings | {summary['questions_with_savings']} / {summary['question_count']} |")
    lines.append(f"| Mean % saved per question | {summary['pct_saved_mean'] * 100:.2f}% |")
    lines.append(f"| Median % saved per question | {summary['pct_saved_median'] * 100:.2f}% |")
    lines.append(f"| p95 % saved per question | {summary['pct_saved_p95'] * 100:.2f}% |")
    lines.append(f"| Mean tokens saved per question | {summary['delta_mean']:.1f} |")
    lines.append(f"| Median tokens saved per question | {summary['delta_median']:.1f} |")
    lines.append(f"| p95 tokens saved per question | {summary['delta_p95']:.0f} |")
    lines.append(f"| Share of retrieved chunks in summary mode | {summary['summary_chunks_share'] * 100:.2f}% |")
    lines.append("")

    lines.append("## Histogram of pct_saved per question")
    lines.append("")
    lines.append("```")
    lines.append(_ascii_histogram([m.pct_saved for m in metrics]))
    lines.append("```")
    lines.append("")

    if metrics:
        sorted_by_savings = sorted(metrics, key=lambda m: m.delta_tokens, reverse=True)
        lines.append("## Top 10 questions by absolute savings")
        lines.append("")
        lines.append("| # | id | tokens_full | tokens_hybrid | delta | pct | chunks (summary/total) |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, m in enumerate(sorted_by_savings[:10], start=1):
            lines.append(
                f"| {i} | {m.question_id} | {m.tokens_full:,} | {m.tokens_hybrid:,} | "
                f"{m.delta_tokens:,} | {m.pct_saved * 100:.2f}% | "
                f"{m.chunks_summary_mode_count}/{m.chunks_total} |"
            )
        lines.append("")

        lines.append("## Bottom 10 questions by absolute savings")
        lines.append("")
        lines.append("| # | id | tokens_full | tokens_hybrid | delta | pct | chunks (summary/total) |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, m in enumerate(sorted_by_savings[-10:], start=1):
            lines.append(
                f"| {i} | {m.question_id} | {m.tokens_full:,} | {m.tokens_hybrid:,} | "
                f"{m.delta_tokens:,} | {m.pct_saved * 100:.2f}% | "
                f"{m.chunks_summary_mode_count}/{m.chunks_total} |"
            )
        lines.append("")

        # Risk callouts: questions where >50% of retrieved chunks are summary-mode.
        risky = [
            m for m in metrics
            if m.chunks_total > 0
            and (m.chunks_summary_mode_count / m.chunks_total) > 0.5
        ]
        lines.append("## Risk callouts: questions with >50% summary-mode chunks")
        lines.append("")
        if not risky:
            lines.append("(none — change is conservative across this set.)")
        else:
            lines.append(
                "These are the questions where the swap matters most. "
                "Flag for human review of the actual LLM answer when the "
                "follow-up A/B harness runs."
            )
            lines.append("")
            lines.append("| id | summary/total | pct_saved |")
            lines.append("|---|---|---|")
            for m in risky:
                lines.append(
                    f"| {m.question_id} | "
                    f"{m.chunks_summary_mode_count}/{m.chunks_total} | "
                    f"{m.pct_saved * 100:.2f}% |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke fixture (no DB, no LLM)
# ---------------------------------------------------------------------------


def _smoke_results():
    """Three hand-crafted RetrievalResults with mixed embedding_mode."""
    from mtss.models.chunk import RetrievalResult

    return [
        RetrievalResult(
            text="The hull was inspected on 2025-04-01 and passed all checks. "
                 "Crew confirmed no leakage observed during sea trials.",
            score=0.9,
            chunk_id="aaaaaaaaaaaa",
            doc_id="doc-001",
            source_id="src-001",
            source_title="Hull Inspection Report",
            section_path=[],
            context_summary="Hull inspection report covering sea trial findings.",
            embedding_mode="full",
        ),
        RetrievalResult(
            # Sensor log payload — repetitive numeric noise; in production
            # this would be a multi-thousand-line dump.
            text="\n".join(
                f"2025-04-01T{h:02d}:00:00Z RPM=82.{h % 10} "
                f"TEMP=4{h % 10}.5 PRESS=12.{h % 10}"
                for h in range(0, 24)
            ),
            score=0.82,
            chunk_id="bbbbbbbbbbbb",
            doc_id="doc-002",
            source_id="src-002",
            source_title="Engine Sensor Log",
            section_path=[],
            context_summary=(
                "24-hour engine sensor log: RPM stable around 82, temp 40-49C, "
                "no anomalies recorded."
            ),
            embedding_mode="summary",
        ),
        RetrievalResult(
            text="filename: empty_attachment.pdf",
            score=0.55,
            chunk_id="cccccccccccc",
            doc_id="doc-003",
            source_id="src-003",
            source_title="empty_attachment.pdf",
            section_path=[],
            context_summary=None,
            embedding_mode="metadata_only",
        ),
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def _run_real(
    *,
    questions: List[_Question],
    top_k: int,
    rerank_top_n: int,
    use_rerank: bool,
    sim_threshold: float,
) -> tuple[List[QuestionMetric], str]:
    """Run live retrieval against Supabase and measure each question.

    Returns (metrics, mode_source) where mode_source is a short marker
    for the report explaining where embedding_mode came from.
    """
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.rag.citation_processor import CitationProcessor
    from mtss.rag.reranker import Reranker
    from mtss.rag.retriever import Retriever
    from mtss.storage.supabase_client import SupabaseClient

    db = SupabaseClient()
    embeddings = EmbeddingGenerator()
    reranker = Reranker()
    retriever = Retriever(db=db, embeddings=embeddings, reranker=reranker)
    cp = CitationProcessor()
    count_tokens = _get_token_counter()

    metrics: List[QuestionMetric] = []
    try:
        for q in questions:
            metadata_filter = None
            if q.vessel_filter:
                # Goldens carry vessel_class — production retrieval
                # filters by vessel_ids; without that mapping we skip
                # the filter rather than misroute. Logged for awareness.
                logger.info(
                    "%s: vessel_filter=%r ignored (no class->ids mapping in harness)",
                    q.id, q.vessel_filter,
                )
            try:
                results = await retriever.retrieve(
                    query=q.question,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                    use_rerank=use_rerank,
                    similarity_threshold=sim_threshold,
                    metadata_filter=metadata_filter,
                )
            except Exception as exc:
                logger.error("retrieve failed for %s: %s", q.id, exc)
                continue
            if not results:
                logger.info("%s: no results", q.id)
                continue
            await _hydrate_embedding_modes(db, results)
            metric = _measure_question(
                q,
                results,
                citation_processor=cp,
                count_tokens=count_tokens,
            )
            logger.info(
                "%s: tokens %d -> %d (saved %d, %.1f%%) summary-chunks=%d/%d",
                q.id, metric.tokens_full, metric.tokens_hybrid,
                metric.delta_tokens, metric.pct_saved * 100,
                metric.chunks_summary_mode_count, metric.chunks_total,
            )
            metrics.append(metric)
    finally:
        await db.close()

    return metrics, "live retrieval (Supabase) + side-loaded chunks.embedding_mode"


def _run_smoke() -> tuple[List[QuestionMetric], str]:
    """Run the in-memory fixture path so the report pipeline can be smoke-tested.

    Patches out ``ArchiveStorage`` so the script runs without Supabase
    credentials — the smoke path doesn't touch the network.
    """
    from unittest.mock import patch

    with patch("mtss.rag.citation_processor.ArchiveStorage"):
        from mtss.rag.citation_processor import CitationProcessor

        cp = CitationProcessor()
    count_tokens = _get_token_counter()
    fixture = _smoke_results()

    fake_q = _Question(
        id="smoke-q01",
        question="Smoke fixture — what is the hull inspection status?",
    )
    metric = _measure_question(
        fake_q,
        fixture,
        citation_processor=cp,
        count_tokens=count_tokens,
    )
    return [metric], "smoke fixture (no DB)"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to a YAML/JSON/JSONL questions file.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("reports/context-summary-measurement"),
        help="Directory to write the timestamped markdown + json under.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap N questions.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rerank-top-n", type=int, default=8)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument(
        "--sim-threshold", type=float, default=0.3,
        help="Similarity threshold passed through to the retriever.",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Skip Supabase. Run against the 3-chunk in-memory fixture so "
             "the report pipeline can be exercised without prod credentials.",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.smoke:
        metrics, mode_source = _run_smoke()
        questions_path_repr = "(smoke fixture — REPLACE WITH REAL GOLDENS BEFORE PUBLISHING)"
    else:
        if not args.questions:
            print(
                "ERROR: --questions is required (or pass --smoke). See --help.",
                file=sys.stderr,
            )
            return 2
        if not args.questions.exists():
            print(f"ERROR: {args.questions} not found.", file=sys.stderr)
            return 2

        questions = _load_questions(args.questions)
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            print("ERROR: no questions loaded.", file=sys.stderr)
            return 2

        if not os.environ.get("DATABASE_URL") and not os.environ.get("SUPABASE_URL"):
            logger.warning(
                "Neither DATABASE_URL nor SUPABASE_URL is set in the env; "
                "live retrieval will likely fail. Source your .env first."
            )

        metrics, mode_source = asyncio.run(
            _run_real(
                questions=questions,
                top_k=args.top_k,
                rerank_top_n=args.rerank_top_n,
                use_rerank=not args.no_rerank,
                sim_threshold=args.sim_threshold,
            )
        )
        questions_path_repr = str(args.questions)

    summary = _summarise(metrics)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "timestamp": timestamp,
        "questions_path": questions_path_repr,
        "top_k": args.top_k,
        "rerank_top_n": args.rerank_top_n,
        "use_rerank": (not args.no_rerank) and not args.smoke,
        "mode_source": mode_source,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / f"{timestamp}.md"
    json_path = args.output / f"{timestamp}.json"

    md_path.write_text(_render_report(metrics=metrics, summary=summary, config=config),
                       encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "config": config,
                "summary": summary,
                "metrics": [m.to_dict() for m in metrics],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    print(
        f"questions={summary['question_count']} "
        f"tokens_full={summary['tokens_full_total']} "
        f"tokens_hybrid={summary['tokens_hybrid_total']} "
        f"saved={summary['delta_total']} "
        f"({summary['pct_saved_total'] * 100:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
