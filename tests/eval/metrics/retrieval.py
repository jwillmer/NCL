"""Retrieval-quality deterministic scoring for RAG eval runs.

Given a `RunResult.retrieval` (rank-ordered chunks) and a
`GoldenQuestion.expected_chunk_ids` (the labeled relevant set), computes
recall@{5,10,20}, reciprocal rank of the first relevant hit in the top
20, and nDCG@10 with binary relevance. Returns all None when the golden
has no labeled chunks.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..types import GoldenQuestion, RetrievedChunk, RunResult


def _top_k_ids(retrieval: List[RetrievedChunk], k: int) -> List[str]:
    """Top-k chunk_ids in rank order (rank 1 = most relevant)."""
    ordered = sorted(retrieval, key=lambda r: r.rank)
    return [c.chunk_id for c in ordered[:k]]


def _recall_at_k(relevant: set[str], retrieved_top_k: List[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved_top_k if cid in relevant)
    return hits / len(relevant)


def _mrr(relevant: set[str], retrieved: List[str]) -> float:
    """Reciprocal rank of the first relevant hit (1-indexed). 0 if none."""
    for idx, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(relevant: set[str], retrieved_top_k: List[str], k: int) -> float:
    """Binary-relevance nDCG@k. Standard formula:
        DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
        IDCG@k = DCG for the ideal ranking (all relevants first, up to k).
    Returns 0.0 when there are no relevant items.
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved_top_k[:k], start=1):
        rel = 1.0 if cid in relevant else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_retrieval(run: RunResult, golden: GoldenQuestion) -> dict:
    """Score retrieval quality against a golden's expected_chunk_ids.

    When `golden.expected_chunk_ids` is empty, all metrics are None —
    retrieval for that question is evaluated only by run-to-run stability
    elsewhere in the pipeline.
    """
    expected = list(golden.expected_chunk_ids or [])
    if not expected:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "recall_at_20": None,
            "mrr": None,
            "ndcg_at_10": None,
        }

    relevant: set[str] = set(expected)
    top5 = _top_k_ids(run.retrieval, 5)
    top10 = _top_k_ids(run.retrieval, 10)
    top20 = _top_k_ids(run.retrieval, 20)

    recall_5: Optional[float] = _recall_at_k(relevant, top5)
    recall_10: Optional[float] = _recall_at_k(relevant, top10)
    recall_20: Optional[float] = _recall_at_k(relevant, top20)
    mrr: Optional[float] = _mrr(relevant, top20)
    ndcg10: Optional[float] = _ndcg_at_k(relevant, top10, 10)

    return {
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "recall_at_20": recall_20,
        "mrr": mrr,
        "ndcg_at_10": ndcg10,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    from ..types import GoldenQuestion as _GQ
    from ..types import RunMetrics, RunResult as _RR

    retrieval = [
        RetrievedChunk(rank=1, chunk_id="aaa", doc_id="d1", score=0.9),
        RetrievedChunk(rank=2, chunk_id="bbb", doc_id="d2", score=0.8),
        RetrievedChunk(rank=3, chunk_id="ccc", doc_id="d3", score=0.7),
        RetrievedChunk(rank=4, chunk_id="ddd", doc_id="d4", score=0.6),
        RetrievedChunk(rank=5, chunk_id="eee", doc_id="d5", score=0.5),
    ]
    run = _RR(
        question_id="q1",
        question="?",
        response="",
        retrieval=retrieval,
        metrics=RunMetrics(latency_ms=1),
    )
    golden = _GQ(
        id="q1",
        category="test",
        question="?",
        reference_answer="",
        expected_chunk_ids=["bbb", "ddd", "zzz"],
    )
    print(score_retrieval(run, golden))
    print(score_retrieval(run, _GQ(id="q2", category="t", question="?", reference_answer="")))
