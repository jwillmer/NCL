"""Shared dataclasses for the eval framework.

These define the contracts between Phase 1 (run), Phase 2 (judge), and Phase 3
(diff). Everything written to disk in `runs/<id>/` round-trips through these
models — no ad-hoc dicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Inputs (read from goldens/questions.yaml)
# =============================================================================


class GoldenQuestion(BaseModel):
    """One labeled question + reference answer + optional retrieval labels."""

    id: str
    category: str
    question: str
    reference_answer: str
    expected_facts: List[str] = Field(
        default_factory=list,
        description="Bulleted key claims used by the human reviewer as a "
        "checklist when comparing the agent's response to the reference. "
        "Each fact is one sentence.",
    )
    vessel_filter: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional filter: {'vessel_id': '...'} | {'vessel_type': 'VLCC'} | "
        "{'vessel_class': 'Canopus'}. Mutually exclusive.",
    )
    expected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Optional labeled relevant chunk_ids for recall@k / MRR. "
        "Empty list = retrieval graded only by stability vs prior runs.",
    )
    notes: Optional[str] = None


# =============================================================================
# Phase 1 outputs (results.jsonl)
# =============================================================================


class RetrievedChunk(BaseModel):
    """One chunk that came back from the retriever, in rank order."""

    rank: int
    chunk_id: str
    doc_id: str
    score: float
    rerank_score: Optional[float] = None
    text_preview: str = Field(default="", description="First 240 chars of chunk text")
    email_subject: Optional[str] = None
    email_date: Optional[str] = None
    file_path: Optional[str] = None
    document_type: Optional[str] = None


class CitationOccurrence(BaseModel):
    """A [C:chunk_id] reference found in the agent's response."""

    chunk_id: str
    char_offset: int
    is_valid: bool = Field(description="Whether chunk_id resolved against citation_map")


class TopicFilterTrace(BaseModel):
    detected: List[str] = Field(default_factory=list)
    matched: List[str] = Field(default_factory=list)
    unmatched: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    should_skip_rag: bool = False


class RunMetrics(BaseModel):
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    # Per-step breakdown populated by ``step_timing.record_step``. Keys:
    # intent_ms, chat_llm1_ms, chat_llm2_ms, topic_filter_ms, embed_ms,
    # search_rerank_ms, validate_ms. Repeats in the same run are summed
    # (e.g. multi-turn chats fire chat_llm1_ms more than once).
    # Optional so pre-instrumentation runs still parse.
    step_latencies_ms: Optional[Dict[str, int]] = None


class TraceLink(BaseModel):
    langfuse_trace_id: Optional[str] = None
    langfuse_url: Optional[str] = None
    session_id: Optional[str] = None


class RunResult(BaseModel):
    """One question's full execution log. One per line in results.jsonl."""

    question_id: str
    question: str
    response: str
    retrieval: List[RetrievedChunk] = Field(default_factory=list)
    topic_filter: Optional[TopicFilterTrace] = None
    citations: List[CitationOccurrence] = Field(default_factory=list)
    incident_count: int = 0
    unique_incidents: int = 0
    metrics: RunMetrics
    trace: TraceLink = Field(default_factory=TraceLink)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RunManifest(BaseModel):
    """runs/<id>/manifest.json — describes the configuration of a run."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    git_sha: Optional[str] = None
    env_file: str
    questions_file: str
    question_count: int
    settings_snapshot: Dict[str, Any] = Field(
        description="Subset of Settings that affect retrieval/generation: "
        "llm_model, embedding_model, retrieval_top_k, rerank_*, hybrid_search, "
        "topic thresholds. Used to compare runs."
    )
    notes: Optional[str] = None


# =============================================================================
# Phase 2 outputs (scores.jsonl)
# =============================================================================


class AutoScores(BaseModel):
    """Deterministic, no-LLM scores. Always computed."""

    citations_count: int = 0
    citations_valid_count: int = 0
    citations_valid_pct: float = 0.0
    citation_grounding_score: float = Field(
        default=0.0,
        description="Mean cosine similarity between each cited chunk's text and "
        "its surrounding claim sentence in the response. 1.0 = perfect grounding.",
    )
    response_length_chars: int = 0
    has_no_results_disclaimer: bool = False
    # Retrieval (only computed if golden has expected_chunk_ids)
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    # Format adherence
    follows_response_format: bool = False
    format_violations: List[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """One question's full scoring. One per line in scores.jsonl."""

    question_id: str
    auto: AutoScores
    overall: float = Field(
        default=0.0,
        description="Weighted aggregate of auto-grader signals, 0-1.",
    )


class ScoreSummary(BaseModel):
    """runs/<id>/summary.json — aggregate scores across all questions."""

    run_id: str
    question_count: int
    scored_count: int
    auto_aggregates: Dict[str, float]
    overall_mean: float = 0.0
    overall_p50: float = 0.0
    overall_p10: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    failures: List[str] = Field(default_factory=list)
