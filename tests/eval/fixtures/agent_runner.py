"""Phase 1 executor: runs one golden question through the LangGraph agent.

Mirrors the production path in src/mtss/api/streaming.py but:
  - Uses MemorySaver (fresh per question — no thread continuation)
  - Captures intermediate state (search_node's citation_map) via astream(updates)
  - Records token counts via a local callback as well as Langfuse
  - Emits a fully-typed RunResult, not an HTTP stream

The graph itself is unmodified — eval runs the same code that serves users.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from mtss.api.agent import create_graph
from mtss.observability import (
    create_trace_id_for_thread,
    flush_langfuse_traces,
    get_langfuse_handler,
    set_session_id,
)
from mtss.observability.step_timing import start_capture, stop_capture

from ..types import (
    CitationOccurrence,
    GoldenQuestion,
    RetrievedChunk,
    RunMetrics,
    RunResult,
    TopicFilterTrace,
    TraceLink,
)
from .callbacks import TokenCounterCallback

logger = logging.getLogger(__name__)

# Matches either the pre-validate marker ``[C:chunk_id]`` *or* the
# post-validate HTML tag ``<cite id="chunk_id" ...>N</cite>`` that the
# validate_response_node writes. We count both so the eval regex stays
# correct regardless of whether validation has run.
CITATION_RE = re.compile(
    r"\[C:([a-f0-9]+)\]|<cite[^>]*\bid=\"([a-f0-9]+)\""
)


async def run_question(
    question: GoldenQuestion,
    *,
    run_id: str,
    session_prefix: str = "eval",
    langfuse_tags: Optional[List[str]] = None,
) -> RunResult:
    """Execute one golden question end-to-end through the LangGraph agent.

    Args:
        question: One golden question to run.
        run_id: Identifier for this eval run (groups all questions together).
        session_prefix: Langfuse session prefix; the full session_id becomes
            ``{prefix}-{run_id}-{question.id}`` so prod analytics filters can
            cleanly exclude eval traffic.
        langfuse_tags: Optional list appended to Langfuse trace metadata.

    Returns:
        A fully populated RunResult, written by the orchestrator to results.jsonl.
    """
    session_id = f"{session_prefix}-{run_id}-{question.id}"
    set_session_id(session_id)

    # Per-question fresh checkpointer: no message carry-over between questions
    graph = create_graph(MemorySaver())

    token_counter = TokenCounterCallback()
    callbacks: List[Any] = [token_counter]
    handler = get_langfuse_handler()
    if handler is not None:
        callbacks.append(handler)

    metadata: Dict[str, Any] = {
        "langfuse_session_id": session_id,
        "eval_run_id": run_id,
        "eval_question_id": question.id,
    }
    if langfuse_tags:
        metadata["langfuse_tags"] = langfuse_tags

    config: Dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "callbacks": callbacks,
        "metadata": metadata,
    }

    # Build initial state
    input_state: Dict[str, Any] = {"messages": [HumanMessage(content=question.question)]}
    vf = question.vessel_filter or {}
    if vid := vf.get("vessel_id"):
        input_state["selected_vessel_id"] = vid
    elif vt := vf.get("vessel_type"):
        input_state["selected_vessel_type"] = vt
    elif vc := vf.get("vessel_class"):
        input_state["selected_vessel_class"] = vc

    # Stream with state-level updates so we can capture search_node's citation_map
    # (the agent clears citation_map before END, so .ainvoke() loses the retrieval list)
    citation_map_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    step_buf = start_capture()
    started = time.monotonic()

    try:
        async for chunk in graph.astream(input_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "search_node" and update.get("citation_map"):
                    citation_map_data = update["citation_map"]
    except Exception as exc:
        logger.exception("Agent run failed for question %s", question.id)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_capture()

    latency_ms = int((time.monotonic() - started) * 1000)

    step_latencies_ms: Optional[Dict[str, int]] = None
    if step_buf:
        agg: Dict[str, int] = {}
        for name, ms in step_buf:
            agg[name] = agg.get(name, 0) + ms
        step_latencies_ms = agg

    # Pull final state to extract response + ToolMessage trace
    final_state = await graph.aget_state(config)
    messages: List[Any] = final_state.values.get("messages", []) if final_state.values else []

    response_text, tool_message = _extract_response_and_tool(messages)
    topic_filter = _extract_topic_filter(tool_message)
    incident_count, unique_incidents = _extract_incident_counts(tool_message)
    retrieval = _build_retrieval(citation_map_data)
    citations = _extract_citations(response_text, citation_map_data)

    trace_id = None
    if handler is not None:
        try:
            trace_id = create_trace_id_for_thread(session_id)
        except Exception:
            trace_id = None

    metrics = RunMetrics(
        latency_ms=latency_ms,
        input_tokens=token_counter.input_tokens,
        output_tokens=token_counter.output_tokens,
        cost_usd=round(token_counter.cost_usd, 6),
        tool_calls=token_counter.tool_calls,
        step_latencies_ms=step_latencies_ms,
    )

    # Best-effort flush so traces are visible in Langfuse before next question
    try:
        flush_langfuse_traces()
    except Exception:
        pass

    return RunResult(
        question_id=question.id,
        question=question.question,
        response=response_text,
        retrieval=retrieval,
        topic_filter=topic_filter,
        citations=citations,
        incident_count=incident_count,
        unique_incidents=unique_incidents,
        metrics=metrics,
        trace=TraceLink(langfuse_trace_id=trace_id, session_id=session_id),
        error=error,
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# Extraction helpers
# =============================================================================


def _extract_response_and_tool(messages: List[Any]) -> tuple[str, Optional[ToolMessage]]:
    """Find the final assistant response + the search tool's response message."""
    response_text = ""
    tool_message: Optional[ToolMessage] = None

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_message = msg
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                # Final response (no further tool calls)
                content = msg.content
                if isinstance(content, list):
                    response_text = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                else:
                    response_text = str(content) if content else ""

    return response_text, tool_message


def _extract_topic_filter(tool_message: Optional[ToolMessage]) -> Optional[TopicFilterTrace]:
    if tool_message is None:
        return None
    try:
        payload = json.loads(tool_message.content)
    except (json.JSONDecodeError, TypeError):
        return None
    info = payload.get("topic_info") or {}
    if not info:
        return None
    return TopicFilterTrace(
        detected=list(info.get("detected") or []),
        matched=list(info.get("matched") or []),
        unmatched=list(info.get("unmatched") or []),
        chunk_count=int(info.get("chunk_count") or 0),
        should_skip_rag=bool(info.get("should_skip") or False),
    )


def _extract_incident_counts(tool_message: Optional[ToolMessage]) -> tuple[int, int]:
    if tool_message is None:
        return 0, 0
    try:
        payload = json.loads(tool_message.content)
    except (json.JSONDecodeError, TypeError):
        return 0, 0
    return (
        int(payload.get("incident_count") or 0),
        int(payload.get("unique_incidents") or 0),
    )


def _build_retrieval(citation_map_data: Optional[Dict[str, Any]]) -> List[RetrievedChunk]:
    """Convert serialized citation_map into rank-ordered RetrievedChunk list.

    Preserved field name ``rank`` because dict insertion order in Python 3.7+
    matches the rerank order returned by the retriever; we just enumerate.
    """
    if not citation_map_data:
        return []

    out: List[RetrievedChunk] = []
    for rank, (chunk_id, serialized) in enumerate(citation_map_data.items(), start=1):
        text = serialized.get("text") or ""
        out.append(RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id,
            doc_id=serialized.get("doc_id") or "",
            score=float(serialized.get("score") or 0.0),
            rerank_score=(
                float(serialized["rerank_score"])
                if serialized.get("rerank_score") is not None
                else None
            ),
            text_preview=text[:240],
            email_subject=serialized.get("email_subject"),
            email_date=serialized.get("email_date"),
            file_path=serialized.get("file_path"),
            document_type=serialized.get("document_type"),
        ))
    return out


def _extract_citations(
    response: str,
    citation_map_data: Optional[Dict[str, Any]],
) -> List[CitationOccurrence]:
    valid_ids = set((citation_map_data or {}).keys())
    occurrences: List[CitationOccurrence] = []
    for match in CITATION_RE.finditer(response):
        # One of the two capture groups will be populated depending on
        # whether the match was [C:...] or <cite id="...">.
        chunk_id = match.group(1) or match.group(2)
        occurrences.append(CitationOccurrence(
            chunk_id=chunk_id,
            char_offset=match.start(),
            is_valid=chunk_id in valid_ids,
        ))
    return occurrences


# =============================================================================
# Bounded concurrency helper for the orchestrator
# =============================================================================


async def run_questions(
    questions: List[GoldenQuestion],
    *,
    run_id: str,
    concurrency: int = 2,
    session_prefix: str = "eval",
    langfuse_tags: Optional[List[str]] = None,
    on_complete: Optional[Any] = None,
) -> List[RunResult]:
    """Run many questions with bounded concurrency.

    ``on_complete(result)`` (sync callable) fires after each question — the
    orchestrator uses it to append-write results.jsonl as it goes, so a
    crashed run still produces partial output.
    """
    # Pre-warm process-wide caches (topics + vessels) so the first query
    # doesn't pay the ~5-7s cold load latency on top of its normal budget.
    try:
        from mtss.processing.entity_cache import warm_caches
        from mtss.storage.supabase_client import SupabaseClient

        await warm_caches(SupabaseClient())
    except Exception as exc:  # pragma: no cover — warming is best-effort
        logger.warning("Cache pre-warm failed (non-fatal): %s", exc)

    sem = asyncio.Semaphore(concurrency)
    results: List[RunResult] = []

    async def _bounded(q: GoldenQuestion) -> RunResult:
        async with sem:
            r = await run_question(
                q,
                run_id=run_id,
                session_prefix=session_prefix,
                langfuse_tags=langfuse_tags,
            )
            if on_complete is not None:
                on_complete(r)
            return r

    tasks = [asyncio.create_task(_bounded(q)) for q in questions]
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
    # Sort by question order for deterministic output
    qid_order = {q.id: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: qid_order.get(r.question_id, 0))
    return results
