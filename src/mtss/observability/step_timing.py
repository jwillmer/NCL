"""Per-step latency capture for the eval harness.

Background:
    Total per-question latency (``results.jsonl:metrics.latency_ms``) is a
    black box. To compare models, routing changes, or prompt tweaks we need
    a breakdown — how much of the 60s went to intent classification vs
    retrieval vs answer generation. This module provides exactly that,
    without changing agent behavior.

Design:
    * A single ``ContextVar`` holds a per-task list of ``(step_name, ms)``
      pairs. The context var is ``None`` by default, so the ``record_step``
      context manager is a zero-work yield in production — literally one
      ``ContextVar.get()`` lookup per call site per request.
    * The eval runner opts in via ``start_capture()`` before starting
      ``graph.astream(...)`` and aggregates the captured tuples afterwards.
    * ContextVars are asyncio-native: they propagate automatically through
      ``asyncio.create_task``/``gather``, so timings recorded inside
      concurrent branches (e.g. the ``asyncio.gather(filter, embed)`` inside
      ``search_node``) land in the same buffer.

Usage (agent code):
    from mtss.observability.step_timing import record_step

    async with record_step("chat_llm2_ms"):
        response = await invoker.ainvoke(...)

Usage (eval runner):
    from mtss.observability.step_timing import start_capture, stop_capture

    buf = start_capture()
    try:
        async for chunk in graph.astream(...):
            ...
    finally:
        stop_capture()

    step_latencies_ms: Dict[str, int] = {}
    for name, ms in buf:
        step_latencies_ms[name] = step_latencies_ms.get(name, 0) + ms

Naming convention:
    All keys end in ``_ms`` and describe the wrapped operation directly
    (``intent_ms``, ``chat_llm1_ms``, ``chat_llm2_ms``, ``retrieval_ms``,
    ``embed_ms``, ``vector_ms``, ``rerank_ms``, ``validate_ms``). A key can
    fire more than once in a run (e.g. intent on a multi-turn conversation);
    the runner sums repeats rather than storing a list, so consumers read
    ``total ms in step X`` directly.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, List, Optional, Tuple

# ``None`` means "capture is off" — the common case in production.
_steps: ContextVar[Optional[List[Tuple[str, int]]]] = ContextVar(
    "mtss_step_latencies", default=None
)


def start_capture() -> List[Tuple[str, int]]:
    """Begin capturing step timings in the current async task.

    Returns the buffer the caller owns. Subsequent ``record_step(...)`` calls
    in this task (and its child tasks via ContextVar propagation) append to
    this buffer. Call ``stop_capture()`` when done to restore the no-op
    default.
    """
    buf: List[Tuple[str, int]] = []
    _steps.set(buf)
    return buf


def stop_capture() -> None:
    """Disable further step capture in the current task."""
    _steps.set(None)


@asynccontextmanager
async def record_step(name: str) -> AsyncIterator[None]:
    """Time the wrapped async block and append ``(name, ms)`` to the buffer.

    When no buffer is active (ContextVar is ``None``), this is a plain
    ``yield`` with a single ContextVar lookup — production overhead is
    effectively nil.
    """
    buf = _steps.get()
    if buf is None:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        buf.append((name, (time.perf_counter_ns() - t0) // 1_000_000))
