"""Per-step latency capture for the eval harness.

Opt-in via ``start_capture()``; zero overhead in production (the CV is
``None`` so ``record_step`` is a yield-only cm). ContextVars propagate
across ``asyncio.create_task``/``gather``, so timings recorded inside
concurrent branches land in the same buffer.

Usage (agent code):
    async with record_step("chat_llm2_ms"):
        response = await invoker.ainvoke(...)

Usage (eval runner):
    buf = start_capture()
    try:
        async for chunk in graph.astream(...): ...
    finally:
        stop_capture()
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, List, Optional, Tuple

_steps: ContextVar[Optional[List[Tuple[str, int]]]] = ContextVar(
    "mtss_step_latencies", default=None
)


def start_capture() -> List[Tuple[str, int]]:
    """Begin capturing step timings; returns the caller-owned buffer."""
    buf: List[Tuple[str, int]] = []
    _steps.set(buf)
    return buf


def stop_capture() -> None:
    _steps.set(None)


@asynccontextmanager
async def record_step(name: str) -> AsyncIterator[None]:
    """Time the wrapped async block and append ``(name, ms)`` to the buffer."""
    buf = _steps.get()
    if buf is None:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        buf.append((name, (time.perf_counter_ns() - t0) // 1_000_000))
