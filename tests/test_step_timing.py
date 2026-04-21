"""Tests for src/mtss/observability/step_timing.py.

The step-timing primitive needs to be:
  * a no-op when no capture is active (prod path — never pay for it)
  * correct when capture IS active (eval path — accurate per-step numbers)
  * async-safe across gather/create_task (we record inside gathered coroutines)
"""

from __future__ import annotations

import asyncio

import pytest

from mtss.observability.step_timing import (
    _steps,
    record_step,
    start_capture,
    stop_capture,
)


@pytest.fixture(autouse=True)
def _reset_context_var():
    """Every test starts with capture OFF."""
    token = _steps.set(None)
    yield
    _steps.reset(token)


class TestNoOpWhenCaptureOff:
    @pytest.mark.asyncio
    async def test_record_step_is_noop_when_capture_off(self):
        """Without start_capture, record_step just yields — no state changes."""
        assert _steps.get() is None
        async with record_step("anything"):
            await asyncio.sleep(0)
        # Still None; no buffer was silently created.
        assert _steps.get() is None


class TestCaptureActive:
    @pytest.mark.asyncio
    async def test_single_step_records_ms(self):
        buf = start_capture()
        try:
            async with record_step("thing_ms"):
                await asyncio.sleep(0.01)
        finally:
            stop_capture()

        assert len(buf) == 1
        name, ms = buf[0]
        assert name == "thing_ms"
        # Should be >= 10ms, but allow generous ceiling for CI jitter.
        assert ms >= 5, f"expected at least 5ms, got {ms}"
        assert ms < 5000, f"unreasonably slow: {ms}ms"

    @pytest.mark.asyncio
    async def test_multiple_steps_preserve_order(self):
        buf = start_capture()
        try:
            async with record_step("first"):
                pass
            async with record_step("second"):
                pass
        finally:
            stop_capture()
        assert [name for name, _ in buf] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_repeated_step_name_both_captured(self):
        """A step can fire twice (e.g. chat_llm1 on a multi-turn chat)."""
        buf = start_capture()
        try:
            async with record_step("chat_llm1_ms"):
                pass
            async with record_step("chat_llm1_ms"):
                pass
        finally:
            stop_capture()
        names = [n for n, _ in buf]
        assert names.count("chat_llm1_ms") == 2

    @pytest.mark.asyncio
    async def test_stop_capture_disables_further_recording(self):
        buf = start_capture()
        async with record_step("before_stop"):
            pass
        stop_capture()
        async with record_step("after_stop"):
            pass
        assert [n for n, _ in buf] == ["before_stop"]


class TestAsyncPropagation:
    """ContextVars propagate into asyncio.gather / create_task so timings
    inside concurrent branches land in the caller's buffer."""

    @pytest.mark.asyncio
    async def test_gather_tasks_share_buffer(self):
        async def _work(name):
            async with record_step(name):
                await asyncio.sleep(0)

        buf = start_capture()
        try:
            await asyncio.gather(_work("left"), _work("right"))
        finally:
            stop_capture()
        names = sorted(n for n, _ in buf)
        assert names == ["left", "right"]

    @pytest.mark.asyncio
    async def test_exception_still_records_timing(self):
        """A raised exception still records the step duration via finally."""
        buf = start_capture()
        try:
            try:
                async with record_step("explodes_ms"):
                    raise ValueError("boom")
            except ValueError:
                pass
        finally:
            stop_capture()
        assert len(buf) == 1
        assert buf[0][0] == "explodes_ms"
