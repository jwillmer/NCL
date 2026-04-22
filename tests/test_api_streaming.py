"""Tests for the /api/agent streaming endpoint.

Covers both request validation and the AI SDK v6 UI Message Stream wire
format emitted by ``src/mtss/api/streaming.py``. The wire format is a
regression-prone contract: v5/v6 clients silently drop legacy ``0:/2:/d:``
frames, so we pin the exact event shapes the backend produces.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestAgentRequestValidation:
    """POST /api/agent — AgentRequest Pydantic validation."""

    @pytest.mark.asyncio
    async def test_streaming_rejects_non_uuid_thread_id(self, client, auth_headers):
        """thread_id must be a valid UUID — non-UUID body → 422."""
        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
                ],
                "thread_id": "not-a-uuid",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_streaming_rejects_missing_thread_id(self, client, auth_headers):
        """thread_id is required — missing → 422."""
        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
                ],
            },
        )
        assert response.status_code == 422


def _stub_graph(events: list[dict]) -> MagicMock:
    """Build a fake agent graph that yields ``events`` from astream_events.

    Also stubs ``aget_state`` so the new-thread branch is taken (no prior
    checkpoint → full message history sent through input_state).
    """

    async def _astream(input_state, config, version):  # noqa: ARG001 - signature mirrors langgraph
        for ev in events:
            yield ev

    async def _aget_state(config):  # noqa: ARG001
        return SimpleNamespace(values={})

    graph = MagicMock()
    graph.astream_events = _astream
    graph.aget_state = _aget_state
    return graph


def _parse_sse(body: str) -> list[dict | str]:
    """Parse an SSE body into a list of parsed events.

    Dict for ``data: {...}`` JSON events, str for the terminating
    ``[DONE]`` sentinel.
    """
    out: list[dict | str] = []
    for raw in body.split("\n\n"):
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            out.append("[DONE]")
        else:
            out.append(json.loads(payload))
    return out


class TestUIMessageStreamWireFormat:
    """Pin the exact SSE events emitted to the v6 AI SDK client."""

    @pytest.mark.asyncio
    async def test_text_delta_stream(self, app, client, auth_headers):
        """on_chat_model_stream chunks → text-start, repeated text-delta, text-end."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="Hello ")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="world")},
            },
        ]
        app.state.agent_graph = _stub_graph(events)

        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"

        parsed = _parse_sse(response.text)
        types = [p.get("type") if isinstance(p, dict) else p for p in parsed]

        assert types[0] == "start"
        assert "messageId" in parsed[0]
        assert "text-start" in types
        # Two deltas, preserving chunk order
        deltas = [p for p in parsed if isinstance(p, dict) and p.get("type") == "text-delta"]
        assert [d["delta"] for d in deltas] == ["Hello ", "world"]
        # All text-delta events share the id from text-start
        text_start = next(p for p in parsed if isinstance(p, dict) and p.get("type") == "text-start")
        assert all(d["id"] == text_start["id"] for d in deltas)
        # Bookend + terminator
        assert "text-end" in types
        assert "finish" in types
        assert parsed[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_custom_data_parts(self, app, client, auth_headers):
        """emit_filter_update + manually_emit_state → data-filter + data-progress parts."""
        events = [
            {
                "event": "on_custom_event",
                "name": "manually_emit_state",
                "data": {"search_progress": "searching vessels"},
            },
            {
                "event": "on_custom_event",
                "name": "emit_filter_update",
                "data": {"vessel_id": None, "vessel_type": "vlcc", "vessel_class": None},
            },
            {
                "event": "on_custom_event",
                "name": "chat_token",
                "data": {"text": "ok"},
            },
        ]
        app.state.agent_graph = _stub_graph(events)

        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "parts": [{"type": "text", "text": "q"}]}],
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
            },
        )
        assert response.status_code == 200

        parsed = _parse_sse(response.text)
        progress = next(p for p in parsed if isinstance(p, dict) and p.get("type") == "data-progress")
        assert progress["data"] == "searching vessels"

        filter_part = next(p for p in parsed if isinstance(p, dict) and p.get("type") == "data-filter")
        assert filter_part["data"] == {
            "vessel_id": None,
            "vessel_type": "vlcc",
            "vessel_class": None,
        }

        deltas = [p for p in parsed if isinstance(p, dict) and p.get("type") == "text-delta"]
        assert [d["delta"] for d in deltas] == ["ok"]

    @pytest.mark.asyncio
    async def test_no_tokens_skips_text_bracket(self, app, client, auth_headers):
        """If the graph never emits a token, we must NOT bracket with text-start/end."""
        app.state.agent_graph = _stub_graph([])

        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "parts": [{"type": "text", "text": "q"}]}],
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
            },
        )
        assert response.status_code == 200

        parsed = _parse_sse(response.text)
        types = [p.get("type") if isinstance(p, dict) else p for p in parsed]
        assert "text-start" not in types
        assert "text-end" not in types
        assert types[0] == "start"
        assert "finish" in types
        assert parsed[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_stream_exception_emits_error(self, app, client, auth_headers):
        """A crash mid-stream yields an error event, not a half-open stream."""

        async def _astream(input_state, config, version):  # noqa: ARG001
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="partial ")},
            }
            raise RuntimeError("graph crashed")

        graph = MagicMock()
        graph.astream_events = _astream
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
        app.state.agent_graph = graph

        response = await client.post(
            "/api/agent",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "parts": [{"type": "text", "text": "q"}]}],
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
            },
        )
        assert response.status_code == 200

        parsed = _parse_sse(response.text)
        types = [p.get("type") if isinstance(p, dict) else p for p in parsed]
        # text-start was emitted (there was a partial delta) → must be closed
        assert "text-start" in types
        assert "text-end" in types
        error_part = next(p for p in parsed if isinstance(p, dict) and p.get("type") == "error")
        assert error_part["errorText"]
        assert parsed[-1] == "[DONE]"
