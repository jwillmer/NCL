"""Tests for SSE-aware GZip middleware.

The streaming ``/api/agent`` endpoint uses Server-Sent Events; stock
GZipMiddleware buffers enough of the response to decide on compression,
which breaks the Vercel AI SDK stream consumer. Our ``SSEAwareGZipMiddleware``
bypasses compression for that path while still compressing normal JSON.
"""

from __future__ import annotations

import pytest


class TestGZipOnJsonEndpoints:
    """Regular JSON endpoints must be compressed when large enough."""

    @pytest.mark.asyncio
    async def test_large_json_response_is_gzipped(self, client, auth_headers, app):
        """The list endpoint returning many conversations should be gzipped.

        The minimum_size threshold is 500 bytes; a single row of mock
        JSON is ~400 bytes, so 10 rows blow past the threshold.
        """
        from unittest.mock import MagicMock
        from uuid import uuid4

        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        rows = [
            {
                "id": str(uuid4()),
                "thread_id": str(uuid4()),
                "user_id": str(uuid4()),
                "title": f"Conversation number {i} with a reasonably long title to push payload over gzip minimum",
                "vessel_id": None,
                "vessel_type": None,
                "vessel_class": None,
                "is_archived": False,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "last_message_at": "2024-01-02T00:00:00+00:00",
            }
            for i in range(10)
        ]
        mock_table.execute.return_value = MagicMock(data=rows, count=len(rows))

        response = await client.get(
            "/api/conversations",
            headers={**auth_headers, "Accept-Encoding": "gzip"},
        )

        assert response.status_code == 200
        assert response.headers.get("content-encoding") == "gzip"

        # Reset
        mock_table.execute.return_value = MagicMock(data=[], count=0)


class TestGZipSkipsSSE:
    """Requests to the streaming agent path must never be gzipped."""

    @pytest.mark.asyncio
    async def test_agent_endpoint_bypasses_gzip(self, client, auth_headers):
        """POST /api/agent must not advertise Content-Encoding: gzip."""
        response = await client.post(
            "/api/agent",
            headers={**auth_headers, "Accept-Encoding": "gzip"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        # Regardless of whether the mocked graph returns 200 / 500, the
        # gzip middleware MUST NOT have compressed the response.
        assert response.headers.get("content-encoding") != "gzip"
