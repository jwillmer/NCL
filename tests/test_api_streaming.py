"""Tests for the /api/agent streaming endpoint — request validation."""

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
