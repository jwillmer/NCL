"""Tests for feedback API input validation."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestFeedbackValidation:
    """POST /api/feedback — verify Pydantic validation on request body."""

    @pytest.mark.asyncio
    async def test_valid_thumbs_up(self, client, auth_headers, app):
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        mock_table.execute.return_value = MagicMock(data=[{"id": 1}])

        with patch("mtss.api.feedback.get_langfuse_client", return_value=None):
            response = await client.post(
                "/api/feedback",
                headers=auth_headers,
                json={
                    "thread_id": str(uuid4()),
                    "message_id": "msg-001",
                    "value": 1,
                },
            )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_invalid_value_rejected(self, client, auth_headers):
        """Feedback value must be 0 or 1, not 2."""
        response = await client.post(
            "/api/feedback",
            headers=auth_headers,
            json={
                "thread_id": str(uuid4()),
                "message_id": "msg-001",
                "value": 2,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_thread_id_rejected(self, client, auth_headers):
        """thread_id is required."""
        response = await client.post(
            "/api/feedback",
            headers=auth_headers,
            json={"message_id": "msg-001", "value": 1},
        )
        assert response.status_code == 422
