"""Tests for conversations API: CRUD operations, auth isolation, validation."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


def _mock_conversation_row(**overrides):
    """Build a mock conversation database row."""
    defaults = {
        "id": str(uuid4()),
        "thread_id": str(uuid4()),
        "user_id": str(uuid4()),
        "title": "Test Conversation",
        "vessel_id": None,
        "vessel_type": None,
        "vessel_class": None,
        "is_archived": False,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "last_message_at": None,
    }
    defaults.update(overrides)
    return defaults


class TestListConversations:
    """GET /api/conversations"""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, client, auth_headers):
        response = await client.get("/api/conversations", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        # `total` is intentionally None now (count="exact" was dropped for
        # performance — it forces a full-table scan). Clients should rely
        # on `has_more` instead.
        assert data["total"] is None
        assert data["has_more"] is False

    @pytest.mark.asyncio
    async def test_filters_by_user_id(self, client, auth_headers, app):
        """Must filter conversations by the authenticated user's ID."""
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value

        await client.get("/api/conversations", headers=auth_headers)

        # Verify .eq("user_id", "test-user-id") was called
        mock_table.eq.assert_any_call("user_id", "test-user-id")

    @pytest.mark.asyncio
    async def test_with_results(self, client, auth_headers, app):
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        rows = [_mock_conversation_row(), _mock_conversation_row()]
        mock_table.execute.return_value = MagicMock(data=rows, count=2)

        response = await client.get("/api/conversations", headers=auth_headers)
        assert response.status_code == 200
        assert len(response.json()["items"]) == 2

        # Reset for other tests
        mock_table.execute.return_value = MagicMock(data=[], count=0)


class TestCreateConversation:
    """POST /api/conversations"""

    @pytest.mark.asyncio
    async def test_create_minimal(self, client, auth_headers, app):
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        row = _mock_conversation_row()
        # insert().execute() returns list with single item
        mock_table.execute.return_value = MagicMock(data=[row])

        response = await client.post("/api/conversations", headers=auth_headers, json={})
        assert response.status_code == 201

        # Reset
        mock_table.execute.return_value = MagicMock(data=[], count=0)

    @pytest.mark.asyncio
    async def test_title_max_length_validation(self, client, auth_headers):
        """Title exceeding max_length should be rejected."""
        response = await client.post(
            "/api/conversations",
            headers=auth_headers,
            json={"title": "x" * 201},
        )
        assert response.status_code == 422


class TestGetConversation:
    """GET /api/conversations/{thread_id}"""

    @pytest.mark.asyncio
    async def test_not_found(self, client, auth_headers):
        """Non-existent conversation returns 404."""
        response = await client.get(
            f"/api/conversations/{uuid4()}", headers=auth_headers
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_found(self, client, auth_headers, app):
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        thread_id = str(uuid4())
        # maybe_single().execute() returns dict directly as .data (not a list)
        row = _mock_conversation_row(thread_id=thread_id)
        mock_table.execute.return_value = MagicMock(data=row)

        response = await client.get(
            f"/api/conversations/{thread_id}", headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["thread_id"] == thread_id

        # Reset
        mock_table.execute.return_value = MagicMock(data=[], count=0)


class TestThreadIdValidation:
    """Path params validated as UUID (cache-key pollution guard)."""

    @pytest.mark.asyncio
    async def test_conversations_rejects_non_uuid_thread_id(self, client, auth_headers):
        """GET /api/conversations/{thread_id} with non-UUID path param → 422."""
        response = await client.get(
            "/api/conversations/not-a-uuid", headers=auth_headers
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_conversations_patch_rejects_non_uuid_thread_id(self, client, auth_headers):
        response = await client.patch(
            "/api/conversations/not-a-uuid",
            headers=auth_headers,
            json={"title": "new"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_conversations_delete_rejects_non_uuid_thread_id(self, client, auth_headers):
        response = await client.delete(
            "/api/conversations/not-a-uuid", headers=auth_headers
        )
        assert response.status_code == 422


class TestDeleteConversation:
    """DELETE /api/conversations/{thread_id}"""

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client, auth_headers):
        response = await client.delete(
            f"/api/conversations/{uuid4()}", headers=auth_headers
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_success(self, client, auth_headers, app):
        mock_rest = app.state.mock_supabase_rest_client
        mock_table = mock_rest.table.return_value
        thread_id = str(uuid4())
        # maybe_single returns dict for ownership check, then delete chains
        row = _mock_conversation_row(thread_id=thread_id)
        mock_table.execute.return_value = MagicMock(data=row)

        response = await client.delete(
            f"/api/conversations/{thread_id}", headers=auth_headers
        )
        # Should succeed (200 or 204)
        assert response.status_code in (200, 204)

        # Reset
        mock_table.execute.return_value = MagicMock(data=[], count=0)
