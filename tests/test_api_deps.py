"""Tests for the FastAPI dependency-injection helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_get_supabase_client_returns_singleton(comprehensive_mock_settings):
    """Two calls to get_supabase_client() must return the same instance.

    The whole point of the helper is to avoid per-request SupabaseClient()
    construction. If this invariant breaks, every API handler pays the
    PostgREST + asyncpg-pool bootstrap cost again.
    """
    from mtss.api.deps import get_supabase_client

    fake_rest_client = MagicMock()

    def patched_init(self):
        self.client = fake_rest_client
        self.db_url = "postgresql://test:test@localhost/test"
        self._docs = MagicMock()
        self._search = MagicMock()
        self._domain = MagicMock()
        self._pool = None

    get_supabase_client.cache_clear()
    with (
        patch("mtss.storage.supabase_client.get_settings", return_value=comprehensive_mock_settings),
        patch("mtss.storage.supabase_client.create_client", return_value=fake_rest_client),
        patch("mtss.storage.supabase_client.SupabaseClient.__init__", patched_init),
    ):
        first = get_supabase_client()
        second = get_supabase_client()

    assert first is second, "get_supabase_client() must return the same instance"


def test_get_supabase_client_cache_clear_returns_new_instance(comprehensive_mock_settings):
    """cache_clear() must produce a fresh instance on the next call.

    Tests rely on this to get a clean mock for each API test.
    """
    from mtss.api.deps import get_supabase_client

    fake_rest_client = MagicMock()

    def patched_init(self):
        self.client = fake_rest_client
        self.db_url = "postgresql://test:test@localhost/test"
        self._docs = MagicMock()
        self._search = MagicMock()
        self._domain = MagicMock()
        self._pool = None

    get_supabase_client.cache_clear()
    with (
        patch("mtss.storage.supabase_client.get_settings", return_value=comprehensive_mock_settings),
        patch("mtss.storage.supabase_client.create_client", return_value=fake_rest_client),
        patch("mtss.storage.supabase_client.SupabaseClient.__init__", patched_init),
    ):
        first = get_supabase_client()
        get_supabase_client.cache_clear()
        second = get_supabase_client()

    assert first is not second, "cache_clear() must yield a fresh instance"
