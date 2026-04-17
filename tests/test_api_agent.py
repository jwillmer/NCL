"""Tests for src/mtss/api/agent.py helpers.

Focus: the bounded LRU cache behind ``_get_vessel_info``. The cache must stay
within ``_VESSEL_CACHE_MAXSIZE`` entries so long-running API processes don't
grow unboundedly as new vessel_ids are queried, and it must promote hits so
that the least-recently-used vessel is evicted (not an arbitrary one).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from mtss.api import agent as agent_module


def _mk_vessel(name: str):
    """Build a minimal Vessel instance for caching tests."""
    from mtss.models.vessel import Vessel

    return Vessel(name=name, vessel_type="VLCC", vessel_class="Canopus Class")


class _FakeSupabaseClient:
    """Stand-in for SupabaseClient.

    Returns a fresh Vessel per call. The test only cares that the cache avoids
    re-invoking the client for already-cached ids, so we don't need to match
    the real get_vessel_by_id contract beyond returning a Vessel or None.
    """

    def __init__(self):
        self.get_vessel_by_id = AsyncMock(
            side_effect=lambda vid: _mk_vessel(f"vessel-{vid}")
        )


@pytest.fixture(autouse=True)
def _reset_vessel_cache():
    """Every test starts with an empty vessel cache."""
    agent_module._vessel_cache.clear()
    yield
    agent_module._vessel_cache.clear()


class TestVesselCacheLRU:
    """LRU eviction + recency promotion for _get_vessel_info."""

    @pytest.mark.asyncio
    async def test_vessel_cache_evicts_oldest_at_capacity(self, monkeypatch):
        """With maxsize=2, inserting a 3rd id must evict the oldest."""
        monkeypatch.setattr(agent_module, "_VESSEL_CACHE_MAXSIZE", 2)

        vid_a = str(uuid4())
        vid_b = str(uuid4())
        vid_c = str(uuid4())

        fake_client = _FakeSupabaseClient()
        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            await agent_module._get_vessel_info(vid_a)
            await agent_module._get_vessel_info(vid_b)
            await agent_module._get_vessel_info(vid_c)

        # The oldest (A) must have been evicted; B and C remain.
        assert vid_a not in agent_module._vessel_cache
        assert vid_b in agent_module._vessel_cache
        assert vid_c in agent_module._vessel_cache
        assert len(agent_module._vessel_cache) == 2

        # Sanity: one DB call per distinct id, no redundant fetches.
        assert fake_client.get_vessel_by_id.call_count == 3

    @pytest.mark.asyncio
    async def test_vessel_cache_hit_promotes_to_recent(self, monkeypatch):
        """Sequence A, B, A, C must evict B (not A)."""
        monkeypatch.setattr(agent_module, "_VESSEL_CACHE_MAXSIZE", 2)

        vid_a = str(uuid4())
        vid_b = str(uuid4())
        vid_c = str(uuid4())

        fake_client = _FakeSupabaseClient()
        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            await agent_module._get_vessel_info(vid_a)
            await agent_module._get_vessel_info(vid_b)
            # Hitting A again should promote it to most-recently-used.
            await agent_module._get_vessel_info(vid_a)
            # Adding C now must evict the LRU entry, which is B.
            await agent_module._get_vessel_info(vid_c)

        assert vid_a in agent_module._vessel_cache, "A was promoted on hit; must survive"
        assert vid_b not in agent_module._vessel_cache, "B was least recent; must be evicted"
        assert vid_c in agent_module._vessel_cache
        assert len(agent_module._vessel_cache) == 2

        # A was fetched exactly once (second access was a cache hit).
        fetched_ids = [call.args[0] for call in fake_client.get_vessel_by_id.call_args_list]
        assert fetched_ids.count(UUID(vid_a)) == 1
        assert fake_client.get_vessel_by_id.call_count == 3

    @pytest.mark.asyncio
    async def test_vessel_cache_none_on_empty_id(self):
        """An empty/None vessel_id short-circuits and does not hit the DB."""
        with patch.object(agent_module, "SupabaseClient") as mock_cls:
            result = await agent_module._get_vessel_info(None)
        assert result is None
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_vessel_cache_db_failure_not_cached(self, monkeypatch):
        """DB errors must not poison the cache — next call retries the lookup."""
        vid = str(uuid4())

        failing_client = _FakeSupabaseClient()
        failing_client.get_vessel_by_id = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(agent_module, "SupabaseClient", return_value=failing_client):
            result = await agent_module._get_vessel_info(vid)

        assert result is None
        assert vid not in agent_module._vessel_cache
