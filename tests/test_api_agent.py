"""Tests for src/mtss/api/agent.py helpers.

Focus: ``_get_vessel_info`` delegates to the process-wide ``VesselCache``
(a full in-memory mirror of the ~50-row vessels table, loaded lazily and
refreshed on a 5-minute TTL). The helper short-circuits on empty /
malformed ids and returns ``None`` without hitting the DB; valid ids are
served from the cached map after a single cache load.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from mtss.api import agent as agent_module
from mtss.processing import entity_cache


def _mk_vessel(vessel_id: UUID, name: str = "test-vessel"):
    """Build a minimal Vessel instance for caching tests."""
    from mtss.models.vessel import Vessel

    return Vessel(
        id=vessel_id, name=name, vessel_type="VLCC", vessel_class="Canopus Class"
    )


@pytest.fixture(autouse=True)
def _reset_vessel_cache():
    """Every test starts with an empty process-wide VesselCache."""
    entity_cache._vessel_cache = entity_cache.VesselCache()
    yield
    entity_cache._vessel_cache = entity_cache.VesselCache()


class TestGetVesselInfo:
    """Behavior of ``_get_vessel_info`` against the shared VesselCache."""

    @pytest.mark.asyncio
    async def test_empty_vessel_id_short_circuits(self):
        """An empty/None vessel_id short-circuits and does not touch DB or cache."""
        with patch.object(agent_module, "SupabaseClient") as mock_cls:
            result = await agent_module._get_vessel_info(None)
        assert result is None
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_uuid_vessel_id_returns_none(self):
        """Malformed vessel_id returns None without loading the cache."""
        with patch.object(agent_module, "SupabaseClient") as mock_cls:
            result = await agent_module._get_vessel_info("not-a-uuid")
        assert result is None
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_id_served_from_cache_after_load(self):
        """A valid UUID triggers a single cache load + returns the mapped Vessel."""
        vid = uuid4()
        vessel = _mk_vessel(vid, name="Test Canopus")

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            result1 = await agent_module._get_vessel_info(str(vid))
            # Second lookup must be served purely from the warmed cache.
            result2 = await agent_module._get_vessel_info(str(vid))

        assert result1 is vessel
        assert result2 is vessel
        # Cache is loaded exactly once: single DB call for many lookups.
        fake_client.get_all_vessels.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Unknown vessel id: cache loads, does not contain it, returns None."""
        vid = uuid4()

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            result = await agent_module._get_vessel_info(str(vid))

        assert result is None

    @pytest.mark.asyncio
    async def test_db_failure_returns_none_without_crashing(self):
        """Cache-load errors are swallowed: helper returns None, caller proceeds."""
        vid = uuid4()

        failing_client = AsyncMock()
        failing_client.get_all_vessels = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(agent_module, "SupabaseClient", return_value=failing_client):
            result = await agent_module._get_vessel_info(str(vid))

        assert result is None
