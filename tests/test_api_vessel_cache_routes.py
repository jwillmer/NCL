"""Tests that /api/vessels* endpoints read from VesselCache, not DB.

Before PR2.4 each call constructed a fresh SupabaseClient and went to the
DB; now they hit the in-memory VesselCache. These tests lock that in so a
future refactor can't silently regress to per-request DB round-trips.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from mtss.models.vessel import Vessel


def _make_vessel(name: str, vtype: str, vclass: str) -> Vessel:
    return Vessel(
        id=uuid4(),
        name=name,
        vessel_type=vtype,
        vessel_class=vclass,
        aliases=[],
    )


@pytest.fixture
def mock_vessels():
    """Fixed set of vessels for cache tests."""
    return [
        _make_vessel("MARAN CASTOR", "VLCC", "Canopus Class"),
        _make_vessel("MARAN POLLUX", "VLCC", "Canopus Class"),
        _make_vessel("MARAN GAS ALTHEA", "LNG", "Althea Class"),
    ]


class TestVesselEndpointsReadFromCache:
    """The vessel endpoints must hit VesselCache, not the underlying repo."""

    @pytest.mark.asyncio
    async def test_list_vessels_uses_cache(self, client, auth_headers, app, mock_vessels):
        from mtss.processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        # Seed cache directly so ensure_loaded() treats it as fresh.
        cache._by_id = {v.id: v for v in mock_vessels}
        cache._by_norm = {v.name.lower(): v for v in mock_vessels}
        import time as _time
        cache._loaded_at = _time.time()

        with patch.object(cache, "ensure_loaded", new=AsyncMock(return_value=None)) as mock_ensure:
            response = await client.get("/api/vessels", headers=auth_headers)

        assert response.status_code == 200
        names = [row["name"] for row in response.json()]
        assert names == sorted(names, key=str.lower), "vessels returned in case-insensitive name order"
        assert set(names) == {v.name for v in mock_vessels}
        mock_ensure.assert_awaited_once()
        assert response.headers.get("cache-control") == "private, max-age=300, stale-while-revalidate=600"

    @pytest.mark.asyncio
    async def test_list_vessel_types_uses_cache(self, client, auth_headers, mock_vessels):
        from mtss.processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        cache._by_id = {v.id: v for v in mock_vessels}
        cache._by_norm = {v.name.lower(): v for v in mock_vessels}
        import time as _time
        cache._loaded_at = _time.time()

        with patch.object(cache, "ensure_loaded", new=AsyncMock(return_value=None)) as mock_ensure:
            response = await client.get("/api/vessel-types", headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == sorted({"VLCC", "LNG"})
        mock_ensure.assert_awaited_once()
        assert response.headers.get("cache-control") == "private, max-age=300, stale-while-revalidate=600"

    @pytest.mark.asyncio
    async def test_list_vessel_classes_uses_cache(self, client, auth_headers, mock_vessels):
        from mtss.processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        cache._by_id = {v.id: v for v in mock_vessels}
        cache._by_norm = {v.name.lower(): v for v in mock_vessels}
        import time as _time
        cache._loaded_at = _time.time()

        with patch.object(cache, "ensure_loaded", new=AsyncMock(return_value=None)) as mock_ensure:
            response = await client.get("/api/vessel-classes", headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == sorted({"Canopus Class", "Althea Class"})
        mock_ensure.assert_awaited_once()
        assert response.headers.get("cache-control") == "private, max-age=300, stale-while-revalidate=600"
