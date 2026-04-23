"""Regression tests for DomainRepository.upsert_vessel.

Guards the UUID-stability invariant: upserting a vessel whose name already
exists must NOT rewrite the existing row's id. The prior implementation sent
the Pydantic model's freshly-minted uuid4 on every upsert, which made
Postgres UPDATE the id column on name conflict — orphaning every
`chunk.metadata.vessel_ids` reference and tripping the FK from
`conversations.vessel_id`. See domain.py:upsert_vessel docstring.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from mtss.models.vessel import Vessel
from mtss.storage.repositories.domain import DomainRepository


class FakeTable:
    """Minimal supabase-py table stub capturing upsert payloads for assertion."""

    def __init__(self, existing_rows: list[dict] | None = None) -> None:
        self.existing_rows = existing_rows or []
        self.upsert_calls: list[dict] = []
        self._select_filters: dict = {}

    def select(self, *_args, **_kwargs):
        self._select_filters = {}
        return self

    def eq(self, column: str, value):
        self._select_filters[column] = value
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        # Select path.
        if self._select_filters:
            name = self._select_filters.get("name")
            matches = [r for r in self.existing_rows if r.get("name") == name]
            self._select_filters = {}
            return SimpleNamespace(data=matches)
        return SimpleNamespace(data=[])

    def upsert(self, data, on_conflict: str | None = None):
        self.upsert_calls.append(data)
        return self


def _make_repo(table: FakeTable) -> DomainRepository:
    client = MagicMock()
    client.table.return_value = table
    # BaseRepository.__init__ expects (client, db_url); db_url unused here.
    return DomainRepository(client, db_url="postgresql://unused")


@pytest.mark.asyncio
async def test_upsert_reuses_existing_uuid():
    existing_id = str(uuid4())
    table = FakeTable(existing_rows=[{"id": existing_id, "name": "MARAN HELEN"}])
    repo = _make_repo(table)

    new_model = Vessel(name="MARAN HELEN", vessel_type="SUEZMAX", vessel_class="Hermione")
    assert str(new_model.id) != existing_id  # new model has a fresh uuid4

    result = await repo.upsert_vessel(new_model)

    assert table.upsert_calls, "upsert was not invoked"
    payload = table.upsert_calls[-1]
    assert payload["id"] == existing_id, "must preserve existing row UUID"
    assert payload["name"] == "MARAN HELEN"
    assert result.id == UUID(existing_id), "returned model must reflect resolved UUID"


@pytest.mark.asyncio
async def test_upsert_inserts_with_model_uuid_when_new():
    table = FakeTable(existing_rows=[])
    repo = _make_repo(table)

    new_model = Vessel(name="MARAN CONQUEROR", vessel_type="BULK CARRIER", vessel_class="Capesize")

    await repo.upsert_vessel(new_model)

    payload = table.upsert_calls[-1]
    assert payload["id"] == str(new_model.id), "new vessels keep minted UUID"
    assert payload["vessel_type"] == "BULK CARRIER"
    assert payload["vessel_class"] == "Capesize"
