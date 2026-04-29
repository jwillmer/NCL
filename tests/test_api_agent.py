"""Tests for src/mtss/api/agent.py helpers.

Focus: ``_get_vessel_info`` delegates to the process-wide ``VesselCache``
(a full in-memory mirror of the ~50-row vessels table, loaded lazily and
refreshed on a 5-minute TTL). The helper short-circuits on empty /
malformed ids and returns ``None`` without hitting the DB; valid ids are
served from the cached map after a single cache load.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
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


class TestResolveFilterValue:
    """Behavior of ``_resolve_filter_value`` — the pure name-to-filter mapping
    used by set_filter_node. Covers each `kind` branch plus empty/unknown
    inputs. Wraps the shared VesselCache so no DB round-trip is needed after
    the first cache load."""

    @pytest.mark.asyncio
    async def test_clear_returns_all_null(self):
        """kind='clear' resolves to (None, None, None) regardless of value."""
        with patch.object(agent_module, "SupabaseClient") as mock_cls:
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "clear", None
            )
        assert (vid, vtype, vclass) == (None, None, None)
        assert "clear" in msg.lower()
        # 'clear' must not load the cache — it has nothing to resolve.
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_vessel_resolves_by_name(self):
        """kind='vessel' resolves by cached vessel name and returns UUID string."""
        vid = uuid4()
        vessel = _mk_vessel(vid, name="Maran Canopus")

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            resolved_id, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel", "maran canopus"
            )

        assert resolved_id == str(vid)
        assert vtype is None and vclass is None
        assert "Maran Canopus" in msg

    @pytest.mark.asyncio
    async def test_vessel_unknown_name_returns_message(self):
        """Unknown vessel name: no filter set, informative message for the LLM."""
        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel", "Nonexistent"
            )

        assert (vid, vtype, vclass) == (None, None, None)
        assert "no vessel" in msg.lower()

    @pytest.mark.asyncio
    async def test_vessel_type_matches_case_insensitive(self):
        """kind='vessel_type' matches a distinct type from the cached vessels."""
        vessel = _mk_vessel(uuid4(), name="V1")

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel_type", "vlcc"
            )

        # Preserves canonical casing from the source vessel row.
        assert vtype == "VLCC"
        assert vid is None and vclass is None
        assert "VLCC" in msg

    @pytest.mark.asyncio
    async def test_vessel_class_matches_case_insensitive(self):
        """kind='vessel_class' matches a cached vessel_class value."""
        vessel = _mk_vessel(uuid4(), name="V1")

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel_class", "canopus class"
            )

        assert vclass == "Canopus Class"
        assert vid is None and vtype is None

    @pytest.mark.asyncio
    async def test_unknown_type_returns_message(self):
        """Unknown type/class doesn't partially apply — all three stay None."""
        vessel = _mk_vessel(uuid4(), name="V1")

        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel_type", "ULCC"
            )

        assert (vid, vtype, vclass) == (None, None, None)
        assert "no vessel type" in msg.lower()

    @pytest.mark.asyncio
    async def test_empty_value_rejected(self):
        """Non-'clear' kind with empty value: no-op, no cache load."""
        with patch.object(agent_module, "SupabaseClient") as mock_cls:
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "vessel", ""
            )
        assert (vid, vtype, vclass) == (None, None, None)
        assert "empty" in msg.lower()
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_kind_rejected(self):
        """An unrecognised kind resolves to all-None with an explanatory message."""
        vessel = _mk_vessel(uuid4(), name="V1")
        fake_client = AsyncMock()
        fake_client.get_all_vessels = AsyncMock(return_value=[vessel])

        with patch.object(agent_module, "SupabaseClient", return_value=fake_client):
            vid, vtype, vclass, msg = await agent_module._resolve_filter_value(
                "flag", "x"
            )
        assert (vid, vtype, vclass) == (None, None, None)
        assert "unknown filter kind" in msg.lower()


class TestEmitFilterUpdate:
    """``emit_filter_update`` dispatches the LangGraph custom event that
    streaming.py converts into a `data-filter` part on the UI stream.
    Streaming wire-format (`2:["filter", {...}]`) is exercised end-to-end
    by the progress-forwarding path — here we just verify the event shape."""

    @pytest.mark.asyncio
    async def test_dispatches_expected_payload(self):
        with patch.object(agent_module, "adispatch_custom_event") as dispatch:
            dispatch.return_value = None
            dispatch.side_effect = AsyncMock()
            await agent_module.emit_filter_update(
                config={"callbacks": []},
                vessel_id="abc",
                vessel_type=None,
                vessel_class=None,
            )
        dispatch.assert_called_once()
        kwargs = dispatch.call_args.kwargs
        assert kwargs["name"] == "emit_filter_update"
        assert kwargs["data"] == {
            "vessel_id": "abc",
            "vessel_type": None,
            "vessel_class": None,
        }


class TestEmitCitations:
    """``emit_citations`` dispatches the LangGraph custom event that
    streaming.py converts into a `data-citations` part. Verifies the
    wire shape v1: {version, citations, invalid_chunk_ids}."""

    @pytest.mark.asyncio
    async def test_dispatches_expected_payload(self):
        payload = {
            "version": 1,
            "citations": [{"chunk_id": "aabbccddeeff", "index": 1}],
            "invalid_chunk_ids": ["deadbeefdead"],
        }
        with patch.object(agent_module, "adispatch_custom_event") as dispatch:
            dispatch.side_effect = AsyncMock()
            await agent_module.emit_citations(
                config={"callbacks": []}, payload=payload
            )
        dispatch.assert_called_once()
        kwargs = dispatch.call_args.kwargs
        assert kwargs["name"] == "emit_citations"
        assert kwargs["data"] == payload


class TestChatNodeCitationPaths:
    """chat_node's final branch — after the LLM returns plain text and
    citation_map is present from search_node — picks one of two paths
    based on settings.async_citations_enabled.

    Sync (default): full substitution to <cite> tags before returning.
    Async (flag on): keep valid [C:xxx] markers; emit data-citations frame
    so the frontend patches the markers post-stream."""

    @staticmethod
    def _build_state(response_content: str, citation_map_data):
        from langchain_core.messages import HumanMessage

        return {
            "messages": [HumanMessage(content="why?")],
            "search_progress": "",
            "selected_vessel_id": None,
            "selected_vessel_type": None,
            "selected_vessel_class": None,
            "citation_map": citation_map_data,
            "filter_pending_search": False,
        }

    @staticmethod
    def _make_retrieval_result(chunk_id: str):
        from mtss.models.chunk import RetrievalResult

        return RetrievalResult(
            text="Sample text",
            score=0.9,
            chunk_id=chunk_id,
            doc_id="doc001",
            source_id="src001",
            source_title="Inspection Report",
            section_path=[],
            page_number=3,
            archive_browse_uri="/archive/abc/index.md",
            archive_download_uri="/archive/abc/file.pdf",
        )

    @pytest.mark.asyncio
    async def test_async_path_keeps_valid_markers_and_emits_citations(
        self, monkeypatch
    ):
        """Flag on: AIMessage keeps valid [C:xxx], invalid stripped,
        emit_citations dispatched once with the v1 payload shape."""
        from langchain_core.messages import AIMessage

        valid_id = "aabbccddeeff"
        invalid_id = "deadbeefdead"
        valid_result = self._make_retrieval_result(valid_id)
        citation_map_data = {valid_id: valid_result.to_dict()}

        # LLM produces text with one valid + one invalid citation marker.
        llm_text = (
            f"The hull is OK [C:{valid_id}]. "
            f"Some other detail [C:{invalid_id}]."
        )

        async def fake_call_chat_llm(**kwargs):
            return AIMessage(content=llm_text)

        monkeypatch.setattr(agent_module, "_call_chat_llm", fake_call_chat_llm)

        # Patch settings to enable async citations.
        from mtss import config as config_module

        cur_settings = config_module.get_settings()
        # Pydantic settings are immutable by default; build a copy.
        new_settings = cur_settings.model_copy(
            update={"async_citations_enabled": True}
        )
        monkeypatch.setattr(agent_module, "get_settings", lambda: new_settings)

        # Patch CitationProcessor archive verification: skip network HEAD.
        # Also stub ArchiveStorage so the constructor doesn't need a live
        # Supabase URL/key at test time.
        from mtss.rag import citation_processor as cp_module

        monkeypatch.setattr(cp_module, "ArchiveStorage", MagicMock)

        async def fake_verify_async(self, uris):
            return {u: True for u in uris}

        monkeypatch.setattr(
            cp_module.CitationProcessor,
            "verify_archives_async",
            fake_verify_async,
        )

        emitted_citations: list = []
        emitted_state: list = []

        async def fake_emit_citations(config, payload):
            emitted_citations.append(payload)

        async def fake_emit_state(config, state):
            emitted_state.append(dict(state))

        monkeypatch.setattr(agent_module, "emit_citations", fake_emit_citations)
        monkeypatch.setattr(agent_module, "emit_state", fake_emit_state)

        state = self._build_state(llm_text, citation_map_data)
        cmd = await agent_module.chat_node(state, config={"callbacks": []})

        # END-bound Command with the rewired AIMessage.
        out_msg = cmd.update["messages"][0]
        assert isinstance(out_msg, AIMessage)
        # Valid marker preserved verbatim — frontend patches it post-stream.
        assert f"[C:{valid_id}]" in out_msg.content
        # Invalid marker stripped.
        assert f"[C:{invalid_id}]" not in out_msg.content
        # No <cite> tags in async path; substitution happens client-side.
        assert "<cite" not in out_msg.content

        # emit_citations dispatched exactly once with the v1 wire shape.
        assert len(emitted_citations) == 1
        payload = emitted_citations[0]
        assert payload["version"] == 1
        assert isinstance(payload["citations"], list)
        assert len(payload["citations"]) == 1
        assert payload["citations"][0]["chunk_id"] == valid_id
        assert payload["invalid_chunk_ids"] == [invalid_id]

    @pytest.mark.asyncio
    async def test_sync_path_substitutes_cite_tags(self, monkeypatch):
        """Flag off (default): full substitution to <cite> tags;
        no emit_citations dispatched. Backward-compatible behaviour."""
        from langchain_core.messages import AIMessage

        valid_id = "aabbccddeeff"
        valid_result = self._make_retrieval_result(valid_id)
        citation_map_data = {valid_id: valid_result.to_dict()}

        llm_text = f"The hull is OK [C:{valid_id}]."

        async def fake_call_chat_llm(**kwargs):
            return AIMessage(content=llm_text)

        monkeypatch.setattr(agent_module, "_call_chat_llm", fake_call_chat_llm)

        from mtss import config as config_module

        cur_settings = config_module.get_settings()
        new_settings = cur_settings.model_copy(
            update={"async_citations_enabled": False}
        )
        monkeypatch.setattr(agent_module, "get_settings", lambda: new_settings)

        from mtss.rag import citation_processor as cp_module

        # Stub ArchiveStorage to avoid needing live Supabase credentials.
        monkeypatch.setattr(cp_module, "ArchiveStorage", MagicMock)
        # Sync path uses verify_archive_exists — patch storage.file_exists.
        monkeypatch.setattr(
            cp_module.CitationProcessor,
            "verify_archive_exists",
            lambda self, uri: True,
        )

        emitted_citations: list = []

        async def fake_emit_citations(config, payload):
            emitted_citations.append(payload)

        async def fake_emit_state(config, state):
            return None

        monkeypatch.setattr(agent_module, "emit_citations", fake_emit_citations)
        monkeypatch.setattr(agent_module, "emit_state", fake_emit_state)

        state = self._build_state(llm_text, citation_map_data)
        cmd = await agent_module.chat_node(state, config={"callbacks": []})

        out_msg = cmd.update["messages"][0]
        assert isinstance(out_msg, AIMessage)
        # Sync path produces a fully-rendered <cite> tag.
        assert f"[C:{valid_id}]" not in out_msg.content
        assert "<cite" in out_msg.content
        assert f'id="{valid_id}"' in out_msg.content
        # No emit_citations dispatched in sync path.
        assert emitted_citations == []
