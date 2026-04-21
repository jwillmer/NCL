"""Tests for JSONL roundtrip fidelity and import idempotency.

Validates that data survives: Model → _doc_to_dict → JSON → _dict_to_document → Model
without losing fields, especially timestamps (previously a real bug).
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from mtss.models.chunk import Chunk
from mtss.models.document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    EmailMetadata,
    ProcessingStatus,
)
from mtss.models.topic import Topic


def _roundtrip_doc(doc: Document) -> Document:
    """Serialize a Document to dict, dump/load JSON, deserialize back."""
    from mtss.models.serializers import dict_to_document, doc_to_dict

    d = doc_to_dict(doc)
    json_str = json.dumps(d, default=str)
    d2 = json.loads(json_str)
    return dict_to_document(d2)


def _roundtrip_chunk(chunk: Chunk) -> Chunk:
    """Serialize a Chunk to dict, dump/load JSON, deserialize back."""
    from mtss.models.serializers import chunk_to_dict, dict_to_chunk

    d = chunk_to_dict(chunk)
    json_str = json.dumps(d, default=str)
    d2 = json.loads(json_str)
    return dict_to_chunk(d2)


def _roundtrip_topic(topic: Topic) -> Topic:
    """Serialize a Topic to dict, dump/load JSON, deserialize back."""
    from mtss.models.serializers import dict_to_topic, topic_to_dict

    d = topic_to_dict(topic)
    json_str = json.dumps(d, default=str)
    d2 = json.loads(json_str)
    return dict_to_topic(d2)


class TestDocumentRoundtrip:
    """Verify Document fields survive JSONL serialization roundtrip."""

    def test_email_document_preserves_all_fields(self, sample_document):
        result = _roundtrip_doc(sample_document)

        assert result.id == sample_document.id
        assert result.source_id == sample_document.source_id
        assert result.doc_id == sample_document.doc_id
        assert result.document_type == sample_document.document_type
        assert result.file_path == sample_document.file_path
        assert result.file_name == sample_document.file_name
        assert result.file_hash == sample_document.file_hash
        assert result.parent_id == sample_document.parent_id
        assert result.root_id == sample_document.root_id
        assert result.depth == sample_document.depth
        assert result.path == sample_document.path
        assert result.source_title == sample_document.source_title
        assert result.archive_browse_uri == sample_document.archive_browse_uri
        assert result.archive_download_uri == sample_document.archive_download_uri
        assert result.status == sample_document.status
        assert result.ingest_version == sample_document.ingest_version
        assert result.content_version == sample_document.content_version

    def test_email_metadata_preserved(self, sample_document):
        result = _roundtrip_doc(sample_document)

        assert result.email_metadata is not None
        orig = sample_document.email_metadata
        assert result.email_metadata.subject == orig.subject
        assert result.email_metadata.participants == orig.participants
        assert result.email_metadata.initiator == orig.initiator
        assert result.email_metadata.date_start == orig.date_start
        assert result.email_metadata.message_count == orig.message_count

    def test_attachment_metadata_preserved(self, sample_attachment_document):
        result = _roundtrip_doc(sample_attachment_document)

        assert result.attachment_metadata is not None
        orig = sample_attachment_document.attachment_metadata
        assert result.attachment_metadata.content_type == orig.content_type
        assert result.attachment_metadata.size_bytes == orig.size_bytes

    def test_timestamps_preserved(self, sample_document):
        """Regression test: created_at/updated_at must survive roundtrip."""
        original_created = sample_document.created_at
        original_updated = sample_document.updated_at

        result = _roundtrip_doc(sample_document)

        # Timestamps should match (within microsecond due to ISO format)
        assert result.created_at.replace(microsecond=0) == original_created.replace(microsecond=0)
        assert result.updated_at.replace(microsecond=0) == original_updated.replace(microsecond=0)

    def test_backward_compat_missing_timestamps(self):
        """Old JSONL files without created_at/updated_at should still import."""
        from mtss.cli.import_cmd import _dict_to_document

        d = {
            "id": str(uuid4()),
            "document_type": "email",
            "file_path": "/test.eml",
            "file_name": "test.eml",
            "depth": 0,
            "ingest_version": 1,
            "status": "completed",
            # No created_at, no updated_at
        }
        doc = _dict_to_document(d)
        assert doc.created_at is not None  # Should default, not crash
        assert doc.updated_at is not None


class TestChunkRoundtrip:
    """Verify Chunk fields survive JSONL serialization roundtrip."""

    def test_preserves_all_fields(self, sample_chunk):
        result = _roundtrip_chunk(sample_chunk)

        assert result.id == sample_chunk.id
        assert result.document_id == sample_chunk.document_id
        assert result.chunk_id == sample_chunk.chunk_id
        assert result.content == sample_chunk.content
        assert result.chunk_index == sample_chunk.chunk_index
        assert result.context_summary == sample_chunk.context_summary
        assert result.section_path == sample_chunk.section_path
        assert result.section_title == sample_chunk.section_title
        assert result.source_title == sample_chunk.source_title
        assert result.source_id == sample_chunk.source_id
        assert result.line_from == sample_chunk.line_from
        assert result.line_to == sample_chunk.line_to
        assert result.char_start == sample_chunk.char_start
        assert result.char_end == sample_chunk.char_end

    def test_embedding_preserved(self, sample_chunk):
        result = _roundtrip_chunk(sample_chunk)
        assert result.embedding == sample_chunk.embedding
        assert len(result.embedding) == 1536

    def test_metadata_preserved(self, sample_chunk):
        result = _roundtrip_chunk(sample_chunk)
        assert result.metadata == sample_chunk.metadata

    def test_embedding_mode_preserved(self, sample_chunk):
        sample_chunk.embedding_mode = "summary"
        result = _roundtrip_chunk(sample_chunk)
        assert result.embedding_mode == "summary"

    def test_embedding_mode_none_when_unset(self, sample_chunk):
        sample_chunk.embedding_mode = None
        result = _roundtrip_chunk(sample_chunk)
        assert result.embedding_mode is None


class TestDocumentEmbeddingModeRoundtrip:
    """Verify Document.embedding_mode survives JSONL round-trip."""

    def test_full_mode_round_trips(self, sample_document):
        from mtss.models.document import EmbeddingMode

        sample_document.embedding_mode = EmbeddingMode.FULL
        result = _roundtrip_doc(sample_document)
        assert result.embedding_mode == EmbeddingMode.FULL

    def test_summary_mode_round_trips(self, sample_document):
        from mtss.models.document import EmbeddingMode

        sample_document.embedding_mode = EmbeddingMode.SUMMARY
        result = _roundtrip_doc(sample_document)
        assert result.embedding_mode == EmbeddingMode.SUMMARY

    def test_metadata_only_mode_round_trips(self, sample_document):
        from mtss.models.document import EmbeddingMode

        sample_document.embedding_mode = EmbeddingMode.METADATA_ONLY
        result = _roundtrip_doc(sample_document)
        assert result.embedding_mode == EmbeddingMode.METADATA_ONLY

    def test_none_when_unset(self, sample_document):
        sample_document.embedding_mode = None
        result = _roundtrip_doc(sample_document)
        assert result.embedding_mode is None


class TestTopicRoundtrip:
    """Verify Topic fields survive JSONL serialization roundtrip."""

    def test_preserves_all_fields(self):
        topic = Topic(
            name="cargo damage",
            display_name="Cargo Damage",
            description="Damage to cargo during transit",
            embedding=[0.1] * 1536,
            chunk_count=5,
            document_count=2,
        )
        result = _roundtrip_topic(topic)

        assert result.id == topic.id
        assert result.name == topic.name
        assert result.display_name == topic.display_name
        assert result.description == topic.description
        assert result.embedding == topic.embedding
        assert result.chunk_count == topic.chunk_count
        assert result.document_count == topic.document_count

    def test_timestamps_preserved(self):
        """Regression test: topic timestamps must survive roundtrip."""
        topic = Topic(
            name="engine failure",
            display_name="Engine Failure",
            embedding=[0.2] * 10,
        )
        original_created = topic.created_at

        result = _roundtrip_topic(topic)

        assert result.created_at.replace(microsecond=0) == original_created.replace(microsecond=0)

    def test_backward_compat_missing_timestamps(self):
        """Old JSONL without created_at/updated_at should still import."""
        from mtss.cli.import_cmd import _dict_to_topic

        d = {
            "id": str(uuid4()),
            "name": "test topic",
            "display_name": "Test Topic",
        }
        topic = _dict_to_topic(d)
        assert topic.created_at is not None


class TestImportIdempotency:
    """Verify import skips already-existing records."""

    @pytest.mark.asyncio
    async def test_topics_dedup_by_name(self, tmp_path):
        """Import should skip topics that already exist in DB with same counts."""
        from mtss.cli.import_cmd import _import_topics

        # Mock async context manager for pool.acquire()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"name": "existing", "chunk_count": 0, "document_count": 0, "description": None}
        ])
        mock_conn.execute = AsyncMock()

        class FakeAcquire:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = FakeAcquire()
        mock_db = MagicMock()
        mock_db.get_pool = AsyncMock(return_value=mock_pool)
        mock_db.insert_topic = AsyncMock()

        # Seed ingest.db with the existing topic.
        from mtss.storage.sqlite_client import SqliteStorageClient

        client = SqliteStorageClient(output_dir=tmp_path)
        try:
            now = "2026-04-20T00:00:00"
            client._conn.execute(
                "INSERT INTO topics(id, name, display_name, chunk_count, document_count, "
                "created_at, updated_at) VALUES (?, ?, ?, 0, 0, ?, ?)",
                (str(uuid4()), "existing", "Existing", now, now),
            )
        finally:
            client._conn.close()

        totals = {"topics": 0}
        changes = {"failed": 0, "topics_removed": 0}

        await _import_topics(mock_db, tmp_path, totals, changes, dry_run=False, verbose=False)

        # Topic already exists with same counts, should not be inserted or updated
        mock_db.insert_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_documents_dedup_by_doc_id(self, tmp_path):
        """Import should skip documents that already exist in DB."""
        from mtss.cli.import_cmd import _import_documents

        # Mock async context manager for pool.acquire()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"doc_id": "existing-doc"}
        ])

        class FakeAcquire:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = FakeAcquire()
        mock_db = MagicMock()
        mock_db.get_pool = AsyncMock(return_value=mock_pool)
        mock_db.persist_ingest_result = AsyncMock()

        from mtss.storage.sqlite_client import SqliteStorageClient

        doc_id = str(uuid4())
        client = SqliteStorageClient(output_dir=tmp_path)
        try:
            now = "2026-04-20T00:00:00"
            client._conn.execute(
                "INSERT INTO documents("
                "id, doc_id, source_id, document_type, status, file_path, file_name, "
                "depth, content_version, ingest_version, root_id, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id, "existing-doc", "test.eml", "email", "completed",
                    "/test.eml", "test.eml", 0, 1, 1, doc_id, now, now,
                ),
            )
        finally:
            client._conn.close()

        totals = {"documents": 0, "chunks": 0}
        changes = {"new_documents": 0, "new_chunks": 0, "failed": 0}

        await _import_documents(mock_db, tmp_path, totals, changes, dry_run=False, verbose=False)

        # Doc already exists, should not be re-imported
        mock_db.persist_ingest_result.assert_not_called()


class TestImportTopicsBatching:
    """Verify _import_topics uses a single batched UPSERT instead of N calls."""

    @pytest.mark.asyncio
    async def test_import_topics_batches_upserts(self, tmp_path):
        """Insert + update topics must funnel through one ``conn.executemany``.

        The refactor removed per-row ``db.insert_topic`` calls in favour of a
        single ``INSERT ... ON CONFLICT (name) DO UPDATE`` over the combined
        set. We pin both: executemany is called exactly once with every
        changed row, and ``db.insert_topic`` is never called.
        """
        from mtss.cli.import_cmd import _import_topics

        # Remote shows one pre-existing topic with *different* counts so it
        # lands in the update bucket, plus no row for "brand_new" so that
        # lands in the insert bucket.
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                "name": "existing",
                "chunk_count": 0,
                "document_count": 0,
                "description": None,
            }
        ])
        mock_conn.execute = AsyncMock()
        mock_conn.executemany = AsyncMock()

        class FakeAcquire:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = FakeAcquire()
        mock_db = MagicMock()
        mock_db.get_pool = AsyncMock(return_value=mock_pool)
        mock_db.insert_topic = AsyncMock()

        # Seed ingest.db so _read_jsonl("topics.jsonl") returns both rows.
        from mtss.storage.sqlite_client import SqliteStorageClient

        client = SqliteStorageClient(output_dir=tmp_path)
        try:
            now = "2026-04-20T00:00:00"
            # "existing" — chunk_count diff triggers an update.
            client._conn.execute(
                "INSERT INTO topics(id, name, display_name, chunk_count, "
                "document_count, created_at, updated_at) "
                "VALUES (?, ?, ?, 7, 3, ?, ?)",
                (str(uuid4()), "existing", "Existing", now, now),
            )
            # "brand_new" — not on remote → insert bucket.
            client._conn.execute(
                "INSERT INTO topics(id, name, display_name, chunk_count, "
                "document_count, created_at, updated_at) "
                "VALUES (?, ?, ?, 2, 1, ?, ?)",
                (str(uuid4()), "brand_new", "Brand New", now, now),
            )
        finally:
            client._conn.close()

        totals = {"topics": 0}
        changes = {"failed": 0, "topics_removed": 0}

        await _import_topics(
            mock_db, tmp_path, totals, changes, dry_run=False, verbose=False
        )

        # Batched upsert called exactly once with both rows.
        assert mock_conn.executemany.await_count == 1
        call = mock_conn.executemany.await_args
        sql = call.args[0]
        rows = call.args[1]
        assert "INSERT INTO topics" in sql
        assert "ON CONFLICT (name)" in sql
        row_names = {r[0] for r in rows}
        assert row_names == {"existing", "brand_new"}
        # Per-row insert_topic must not be used anymore.
        mock_db.insert_topic.assert_not_called()


class TestRewriteRemoteTopicIds:
    """Verify the merge-plan-driven rewrite_chunk_topic_ids RPC call.

    The helper does more than just ``SELECT rewrite_chunk_topic_ids``: it
    resolves names → remote UUIDs, previews the blast radius (a second
    ``fetch``), writes a rollback backup JSON, prompts for confirmation
    (skippable via ``IMPORT_REWRITE_ASSUME_YES``), and only then invokes
    the RPC. The tests exercise the real flow end-to-end.
    """

    @staticmethod
    def _plan_path(tmp_path, plan: list[dict]) -> "Path":
        p = tmp_path / "plan.json"
        p.write_text(json.dumps(plan), encoding="utf-8")
        return p

    @pytest.mark.asyncio
    async def test_rewrite_remote_topic_ids_resolves_uuids_and_calls_rpc(
        self, tmp_path, monkeypatch
    ):
        """Plan names → remote UUIDs → jsonb mapping → RPC invocation."""
        from mtss.cli.import_cmd import _rewrite_remote_topic_ids

        absorbed_uuid = str(uuid4())
        keeper_uuid = str(uuid4())
        affected_chunk_id = uuid4()
        mock_conn = MagicMock()
        # Two distinct fetch() calls: first resolves names, second previews
        # the blast radius. Return rows must carry ``topic_ids`` because
        # the helper writes them into a rollback JSON snapshot.
        mock_conn.fetch = AsyncMock(side_effect=[
            [
                {"name": "absorbed", "id": absorbed_uuid},
                {"name": "keeper", "id": keeper_uuid},
            ],
            [
                {"id": affected_chunk_id, "topic_ids": [absorbed_uuid]},
            ],
        ])
        mock_conn.fetchval = AsyncMock(return_value=1)

        class FakeAcquire:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = FakeAcquire()
        mock_db = MagicMock()
        mock_db.get_pool = AsyncMock(return_value=mock_pool)

        plan_path = self._plan_path(tmp_path, [
            {
                "keeper_id": "local-keeper",
                "keeper_name": "keeper",
                "keeper_display_name": "Keeper",
                "keeper_chunks": 9,
                "keeper_docs": 3,
                "absorbed_id": "local-absorbed",
                "absorbed_name": "absorbed",
                "absorbed_display_name": "Absorbed",
                "absorbed_chunks": 2,
                "absorbed_docs": 1,
                "similarity": 0.91,
            },
        ])

        changes = {"failed": 0, "chunks_rewritten": 0}

        # Bypass the interactive confirm via the documented env override.
        monkeypatch.setenv("IMPORT_REWRITE_ASSUME_YES", "1")

        await _rewrite_remote_topic_ids(
            mock_db, plan_path, changes, dry_run=False, verbose=False
        )

        mock_conn.fetchval.assert_awaited_once()
        args = mock_conn.fetchval.await_args.args
        assert args[0] == "SELECT rewrite_chunk_topic_ids($1::jsonb)"
        mapping = json.loads(args[1])
        assert mapping == {absorbed_uuid: keeper_uuid}
        assert changes["chunks_rewritten"] == 1
        # Backup snapshot must land next to the plan for rollback use.
        backup_path = plan_path.with_name(f"{plan_path.stem}.backup.json")
        assert backup_path.exists()

    @pytest.mark.asyncio
    async def test_rewrite_remote_topic_ids_dry_run_skips_rpc(self, tmp_path):
        """--dry-run must resolve UUIDs + write backup but never invoke the RPC."""
        from mtss.cli.import_cmd import _rewrite_remote_topic_ids

        absorbed_uuid = str(uuid4())
        keeper_uuid = str(uuid4())
        affected_chunk_id = uuid4()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(side_effect=[
            [
                {"name": "absorbed", "id": absorbed_uuid},
                {"name": "keeper", "id": keeper_uuid},
            ],
            [
                {"id": affected_chunk_id, "topic_ids": [absorbed_uuid]},
            ],
        ])
        mock_conn.fetchval = AsyncMock(return_value=99)

        class FakeAcquire:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = FakeAcquire()
        mock_db = MagicMock()
        mock_db.get_pool = AsyncMock(return_value=mock_pool)

        plan_path = self._plan_path(tmp_path, [
            {
                "keeper_id": "local-keeper",
                "keeper_name": "keeper",
                "absorbed_id": "local-absorbed",
                "absorbed_name": "absorbed",
                "similarity": 0.93,
            },
        ])

        changes = {"failed": 0, "chunks_rewritten": 0}

        await _rewrite_remote_topic_ids(
            mock_db, plan_path, changes, dry_run=True, verbose=False
        )

        mock_conn.fetchval.assert_not_awaited()
        assert changes["chunks_rewritten"] == 0
