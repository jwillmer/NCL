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

        # Write JSONL file
        topic_data = {"id": str(uuid4()), "name": "existing", "display_name": "Existing"}
        (tmp_path / "topics.jsonl").write_text(json.dumps(topic_data) + "\n")

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

        doc_id = str(uuid4())
        doc = {
            "id": doc_id,
            "doc_id": "existing-doc",
            "document_type": "email",
            "file_path": "/test.eml",
            "file_name": "test.eml",
            "depth": 0,
            "root_id": None,
            "parent_id": None,
            "ingest_version": 1,
            "status": "completed",
        }
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")
        (tmp_path / "chunks.jsonl").write_text("")

        totals = {"documents": 0, "chunks": 0}
        changes = {"new_documents": 0, "new_chunks": 0, "failed": 0}

        await _import_documents(mock_db, tmp_path, totals, changes, dry_run=False, verbose=False)

        # Doc already exists, should not be re-imported
        mock_db.persist_ingest_result.assert_not_called()


class TestUploadWithRetry:
    """Verify archive upload retry behavior for transient Supabase failures."""

    def test_succeeds_on_first_attempt(self):
        from mtss.cli import import_cmd

        storage = MagicMock()
        storage.upload_file = MagicMock(return_value="key")

        with patch.object(import_cmd, "time") as mock_time:
            ok = import_cmd._upload_with_retry(storage, "doc/email.eml", b"body", "message/rfc822")

        assert ok is True
        assert storage.upload_file.call_count == 1
        mock_time.sleep.assert_not_called()

    def test_retries_transient_jsondecodeerror_then_succeeds(self):
        """Simulates the real failure: storage3 raises JSONDecodeError on
        non-JSON gateway response, second attempt succeeds."""
        from json import JSONDecodeError

        from mtss.cli import import_cmd

        storage = MagicMock()
        storage.upload_file = MagicMock(
            side_effect=[
                JSONDecodeError("Expecting value", "", 0),
                "key",
            ]
        )

        with patch.object(import_cmd, "time") as mock_time:
            ok = import_cmd._upload_with_retry(storage, "doc/email.eml", b"body", "message/rfc822")

        assert ok is True
        assert storage.upload_file.call_count == 2
        assert mock_time.sleep.call_count == 1
        # First retry delay = base * 2^0 = base
        assert mock_time.sleep.call_args_list[0].args[0] == import_cmd._UPLOAD_BACKOFF_BASE

    def test_returns_false_after_all_attempts_exhausted(self):
        from mtss.cli import import_cmd

        storage = MagicMock()
        storage.upload_file = MagicMock(side_effect=RuntimeError("still broken"))

        with patch.object(import_cmd, "time"):
            ok = import_cmd._upload_with_retry(storage, "doc/email.eml", b"body", "message/rfc822")

        assert ok is False
        assert storage.upload_file.call_count == import_cmd._UPLOAD_MAX_ATTEMPTS

    def test_backoff_doubles_between_attempts(self):
        from mtss.cli import import_cmd

        storage = MagicMock()
        storage.upload_file = MagicMock(side_effect=RuntimeError("always fails"))

        with patch.object(import_cmd, "time") as mock_time:
            import_cmd._upload_with_retry(storage, "k", b"b", "application/octet-stream")

        # MAX_ATTEMPTS attempts => MAX_ATTEMPTS - 1 sleeps between them
        delays = [c.args[0] for c in mock_time.sleep.call_args_list]
        assert len(delays) == import_cmd._UPLOAD_MAX_ATTEMPTS - 1
        base = import_cmd._UPLOAD_BACKOFF_BASE
        expected = [base * (2 ** i) for i in range(import_cmd._UPLOAD_MAX_ATTEMPTS - 1)]
        assert delays == expected
