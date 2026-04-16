"""Tests for topic count synchronization in the MTSS import pipeline.

Validates that topic chunk_count and document_count stay consistent across:
- Local ingest (increment_topic_counts accumulation)
- flush() JSONL serialization (recomputed from actual chunk metadata)
- import_cmd _import_topics (DB write)
- Hypothetical recompute from actual chunk metadata

flush() recomputes topic counts from actual chunk metadata references,
preventing stale accumulation across multiple ingest runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from mtss.models.chunk import Chunk
from mtss.models.document import Document, DocumentType, ProcessingStatus
from mtss.models.topic import Topic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_topic(
    name: str,
    chunk_count: int = 0,
    document_count: int = 0,
    topic_id: UUID | None = None,
) -> Topic:
    """Create a Topic with given counts."""
    return Topic(
        id=topic_id or uuid4(),
        name=name,
        display_name=name.title(),
        chunk_count=chunk_count,
        document_count=document_count,
    )


def _make_document(doc_id: UUID | None = None) -> Document:
    """Create a minimal Document for chunk ownership."""
    did = doc_id or uuid4()
    return Document(
        id=did,
        document_type=DocumentType.EMAIL,
        file_path="/test.eml",
        file_name="test.eml",
        depth=0,
        status=ProcessingStatus.COMPLETED,
    )


def _make_chunk(
    document_id: UUID,
    topic_ids: list[str] | None = None,
    chunk_index: int = 0,
) -> Chunk:
    """Create a Chunk with topic_ids in metadata."""
    metadata: Dict[str, Any] = {"type": "email_body"}
    if topic_ids is not None:
        metadata["topic_ids"] = topic_ids
    return Chunk(
        id=uuid4(),
        document_id=document_id,
        chunk_id=f"chunk_{uuid4().hex[:12]}",
        content=f"Chunk content {chunk_index}",
        chunk_index=chunk_index,
        metadata=metadata,
    )


def _count_topic_refs_from_chunks(
    chunks: list[Chunk],
    topic_id: str,
) -> dict[str, int]:
    """Recompute chunk_count and document_count from actual chunk metadata.

    This mirrors what _recompute_topic_counts SQL would do:
    - chunk_count = number of chunks whose metadata->'topic_ids' contains topic_id
    - document_count = number of distinct document_ids for those chunks
    """
    matching_chunks = [
        c for c in chunks
        if topic_id in (c.metadata.get("topic_ids") or [])
    ]
    doc_ids = {str(c.document_id) for c in matching_chunks}
    return {
        "chunk_count": len(matching_chunks),
        "document_count": len(doc_ids),
    }


# ---------------------------------------------------------------------------
# Test: flush() recomputes topic counts from actual chunk metadata
# ---------------------------------------------------------------------------


class TestFlushTopicCounts:
    """Test that flush() recomputes topic counts from actual chunk references."""

    @pytest.fixture
    def local_client(self, tmp_path):
        """Create a fresh LocalStorageClient."""
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_flush_recomputes_counts_from_chunks(self, local_client):
        """flush() should recompute counts from actual chunk metadata, not use stale increments."""
        topic = _make_topic("cargo damage")
        await local_client.insert_topic(topic)

        doc = _make_document()
        await local_client.insert_document(doc)

        # Create 3 chunks referencing this topic
        chunks = [
            _make_chunk(doc.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(3)
        ]
        await local_client.insert_chunks(chunks)

        # Stale increment (as if from a prior run) — should be overridden by recompute
        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=10, document_delta=5
        )

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        assert len(topics_data) == 1
        # Recomputed from actual chunks, not stale increment
        assert topics_data[0]["chunk_count"] == 3
        assert topics_data[0]["document_count"] == 1

    @pytest.mark.asyncio
    async def test_flush_counts_match_actual_chunk_refs(self, local_client):
        """flush counts should exactly match actual chunk references."""
        topic = _make_topic("engine failure")
        await local_client.insert_topic(topic)

        doc = _make_document()
        await local_client.insert_document(doc)

        # Create 3 chunks that reference this topic
        chunks = [
            _make_chunk(doc.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(3)
        ]
        await local_client.insert_chunks(chunks)

        # Stale increment says 5, but only 3 chunks exist
        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=5, document_delta=2
        )

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        assert len(topics_data) == 1

        # Recompute from actual chunks
        all_chunks = list(local_client._chunks.values())
        actual = _count_topic_refs_from_chunks(all_chunks, str(topic.id))

        flushed_chunk_count = topics_data[0]["chunk_count"]
        flushed_doc_count = topics_data[0]["document_count"]

        # After fix: flushed counts match actual chunk references
        assert flushed_chunk_count == actual["chunk_count"] == 3
        assert flushed_doc_count == actual["document_count"] == 1

    @pytest.mark.asyncio
    async def test_flush_counts_correct_for_single_run(self, local_client):
        """When only one ingest run occurs, counts should be consistent."""
        topic = _make_topic("hull inspection")
        await local_client.insert_topic(topic)

        doc = _make_document()
        await local_client.insert_document(doc)

        chunks = [
            _make_chunk(doc.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(4)
        ]
        await local_client.insert_chunks(chunks)

        # Correct increment for this single run
        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=4, document_delta=1
        )

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        all_chunks = list(local_client._chunks.values())
        actual = _count_topic_refs_from_chunks(all_chunks, str(topic.id))

        assert topics_data[0]["chunk_count"] == actual["chunk_count"] == 4
        assert topics_data[0]["document_count"] == actual["document_count"] == 1

    @pytest.mark.asyncio
    async def test_flush_preserves_prior_topic_counts(self, tmp_path):
        """When a second run starts, prior topic counts are recomputed from all chunks."""
        from mtss.storage.local_client import LocalStorageClient

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        topic_id = str(uuid4())
        doc_id = str(uuid4())

        # Simulate first run output: topic with counts AND chunks referencing it
        prior_topic = {
            "id": topic_id,
            "name": "cargo damage",
            "display_name": "Cargo Damage",
            "chunk_count": 10,
            "document_count": 3,
        }
        prior_doc = {
            "id": doc_id,
            "doc_id": f"doc_{doc_id[:8]}",
            "document_type": "email",
            "file_path": "/test.eml",
            "file_name": "test.eml",
            "depth": 0,
            "status": "completed",
        }
        prior_chunks = []
        for i in range(10):
            prior_chunks.append({
                "id": str(uuid4()),
                "document_id": doc_id,
                "chunk_id": f"chunk_{uuid4().hex[:12]}",
                "content": f"Chunk {i}",
                "chunk_index": i,
                "metadata": {"type": "email_body", "topic_ids": [topic_id]},
            })

        (output_dir / "topics.jsonl").write_text(json.dumps(prior_topic) + "\n")
        (output_dir / "documents.jsonl").write_text(json.dumps(prior_doc) + "\n")
        (output_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in prior_chunks) + "\n"
        )

        # Create client (loads prior data)
        client = LocalStorageClient(output_dir)
        loaded_topic = client._topics.get(topic_id)
        assert loaded_topic is not None

        # Flush without adding anything new
        client.flush()

        topics_data = _read_topics_jsonl(output_dir)
        assert len(topics_data) == 1
        # Recomputed from actual 10 chunks, all from 1 document
        assert topics_data[0]["chunk_count"] == 10
        assert topics_data[0]["document_count"] == 1  # All chunks share same doc_id


# ---------------------------------------------------------------------------
# Test: increment_topic_counts behavior
# ---------------------------------------------------------------------------


class TestIncrementTopicCounts:
    """Test the increment_topic_counts method in isolation."""

    @pytest.fixture
    def local_client(self, tmp_path):
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_increment_adds_to_zero(self, local_client):
        """Incrementing from zero should set the exact delta."""
        topic = _make_topic("new topic", chunk_count=0, document_count=0)
        await local_client.insert_topic(topic)

        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=3, document_delta=1
        )

        t = local_client._topics[str(topic.id)]
        assert t.chunk_count == 3
        assert t.document_count == 1

    @pytest.mark.asyncio
    async def test_increment_accumulates(self, local_client):
        """Multiple increments should accumulate."""
        topic = _make_topic("accumulating topic")
        await local_client.insert_topic(topic)

        for _ in range(5):
            await local_client.increment_topic_counts(
                [topic.id], chunk_delta=2, document_delta=1
            )

        t = local_client._topics[str(topic.id)]
        assert t.chunk_count == 10  # 2 * 5
        assert t.document_count == 5  # 1 * 5

    @pytest.mark.asyncio
    async def test_increment_unknown_topic_is_noop(self, local_client):
        """Incrementing a non-existent topic should not raise."""
        fake_id = uuid4()
        # Should not raise
        await local_client.increment_topic_counts(
            [fake_id], chunk_delta=5, document_delta=1
        )

    @pytest.mark.asyncio
    async def test_increment_multiple_topics(self, local_client):
        """Should increment all specified topics."""
        t1 = _make_topic("topic a")
        t2 = _make_topic("topic b")
        await local_client.insert_topic(t1)
        await local_client.insert_topic(t2)

        await local_client.increment_topic_counts(
            [t1.id, t2.id], chunk_delta=3, document_delta=1
        )

        assert local_client._topics[str(t1.id)].chunk_count == 3
        assert local_client._topics[str(t2.id)].chunk_count == 3
        assert local_client._topics[str(t1.id)].document_count == 1
        assert local_client._topics[str(t2.id)].document_count == 1

    @pytest.mark.asyncio
    async def test_increment_handles_none_counts(self, local_client):
        """Should handle topics where chunk_count/document_count is None."""
        topic = _make_topic("none counts")
        topic.chunk_count = None  # type: ignore[assignment]
        topic.document_count = None  # type: ignore[assignment]
        # Insert directly into dict to bypass validation
        local_client._topics[str(topic.id)] = topic

        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=2, document_delta=1
        )

        t = local_client._topics[str(topic.id)]
        assert t.chunk_count == 2  # (None or 0) + 2
        assert t.document_count == 1


# ---------------------------------------------------------------------------
# Test: recompute logic (Python equivalent of DB SQL)
# ---------------------------------------------------------------------------


class TestRecomputeTopicCounts:
    """Test the recompute logic that should produce correct counts.

    In production, this would be SQL like:
        UPDATE topics SET chunk_count = sub.cc, document_count = sub.dc
        FROM (
            SELECT t.id, COUNT(*) as cc,
                   COUNT(DISTINCT c.document_id) as dc
            FROM topics t
            JOIN chunks c ON c.metadata->'topic_ids' ? t.id::text
            GROUP BY t.id
        ) sub
        WHERE topics.id = sub.id;

    We test the equivalent Python logic here.
    """

    def test_recompute_single_topic_single_chunk(self):
        """One topic referenced by one chunk."""
        topic_id = str(uuid4())
        doc_id = uuid4()
        chunks = [_make_chunk(doc_id, topic_ids=[topic_id])]

        counts = _count_topic_refs_from_chunks(chunks, topic_id)
        assert counts["chunk_count"] == 1
        assert counts["document_count"] == 1

    def test_recompute_topic_with_multiple_chunks(self):
        """One topic referenced by multiple chunks from the same document."""
        topic_id = str(uuid4())
        doc_id = uuid4()
        chunks = [
            _make_chunk(doc_id, topic_ids=[topic_id], chunk_index=i)
            for i in range(5)
        ]

        counts = _count_topic_refs_from_chunks(chunks, topic_id)
        assert counts["chunk_count"] == 5
        assert counts["document_count"] == 1  # All from same doc

    def test_recompute_topic_with_multiple_documents(self):
        """One topic referenced by chunks from different documents."""
        topic_id = str(uuid4())
        doc1, doc2, doc3 = uuid4(), uuid4(), uuid4()
        chunks = [
            _make_chunk(doc1, topic_ids=[topic_id], chunk_index=0),
            _make_chunk(doc1, topic_ids=[topic_id], chunk_index=1),
            _make_chunk(doc2, topic_ids=[topic_id], chunk_index=0),
            _make_chunk(doc3, topic_ids=[topic_id], chunk_index=0),
        ]

        counts = _count_topic_refs_from_chunks(chunks, topic_id)
        assert counts["chunk_count"] == 4
        assert counts["document_count"] == 3

    def test_recompute_topic_with_no_chunks(self):
        """Topic that is not referenced by any chunk."""
        topic_id = str(uuid4())
        other_topic = str(uuid4())
        doc_id = uuid4()
        chunks = [_make_chunk(doc_id, topic_ids=[other_topic])]

        counts = _count_topic_refs_from_chunks(chunks, topic_id)
        assert counts["chunk_count"] == 0
        assert counts["document_count"] == 0

    def test_recompute_empty_chunks(self):
        """No chunks at all."""
        topic_id = str(uuid4())
        counts = _count_topic_refs_from_chunks([], topic_id)
        assert counts["chunk_count"] == 0
        assert counts["document_count"] == 0

    def test_recompute_chunk_with_no_topic_ids(self):
        """Chunks without topic_ids in metadata should be ignored."""
        topic_id = str(uuid4())
        doc_id = uuid4()
        chunks = [
            _make_chunk(doc_id, topic_ids=None),  # No topic_ids
            _make_chunk(doc_id, topic_ids=[topic_id]),
        ]

        counts = _count_topic_refs_from_chunks(chunks, topic_id)
        assert counts["chunk_count"] == 1
        assert counts["document_count"] == 1

    def test_recompute_chunk_with_multiple_topic_ids(self):
        """A single chunk can reference multiple topics."""
        topic_a = str(uuid4())
        topic_b = str(uuid4())
        doc_id = uuid4()
        chunks = [
            _make_chunk(doc_id, topic_ids=[topic_a, topic_b]),
            _make_chunk(doc_id, topic_ids=[topic_a]),
        ]

        counts_a = _count_topic_refs_from_chunks(chunks, topic_a)
        counts_b = _count_topic_refs_from_chunks(chunks, topic_b)

        assert counts_a["chunk_count"] == 2  # Both chunks
        assert counts_a["document_count"] == 1
        assert counts_b["chunk_count"] == 1  # Only first chunk
        assert counts_b["document_count"] == 1


# ---------------------------------------------------------------------------
# Test: end-to-end flush + import + recompute consistency
# ---------------------------------------------------------------------------


class TestFlushImportRecomputeConsistency:
    """Test that after flush + import + recompute, local and remote match."""

    @pytest.fixture
    def local_client(self, tmp_path):
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_single_run_counts_are_consistent(self, local_client):
        """After a single ingest run, local flush counts match recomputed counts."""
        topic = _make_topic("consistent topic")
        await local_client.insert_topic(topic)

        doc = _make_document()
        await local_client.insert_document(doc)

        chunks = [
            _make_chunk(doc.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(3)
        ]
        await local_client.insert_chunks(chunks)
        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=3, document_delta=1
        )

        local_client.flush()

        # Read local JSONL
        topics_data = _read_topics_jsonl(local_client.output_dir)
        local_chunk_count = topics_data[0]["chunk_count"]
        local_doc_count = topics_data[0]["document_count"]

        # Recompute from chunks (simulating what DB would do)
        recomputed = _count_topic_refs_from_chunks(
            list(local_client._chunks.values()), str(topic.id)
        )

        assert local_chunk_count == recomputed["chunk_count"]
        assert local_doc_count == recomputed["document_count"]

    @pytest.mark.asyncio
    async def test_multi_run_counts_stay_consistent(self, tmp_path):
        """After multiple runs, flush recomputes counts from all chunks on disk.

        This verifies the fix: counts don't drift across runs because flush
        recomputes from actual chunk metadata rather than accumulating.
        """
        from mtss.storage.local_client import LocalStorageClient

        output_dir = tmp_path / "output"

        # --- Run 1 ---
        client1 = LocalStorageClient(output_dir)
        topic = _make_topic("divergent topic")
        await client1.insert_topic(topic)

        doc1 = _make_document()
        await client1.insert_document(doc1)

        chunks_run1 = [
            _make_chunk(doc1.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(3)
        ]
        await client1.insert_chunks(chunks_run1)
        await client1.increment_topic_counts(
            [topic.id], chunk_delta=3, document_delta=1
        )
        client1.flush()

        # After run 1: recomputed from 3 chunks, 1 doc
        topics_after_run1 = _read_topics_jsonl(output_dir)
        assert topics_after_run1[0]["chunk_count"] == 3
        assert topics_after_run1[0]["document_count"] == 1

        # --- Run 2 (adds more chunks for a new document) ---
        client2 = LocalStorageClient(output_dir)

        doc2 = _make_document()
        await client2.insert_document(doc2)

        chunks_run2 = [
            _make_chunk(doc2.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(2)
        ]
        await client2.insert_chunks(chunks_run2)
        await client2.increment_topic_counts(
            [UUID(str(topic.id))], chunk_delta=2, document_delta=1
        )
        client2.flush()

        # After run 2: recomputed from ALL chunks (3 from run1 + 2 from run2)
        topics_after_run2 = _read_topics_jsonl(output_dir)
        assert topics_after_run2[0]["chunk_count"] == 5  # 3 + 2
        assert topics_after_run2[0]["document_count"] == 2  # 2 distinct docs

    @pytest.mark.asyncio
    async def test_import_topics_writes_local_counts_to_db(self, tmp_path):
        """Import writes whatever counts are in JSONL to the DB.

        After the flush fix, JSONL counts are recomputed (accurate), so import
        writes correct counts to the DB.
        """
        from mtss.models.serializers import dict_to_topic

        topic_id = str(uuid4())
        topic_data = {
            "id": topic_id,
            "name": "test import",
            "display_name": "Test Import",
            "chunk_count": 42,
            "document_count": 7,
        }
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "topics.jsonl").write_text(
            json.dumps(topic_data) + "\n"
        )

        # Simulate what _import_topics does: read JSONL, deserialize, insert
        topics_jsonl = _read_topics_jsonl(output_dir)
        assert len(topics_jsonl) == 1

        mock_db = MagicMock()
        mock_db.get_topic_by_name = AsyncMock(return_value=None)
        mock_db.insert_topic = AsyncMock()

        for td in topics_jsonl:
            existing = await mock_db.get_topic_by_name(td["name"])
            if not existing:
                topic = dict_to_topic(td)
                await mock_db.insert_topic(topic)

        inserted_topic = mock_db.insert_topic.call_args[0][0]
        assert inserted_topic.chunk_count == 42
        assert inserted_topic.document_count == 7

    @pytest.mark.asyncio
    async def test_import_topics_skips_existing(self, tmp_path):
        """Import should skip topics that already exist in DB."""
        from mtss.models.serializers import dict_to_topic

        topic_data = {
            "id": str(uuid4()),
            "name": "existing topic",
            "display_name": "Existing Topic",
            "chunk_count": 10,
            "document_count": 3,
        }
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "topics.jsonl").write_text(
            json.dumps(topic_data) + "\n"
        )

        existing = _make_topic("existing topic")
        mock_db = MagicMock()
        mock_db.get_topic_by_name = AsyncMock(return_value=existing)
        mock_db.insert_topic = AsyncMock()

        # Simulate _import_topics logic
        topics_jsonl = _read_topics_jsonl(output_dir)
        skipped = 0
        for td in topics_jsonl:
            existing_topic = await mock_db.get_topic_by_name(td["name"])
            if existing_topic:
                skipped += 1
            else:
                topic = dict_to_topic(td)
                await mock_db.insert_topic(topic)

        assert skipped == 1
        mock_db.insert_topic.assert_not_called()


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestTopicCountEdgeCases:
    """Test edge cases in topic count tracking."""

    @pytest.fixture
    def local_client(self, tmp_path):
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_topic_with_zero_chunks_is_dropped_by_flush(self, local_client):
        """A topic with zero chunks/documents is dropped as orphan during flush."""
        topic = _make_topic("orphan topic")
        await local_client.insert_topic(topic)

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        assert len(topics_data) == 0

        recomputed = _count_topic_refs_from_chunks(
            list(local_client._chunks.values()), str(topic.id)
        )
        assert recomputed["chunk_count"] == 0
        assert recomputed["document_count"] == 0

    @pytest.mark.asyncio
    async def test_topic_with_zero_chunks_recompute(self):
        """Recompute logic should return zero for a topic with no chunk refs."""
        topic_id = str(uuid4())
        other_topic = str(uuid4())
        doc_id = uuid4()

        chunks = [
            _make_chunk(doc_id, topic_ids=[other_topic], chunk_index=0),
            _make_chunk(doc_id, topic_ids=[other_topic], chunk_index=1),
        ]

        recomputed = _count_topic_refs_from_chunks(chunks, topic_id)
        assert recomputed["chunk_count"] == 0
        assert recomputed["document_count"] == 0

    @pytest.mark.asyncio
    async def test_chunk_with_multiple_topic_ids(self, local_client):
        """A chunk referencing multiple topics should count for all of them."""
        t1 = _make_topic("topic alpha")
        t2 = _make_topic("topic beta")
        t3 = _make_topic("topic gamma")
        await local_client.insert_topic(t1)
        await local_client.insert_topic(t2)
        await local_client.insert_topic(t3)

        doc = _make_document()
        await local_client.insert_document(doc)

        chunk = _make_chunk(
            doc.id,
            topic_ids=[str(t1.id), str(t2.id), str(t3.id)],
        )
        await local_client.insert_chunks([chunk])

        await local_client.increment_topic_counts(
            [t1.id, t2.id, t3.id], chunk_delta=1, document_delta=1
        )

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        topics_by_name = {t["name"]: t for t in topics_data}

        for name in ["topic alpha", "topic beta", "topic gamma"]:
            assert topics_by_name[name]["chunk_count"] == 1
            assert topics_by_name[name]["document_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_chunks_same_doc_single_topic(self, local_client):
        """Multiple chunks from same doc should give chunk_count > document_count."""
        topic = _make_topic("frequent topic")
        await local_client.insert_topic(topic)

        doc = _make_document()
        await local_client.insert_document(doc)

        chunks = [
            _make_chunk(doc.id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(10)
        ]
        await local_client.insert_chunks(chunks)

        await local_client.increment_topic_counts(
            [topic.id], chunk_delta=10, document_delta=1
        )

        local_client.flush()

        topics_data = _read_topics_jsonl(local_client.output_dir)
        all_chunks = list(local_client._chunks.values())
        recomputed = _count_topic_refs_from_chunks(all_chunks, str(topic.id))

        assert topics_data[0]["chunk_count"] == 10
        assert topics_data[0]["document_count"] == 1
        assert recomputed["chunk_count"] == 10
        assert recomputed["document_count"] == 1

    @pytest.mark.asyncio
    async def test_persist_ingest_result_increments_counts(self, local_client):
        """persist_ingest_result should call increment_topic_counts correctly."""
        topic = _make_topic("persist topic")
        await local_client.insert_topic(topic)

        doc_id = uuid4()
        email_doc = Document(
            id=doc_id,
            document_type=DocumentType.EMAIL,
            file_path="/test.eml",
            file_name="test.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )

        chunks = [
            _make_chunk(doc_id, topic_ids=[str(topic.id)], chunk_index=i)
            for i in range(3)
        ]

        await local_client.persist_ingest_result(
            email_doc=email_doc,
            attachment_docs=[],
            chunks=chunks,
            topic_ids=[topic.id],
            chunk_delta=3,
        )

        t = local_client._topics[str(topic.id)]
        assert t.chunk_count == 3
        assert t.document_count == 1

    @pytest.mark.asyncio
    async def test_persist_ingest_result_no_topics(self, local_client):
        """persist_ingest_result with no topics should not change any counts."""
        doc_id = uuid4()
        email_doc = Document(
            id=doc_id,
            document_type=DocumentType.EMAIL,
            file_path="/test.eml",
            file_name="test.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )

        chunks = [_make_chunk(doc_id, chunk_index=0)]

        await local_client.persist_ingest_result(
            email_doc=email_doc,
            attachment_docs=[],
            chunks=chunks,
            topic_ids=None,
            chunk_delta=0,
        )

        assert len(local_client._topics) == 0

    @pytest.mark.asyncio
    async def test_topic_roundtrip_preserves_counts(self):
        """Topic serialization roundtrip should preserve chunk/document counts."""
        from mtss.models.serializers import dict_to_topic, topic_to_dict

        topic = _make_topic("roundtrip topic", chunk_count=42, document_count=7)
        d = topic_to_dict(topic)
        json_str = json.dumps(d, default=str)
        d2 = json.loads(json_str)
        restored = dict_to_topic(d2)

        assert restored.chunk_count == 42
        assert restored.document_count == 7

    @pytest.mark.asyncio
    async def test_topic_roundtrip_default_zero_counts(self):
        """Topic without explicit counts should default to 0."""
        from mtss.models.serializers import dict_to_topic

        d = {
            "id": str(uuid4()),
            "name": "no counts",
            "display_name": "No Counts",
        }
        restored = dict_to_topic(d)

        assert restored.chunk_count == 0
        assert restored.document_count == 0


# ---------------------------------------------------------------------------
# Test: validate comparison logic
# ---------------------------------------------------------------------------


class TestValidateComparison:
    """Test the comparison logic that would detect mismatches."""

    def test_detect_mismatch_after_recompute(self):
        """Simulated validation: local stale counts vs recomputed DB counts."""
        topic_id = str(uuid4())
        doc1, doc2 = uuid4(), uuid4()

        # Local JSONL has accumulated counts from two runs
        local_counts = {"chunk_count": 15, "document_count": 5}

        # But actual chunks in DB tell a different story
        chunks = [
            _make_chunk(doc1, topic_ids=[topic_id], chunk_index=i)
            for i in range(7)
        ] + [
            _make_chunk(doc2, topic_ids=[topic_id], chunk_index=i)
            for i in range(3)
        ]

        recomputed = _count_topic_refs_from_chunks(chunks, topic_id)
        assert recomputed["chunk_count"] == 10
        assert recomputed["document_count"] == 2

        # Mismatch detected
        assert local_counts["chunk_count"] != recomputed["chunk_count"]
        assert local_counts["document_count"] != recomputed["document_count"]

    def test_no_mismatch_when_counts_correct(self):
        """When local counts match actual chunk refs, no mismatch."""
        topic_id = str(uuid4())
        doc_id = uuid4()

        chunks = [
            _make_chunk(doc_id, topic_ids=[topic_id], chunk_index=i)
            for i in range(5)
        ]

        recomputed = _count_topic_refs_from_chunks(chunks, topic_id)
        local_counts = {"chunk_count": 5, "document_count": 1}

        assert local_counts["chunk_count"] == recomputed["chunk_count"]
        assert local_counts["document_count"] == recomputed["document_count"]


# ---------------------------------------------------------------------------
# Test: chunk metadata topic_ids storage
# ---------------------------------------------------------------------------


class TestChunkMetadataTopicIds:
    """Test how topic_ids are stored in chunk metadata."""

    @pytest.fixture
    def local_client(self, tmp_path):
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_update_chunks_topic_ids(self, local_client):
        """update_chunks_topic_ids should set topic_ids in chunk metadata."""
        doc_id = uuid4()
        doc = _make_document(doc_id)
        await local_client.insert_document(doc)

        chunk = _make_chunk(doc_id, topic_ids=None)
        await local_client.insert_chunks([chunk])

        topic_ids = [uuid4(), uuid4()]
        await local_client.update_chunks_topic_ids(doc_id, topic_ids)

        updated_chunk = list(local_client._chunks.values())[0]
        assert "topic_ids" in updated_chunk.metadata
        assert len(updated_chunk.metadata["topic_ids"]) == 2
        assert str(topic_ids[0]) in updated_chunk.metadata["topic_ids"]

    @pytest.mark.asyncio
    async def test_chunk_metadata_topic_ids_are_strings(self, local_client):
        """topic_ids in metadata should be stored as strings (for JSON compat)."""
        doc_id = uuid4()
        topic_ids = [uuid4(), uuid4()]
        chunk = _make_chunk(doc_id, topic_ids=[str(t) for t in topic_ids])
        await local_client.insert_chunks([chunk])

        stored = list(local_client._chunks.values())[0]
        for tid in stored.metadata["topic_ids"]:
            assert isinstance(tid, str)

    @pytest.mark.asyncio
    async def test_chunk_topic_ids_survive_jsonl_roundtrip(self, local_client):
        """topic_ids in chunk metadata should survive JSONL serialization."""
        from mtss.models.serializers import chunk_to_dict, dict_to_chunk

        topic_ids = [str(uuid4()), str(uuid4())]
        doc_id = uuid4()
        chunk = _make_chunk(doc_id, topic_ids=topic_ids)

        d = chunk_to_dict(chunk)
        json_str = json.dumps(d, default=str)
        d2 = json.loads(json_str)
        restored = dict_to_chunk(d2)

        assert restored.metadata.get("topic_ids") == topic_ids


# ---------------------------------------------------------------------------
# JSONL reader helper
# ---------------------------------------------------------------------------


def _read_topics_jsonl(output_dir: Path) -> list[dict]:
    """Read topics from a JSONL file."""
    topics_path = output_dir / "topics.jsonl"
    if not topics_path.exists():
        return []
    results = []
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results
