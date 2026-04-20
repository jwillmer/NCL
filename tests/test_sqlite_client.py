"""Unit tests for SqliteStorageClient.

Covers:
- Schema init + WAL/FK pragmas
- Document/chunk/topic insert + lookup + UUID remap on doc_id dedup
- UNIQUE(chunk_id) constraint surfaces dupes as errors
- FK CASCADE: delete_document_for_reprocess removes chunks + chunk_topics
- Topic membership via chunk_topics (single source of truth)
- merge_similar_topics transactional move of memberships
- log_ingest_event + log_unsupported_file land in ingest_events table
- flush() recomputes topic counts + drops orphan topics
- iter_* helpers stream rows with decoded embeddings + topic_ids

Fixtures use ``:memory:`` DBs so tests stay fast.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mtss.models.chunk import Chunk
from mtss.models.document import (
    Document,
    DocumentType,
    EmailMetadata,
    ProcessingStatus,
)
from mtss.models.topic import Topic
from mtss.storage.sqlite_client import (
    SqliteStorageClient,
    _decode_embedding,
    _encode_embedding,
)


# ── fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_client(tmp_path: Path) -> SqliteStorageClient:
    """Fresh SqliteStorageClient backed by a temp directory."""
    return SqliteStorageClient(tmp_path)


@pytest.fixture
def email_doc() -> Document:
    uid = UUID("aaaaaaaa-0000-0000-0000-000000000001")
    return Document(
        id=uid,
        parent_id=None,
        root_id=uid,
        depth=0,
        path=[str(uid)],
        document_type=DocumentType.EMAIL,
        file_path="/data/emails/test.eml",
        file_name="test.eml",
        file_hash="hash-a",
        source_id="test.eml",
        doc_id="doc-a",
        content_version=1,
        ingest_version=1,
        source_title="Test subject",
        email_metadata=EmailMetadata(
            subject="Test subject",
            participants=["a@x", "b@x"],
            initiator="a@x",
            message_count=1,
        ),
        status=ProcessingStatus.PENDING,
    )


@pytest.fixture
def chunk_for(email_doc: Document) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=email_doc.id,
        chunk_id="chunk-a-1",
        content="hello world",
        chunk_index=0,
        char_start=0,
        char_end=11,
        line_from=1,
        line_to=1,
        section_path=["Intro"],
        embedding=[0.1] * 8,
        metadata={"source_file": "test.eml"},
    )


# ── schema + pragmas ────────────────────────────────────────────────

def test_schema_tables_created(tmp_client: SqliteStorageClient):
    names = {
        row["name"]
        for row in tmp_client._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {
        "documents",
        "chunks",
        "topics",
        "chunk_topics",
        "ingest_events",
        "processing_log",
        "run_history",
        "manifest",
    } <= names


def test_wal_and_fk_pragmas(tmp_client: SqliteStorageClient):
    mode = tmp_client._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    fk = tmp_client._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_schema_version_seeded(tmp_client: SqliteStorageClient):
    val = tmp_client._conn.execute(
        "SELECT value FROM manifest WHERE key = 'schema_version'"
    ).fetchone()
    assert val is not None
    assert int(val[0]) >= 1


def test_processing_log_schema_matches_between_client_and_tracker(tmp_path: Path):
    """SqliteStorageClient and SqliteProgressTracker must produce the exact
    same ``processing_log`` schema regardless of which one initialises the
    DB first.

    Regression: the tracker historically declared its own
    ``CREATE TABLE`` that omitted the ``ingest_version`` column. Whichever
    class won the init race fixed the schema; writes from the loser to the
    missing column silently no-op'd. Both classes now share
    ``PROCESSING_LOG_SCHEMA_SQL``, so a fresh init from either side must
    yield identical column names and identical indexes.
    """
    from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

    def _columns(conn: sqlite3.Connection) -> list[str]:
        return [r[1] for r in conn.execute("PRAGMA table_info(processing_log)")]

    def _indexes(conn: sqlite3.Connection) -> set[str]:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='processing_log' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }

    client_dir = tmp_path / "client"
    client = SqliteStorageClient(client_dir)
    client_columns = _columns(client._conn)
    client_indexes = _indexes(client._conn)

    tracker_dir = tmp_path / "tracker"
    tracker = SqliteProgressTracker(tracker_dir)
    tracker_columns = _columns(tracker._conn)
    tracker_indexes = _indexes(tracker._conn)

    assert client_columns == tracker_columns
    assert client_indexes == tracker_indexes
    # ``ingest_version`` is the column the tracker historically dropped —
    # pin it explicitly so the regression cannot regress quietly.
    assert "ingest_version" in tracker_columns


def test_processing_log_tracker_then_client_yields_full_schema(tmp_path: Path):
    """Tracker-first init must leave the table with every column the
    client expects to write. Catches any future drift where the tracker
    is upgraded but the shared constant isn't.
    """
    from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

    SqliteProgressTracker(tmp_path)
    client = SqliteStorageClient(tmp_path)

    columns = [
        r[1] for r in client._conn.execute("PRAGMA table_info(processing_log)")
    ]
    expected = {
        "file_path",
        "file_hash",
        "status",
        "started_at",
        "completed_at",
        "duration_seconds",
        "attempts",
        "error",
        "ingest_version",
    }
    assert expected.issubset(set(columns))


# ── embedding codec ─────────────────────────────────────────────────

def test_embedding_roundtrip():
    vec = [0.1, -0.5, 3.14, 0.0]
    blob, dim = _encode_embedding(vec)
    assert dim == 4
    out = _decode_embedding(blob, dim)
    assert out is not None
    assert len(out) == 4
    for a, b in zip(vec, out):
        assert abs(a - b) < 1e-6


def test_embedding_none_roundtrip():
    blob, dim = _encode_embedding(None)
    assert blob is None and dim is None
    assert _decode_embedding(None, None) is None


# ── documents ────────────────────────────────────────────────────────

async def test_insert_and_lookup_document(tmp_client: SqliteStorageClient, email_doc: Document):
    stored = await tmp_client.insert_document(email_doc)
    assert stored.id == email_doc.id

    by_hash = await tmp_client.get_document_by_hash("hash-a")
    by_doc_id = await tmp_client.get_document_by_doc_id("doc-a")
    by_source = await tmp_client.get_document_by_source_id("test.eml")
    by_id = await tmp_client.get_document_by_id(email_doc.id)

    for hit in (by_hash, by_doc_id, by_source, by_id):
        assert hit is not None
        assert str(hit.id) == str(email_doc.id)


async def test_insert_document_dedups_by_doc_id(
    tmp_client: SqliteStorageClient, email_doc: Document
):
    first = await tmp_client.insert_document(email_doc)

    # Different UUID, same doc_id → returns the original; no duplicate row.
    dup = email_doc.model_copy(update={"id": uuid4(), "file_hash": "hash-a-dup"})
    second = await tmp_client.insert_document(dup)
    assert str(second.id) == str(first.id)

    count = tmp_client._conn.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc-a'"
    ).fetchone()[0]
    assert count == 1


# ── chunks + FK cascade ─────────────────────────────────────────────

async def test_insert_chunk_and_lookup_by_document(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_chunks([chunk_for])

    chunks = await tmp_client.get_chunks_by_document(email_doc.id)
    assert len(chunks) == 1
    assert str(chunks[0].id) == str(chunk_for.id)

    # Embedding BLOB round-trips
    loaded = list(tmp_client.iter_chunks())
    assert len(loaded) == 1
    assert loaded[0]["embedding"] is not None
    assert abs(loaded[0]["embedding"][0] - 0.1) < 1e-6


async def test_chunk_id_unique_constraint(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    """Inserting two chunks with the same chunk_id via separate calls must
    replace the first, not duplicate — the root of the orphan bug that
    forced this migration. Covers INSERT OR REPLACE semantics."""
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_chunks([chunk_for])
    # Second insert with the same chunk_id but a new UUID id.
    dup = chunk_for.model_copy(update={"id": uuid4(), "content": "replaced"})
    await tmp_client.insert_chunks([dup])

    count = tmp_client._conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_id = ?", (chunk_for.chunk_id,)
    ).fetchone()[0]
    assert count == 1


async def test_delete_document_cascades_to_chunks(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    """force_reparse relies on FK CASCADE to atomically wipe chunks."""
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_chunks([chunk_for])
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    tmp_client.delete_document_for_reprocess(email_doc.id)

    assert tmp_client._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


async def test_delete_document_cascades_to_chunk_topics(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    topic = Topic(id=uuid4(), name="safety", display_name="Safety", embedding=[0.1] * 8)
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_topic(topic)
    chunk_for.metadata["topic_ids"] = [str(topic.id)]
    await tmp_client.insert_chunks([chunk_for])

    ct_before = tmp_client._conn.execute("SELECT COUNT(*) FROM chunk_topics").fetchone()[0]
    assert ct_before == 1

    tmp_client.delete_document_for_reprocess(email_doc.id)
    ct_after = tmp_client._conn.execute("SELECT COUNT(*) FROM chunk_topics").fetchone()[0]
    assert ct_after == 0
    # Topic itself not deleted, only its membership:
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0] == 1


async def test_chunk_topics_is_single_source_of_truth(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    """Topic IDs must NOT be stored inside chunks.metadata_json — junction
    table only. Otherwise Wave A/B style drift becomes possible again."""
    topic = Topic(id=uuid4(), name="t", display_name="T", embedding=[0.1] * 8)
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_topic(topic)
    chunk_for.metadata["topic_ids"] = [str(topic.id)]
    await tmp_client.insert_chunks([chunk_for])

    row = tmp_client._conn.execute(
        "SELECT metadata_json FROM chunks WHERE id = ?", (str(chunk_for.id),)
    ).fetchone()
    stored = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    assert "topic_ids" not in stored

    # iter_chunks reconstructs topic_ids from the junction table.
    rebuilt = list(tmp_client.iter_chunks())
    assert rebuilt[0]["metadata"]["topic_ids"] == [str(topic.id)]


# ── topics ───────────────────────────────────────────────────────────

async def test_insert_topic_and_lookup(tmp_client: SqliteStorageClient):
    topic = Topic(id=uuid4(), name="maintenance", display_name="Maintenance", embedding=[0.1] * 8)
    await tmp_client.insert_topic(topic)
    by_id = await tmp_client.get_topic_by_id(topic.id)
    by_name = await tmp_client.get_topic_by_name("maintenance")
    assert by_id is not None and str(by_id.id) == str(topic.id)
    assert by_name is not None and str(by_name.id) == str(topic.id)


async def test_find_similar_topics(tmp_client: SqliteStorageClient):
    t1 = Topic(id=uuid4(), name="a", display_name="A", embedding=[1.0, 0.0, 0.0, 0.0])
    t2 = Topic(id=uuid4(), name="b", display_name="B", embedding=[0.99, 0.1, 0.0, 0.0])
    t3 = Topic(id=uuid4(), name="c", display_name="C", embedding=[0.0, 1.0, 0.0, 0.0])
    for t in (t1, t2, t3):
        await tmp_client.insert_topic(t)
    hits = await tmp_client.find_similar_topics([1.0, 0.0, 0.0, 0.0], threshold=0.9)
    names = {h["name"] for h in hits}
    assert "a" in names and "b" in names
    assert "c" not in names


async def test_merge_similar_topics_moves_memberships(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    keeper = Topic(id=uuid4(), name="keeper", display_name="Keeper", embedding=[1.0, 0.0, 0.0, 0.0], chunk_count=5)
    absorbed = Topic(id=uuid4(), name="absorbed", display_name="Absorbed", embedding=[0.99, 0.01, 0.0, 0.0], chunk_count=2)
    await tmp_client.insert_topic(keeper)
    await tmp_client.insert_topic(absorbed)
    await tmp_client.insert_document(email_doc)
    chunk_for.metadata["topic_ids"] = [str(absorbed.id)]
    await tmp_client.insert_chunks([chunk_for])

    merges = tmp_client.merge_similar_topics(threshold=0.95)
    assert len(merges) == 1

    # Absorbed gone; keeper inherits the membership.
    assert tmp_client._conn.execute(
        "SELECT COUNT(*) FROM topics WHERE id = ?", (str(absorbed.id),)
    ).fetchone()[0] == 0
    kept_rows = tmp_client._conn.execute(
        "SELECT topic_id FROM chunk_topics"
    ).fetchall()
    assert len(kept_rows) == 1
    assert kept_rows[0]["topic_id"] == str(keeper.id)


# ── events ───────────────────────────────────────────────────────────

def test_log_ingest_event_inserts_row(tmp_client: SqliteStorageClient):
    tmp_client.log_ingest_event(
        document_id=uuid4(),
        event_type="foo",
        severity="warning",
        message="m",
    )
    count = tmp_client._conn.execute(
        "SELECT COUNT(*) FROM ingest_events WHERE event_type = 'foo'"
    ).fetchone()[0]
    assert count == 1


async def test_log_unsupported_file_canonicalises_event_type(
    tmp_client: SqliteStorageClient, tmp_path: Path
):
    p = tmp_path / "x.bin"
    p.write_bytes(b"nope")
    await tmp_client.log_unsupported_file(
        file_path=p,
        reason="unsupported_format: application/octet-stream",
    )
    row = tmp_client._conn.execute(
        "SELECT event_type, reason FROM ingest_events LIMIT 1"
    ).fetchone()
    assert row["event_type"] == "unsupported_format"
    assert "application/octet-stream" in row["reason"]


# ── flush ───────────────────────────────────────────────────────────

async def test_flush_recomputes_topic_counts_and_drops_orphans(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    used = Topic(id=uuid4(), name="used", display_name="Used", embedding=[0.1] * 4)
    orphan = Topic(id=uuid4(), name="orphan", display_name="Orphan", embedding=[0.2] * 4)
    for t in (used, orphan):
        await tmp_client.insert_topic(t)
    await tmp_client.insert_document(email_doc)
    chunk_for.metadata["topic_ids"] = [str(used.id)]
    await tmp_client.insert_chunks([chunk_for])

    tmp_client.flush()

    rows = list(tmp_client._conn.execute(
        "SELECT name, chunk_count, document_count FROM topics"
    ))
    assert len(rows) == 1
    assert rows[0]["name"] == "used"
    assert rows[0]["chunk_count"] == 1
    assert rows[0]["document_count"] == 1


# ── transactional safety ────────────────────────────────────────────

async def test_force_reparse_is_atomic(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    """Re-inserting after a reparse must leave exactly one copy per chunk_id.

    Mirrors the production flow: delete_document_for_reprocess → insert
    fresh. The original LocalClient flush could drop this into a partial
    state on Windows file-lock failure; SQLite cannot.
    """
    await tmp_client.insert_document(email_doc)
    await tmp_client.insert_chunks([chunk_for])

    # Simulate a force_reparse cycle.
    tmp_client.delete_document_for_reprocess(email_doc.id)

    fresh = email_doc.model_copy(update={"id": uuid4()})
    chunk_new = chunk_for.model_copy(update={"id": uuid4(), "document_id": fresh.id})
    await tmp_client.insert_document(fresh)
    await tmp_client.insert_chunks([chunk_new])

    # Exactly one doc, exactly one chunk, no orphan residue.
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    # No dangling chunk_topics row.
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM chunk_topics").fetchone()[0] == 0


# ── iterators + persist_ingest_result ───────────────────────────────

async def test_persist_ingest_result_atomic(
    tmp_client: SqliteStorageClient, email_doc: Document, chunk_for: Chunk
):
    attachment_id = uuid4()
    attachment = email_doc.model_copy(update={
        "id": attachment_id,
        "parent_id": email_doc.id,
        "root_id": email_doc.id,
        "document_type": DocumentType.ATTACHMENT_PDF,
        "file_name": "a.pdf",
        "file_hash": "hash-att",
        "doc_id": "doc-att",
        "source_id": "test.eml/a.pdf",
    })
    await tmp_client.persist_ingest_result(
        email_doc=email_doc,
        attachment_docs=[attachment],
        chunks=[chunk_for],
        topic_ids=[],
        chunk_delta=1,
    )

    assert tmp_client._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    assert tmp_client._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


def test_iter_documents_streams_all_rows(
    tmp_client: SqliteStorageClient, email_doc: Document
):
    import asyncio
    asyncio.run(tmp_client.insert_document(email_doc))
    rows = list(tmp_client.iter_documents())
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "doc-a"
