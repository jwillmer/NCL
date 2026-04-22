"""Tests for mtss maintenance CLI commands (reindex-chunks, etc.)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from mtss.cli import maintenance_cmd


def _make_chunk_stub():
    """Create a minimal chunk-like object the reindex path can mutate."""
    chunk = MagicMock()
    chunk.char_start = 0
    chunk.char_end = 100
    chunk.content = "x" * 100
    chunk.context_summary = None
    chunk.embedding_text = None
    return chunk


def _make_doc_row(
    doc_uuid: str,
    *,
    depth: int = 0,
    root_id: str | None = None,
    archive_uri: str = "test_folder/body.md",
) -> dict:
    return {
        "id": doc_uuid,
        "doc_id": f"docid-{doc_uuid[:8]}",
        "source_id": f"test.eml/{doc_uuid[:8]}",
        "source_title": f"Doc {doc_uuid[:8]}",
        "archive_browse_uri": f"/archive/{archive_uri}",
        "archive_download_uri": f"/archive/{archive_uri}",
        "depth": depth,
        "root_id": root_id,
    }


def _build_db_mock(chunks_metadata_rows: list[dict]) -> MagicMock:
    """Build a fake SupabaseClient whose ``client.table('chunks')`` query for
    topic_ids returns the given ``metadata`` rows, and whose
    ``update_chunks_topic_ids`` / ``delete_chunks_by_document`` /
    ``insert_chunks`` are AsyncMocks we can assert on."""
    db = MagicMock()
    db.close = AsyncMock()
    db.delete_chunks_by_document = AsyncMock(return_value=0)
    db.insert_chunks = AsyncMock(return_value=None)
    db.update_chunks_topic_ids = AsyncMock(return_value=1)

    # Model the supabase-py fluent query builder for
    # db.client.table("chunks").select(...).eq(...).limit(...).execute()
    exec_result = MagicMock()
    exec_result.data = chunks_metadata_rows
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.limit.return_value = query
    query.execute.return_value = exec_result

    table = MagicMock()
    table.select.return_value.eq.return_value.limit.return_value.execute.return_value = (
        exec_result
    )
    db.client.table.return_value = table
    return db


@pytest.mark.asyncio
async def test_reindex_stamps_topic_ids_for_email_body_case():
    """Email body (depth=0) reindex should stamp its own topic_ids back onto
    the regenerated chunks."""
    doc_uuid = str(uuid4())
    topic_a = str(uuid4())
    topic_b = str(uuid4())
    doc_row = _make_doc_row(doc_uuid, depth=0, root_id=None)

    db = _build_db_mock(
        chunks_metadata_rows=[{"metadata": {"topic_ids": [topic_a, topic_b]}}]
    )

    with (
        patch.object(maintenance_cmd, "_get_documents_by_id", new=AsyncMock(return_value=[doc_row])),
        patch("mtss.storage.supabase_client.SupabaseClient", return_value=db),
        patch("mtss.storage.archive_storage.ArchiveStorage") as storage_cls,
        patch("mtss.parsers.chunker.DocumentChunker") as chunker_cls,
        patch("mtss.parsers.chunker.ContextGenerator") as ctx_cls,
        patch("mtss.processing.embeddings.EmbeddingGenerator") as embed_cls,
    ):
        storage_cls.return_value.download_file.return_value = b"# body\n\nhello"
        chunker_cls.return_value.chunk_text.return_value = [_make_chunk_stub()]
        ctx_cls.return_value.generate_context = AsyncMock(return_value="ctx")
        ctx_cls.return_value.build_embedding_text.return_value = "ctx\nhello"
        embed_cls.return_value.embed_chunks = AsyncMock(
            side_effect=lambda chunks: chunks
        )

        await maintenance_cmd._reindex_chunks(
            doc_id=doc_uuid, missing_lines=False, dry_run=False, limit=1
        )

    db.update_chunks_topic_ids.assert_awaited_once()
    called_uuid, called_topics = db.update_chunks_topic_ids.await_args.args
    assert called_uuid == UUID(doc_uuid)
    assert called_topics == [topic_a, topic_b]


@pytest.mark.asyncio
async def test_reindex_stamps_topic_ids_for_attachment_case():
    """Attachment (depth>0) reindex resolves the parent email via root_id and
    stamps the email's topic_ids onto the regenerated attachment chunks."""
    attachment_uuid = str(uuid4())
    root_uuid = str(uuid4())
    topic_a = str(uuid4())
    doc_row = _make_doc_row(
        attachment_uuid, depth=1, root_id=root_uuid,
        archive_uri="test_folder/attachment.md",
    )

    db = _build_db_mock(
        chunks_metadata_rows=[{"metadata": {"topic_ids": [topic_a]}}]
    )

    with (
        patch.object(maintenance_cmd, "_get_documents_by_id", new=AsyncMock(return_value=[doc_row])),
        patch("mtss.storage.supabase_client.SupabaseClient", return_value=db),
        patch("mtss.storage.archive_storage.ArchiveStorage") as storage_cls,
        patch("mtss.parsers.chunker.DocumentChunker") as chunker_cls,
        patch("mtss.parsers.chunker.ContextGenerator") as ctx_cls,
        patch("mtss.processing.embeddings.EmbeddingGenerator") as embed_cls,
    ):
        storage_cls.return_value.download_file.return_value = b"# attach\n\nbody"
        chunker_cls.return_value.chunk_text.return_value = [_make_chunk_stub()]
        ctx_cls.return_value.generate_context = AsyncMock(return_value="ctx")
        ctx_cls.return_value.build_embedding_text.return_value = "ctx\nbody"
        embed_cls.return_value.embed_chunks = AsyncMock(
            side_effect=lambda chunks: chunks
        )

        await maintenance_cmd._reindex_chunks(
            doc_id=attachment_uuid, missing_lines=False, dry_run=False, limit=1
        )

    # The topic-ids query must have been issued against the ROOT (email)
    # document, not the attachment's own id.
    table_call_args = [
        c.args[0] if c.args else None
        for c in db.client.table.call_args_list
    ]
    assert "chunks" in table_call_args
    eq_calls = (
        db.client.table.return_value.select.return_value.eq.call_args_list
    )
    assert eq_calls, "expected an .eq() filter on the chunks query"
    # Most recent .eq() call is the one used to fetch parent topic_ids.
    eq_kwargs = eq_calls[-1]
    assert eq_kwargs.args == ("document_id", root_uuid)

    db.update_chunks_topic_ids.assert_awaited_once()
    called_uuid, called_topics = db.update_chunks_topic_ids.await_args.args
    assert called_uuid == UUID(attachment_uuid)
    assert called_topics == [topic_a]


@pytest.mark.asyncio
async def test_reindex_skips_topic_stamp_when_parent_has_no_topics():
    """If the parent email has no topic_ids (legitimately empty, 3.9% of v=6
    emails), we must NOT touch update_chunks_topic_ids — passing an empty list
    would clear the junction for a no-op."""
    doc_uuid = str(uuid4())
    doc_row = _make_doc_row(doc_uuid, depth=0)

    db = _build_db_mock(
        # Parent chunk exists but has no topic_ids in metadata
        chunks_metadata_rows=[{"metadata": {}}]
    )

    with (
        patch.object(maintenance_cmd, "_get_documents_by_id", new=AsyncMock(return_value=[doc_row])),
        patch("mtss.storage.supabase_client.SupabaseClient", return_value=db),
        patch("mtss.storage.archive_storage.ArchiveStorage") as storage_cls,
        patch("mtss.parsers.chunker.DocumentChunker") as chunker_cls,
        patch("mtss.parsers.chunker.ContextGenerator") as ctx_cls,
        patch("mtss.processing.embeddings.EmbeddingGenerator") as embed_cls,
    ):
        storage_cls.return_value.download_file.return_value = b"# body"
        chunker_cls.return_value.chunk_text.return_value = [_make_chunk_stub()]
        ctx_cls.return_value.generate_context = AsyncMock(return_value="ctx")
        ctx_cls.return_value.build_embedding_text.return_value = "ctx"
        embed_cls.return_value.embed_chunks = AsyncMock(
            side_effect=lambda chunks: chunks
        )

        await maintenance_cmd._reindex_chunks(
            doc_id=doc_uuid, missing_lines=False, dry_run=False, limit=1
        )

    db.update_chunks_topic_ids.assert_not_awaited()
