"""Tests for mtss re-embed CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


PROSE_MD = (
    "# Report\n\n"
    + (
        "This audit covers operational findings across the inspection "
        "period. Recommendations are included. "
    )
    * 20
)


def _make_doc_row(
    doc_id: str,
    uuid_str: str,
    archive_uri: str,
    embedding_mode: str | None = None,
    document_type: str = "attachment_pdf",
) -> dict:
    """Shape-compatible documents.jsonl row."""
    return {
        "id": uuid_str,
        "doc_id": doc_id,
        "source_id": f"test.eml/{doc_id}.pdf",
        "content_version": 1,
        "ingest_version": 5,
        "document_type": document_type,
        "file_path": f"/data/attachments/{doc_id}.pdf",
        "file_name": f"{doc_id}.pdf",
        "file_hash": "hash123",
        "parent_id": None,
        "root_id": None,
        "depth": 1,
        "path": [],
        "source_title": f"{doc_id}.pdf",
        "archive_path": archive_uri.rsplit("/", 1)[0],
        "archive_browse_uri": f"/archive/{archive_uri}",
        "archive_download_uri": f"/archive/{archive_uri.replace('.md', '')}",
        "status": "completed",
        "embedding_mode": embedding_mode,
        "error_message": None,
        "processed_at": "2026-04-18T12:00:00",
        "created_at": "2026-04-18T12:00:00",
        "updated_at": "2026-04-18T12:00:00",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Seed the SQLite ingest.db with the given doc rows and stamp each
    archive file on disk. Keeps the old function name so existing tests don't
    churn — but the JSONL is gone; the re-embed command reads from SQLite now.
    """
    from mtss.storage.sqlite_client import SqliteStorageClient

    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    client = SqliteStorageClient(output_dir=output_dir)
    try:
        conn = client._conn
        # Clear any prior seeds from earlier calls within the same test.
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        for r in rows:
            row = {
                "id":              r["id"],
                "doc_id":          r.get("doc_id"),
                "source_id":       r.get("source_id"),
                "document_type":   r.get("document_type"),
                "status":          r.get("status") or "pending",
                "error_message":   r.get("error_message"),
                "file_hash":       r.get("file_hash"),
                "file_name":       r.get("file_name"),
                "file_path":       r.get("file_path"),
                "parent_id":       r.get("parent_id"),
                "root_id":         r.get("root_id") or r["id"],
                "depth":           r.get("depth", 0),
                "content_version": r.get("content_version", 1),
                "ingest_version":  r.get("ingest_version", 1),
                "archive_path":    r.get("archive_path"),
                "title":           r.get("source_title"),
                "source_title":    r.get("source_title"),
                "mime_type":       None,
                "content_type":    None,
                "size_bytes":      None,
                "embedding_mode":  r.get("embedding_mode"),
                "archive_browse_uri":   r.get("archive_browse_uri"),
                "archive_download_uri": r.get("archive_download_uri"),
                "metadata_json":   None,
                "processed_at":    r.get("processed_at"),
                "created_at":      r.get("created_at"),
                "updated_at":      r.get("updated_at"),
            }
            cols = list(row.keys())
            placeholders = ",".join(["?"] * len(cols))
            conn.execute(
                f"INSERT INTO documents ({','.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
    finally:
        conn.close()

    # Re-embed reads each doc's markdown from output_dir/archive/<archive_uri>.
    # Write the prose fixture there for every doc so the live path exercises the
    # disk reader rather than being stubbed out.
    archive_root = output_dir / "archive"
    for r in rows:
        uri = r.get("archive_browse_uri")
        if not uri:
            continue
        rel = uri.removeprefix("/archive/")
        dest = archive_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(PROSE_MD, encoding="utf-8")


def _chunks_for_doc(output_dir: Path, doc_id: str) -> list[dict]:
    """Read back chunk rows for one doc_id from ingest.db (tests use this)."""
    from mtss.storage.sqlite_client import SqliteStorageClient

    client = SqliteStorageClient(output_dir=output_dir)
    try:
        rows = list(client._conn.execute(
            "SELECT c.* FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.doc_id = ? ORDER BY c.chunk_index",
            (doc_id,),
        ))
        return [dict(r) for r in rows]
    finally:
        client._conn.close()


def _doc_embedding_mode(output_dir: Path, doc_id: str) -> str | None:
    from mtss.storage.sqlite_client import SqliteStorageClient

    client = SqliteStorageClient(output_dir=output_dir)
    try:
        row = client._conn.execute(
            "SELECT embedding_mode FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["embedding_mode"] if row else None
    finally:
        client._conn.close()


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def mocked_storage_and_llm(comprehensive_mock_settings):
    """Patch the LLM + embedding components. Archive markdown is now written
    to disk by _write_jsonl (re-embed reads local files directly)."""
    context_mock = MagicMock()
    context_mock.generate_context = AsyncMock(return_value="ctx summary")
    context_mock.build_embedding_text = MagicMock(
        side_effect=lambda ctx, content: f"{ctx}\n\n{content}" if ctx else content
    )

    embed_mock = MagicMock()
    async def _embed(chunks):
        for c in chunks:
            c.embedding = [0.1] * 1536
        return chunks
    embed_mock.embed_chunks = AsyncMock(side_effect=_embed)

    with patch(
        "mtss.parsers.chunker.ContextGenerator", return_value=context_mock
    ), patch(
        "mtss.processing.embeddings.EmbeddingGenerator", return_value=embed_mock
    ), patch(
        "mtss.ingest.embedding_decider.get_settings",
        return_value=comprehensive_mock_settings,
    ):
        yield {
            "context": context_mock,
            "embed": embed_mock,
            "markdown": PROSE_MD,
        }


# ---------- TestDryRun ---------- #


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/attachments/a.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(
            output_dir=output_dir, doc_id="doc_a", dry_run=True
        )
        assert stats.docs_considered == 1
        assert stats.docs_committed == 0
        assert _chunks_for_doc(output_dir, "doc_a") == []

    @pytest.mark.asyncio
    async def test_dry_run_populates_mode_distribution(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row(f"doc_{i}", str(uuid4()), f"doc_{i}/x.md") for i in range(3)]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(output_dir=output_dir, dry_run=True)
        assert stats.docs_considered == 3
        assert sum(stats.mode_distribution.values()) == 3


# ---------- TestLimit ---------- #


class TestLimit:
    @pytest.mark.asyncio
    async def test_limit_caps_committed_docs(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row(f"doc_{i}", str(uuid4()), f"doc_{i}/x.md") for i in range(5)]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(output_dir=output_dir, limit=2, dry_run=False)
        assert stats.docs_committed == 2


# ---------- TestIdempotency ---------- #


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_skips_when_mode_already_matches(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [
            _make_doc_row(
                "doc_a", str(uuid4()), "doc_a/x.md", embedding_mode="full"
            )
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(output_dir=output_dir, doc_id="doc_a")
        assert stats.docs_skipped_idempotent == 1
        assert stats.docs_committed == 0

    @pytest.mark.asyncio
    async def test_force_overrides_idempotent_skip(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [
            _make_doc_row(
                "doc_a", str(uuid4()), "doc_a/x.md", embedding_mode="full"
            )
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(
            output_dir=output_dir, doc_id="doc_a", force=True
        )
        assert stats.docs_committed == 1
        assert stats.docs_skipped_idempotent == 0


# ---------- TestForcedMode ---------- #


class TestForcedMode:
    @pytest.mark.asyncio
    async def test_forcing_summary_bypasses_decider(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(
            output_dir=output_dir, doc_id="doc_a", mode="summary"
        )
        assert stats.docs_committed == 1
        assert stats.mode_distribution.get("summary") == 1

    @pytest.mark.asyncio
    async def test_forcing_metadata_only_produces_one_chunk(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(
            output_dir=output_dir, doc_id="doc_a", mode="metadata_only"
        )
        assert stats.docs_committed == 1
        assert stats.total_chunks_written == 1


# ---------- TestPersistence ---------- #


class TestPersistence:
    @pytest.mark.asyncio
    async def test_writes_chunks_jsonl(self, output_dir, mocked_storage_and_llm):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a")
        chunks = _chunks_for_doc(output_dir, "doc_a")
        assert len(chunks) >= 1
        assert chunks[0]["embedding_mode"] in {"full", "summary", "metadata_only"}

    @pytest.mark.asyncio
    async def test_updates_documents_embedding_mode(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a")
        assert _doc_embedding_mode(output_dir, "doc_a") is not None

    @pytest.mark.asyncio
    async def test_full_mode_chunk_ids_deterministic(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a", mode="full")
        first_ids = [c["chunk_id"] for c in _chunks_for_doc(output_dir, "doc_a")]

        # Reset docs (keep them with embedding_mode=None so not idempotent).
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a", mode="full")
        second_ids = [c["chunk_id"] for c in _chunks_for_doc(output_dir, "doc_a")]

        assert first_ids == second_ids


# ---------- TestImageFilter ---------- #


class TestImageFilter:
    @pytest.mark.asyncio
    async def test_image_attachments_skipped_by_default(
        self, output_dir, mocked_storage_and_llm
    ):
        """Image attachments hold vision-API descriptions that don't change
        between runs — re-embedding them is wasteful, skip by default."""
        from mtss.cli.reembed_cmd import reembed_run

        docs = [
            _make_doc_row("pdf_a", str(uuid4()), "pdf_a/x.md"),
            _make_doc_row(
                "img_a", str(uuid4()), "img_a/x.md",
                document_type="attachment_image",
            ),
            _make_doc_row(
                "img_b", str(uuid4()), "img_b/x.md",
                document_type="attachment_image",
            ),
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(output_dir=output_dir, dry_run=True)
        # Only the PDF survives the filter.
        assert stats.docs_considered == 1

    @pytest.mark.asyncio
    async def test_include_images_flag_processes_them(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [
            _make_doc_row("pdf_a", str(uuid4()), "pdf_a/x.md"),
            _make_doc_row(
                "img_a", str(uuid4()), "img_a/x.md",
                document_type="attachment_image",
            ),
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        stats = await reembed_run(
            output_dir=output_dir, dry_run=True, include_images=True
        )
        assert stats.docs_considered == 2
