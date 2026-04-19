"""Tests for mtss re-embed CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def mocked_storage_and_llm(comprehensive_mock_settings):
    """Patch archive storage to return a known markdown, and LLM/embeddings."""
    prose_md = (
        "# Report\n\n"
        + (
            "This audit covers operational findings across the inspection "
            "period. Recommendations are included. "
        )
        * 20
    )

    def _download(path):
        return prose_md.encode("utf-8")

    storage_mock = MagicMock()
    storage_mock.download_file = MagicMock(side_effect=_download)

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
        "mtss.storage.archive_storage.ArchiveStorage", return_value=storage_mock
    ), patch(
        "mtss.parsers.chunker.ContextGenerator", return_value=context_mock
    ), patch(
        "mtss.processing.embeddings.EmbeddingGenerator", return_value=embed_mock
    ), patch(
        "mtss.ingest.embedding_decider.get_settings",
        return_value=comprehensive_mock_settings,
    ):
        yield {
            "storage": storage_mock,
            "context": context_mock,
            "embed": embed_mock,
            "markdown": prose_md,
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
        assert not (output_dir / "chunks.jsonl").exists()

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
        assert (output_dir / "chunks.jsonl").exists()
        lines = (output_dir / "chunks.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 1
        first = json.loads(lines[0])
        assert first["embedding_mode"] in {"full", "summary", "metadata_only"}

    @pytest.mark.asyncio
    async def test_updates_documents_embedding_mode(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a")
        reloaded = json.loads(
            (output_dir / "documents.jsonl").read_text().strip().splitlines()[0]
        )
        assert reloaded["embedding_mode"] is not None

    @pytest.mark.asyncio
    async def test_full_mode_chunk_ids_deterministic(
        self, output_dir, mocked_storage_and_llm
    ):
        from mtss.cli.reembed_cmd import reembed_run

        docs = [_make_doc_row("doc_a", str(uuid4()), "doc_a/x.md")]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a", mode="full")
        first_lines = (output_dir / "chunks.jsonl").read_text().splitlines()
        first_ids = [json.loads(ln)["chunk_id"] for ln in first_lines if ln.strip()]

        # Reset docs (keep them with embedding_mode=None so not idempotent).
        _write_jsonl(output_dir / "documents.jsonl", docs)

        await reembed_run(output_dir=output_dir, doc_id="doc_a", mode="full")
        second_lines = (output_dir / "chunks.jsonl").read_text().splitlines()
        second_ids = [json.loads(ln)["chunk_id"] for ln in second_lines if ln.strip()]

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
