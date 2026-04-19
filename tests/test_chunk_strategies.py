"""Tests for chunk-building strategies (full / summary / metadata_only)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------- shared fixtures ---------- #


@pytest.fixture
def fake_doc(sample_attachment_document):
    """Attachment-shaped Document with a stable doc_id for chunk_id asserts."""
    sample_attachment_document.doc_id = "deadbeefdeadbeef"
    sample_attachment_document.source_id = "test-source"
    sample_attachment_document.source_title = "test.pdf"
    return sample_attachment_document


@pytest.fixture
def fake_chunker():
    """Real DocumentChunker — we want the actual splitter behavior."""
    from mtss.parsers.chunker import DocumentChunker

    return DocumentChunker()


@pytest.fixture
def fake_context_generator():
    """ContextGenerator mock returning a fixed summary."""
    cg = MagicMock()
    cg.generate_context = AsyncMock(return_value="This is a test context summary.")
    cg.build_embedding_text = MagicMock(
        side_effect=lambda ctx, content: f"{ctx}\n\n{content}" if ctx else content
    )
    return cg


# ---------- TestBuildChunksFull ---------- #


class TestBuildChunksFull:
    """Full mode: whole markdown chunked normally; embedding_mode stamped."""

    @pytest.mark.asyncio
    async def test_produces_chunks_from_markdown(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_full

        md = "# Title\n\n" + ("This is a prose paragraph. " * 80 + "\n\n") * 5
        chunks = await build_chunks_full(
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stamps_embedding_mode_full_on_every_chunk(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_full

        md = "# Title\n\n" + ("Prose. " * 200 + "\n\n") * 3
        chunks = await build_chunks_full(
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        for c in chunks:
            assert c.embedding_mode == "full"

    @pytest.mark.asyncio
    async def test_applies_context_summary_and_embedding_text(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_full

        md = "# T\n\n" + ("Prose. " * 200 + "\n\n") * 3
        chunks = await build_chunks_full(
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        assert all(c.context_summary == "This is a test context summary." for c in chunks)
        assert all(c.embedding_text and c.embedding_text.startswith(
            "This is a test context summary."
        ) for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_id_determinism_across_runs(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_full

        md = "# T\n\n" + ("Prose. " * 200 + "\n\n") * 3
        run1 = await build_chunks_full(
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        run2 = await build_chunks_full(
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        ids1 = [c.chunk_id for c in run1]
        ids2 = [c.chunk_id for c in run2]
        assert ids1 == ids2
        assert all(cid is not None for cid in ids1)

    @pytest.mark.asyncio
    async def test_empty_markdown_returns_empty_list(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_full

        chunks = await build_chunks_full(
            document=fake_doc,
            markdown="",
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="test.md",
        )
        assert chunks == []


# ---------- TestBuildChunksSummary ---------- #


class TestBuildChunksSummary:
    """Summary mode: single LLM-generated summary chunk; synthesized chunk_id."""

    @pytest.mark.asyncio
    async def test_produces_single_summary_chunk(
        self, fake_doc, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_summary

        chunks = await build_chunks_summary(
            document=fake_doc,
            markdown="sensor data...\n" * 1000,
            context_generator=fake_context_generator,
            source_file="sensor.pdf",
        )
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_summary_chunk_content_is_llm_output(
        self, fake_doc, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_summary

        fake_context_generator.generate_context.return_value = (
            "Sensor log covering week of 2025-06-21. 818 pages of pH/flow/temp readings."
        )
        chunks = await build_chunks_summary(
            document=fake_doc,
            markdown="sensor data...\n" * 1000,
            context_generator=fake_context_generator,
            source_file="sensor.pdf",
        )
        assert chunks[0].content == (
            "Sensor log covering week of 2025-06-21. 818 pages of pH/flow/temp readings."
        )

    @pytest.mark.asyncio
    async def test_stamps_embedding_mode_summary(
        self, fake_doc, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_summary

        chunks = await build_chunks_summary(
            document=fake_doc,
            markdown="sensor data\n" * 500,
            context_generator=fake_context_generator,
            source_file="s.pdf",
        )
        assert chunks[0].embedding_mode == "summary"

    @pytest.mark.asyncio
    async def test_chunk_id_uses_sentinel_positions(
        self, fake_doc, fake_context_generator
    ):
        from mtss.parsers.chunker import build_chunks_summary
        from mtss.utils import compute_chunk_id

        chunks = await build_chunks_summary(
            document=fake_doc,
            markdown="sensor\n" * 100,
            context_generator=fake_context_generator,
            source_file="s.pdf",
        )
        # Deterministic id based on (-1, 0) sentinel.
        expected = compute_chunk_id(fake_doc.doc_id, -1, 0)
        assert chunks[0].chunk_id == expected


# ---------- TestBuildChunksMetadataOnly ---------- #


class TestBuildChunksMetadataOnly:
    """Metadata-only mode: one stub chunk built from filename + metadata."""

    def test_produces_single_chunk(self, fake_doc):
        from mtss.parsers.chunker import build_chunks_metadata_only

        chunks = build_chunks_metadata_only(
            document=fake_doc, source_file="stub.pdf"
        )
        assert len(chunks) == 1

    def test_content_includes_filename_and_type(self, fake_doc):
        from mtss.parsers.chunker import build_chunks_metadata_only

        fake_doc.file_name = "bigscan.pdf"
        chunks = build_chunks_metadata_only(
            document=fake_doc, source_file="bigscan.pdf"
        )
        content = chunks[0].content
        assert "bigscan.pdf" in content

    def test_stamps_embedding_mode_metadata_only(self, fake_doc):
        from mtss.parsers.chunker import build_chunks_metadata_only

        chunks = build_chunks_metadata_only(
            document=fake_doc, source_file="s.pdf"
        )
        assert chunks[0].embedding_mode == "metadata_only"

    def test_chunk_id_uses_sentinel_minus_two(self, fake_doc):
        from mtss.parsers.chunker import build_chunks_metadata_only
        from mtss.utils import compute_chunk_id

        chunks = build_chunks_metadata_only(
            document=fake_doc, source_file="s.pdf"
        )
        expected = compute_chunk_id(fake_doc.doc_id, -2, 0)
        assert chunks[0].chunk_id == expected

    def test_embedding_text_equals_content(self, fake_doc):
        from mtss.parsers.chunker import build_chunks_metadata_only

        chunks = build_chunks_metadata_only(
            document=fake_doc, source_file="s.pdf"
        )
        assert chunks[0].embedding_text == chunks[0].content


# ---------- TestDispatcher ---------- #


class TestBuildChunksForMode:
    """Dispatcher selects the right strategy from EmbeddingMode."""

    @pytest.mark.asyncio
    async def test_full_mode_dispatches_to_full(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.models.document import EmbeddingMode
        from mtss.parsers.chunker import build_chunks_for_mode

        md = "# T\n\n" + ("Prose. " * 200 + "\n\n") * 2
        chunks = await build_chunks_for_mode(
            mode=EmbeddingMode.FULL,
            document=fake_doc,
            markdown=md,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="t.md",
        )
        assert all(c.embedding_mode == "full" for c in chunks)

    @pytest.mark.asyncio
    async def test_summary_mode_dispatches_to_summary(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.models.document import EmbeddingMode
        from mtss.parsers.chunker import build_chunks_for_mode

        chunks = await build_chunks_for_mode(
            mode=EmbeddingMode.SUMMARY,
            document=fake_doc,
            markdown="log " * 100,
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="s.pdf",
        )
        assert len(chunks) == 1
        assert chunks[0].embedding_mode == "summary"

    @pytest.mark.asyncio
    async def test_metadata_only_mode_dispatches_to_metadata_only(
        self, fake_doc, fake_chunker, fake_context_generator
    ):
        from mtss.models.document import EmbeddingMode
        from mtss.parsers.chunker import build_chunks_for_mode

        chunks = await build_chunks_for_mode(
            mode=EmbeddingMode.METADATA_ONLY,
            document=fake_doc,
            markdown="irrelevant",
            chunker=fake_chunker,
            context_generator=fake_context_generator,
            source_file="s.pdf",
        )
        assert len(chunks) == 1
        assert chunks[0].embedding_mode == "metadata_only"
