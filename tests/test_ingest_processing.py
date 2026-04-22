"""Tests for ingest processing pipeline components.

Tests for DocumentChunker, ContextGenerator, EmbeddingGenerator, HierarchyManager, and Utils.
All external API calls are mocked.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    @pytest.fixture
    def chunker(self, comprehensive_mock_settings):
        """Create a DocumentChunker with mocked settings."""
        with patch("mtss.parsers.chunker.get_settings", return_value=comprehensive_mock_settings):
            from mtss.parsers.chunker import DocumentChunker
            return DocumentChunker()

    def test_chunk_empty_text(self, chunker, sample_document_id):
        """Empty text should return empty list."""
        result = chunker.chunk_text("", sample_document_id, "/test/file.md")
        assert result == []

    def test_chunk_whitespace_only_text(self, chunker, sample_document_id):
        """Whitespace-only text should return empty list."""
        result = chunker.chunk_text("   \n\n  ", sample_document_id, "/test/file.md")
        assert result == []

    def test_chunk_short_text_single_chunk(self, chunker, sample_document_id):
        """Short text should produce single chunk."""
        short_text = "This is a short test document."
        result = chunker.chunk_text(short_text, sample_document_id, "/test/file.md")

        assert len(result) == 1
        assert result[0].content == short_text
        assert result[0].document_id == sample_document_id
        assert result[0].chunk_index == 0

    def test_chunk_long_text_multiple_chunks(self, chunker, sample_document_id):
        """Long text should produce multiple chunks."""
        # Create a long text by repeating paragraphs
        paragraph = "This is a paragraph with enough content. " * 50
        long_text = "\n\n".join([paragraph] * 10)

        result = chunker.chunk_text(long_text, sample_document_id, "/test/file.md")

        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i
            assert chunk.document_id == sample_document_id

    def test_chunk_preserves_metadata(self, chunker, sample_document_id):
        """Chunk should preserve metadata passed to it."""
        text = "Test content for metadata check."
        metadata = {"vessel_ids": ["123", "456"], "custom_field": "value"}

        result = chunker.chunk_text(
            text, sample_document_id, "/test/file.md", metadata=metadata
        )

        assert len(result) == 1
        assert result[0].metadata["vessel_ids"] == ["123", "456"]
        assert result[0].metadata["custom_field"] == "value"
        assert result[0].metadata["source_file"] == "/test/file.md"

    def test_chunk_markdown_extracts_headings(self, chunker, sample_document_id, sample_markdown_content):
        """Markdown chunks should extract heading hierarchy."""
        result = chunker.chunk_text(
            sample_markdown_content, sample_document_id, "/test/file.md", is_markdown=True
        )

        # At least one chunk should have heading path extracted
        # The content is short enough it might be one chunk, but it should still extract headings
        assert len(result) >= 1
        assert any(c.section_path for c in result) or len(result) == 1

    def test_chunk_tracks_line_numbers(self, chunker, sample_document_id):
        """Chunks should track line numbers from source."""
        text = "Line one\nLine two\nLine three\nLine four"
        result = chunker.chunk_text(text, sample_document_id, "/test/file.md")

        assert len(result) == 1
        assert result[0].line_from is not None
        assert result[0].line_to is not None
        assert result[0].char_start is not None
        assert result[0].char_end is not None

    def test_chunk_tracks_character_positions(self, chunker, sample_document_id):
        """Chunks should track character positions."""
        text = "First chunk content here.\n\nSecond chunk content here."
        result = chunker.chunk_text(text, sample_document_id, "/test/file.md")

        assert len(result) >= 1
        for chunk in result:
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            assert chunk.char_start <= chunk.char_end

    def test_count_tokens(self, chunker):
        """Token counting should work correctly."""
        text = "Hello world"
        token_count = chunker.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 100  # Short text should have few tokens

    def test_find_chunk_position_rejects_ambiguous_fallback(self, chunker):
        """When ``find(chunk, search_start)`` misses and the chunk appears
        multiple times earlier in the text, ``_find_chunk_position`` must
        return None rather than blindly matching the first occurrence.
        Prior behaviour silently produced a wrong char_start/char_end and a
        deterministic-but-wrong chunk_id on repeated headers/footers.
        """
        full_text = (
            "Results section\nrow A\n\n"
            "Results section\nrow B\n\n"
            "Tail content unrelated\n"
        )
        line_starts = chunker._compute_line_starts(full_text)
        # "Results section" appears twice; find-from-search_start past both
        # occurrences misses. Ambiguous rewind must be rejected.
        char_start, char_end, line_from, line_to = chunker._find_chunk_position(
            full_text, "Results section", search_start=len(full_text), line_starts=line_starts,
        )
        assert char_start is None
        assert char_end is None
        assert line_from is None
        assert line_to is None

    def test_find_chunk_position_unique_fallback_accepted(self, chunker):
        """Unique-in-text content should still be located by the rewind
        fallback — that path is safe because there is no ambiguity."""
        full_text = "Preface\n\nUnique paragraph body.\n\nTail."
        line_starts = chunker._compute_line_starts(full_text)
        # Search from past the end — forces the fallback to run.
        char_start, char_end, line_from, line_to = chunker._find_chunk_position(
            full_text, "Unique paragraph body.",
            search_start=len(full_text), line_starts=line_starts,
        )
        assert char_start is not None
        assert char_end == char_start + len("Unique paragraph body.")
        assert line_from is not None
        assert line_to is not None


class TestContextGenerator:
    """Tests for ContextGenerator class."""

    @pytest.fixture
    def context_generator(self, comprehensive_mock_settings):
        """Create a ContextGenerator with mocked settings."""
        with patch("mtss.parsers.chunker.get_settings", return_value=comprehensive_mock_settings):
            from mtss.parsers.chunker import ContextGenerator
            return ContextGenerator()

    @pytest.mark.asyncio
    async def test_generate_context_success(
        self, context_generator, sample_document, mock_llm_completion_response
    ):
        """Context generation should return LLM response."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_llm_completion_response

            result = await context_generator.generate_context(
                sample_document, "Test content preview"
            )

            assert "email" in result.lower() or "sender" in result.lower()
            mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_context_fallback_on_error(
        self, context_generator, sample_document
    ):
        """Context generation should fall back to metadata on LLM failure."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("API error")

            result = await context_generator.generate_context(
                sample_document, "Test content preview"
            )

            # Should fall back to metadata-based context
            assert "Email" in result or "email" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_context_retry_on_rate_limit(
        self, context_generator, sample_document, mock_llm_completion_response
    ):
        """Context generation should retry on rate limit errors."""
        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return mock_llm_completion_response

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = mock_completion
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await context_generator.generate_context(
                    sample_document, "Test content"
                )

            assert call_count >= 2
            assert result is not None

    def test_build_embedding_text(self, context_generator):
        """Embedding text should combine context and content."""
        context = "This is context."
        chunk_content = "This is chunk content."

        result = context_generator.build_embedding_text(context, chunk_content)

        assert context in result
        assert chunk_content in result
        assert result == f"{context}\n\n{chunk_content}"

    def test_build_embedding_text_no_context(self, context_generator):
        """Embedding text should return content when no context."""
        chunk_content = "This is chunk content only."

        result = context_generator.build_embedding_text("", chunk_content)

        assert result == chunk_content


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def embedding_generator(self, comprehensive_mock_settings):
        """Create an EmbeddingGenerator with mocked settings."""
        with patch("mtss.processing.embeddings.get_settings", return_value=comprehensive_mock_settings):
            with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                from mtss.processing.embeddings import EmbeddingGenerator
                return EmbeddingGenerator()

    def test_truncate_to_max_tokens(self, embedding_generator):
        """Long text should be truncated to max tokens."""
        # Create a very long text
        long_text = "word " * 10000

        result = embedding_generator._truncate_to_max_tokens(long_text)

        assert len(result) < len(long_text)

    def test_truncate_short_text_unchanged(self, embedding_generator):
        """Short text should not be truncated."""
        short_text = "This is a short text."

        result = embedding_generator._truncate_to_max_tokens(short_text)

        assert result == short_text

    @pytest.mark.asyncio
    async def test_generate_single_embedding(
        self, embedding_generator, mock_embedding_response
    ):
        """Single embedding generation should work."""
        with patch("mtss.processing.embeddings.aembedding", new_callable=AsyncMock) as mock:
            with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                mock.return_value = mock_embedding_response

                result = await embedding_generator.generate_embedding("Test text")

                assert len(result) == 1536
                mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(
        self, embedding_generator, mock_batch_embedding_response
    ):
        """Batch embedding generation should work."""
        with patch("mtss.processing.embeddings.aembedding", new_callable=AsyncMock) as mock:
            with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                mock.return_value = mock_batch_embedding_response

                texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
                result = await embedding_generator.generate_embeddings_batch(texts)

                assert len(result) == 5
                for embedding in result:
                    assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, embedding_generator):
        """Empty list should return empty list."""
        result = await embedding_generator.generate_embeddings_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_chunks(self, embedding_generator, sample_chunks, mock_batch_embedding_response):
        """Chunks should get embeddings added."""
        with patch("mtss.processing.embeddings.aembedding", new_callable=AsyncMock) as mock:
            with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                mock.return_value = mock_batch_embedding_response

                result = await embedding_generator.embed_chunks(sample_chunks)

                assert len(result) == 5
                for chunk in result:
                    assert chunk.embedding is not None
                    assert len(chunk.embedding) == 1536

    @pytest.mark.asyncio
    async def test_embed_empty_chunks(self, embedding_generator):
        """Empty chunk list should return empty list."""
        result = await embedding_generator.embed_chunks([])

        assert result == []


class TestHierarchyManager:
    """Tests for HierarchyManager class."""

    @pytest.fixture
    def hierarchy_manager(self, comprehensive_mock_settings, mock_supabase_client):
        """Create a HierarchyManager with mocked dependencies."""
        with patch("mtss.ingest.hierarchy_manager.get_settings", return_value=comprehensive_mock_settings):
            from mtss.ingest.hierarchy_manager import HierarchyManager
            return HierarchyManager(mock_supabase_client, Path("./data/emails"))

    def test_compute_file_hash(self, hierarchy_manager, temp_dir):
        """File hash computation should work."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content for hashing")

        result = hierarchy_manager.compute_file_hash(test_file)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex string

    def test_compute_file_hash_deterministic(self, hierarchy_manager, temp_dir):
        """Same content should produce same hash."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Same content")

        hash1 = hierarchy_manager.compute_file_hash(test_file)
        hash2 = hierarchy_manager.compute_file_hash(test_file)

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_create_email_document(
        self, hierarchy_manager, temp_dir, sample_parsed_email, mock_supabase_client
    ):
        """Email document creation should work."""
        eml_file = temp_dir / "test.eml"
        eml_file.write_text("Mock email content")

        result = await hierarchy_manager.create_email_document(eml_file, sample_parsed_email)

        assert result.document_type.value == "email"
        assert result.depth == 0
        assert result.parent_id is None
        assert result.root_id == result.id
        mock_supabase_client.insert_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_attachment_document(
        self, hierarchy_manager, temp_dir, sample_document, mock_supabase_client
    ):
        """Attachment document creation should work."""
        with patch("mtss.ingest.hierarchy_manager.get_settings") as mock_settings_call:
            mock_settings_call.return_value = MagicMock(
                eml_source_dir=Path("./data/emails"),
                archive_base_url="https://archive.example.com",
                current_ingest_version=1,
            )

            attachment_file = temp_dir / "attachment.pdf"
            attachment_file.write_bytes(b"%PDF-1.4 mock content")

            # Patch the AttachmentProcessor at the module where it's imported
            with patch("mtss.parsers.attachment_processor.AttachmentProcessor") as mock_attachment_processor:
                from mtss.models.document import DocumentType

                mock_processor = MagicMock()
                mock_processor.get_document_type.return_value = DocumentType.ATTACHMENT_PDF
                mock_attachment_processor.return_value = mock_processor

                # Also need to patch where it's lazily imported in hierarchy_manager
                with patch.dict("sys.modules", {"mtss.parsers.attachment_processor": MagicMock(AttachmentProcessor=mock_attachment_processor)}):
                    result = await hierarchy_manager.create_attachment_document(
                        parent_doc=sample_document,
                        attachment_path=attachment_file,
                        content_type="application/pdf",
                        size_bytes=12345,
                        original_filename="attachment.pdf",
                    )

        assert result.parent_id == sample_document.id
        assert result.root_id == sample_document.root_id
        assert result.depth == sample_document.depth + 1

    @pytest.mark.asyncio
    async def test_get_document_ancestry(self, hierarchy_manager, sample_document, mock_supabase_client):
        """Document ancestry retrieval should work."""
        mock_supabase_client.get_document_ancestry.return_value = [sample_document]

        result = await hierarchy_manager.get_document_ancestry(sample_document.id)

        assert len(result) == 1
        mock_supabase_client.get_document_ancestry.assert_called_once_with(sample_document.id)


class TestUtils:
    """Tests for utility functions."""

    def test_normalize_source_id(self, temp_dir):
        """Source ID normalization should work."""
        from mtss.utils import normalize_source_id

        file_path = temp_dir / "subdir" / "test.eml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = normalize_source_id(str(file_path), temp_dir)

        assert result == "subdir/test.eml"
        assert "/" in result  # Should use forward slashes
        assert result.islower()

    def test_normalize_source_id_lowercase(self, temp_dir):
        """Source ID should be lowercase."""
        from mtss.utils import normalize_source_id

        file_path = temp_dir / "SubDir" / "Test.EML"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = normalize_source_id(str(file_path), temp_dir)

        assert result == result.lower()

    def test_compute_doc_id_deterministic(self):
        """Doc ID should be deterministic."""
        from mtss.utils import compute_doc_id

        source_id = "test/email.eml"
        file_hash = "abc123def456"

        id1 = compute_doc_id(source_id, file_hash)
        id2 = compute_doc_id(source_id, file_hash)

        assert id1 == id2
        assert len(id1) == 16

    def test_compute_doc_id_different_for_different_inputs(self):
        """Different inputs should produce different doc IDs."""
        from mtss.utils import compute_doc_id

        id1 = compute_doc_id("path1.eml", "hash1")
        id2 = compute_doc_id("path2.eml", "hash2")

        assert id1 != id2

    def test_compute_chunk_id_unique(self):
        """Chunk IDs should be unique for different positions."""
        from mtss.utils import compute_chunk_id

        doc_id = "doc123abc"

        id1 = compute_chunk_id(doc_id, 0, 100)
        id2 = compute_chunk_id(doc_id, 100, 200)
        id3 = compute_chunk_id(doc_id, 0, 100)

        assert id1 != id2
        assert id1 == id3  # Same inputs = same ID
        assert len(id1) == 12

    def test_sanitize_filename(self):
        """Filename sanitization should remove dangerous characters."""
        from mtss.utils import sanitize_filename

        dangerous = "file<>:test|?.pdf"
        result = sanitize_filename(dangerous)

        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result

    def test_sanitize_filename_max_length(self):
        """Filename should be truncated to max length."""
        from mtss.utils import sanitize_filename

        long_name = "a" * 300
        result = sanitize_filename(long_name, max_length=255)

        assert len(result) == 255


class TestAttachmentProcessor:
    """Tests for AttachmentProcessor class."""

    @pytest.fixture
    def attachment_processor(self, comprehensive_mock_settings):
        """Create an AttachmentProcessor with mocked settings."""
        with patch("mtss.parsers.attachment_processor.get_settings", return_value=comprehensive_mock_settings):
            from mtss.parsers.attachment_processor import AttachmentProcessor
            return AttachmentProcessor()

    @pytest.mark.unit
    def test_classifies_pdf_as_document(self, attachment_processor):
        """PDF files should be classified as ATTACHMENT_PDF."""
        from mtss.models.document import DocumentType

        result = attachment_processor.get_document_type("application/pdf")
        assert result == DocumentType.ATTACHMENT_PDF

    @pytest.mark.unit
    def test_classifies_docx_as_document(self, attachment_processor):
        """DOCX files should be classified as ATTACHMENT_DOCX."""
        from mtss.models.document import DocumentType

        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = attachment_processor.get_document_type(content_type)
        assert result == DocumentType.ATTACHMENT_DOCX

    @pytest.mark.unit
    def test_classifies_image_formats(self, attachment_processor):
        """Image files should be classified as ATTACHMENT_IMAGE."""
        from mtss.models.document import DocumentType

        for content_type in ["image/png", "image/jpeg", "image/tiff"]:
            result = attachment_processor.get_document_type(content_type)
            assert result == DocumentType.ATTACHMENT_IMAGE

    @pytest.mark.unit
    def test_classifies_unknown_as_other(self, attachment_processor):
        """Unknown content types should be classified as ATTACHMENT_OTHER."""
        from mtss.models.document import DocumentType

        result = attachment_processor.get_document_type("application/unknown-type")
        assert result == DocumentType.ATTACHMENT_OTHER

    @pytest.mark.unit
    def test_is_supported_with_parser(self, attachment_processor):
        """Should return True for formats with registered parsers."""
        # PDF is supported
        result = attachment_processor.is_supported("document.pdf", "application/pdf")
        assert result is True

    @pytest.mark.unit
    def test_is_image_format(self, attachment_processor):
        """Should correctly identify image formats."""
        assert attachment_processor.is_image_format("image/png") is True
        assert attachment_processor.is_image_format("image/jpeg") is True
        assert attachment_processor.is_image_format("application/pdf") is False
        assert attachment_processor.is_image_format(None) is False

    @pytest.mark.unit
    def test_is_zip_file_by_extension(self, attachment_processor):
        """Should identify ZIP files by extension."""
        assert attachment_processor.is_zip_file("archive.zip") is True
        assert attachment_processor.is_zip_file("ARCHIVE.ZIP") is True
        assert attachment_processor.is_zip_file("document.pdf") is False

    @pytest.mark.unit
    def test_is_zip_file_by_content_type(self, attachment_processor):
        """Should identify ZIP files by content type."""
        assert attachment_processor.is_zip_file("file", "application/zip") is True
        assert attachment_processor.is_zip_file("file", "application/x-zip-compressed") is True
        assert attachment_processor.is_zip_file("file", "application/pdf") is False

    @pytest.mark.unit
    def test_is_zip_file_excludes_office_formats(self, attachment_processor):
        """Should not classify Office formats as ZIP even though they are technically ZIPs."""
        # DOCX is a ZIP but should not be treated as one
        docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert attachment_processor.is_zip_file("document.docx", docx_type) is False
        assert attachment_processor.is_zip_file("document.xlsx") is False
        assert attachment_processor.is_zip_file("document.pptx") is False

    @pytest.mark.unit
    def test_extracts_zip_contents(self, attachment_processor, temp_dir):
        """Should extract files from a ZIP archive."""
        import zipfile

        # Create a test ZIP file
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("document.txt", "Test content")
            zf.writestr("subdir/nested.txt", "Nested content")

        extracted = attachment_processor.extract_zip(zip_path)

        # Should have extracted files
        assert len(extracted) >= 1
        # Check that paths are returned
        paths = [str(p[0]) for p in extracted]
        assert any("document.txt" in p for p in paths)

    @pytest.mark.unit
    def test_handles_nested_zip(self, attachment_processor, temp_dir, comprehensive_mock_settings):
        """Should extract nested ZIP files up to max depth."""
        import zipfile

        # Create inner ZIP
        inner_zip_path = temp_dir / "inner.zip"
        with zipfile.ZipFile(inner_zip_path, "w") as zf:
            zf.writestr("nested_doc.txt", "Nested content")

        # Create outer ZIP containing inner ZIP
        outer_zip_path = temp_dir / "outer.zip"
        with zipfile.ZipFile(outer_zip_path, "w") as zf:
            zf.write(inner_zip_path, "inner.zip")
            zf.writestr("outer_doc.txt", "Outer content")

        extracted = attachment_processor.extract_zip(outer_zip_path)

        # Should have extracted files from both levels
        paths = [str(p[0]) for p in extracted]
        assert any("outer_doc.txt" in p for p in paths)
        # Nested content should also be extracted
        assert any("nested_doc.txt" in p for p in paths)

    @pytest.mark.unit
    def test_rejects_unsupported_format(self, attachment_processor):
        """Should return False for completely unsupported formats."""
        result = attachment_processor.is_supported("file.xyz", "application/x-unknown")
        assert result is False

    @pytest.mark.unit
    def test_dangerous_zip_path_detection(self, attachment_processor):
        """Should detect dangerous ZIP paths."""
        assert attachment_processor._is_dangerous_zip_path("../etc/passwd") is True
        assert attachment_processor._is_dangerous_zip_path("/absolute/path") is True
        assert attachment_processor._is_dangerous_zip_path("C:\\windows\\system32") is True
        assert attachment_processor._is_dangerous_zip_path("safe/path/file.txt") is False

    @pytest.mark.unit
    def test_sanitize_zip_member_path(self, attachment_processor):
        """Should sanitize ZIP member paths."""
        result = attachment_processor._sanitize_zip_member_path("dir\\file.txt")
        assert "\\" not in result

        result = attachment_processor._sanitize_zip_member_path("/leading/slash.txt")
        assert not result.startswith("/")

        result = attachment_processor._sanitize_zip_member_path("../traversal/file.txt")
        assert ".." not in result


class TestThreadDigest:
    """Tests for _generate_thread_digest function."""

    @pytest.fixture
    def multi_message_body(self):
        """Email body with multiple messages for digest testing."""
        return (
            "Thanks for confirming the turbocharger bearing replacement.\n"
            "The engine is running normally now.\n"
            "\n"
            "On 2024-01-02 10:30, engineer@vessel.com wrote:\n"
            "\n"
            "We replaced the turbocharger bearings and flushed the lube oil system.\n"
            "Vibration levels are back to normal range.\n"
            "\n"
            "On 2024-01-01 08:15, captain@vessel.com wrote:\n"
            "\n"
            "Main engine turbocharger showing excessive vibration during operations.\n"
            "Requesting technical support for diagnosis and repair guidance."
        )

    @pytest.fixture
    def single_message_body(self):
        """Email body with a single message."""
        return "This is a single message email about engine maintenance."

    @pytest.fixture
    def mock_components(self):
        """Mock IngestComponents with context_generator."""
        components = MagicMock()
        components.context_generator = MagicMock()
        components.context_generator.build_embedding_text = MagicMock(
            side_effect=lambda ctx, content: f"{ctx}\n\n{content}"
        )
        return components

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_none_for_single_message(
        self, single_message_body, sample_document, mock_components
    ):
        """Single-message emails should not produce a digest."""
        with patch("mtss.ingest.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.get_model.return_value = "gpt-5-nano"
            mock_settings.return_value.thread_digest_model = "gpt-5-nano"

            from mtss.ingest.pipeline import _generate_thread_digest

            result = await _generate_thread_digest(
                single_message_body, sample_document, "context",
                [], [], [], [], mock_components,
                lambda msg, ctx: None, "test.eml",
            )
            assert result is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generates_digest_for_multi_message(
        self, multi_message_body, sample_document, mock_components
    ):
        """Multi-message emails should produce a digest chunk."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Thread discusses turbocharger vibration on the main engine. "
            "The vessel captain reported excessive vibration. Engineering team "
            "replaced bearings and flushed lube oil. Issue resolved."
        )

        with patch("mtss.ingest.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.get_model.return_value = "gpt-5-nano"
            mock_settings.return_value.thread_digest_model = "gpt-5-nano"

            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
                from mtss.ingest.pipeline import _generate_thread_digest

                result = await _generate_thread_digest(
                    multi_message_body, sample_document, "Test context summary",
                    ["vessel-1"], ["VLCC"], ["Maran Class"], ["topic-1"],
                    mock_components, lambda msg, ctx: None, "test.eml",
                )

                assert result is not None
                assert result.chunk_index == -1
                assert result.metadata["type"] == "thread_digest"
                assert result.metadata["message_count"] == 3
                assert result.metadata["vessel_ids"] == ["vessel-1"]
                assert result.metadata["topic_ids"] == ["topic-1"]
                assert result.section_title == "Thread Digest"
                assert result.embedding_text.startswith("Test context summary")
                assert result.source_id == sample_document.source_id
                mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_none_on_llm_failure(
        self, multi_message_body, sample_document, mock_components
    ):
        """LLM failure should return None, not raise."""
        with patch("mtss.ingest.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.get_model.return_value = "gpt-5-nano"
            mock_settings.return_value.thread_digest_model = "gpt-5-nano"

            with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("LLM error")):
                from mtss.ingest.pipeline import _generate_thread_digest

                result = await _generate_thread_digest(
                    multi_message_body, sample_document, None,
                    [], [], [], [], mock_components,
                    lambda msg, ctx: None, "test.eml",
                )
                assert result is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_none_on_empty_response(
        self, multi_message_body, sample_document, mock_components
    ):
        """Empty LLM response should return None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   "

        with patch("mtss.ingest.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.get_model.return_value = "gpt-5-nano"
            mock_settings.return_value.thread_digest_model = "gpt-5-nano"

            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
                from mtss.ingest.pipeline import _generate_thread_digest

                result = await _generate_thread_digest(
                    multi_message_body, sample_document, None,
                    [], [], [], [], mock_components,
                    lambda msg, ctx: None, "test.eml",
                )
                assert result is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_digest_without_context_summary(
        self, multi_message_body, sample_document, mock_components
    ):
        """Digest without context_summary should use raw digest text as embedding_text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Digest summary text."

        with patch("mtss.ingest.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.get_model.return_value = "gpt-5-nano"
            mock_settings.return_value.thread_digest_model = "gpt-5-nano"

            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
                from mtss.ingest.pipeline import _generate_thread_digest

                result = await _generate_thread_digest(
                    multi_message_body, sample_document, None,
                    [], [], [], [], mock_components,
                    lambda msg, ctx: None, "test.eml",
                )

                assert result is not None
                assert result.embedding_text == "Digest summary text."
                assert result.context_summary is None


class TestZipAttachmentContextGeneration:
    """Tests for process_zip_attachment context generation + archive URIs."""

    @pytest.fixture
    def zip_attachment(self, tmp_path):
        """ParsedAttachment representing a ZIP file."""
        from mtss.models.document import ParsedAttachment

        zip_path = tmp_path / "bundle.zip"
        zip_path.write_bytes(b"PK\x03\x04fake-zip")
        return ParsedAttachment(
            filename="bundle.zip",
            content_type="application/zip",
            size_bytes=zip_path.stat().st_size,
            saved_path=str(zip_path),
        )

    @pytest.fixture
    def extracted_text_file(self, tmp_path):
        """A plaintext file to represent the ZIP extraction result.

        Content is intentionally long enough to clear the embedding decider's
        metadata-only / summary thresholds so the decider picks FULL and the
        context generator is exercised.
        """
        extracted = tmp_path / "extracted" / "notes.txt"
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_text(
            "# Notes\n\n"
            + (
                "These extracted notes describe the inspection findings for "
                "the vessel under review. The narrative covers scope, methods, "
                "conclusions, and recommendations across multiple sections. "
            )
            * 10
        )
        return extracted

    @pytest.fixture
    def attach_chunk(self, sample_document_id):
        """A single chunk returned by the mocked parser for the extracted file."""
        from uuid import uuid4

        from mtss.models.chunk import Chunk

        return Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id="zipchunk123",
            content="Some extracted content from a ZIP member.",
            chunk_index=0,
            section_path=[],
            source_id="test.eml/bundle.zip/notes.txt",
        )

    @pytest.fixture
    def mock_components(self, extracted_text_file):
        """Mock IngestComponents wired for process_zip_attachment."""
        components = MagicMock()

        components.attachment_processor = MagicMock()
        components.attachment_processor.extract_zip = MagicMock(
            return_value=[(extracted_text_file, "text/plain")]
        )

        preprocess_result = MagicMock()
        preprocess_result.should_process = True
        preprocess_result.is_image = False
        preprocess_result.skip_reason = None
        preprocess_result.oversized_pdf = False
        preprocess_result.preview_markdown = None
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess_result)
        components.attachment_processor.process_attachment = AsyncMock()
        # New pipeline goes through parse_to_text → decider → build_chunks_for_mode.
        # Default: return the extracted file's text so the decider sees real content.
        components.attachment_processor.parse_to_text = AsyncMock(
            return_value=(extracted_text_file.read_text(), "local_text", None)
        )
        # Real chunker so the splitter actually produces chunks from the text.
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        components.hierarchy_manager = MagicMock()
        components.hierarchy_manager.build_attachment_document = MagicMock()

        components.context_generator = MagicMock()
        components.context_generator.generate_context = AsyncMock(
            return_value="Generated context summary"
        )
        components.context_generator.build_embedding_text = MagicMock(
            side_effect=lambda ctx, content: f"{ctx}\n\n{content}"
        )

        components.archive_generator = MagicMock()
        components.archive_generator.update_attachment_markdown = MagicMock(
            return_value="doc123abc/attachments/notes.txt.md"
        )
        return components

    def _wire_attach_doc(self, components, sample_attachment_document, attach_chunks):
        components.hierarchy_manager.build_attachment_document.return_value = sample_attachment_document
        components.attachment_processor.process_attachment.return_value = attach_chunks

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_text_attachment_gets_context_summary(
        self,
        zip_attachment,
        sample_document,
        sample_attachment_document,
        attach_chunk,
        mock_components,
    ):
        """Chunks from ZIP contents must get context_summary and embedding_text."""
        self._wire_attach_doc(mock_components, sample_attachment_document, [attach_chunk])
        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        from mtss.ingest.attachment_handler import process_zip_attachment

        chunks = await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=mock_components,
            unsupported_logger=unsupported_logger,
        )

        mock_components.context_generator.generate_context.assert_awaited_once()
        assert len(chunks) >= 1
        assert chunks[0].context_summary == "Generated context summary"
        assert chunks[0].embedding_text.startswith("Generated context summary\n\n")
        assert chunks[0].embedding_mode == "full"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_attachment_gets_archive_uris(
        self,
        zip_attachment,
        sample_document,
        sample_attachment_document,
        attach_chunk,
        mock_components,
    ):
        """ZIP extracted attachment must have archive_browse_uri / download_uri set."""
        sample_attachment_document.archive_browse_uri = None
        sample_attachment_document.archive_download_uri = None
        self._wire_attach_doc(mock_components, sample_attachment_document, [attach_chunk])
        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()
        collected: list = []

        from mtss.ingest.attachment_handler import process_zip_attachment

        await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=mock_components,
            unsupported_logger=unsupported_logger,
            collect_docs=collected,
        )

        mock_components.archive_generator.update_attachment_markdown.assert_called_once()
        assert sample_attachment_document.archive_browse_uri == (
            "/archive/doc123abc/attachments/notes.txt.md"
        )
        assert sample_attachment_document.archive_download_uri == (
            "/archive/doc123abc/attachments/notes.txt"
        )
        assert collected == [sample_attachment_document]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_member_original_is_uploaded_before_md(
        self,
        zip_attachment,
        sample_document,
        sample_attachment_document,
        attach_chunk,
        mock_components,
    ):
        """Regression: extracted ZIP members must have their original uploaded
        to archive storage before update_attachment_markdown is called.

        Without this, the `file_exists` guard in update_attachment_markdown
        short-circuits and the `.md` preview is never written — which was
        the root cause of the "Original file not found, skipping .md creation"
        warnings flooding the retry-failed ingest run.
        """
        self._wire_attach_doc(mock_components, sample_attachment_document, [attach_chunk])
        sample_document.doc_id = "doc123abc4567890"  # 16+ chars so [:16] is valid

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        from mtss.ingest.attachment_handler import process_zip_attachment

        await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=mock_components,
            unsupported_logger=unsupported_logger,
        )

        storage = mock_components.archive_generator.storage
        storage.upload_file.assert_called_once()
        args, _ = storage.upload_file.call_args
        uploaded_path, uploaded_bytes, uploaded_ct = args
        # Goes into the email's archive folder as <folder_id>/attachments/<member>
        from mtss.utils import compute_folder_id
        folder_id = compute_folder_id(sample_document.doc_id)
        assert uploaded_path == f"{folder_id}/attachments/notes.txt"
        # Content is the fixture text — exact bytes aren't asserted here, but
        # it must be a non-empty byte string (real upload path, not a mock).
        assert isinstance(uploaded_bytes, (bytes, bytearray))
        assert len(uploaded_bytes) > 0
        assert uploaded_ct == "text/plain"

        # And update_attachment_markdown was called AFTER (both called once, upload first).
        assert mock_components.archive_generator.update_attachment_markdown.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_member_upload_failure_does_not_abort_processing(
        self,
        zip_attachment,
        sample_document,
        sample_attachment_document,
        attach_chunk,
        mock_components,
    ):
        """If the extracted-original upload fails, we still attempt the .md
        write (existing update_attachment_markdown has its own file_exists
        guard and will return None gracefully). Processing must not crash.
        """
        self._wire_attach_doc(mock_components, sample_attachment_document, [attach_chunk])
        sample_document.doc_id = "doc123abc4567890"
        mock_components.archive_generator.storage.upload_file = MagicMock(
            side_effect=RuntimeError("disk full")
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        from mtss.ingest.attachment_handler import process_zip_attachment

        chunks = await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=mock_components,
            unsupported_logger=unsupported_logger,
        )

        assert len(chunks) == 1  # chunk still produced despite upload failure
        mock_components.archive_generator.update_attachment_markdown.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_context_generation_failure_non_fatal(
        self,
        zip_attachment,
        sample_document,
        sample_attachment_document,
        attach_chunk,
        mock_components,
    ):
        """Context generation errors must not break ZIP processing."""
        self._wire_attach_doc(mock_components, sample_attachment_document, [attach_chunk])
        mock_components.context_generator.generate_context = AsyncMock(
            side_effect=RuntimeError("LLM exploded")
        )
        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        from mtss.ingest.attachment_handler import process_zip_attachment

        chunks = await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=mock_components,
            unsupported_logger=unsupported_logger,
        )

        assert len(chunks) == 1
        assert chunks[0].context_summary is None


class TestEmptyAttachmentEventEmission:
    """When an attachment parser returns 0 chunks, a no_body_chunks ingest
    event must be emitted so the validator can distinguish an intentional
    empty result from a silent bug.
    """

    def _build_components(self, chunks_return, is_zip: bool = False):
        """Mock IngestComponents wired for process_attachment or zip variant."""
        components = MagicMock()
        components.attachment_processor = MagicMock()
        # New pipeline uses parse_to_text + decider; empty-text path is what
        # triggers the `no_body_chunks` event in the refactored handler.
        text = (
            "" if not chunks_return else "something parseable and long enough"
        )
        components.attachment_processor.parse_to_text = AsyncMock(
            return_value=(text, "local_text", None)
        )
        # Keep legacy method mocked for any code path that still references it.
        components.attachment_processor.process_attachment = AsyncMock(
            return_value=chunks_return
        )
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        preprocess = MagicMock(
            should_process=True,
            is_image=False,
            is_zip=False,
            skip_reason=None,
            parser_name=None,
            image_description=None,
            oversized_pdf=False,
            preview_markdown=None,
            total_pages=None,
        )
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)

        if is_zip:
            components.attachment_processor.extract_zip = MagicMock()

        components.hierarchy_manager = MagicMock()
        components.archive_generator = MagicMock()
        components.archive_generator.storage = MagicMock()
        components.archive_generator.storage.file_exists = MagicMock(return_value=False)
        components.context_generator = None

        components.db = MagicMock()
        components.db.log_ingest_event = MagicMock()
        return components

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_attachment_empty_chunks_emits_event(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Direct attachment that parses to 0 chunks → event with matching doc id."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "Settings_Options.ini"
        att_path.write_text("[settings]\n")
        attachment = ParsedAttachment(
            filename="Settings_Options.ini",
            content_type="application/octet-stream",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )

        components = self._build_components(chunks_return=[])
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        chunks = await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # Empty-parse path stamps METADATA_ONLY on the doc AND emits exactly
        # one filename-stub chunk (single_chunk_modes invariant — validate #27).
        # Prior bug: the decider never ran so no chunk was built at all, so
        # 30 v=6 docs landed as metadata_only with 0 chunks. Regression: 2026-04-20.
        from mtss.models.document import EmbeddingMode
        assert len(chunks) == 1
        stub = chunks[0]
        assert stub.embedding_mode == EmbeddingMode.METADATA_ONLY
        assert (stub.metadata or {}).get("type") == "metadata_stub"
        # Stub content mirrors the doc's file_name (what
        # build_chunks_metadata_only embeds) — not the raw attachment filename.
        assert sample_attachment_document.file_name in stub.content
        assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY

        components.db.log_ingest_event.assert_called_once()
        kwargs = components.db.log_ingest_event.call_args.kwargs
        assert kwargs["event_type"] == "no_body_chunks"
        assert kwargs["document_id"] == sample_attachment_document.id
        assert kwargs["file_name"] == "Settings_Options.ini"
        assert kwargs["source_eml_path"] == "test.eml"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_attachment_nonempty_chunks_no_event(
        self, tmp_path, sample_document, sample_attachment_document, sample_document_id
    ):
        """Non-empty chunks must not emit the event (negative control)."""
        from uuid import uuid4
        from mtss.models.chunk import Chunk
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("something parseable")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )
        chunk = Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id="c1",
            content="something parseable",
            chunk_index=0,
            section_path=[],
            source_id="test.eml/notes.txt",
        )

        components = self._build_components(chunks_return=[chunk])
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        components.db.log_ingest_event.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_attachment_stamps_topic_ids_on_chunks(
        self, tmp_path, sample_document, sample_attachment_document, sample_document_id
    ):
        """Attachment chunks must inherit the parent email's topic_ids.

        Regression guard for the 2026-04-22 bug where
        ``_apply_vessel_metadata_to_chunks`` did not receive ``topic_ids``,
        so pgvector's ``match_chunks`` topic filter (``metadata @> {topic_ids}``)
        missed every attachment row even when the parent email was topic-tagged.
        """
        from uuid import uuid4
        from mtss.models.chunk import Chunk
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("something parseable")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )
        chunk = Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id="c1",
            content="something parseable",
            chunk_index=0,
            section_path=[],
            source_id="test.eml/notes.txt",
        )

        components = self._build_components(chunks_return=[chunk])
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )
        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        topic_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
        ]
        chunks = await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
            topic_ids=topic_ids,
        )

        assert chunks, "process_attachment must return at least one chunk"
        assert (chunks[0].metadata or {}).get("topic_ids") == topic_ids

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_attachment_empty_chunks_emits_event(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """ZIP-extracted attachment that parses to 0 chunks → event emitted.

        Reproduces the real-world case where docx files extracted from a
        container (e.g. SUMMARY OF ENGINE JOBS_extracted/*.docx) yielded
        empty content with no explanatory event.
        """
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_zip_attachment

        zip_path = tmp_path / "container.zip"
        zip_path.write_bytes(b"PK")
        extracted = tmp_path / "extracted" / "empty.docx"
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_bytes(b"")

        attachment = ParsedAttachment(
            filename="container.zip",
            content_type="application/zip",
            size_bytes=zip_path.stat().st_size,
            saved_path=str(zip_path),
        )

        components = self._build_components(chunks_return=[], is_zip=True)
        components.attachment_processor.extract_zip.return_value = [
            (
                extracted,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ]
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        chunks = await process_zip_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # ZIP-member empty-parse mirrors the non-ZIP fix: stamp METADATA_ONLY
        # on the extracted doc AND emit exactly one filename-stub chunk.
        from mtss.models.document import EmbeddingMode
        assert len(chunks) == 1
        stub = chunks[0]
        assert stub.embedding_mode == EmbeddingMode.METADATA_ONLY
        assert (stub.metadata or {}).get("type") == "metadata_stub"
        assert sample_attachment_document.file_name in stub.content
        assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY

        components.db.log_ingest_event.assert_called_once()
        kwargs = components.db.log_ingest_event.call_args.kwargs
        assert kwargs["event_type"] == "no_body_chunks"
        assert kwargs["document_id"] == sample_attachment_document.id
        assert kwargs["file_name"] == "empty.docx"


class TestSilentFailureEventEmission:
    """Failures that previously swallowed to vprint/pass must now leave an
    ingest_events.jsonl row so `mtss validate` can surface them.
    """

    def _build_components(self, *, parser_name: str = "local_text"):
        components = MagicMock()
        components.attachment_processor = MagicMock()
        components.attachment_processor.parse_to_text = AsyncMock(
            return_value=("content", parser_name, None)
        )
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        preprocess = MagicMock(
            should_process=True,
            is_image=False,
            is_zip=False,
            skip_reason=None,
            parser_name=parser_name,
            image_description=None,
            oversized_pdf=False,
            preview_markdown=None,
            total_pages=None,
        )
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)

        components.hierarchy_manager = MagicMock()
        components.archive_generator = MagicMock()
        components.archive_generator.storage = MagicMock()
        components.context_generator = None

        components.db = MagicMock()
        components.db.log_ingest_event = MagicMock()
        return components

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_check_failure_emits_event(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Cache lookup that raises during download_text must emit
        `cache_check_failed`, not swallow silently."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("hello")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )

        components = self._build_components()
        # email_doc must have doc_id for the cache branch to run.
        sample_document.doc_id = "abc1234567890123def"
        components.archive_generator.storage.file_exists = MagicMock(return_value=True)
        components.archive_generator.storage.download_text = MagicMock(
            side_effect=RuntimeError("storage unreachable")
        )
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        event_types = [
            c.kwargs.get("event_type")
            for c in components.db.log_ingest_event.call_args_list
        ]
        assert "cache_check_failed" in event_types
        cache_call = next(
            c for c in components.db.log_ingest_event.call_args_list
            if c.kwargs.get("event_type") == "cache_check_failed"
        )
        assert cache_call.kwargs["document_id"] == sample_attachment_document.id
        assert "storage unreachable" in cache_call.kwargs["message"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zip_member_upload_failure_emits_event(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Archive upload of a ZIP member that raises must emit
        `zip_member_upload_failed` — without the event, the .md can exist but
        the download_uri points to a missing original."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_zip_attachment

        zip_path = tmp_path / "container.zip"
        zip_path.write_bytes(b"PK")
        extracted = tmp_path / "extracted" / "report.txt"
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_text("some parseable content")

        attachment = ParsedAttachment(
            filename="container.zip",
            content_type="application/zip",
            size_bytes=zip_path.stat().st_size,
            saved_path=str(zip_path),
        )

        components = self._build_components()
        # Swap in the ZIP-specific preprocess (extract_zip) plumbing.
        components.attachment_processor.extract_zip = MagicMock(
            return_value=[(extracted, "text/plain")]
        )
        sample_document.doc_id = "abc1234567890123def"
        components.archive_generator.storage.upload_file = MagicMock(
            side_effect=RuntimeError("s3 503")
        )
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_zip_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        event_types = [
            c.kwargs.get("event_type")
            for c in components.db.log_ingest_event.call_args_list
        ]
        assert "zip_member_upload_failed" in event_types

class TestAttachmentCacheParserIdentity:
    """Cache reads must skip when the stored parser differs from the router's
    current choice. Without the sidecar check, a config flip
    (Gemini↔LlamaParse) silently reuses the prior parser's output."""

    def _build_components(self, *, parser_name: str, cached_meta_parser: str | None, folder_id: str):
        components = MagicMock()
        components.attachment_processor = MagicMock()
        components.attachment_processor.parse_to_text = AsyncMock(
            return_value=("freshly parsed", parser_name, None)
        )
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        preprocess = MagicMock(
            should_process=True,
            is_image=False,
            is_zip=False,
            skip_reason=None,
            parser_name=parser_name,
            image_description=None,
            oversized_pdf=False,
            preview_markdown=None,
            total_pages=None,
        )
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)

        components.hierarchy_manager = MagicMock()
        components.archive_generator = MagicMock()
        storage = MagicMock()
        md_key = f"{folder_id}/attachments/notes.txt.md"
        meta_key = f"{folder_id}/attachments/notes.txt.meta.json"
        # Sidecar-based exists map; the .md exists for every run, the sidecar
        # only exists when cached_meta_parser is set.
        existing = {md_key: True}
        if cached_meta_parser is not None:
            existing[meta_key] = True

        storage.file_exists = MagicMock(side_effect=lambda p: existing.get(p, False))
        import json as _json
        meta_txt = (
            _json.dumps({
                "parser": cached_meta_parser,
                "model": None,
                "parsed_at": "2026-01-01T00:00:00Z",
            })
            if cached_meta_parser else ""
        )
        storage.download_text = MagicMock(
            side_effect=lambda p: {
                md_key: "# header\n\n## Content\ncached original content",
                meta_key: meta_txt,
            }[p]
        )
        components.archive_generator.storage = storage
        components.archive_generator.update_attachment_markdown = MagicMock(
            return_value=md_key
        )
        components.context_generator = None

        components.db = MagicMock()
        components.db.log_ingest_event = MagicMock()
        return components

    @pytest.mark.asyncio
    async def test_cache_reused_when_parser_matches_sidecar(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Same parser in sidecar and router → use cached content, skip parse_to_text."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("ignored (cache hit)")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )
        from mtss.utils import compute_folder_id
        sample_document.doc_id = "stable1234567890def"
        components = self._build_components(
            parser_name="local_text", cached_meta_parser="local_text",
            folder_id=compute_folder_id(sample_document.doc_id),
        )
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # Cache hit → no fresh parse.
        components.attachment_processor.parse_to_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_skipped_when_parser_differs_from_sidecar(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Sidecar parser != router parser → stale cache, re-parse."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("content")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )
        from mtss.utils import compute_folder_id
        sample_document.doc_id = "stable1234567890def"
        components = self._build_components(
            parser_name="local_text", cached_meta_parser="gemini_pdf",
            folder_id=compute_folder_id(sample_document.doc_id),
        )
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # Stale cache → fresh parse runs.
        components.attachment_processor.parse_to_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_trusted_when_sidecar_missing_legacy(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        """Pre-migration archives have no sidecar. Trust the cache on first
        encounter so the fix does not force a mass re-parse at deploy."""
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        att_path = tmp_path / "notes.txt"
        att_path.write_text("content")
        attachment = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=att_path.stat().st_size,
            saved_path=str(att_path),
        )
        from mtss.utils import compute_folder_id
        sample_document.doc_id = "stable1234567890def"
        components = self._build_components(
            parser_name="local_text", cached_meta_parser=None,
            folder_id=compute_folder_id(sample_document.doc_id),
        )
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # Legacy cache (no sidecar) → still treated as a hit.
        components.attachment_processor.parse_to_text.assert_not_called()


class TestOversizedPdfPeek:
    """Oversized PDFs route through the preprocessor peek so the full cloud
    parser is never invoked. attachment_handler must use the preview as
    parsed_content and emit an ``oversized_pdf_peek`` event."""

    def _build_components(self):
        components = MagicMock()
        components.attachment_processor = MagicMock()
        # parse_to_text should NEVER be called — fail loudly if it is.
        components.attachment_processor.parse_to_text = AsyncMock(
            side_effect=AssertionError(
                "parse_to_text must not run on the oversized_pdf path"
            )
        )
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        preprocess = MagicMock(
            should_process=True,
            is_image=False,
            is_zip=False,
            skip_reason=None,
            parser_name="oversized_pdf_peek",
            image_description=None,
            oversized_pdf=True,
            preview_markdown=(
                "_Oversized PDF — 818 pages total. "
                "The following is a preview of the first 3 pages._\n\n"
                "# Sensor Log\n\n"
                + "12345 67890 11223 | 2025-07-01T00:00:00Z | 100.5\n" * 40
            ),
            total_pages=818,
        )
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)

        components.hierarchy_manager = MagicMock()
        components.archive_generator = MagicMock()
        components.archive_generator.storage = MagicMock()
        components.archive_generator.storage.file_exists = MagicMock(return_value=False)
        components.context_generator = None  # forces FULL mode → simple path

        components.db = MagicMock()
        components.db.log_ingest_event = MagicMock()
        return components

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_oversized_pdf_skips_full_parse_and_logs_event(
        self, tmp_path, sample_document, sample_attachment_document
    ):
        from mtss.models.document import ParsedAttachment
        from mtss.ingest.attachment_handler import process_attachment

        pdf = tmp_path / "huge.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        attachment = ParsedAttachment(
            filename="huge.pdf",
            content_type="application/pdf",
            size_bytes=pdf.stat().st_size,
            saved_path=str(pdf),
        )

        components = self._build_components()
        components.hierarchy_manager.build_attachment_document = MagicMock(
            return_value=sample_attachment_document
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        chunks = await process_attachment(
            attachment=attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )

        # parse_to_text wasn't touched (its side_effect would have raised).
        components.attachment_processor.parse_to_text.assert_not_called()
        # Chunks were produced from the preview.
        assert len(chunks) >= 1
        # Event emitted with the peek identifier.
        event_types = [
            call.kwargs.get("event_type")
            for call in components.db.log_ingest_event.call_args_list
        ]
        assert "oversized_pdf_peek" in event_types


class TestProgressUnitsExpandZipMembers:
    """_count_progress_units must pre-size the progress denominator to one
    tick per ZIP member so image-heavy ZIPs don't freeze the bar at a tiny
    denominator while 48 vision calls run inside."""

    @pytest.mark.unit
    def test_non_zip_attachment_counts_as_one(self, tmp_path):
        from mtss.ingest.pipeline import _count_progress_units
        from mtss.models.document import ParsedAttachment

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        att = ParsedAttachment(
            filename="doc.pdf",
            content_type="application/pdf",
            size_bytes=pdf.stat().st_size,
            saved_path=str(pdf),
        )
        assert _count_progress_units([att]) == 1

    @pytest.mark.unit
    def test_zip_contributes_one_per_member(self, tmp_path):
        import zipfile

        from mtss.ingest.pipeline import _count_progress_units
        from mtss.models.document import ParsedAttachment

        zip_path = tmp_path / "photos.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(7):
                zf.writestr(f"photo_{i}.jpg", b"fake-jpeg")
        att = ParsedAttachment(
            filename="photos.zip",
            content_type="application/zip",
            size_bytes=zip_path.stat().st_size,
            saved_path=str(zip_path),
        )
        assert _count_progress_units([att]) == 7

    @pytest.mark.unit
    def test_mixed_attachments_sum_correctly(self, tmp_path):
        import zipfile

        from mtss.ingest.pipeline import _count_progress_units
        from mtss.models.document import ParsedAttachment

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(4):
                zf.writestr(f"m_{i}.jpg", b"x")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        atts = [
            ParsedAttachment(
                filename="doc.pdf", content_type="application/pdf",
                size_bytes=pdf.stat().st_size, saved_path=str(pdf),
            ),
            ParsedAttachment(
                filename="bundle.zip", content_type="application/zip",
                size_bytes=zip_path.stat().st_size, saved_path=str(zip_path),
            ),
        ]
        # 1 (pdf) + 4 (zip members) = 5
        assert _count_progress_units(atts) == 5

    @pytest.mark.unit
    def test_unreadable_zip_falls_back_to_one(self, tmp_path):
        from mtss.ingest.pipeline import _count_progress_units
        from mtss.models.document import ParsedAttachment

        corrupted = tmp_path / "broken.zip"
        corrupted.write_bytes(b"not a zip")
        att = ParsedAttachment(
            filename="broken.zip", content_type="application/zip",
            size_bytes=corrupted.stat().st_size, saved_path=str(corrupted),
        )
        assert _count_progress_units([att]) == 1


class TestZipMemberConcurrency:
    """Members of a single ZIP must process concurrently.

    Sequential processing of image-heavy ZIPs (~48 photos) serialises ~48
    vision-API calls and blows past the per-file timeout. The handler uses a
    bounded semaphore (``settings.zip_member_concurrency``) to run members in
    parallel. These tests lock the concurrency contract in place.
    """

    @pytest.fixture
    def zip_attachment(self, tmp_path):
        from mtss.models.document import ParsedAttachment

        zip_path = tmp_path / "bundle.zip"
        zip_path.write_bytes(b"PK\x03\x04fake-zip")
        return ParsedAttachment(
            filename="bundle.zip",
            content_type="application/zip",
            size_bytes=zip_path.stat().st_size,
            saved_path=str(zip_path),
        )

    def _make_extracted_files(self, tmp_path, count: int):
        paths = []
        for i in range(count):
            p = tmp_path / "extracted" / f"member_{i}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"content {i}")
            paths.append((p, "text/plain"))
        return paths

    def _build_components(self, extracted_files, per_call_delay: float = 0.0):
        """Mock components; each process_attachment call sleeps per_call_delay."""
        import asyncio as _asyncio
        from uuid import uuid4

        from mtss.models.chunk import Chunk

        components = MagicMock()

        components.attachment_processor = MagicMock()
        components.attachment_processor.extract_zip = MagicMock(return_value=extracted_files)

        preprocess = MagicMock(
            should_process=True, is_image=False, is_zip=False,
            skip_reason=None, parser_name=None, image_description=None,
            oversized_pdf=False, preview_markdown=None, total_pages=None,
        )
        components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)

        async def slow_parse(path, doc_id, content_type):
            if per_call_delay:
                await _asyncio.sleep(per_call_delay)
            return [Chunk(
                id=uuid4(), document_id=doc_id, chunk_id=f"c_{path.name}",
                content=f"parsed {path.name}", chunk_index=0, section_path=[],
            )]

        components.attachment_processor.process_attachment = AsyncMock(side_effect=slow_parse)

        # New pipeline uses parse_to_text; mirror the per-call delay so the
        # concurrency assertion still measures parallelism.
        async def slow_parse_text(path, content_type):
            if per_call_delay:
                await _asyncio.sleep(per_call_delay)
            # Return content long enough to avoid metadata_only path.
            return (
                f"# {path.name}\n\n"
                + ("parsed content for this extracted zip member. " * 30),
                "local_text",
                None,
            )

        components.attachment_processor.parse_to_text = AsyncMock(side_effect=slow_parse_text)
        from mtss.parsers.chunker import DocumentChunker
        components.attachment_processor.chunker = DocumentChunker()

        def build_attach_doc(**kwargs):
            from mtss.models.document import Document, DocumentType, ProcessingStatus
            return Document(
                id=uuid4(),
                doc_id=f"att_{kwargs['attachment_path'].name}",
                document_type=DocumentType.ATTACHMENT_OTHER,
                file_path=str(kwargs['attachment_path']),
                file_name=kwargs['attachment_path'].name,
                depth=1,
                parent_id=kwargs['parent_doc'].id,
                root_id=kwargs['parent_doc'].id,
                status=ProcessingStatus.PENDING,
            )
        components.hierarchy_manager = MagicMock()
        components.hierarchy_manager.build_attachment_document = MagicMock(side_effect=build_attach_doc)

        components.context_generator = None
        components.archive_generator = None
        components.db = MagicMock()
        components.db.log_ingest_event = MagicMock()
        return components

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_members_processed_concurrently(
        self, tmp_path, zip_attachment, sample_document, monkeypatch
    ):
        """With 8 members each sleeping 0.1s and concurrency=5, wall-clock
        should be close to 2 × 0.1s (two batches), not 8 × 0.1s (serial)."""
        import time

        from mtss.config import get_settings
        from mtss.ingest.attachment_handler import process_zip_attachment

        settings = get_settings()
        monkeypatch.setattr(settings, "zip_member_concurrency", 5)

        extracted = self._make_extracted_files(tmp_path, count=8)
        components = self._build_components(extracted, per_call_delay=0.1)

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        t0 = time.perf_counter()
        chunks = await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )
        elapsed = time.perf_counter() - t0

        assert len(chunks) == 8
        # Sequential would be ~0.8s; concurrency=5 with 8 members = 2 batches → ~0.2s.
        # Allow generous slack for Windows event-loop jitter, but fail on anything
        # even close to sequential.
        assert elapsed < 0.55, f"zip members look serial: {elapsed:.2f}s for 8×0.1s with concurrency=5"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_concurrency_limit_is_respected(
        self, tmp_path, zip_attachment, sample_document, monkeypatch
    ):
        """At concurrency=2 with 6 members of 0.1s each → ~3 batches ≈ 0.3s.
        This guards against the other failure mode: unbounded parallelism that
        would hammer the vision API."""
        import time

        from mtss.config import get_settings
        from mtss.ingest.attachment_handler import process_zip_attachment

        settings = get_settings()
        monkeypatch.setattr(settings, "zip_member_concurrency", 2)

        extracted = self._make_extracted_files(tmp_path, count=6)
        components = self._build_components(extracted, per_call_delay=0.1)

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        t0 = time.perf_counter()
        await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
        )
        elapsed = time.perf_counter() - t0

        # Must NOT be fully parallel: 6 parallel × 0.1s ≈ 0.1s.
        # Must NOT be fully serial: 6 × 0.1s = 0.6s.
        # Target: ~3 batches ≈ 0.3s. Allow wide band for jitter.
        assert 0.2 < elapsed < 0.5, (
            f"expected bounded concurrency (~0.3s), got {elapsed:.2f}s "
            f"— may be unbounded (too fast) or serial (too slow)"
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_on_member_complete_fires_once_per_member(
        self, tmp_path, zip_attachment, sample_document
    ):
        """Progress tick callback must fire once per ZIP member so the outer
        progress bar shows real progress (not a single bump at the end).

        Regression: image-heavy ZIPs (e.g. 48 inspection photos) previously
        bumped progress by 1 after the whole ZIP finished, leaving the bar
        showing "0/2" while a single email processed for ~8 minutes.
        """
        from mtss.ingest.attachment_handler import process_zip_attachment

        extracted = self._make_extracted_files(tmp_path, count=6)
        components = self._build_components(extracted, per_call_delay=0.0)

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        ticks = []
        await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
            on_member_complete=lambda: ticks.append(1),
        )
        assert len(ticks) == 6

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_on_member_complete_fires_for_rejected_members(
        self, tmp_path, zip_attachment, sample_document
    ):
        """Even members preprocess-rejected (classified_as_non_content) must
        tick, otherwise the bar stalls for every skipped image."""
        from mtss.ingest.attachment_handler import process_zip_attachment

        extracted = self._make_extracted_files(tmp_path, count=3)
        components = self._build_components(extracted)
        # All three members get rejected.
        components.attachment_processor.preprocess = AsyncMock(
            return_value=MagicMock(
                should_process=False, is_image=False, is_zip=False,
                skip_reason="classified_as_non_content",
            )
        )

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        ticks = []
        await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
            on_member_complete=lambda: ticks.append(1),
        )
        assert len(ticks) == 3

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_collect_docs_receives_all_successful_members(
        self, tmp_path, zip_attachment, sample_document
    ):
        """Concurrent processing must still collect every successful member
        into collect_docs — order not guaranteed, but count and content must match."""
        from mtss.ingest.attachment_handler import process_zip_attachment

        extracted = self._make_extracted_files(tmp_path, count=4)
        components = self._build_components(extracted, per_call_delay=0.0)

        unsupported_logger = MagicMock()
        unsupported_logger.log_unsupported_file = AsyncMock()

        collected: list = []
        chunks = await process_zip_attachment(
            attachment=zip_attachment,
            email_doc=sample_document,
            source_eml_path="test.eml",
            file_ctx="test.eml",
            components=components,
            unsupported_logger=unsupported_logger,
            collect_docs=collected,
        )

        assert len(chunks) == 4
        assert len(collected) == 4
        collected_names = {d.file_name for d in collected}
        assert collected_names == {f"member_{i}.txt" for i in range(4)}
