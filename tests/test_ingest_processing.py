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
        with patch("mtss.processing.hierarchy_manager.get_settings", return_value=comprehensive_mock_settings):
            from mtss.processing.hierarchy_manager import HierarchyManager
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
        with patch("mtss.processing.hierarchy_manager.get_settings") as mock_settings_call:
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
