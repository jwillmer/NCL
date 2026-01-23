"""Integration tests for the full ingest flow.

Tests the complete processing pipeline with all dependencies mocked.
Ensures components work together correctly without external API calls.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestIngestFlowIntegration:
    """Integration tests for the complete ingest flow."""

    @pytest.fixture
    def mock_all_dependencies(
        self,
        comprehensive_mock_settings,
        mock_supabase_client,
        mock_embedding_response,
        mock_llm_completion_response,
    ):
        """Set up all mocks needed for integration tests."""
        patches = {
            "settings": patch(
                "mtss.config.get_settings", return_value=comprehensive_mock_settings
            ),
            "supabase": patch(
                "mtss.storage.supabase_client.create_client",
                return_value=MagicMock(),
            ),
            "embedding": patch(
                "mtss.processing.embeddings.aembedding",
                new_callable=AsyncMock,
                return_value=mock_embedding_response,
            ),
            "completion": patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_llm_completion_response,
            ),
            "langfuse": patch(
                "mtss.processing.embeddings.get_langfuse_metadata",
                return_value={},
            ),
        }
        return patches

    @pytest.mark.asyncio
    async def test_full_email_processing_flow(
        self,
        temp_dir,
        simple_eml_file,
        comprehensive_mock_settings,
        mock_supabase_client,
        mock_embedding_response,
        mock_llm_completion_response,
    ):
        """Test processing an email through the full pipeline."""
        # Set up patches for the processing pipeline
        with patch("mtss.parsers.eml_parser.get_settings", return_value=comprehensive_mock_settings):
            with patch("mtss.parsers.chunker.get_settings", return_value=comprehensive_mock_settings):
                with patch("mtss.processing.embeddings.get_settings", return_value=comprehensive_mock_settings):
                    with patch("mtss.processing.embeddings.aembedding", new_callable=AsyncMock) as mock_embed:
                        with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                            mock_embed.return_value = mock_embedding_response

                            # Parse email
                            from mtss.parsers.eml_parser import EMLParser

                            parser = EMLParser(attachments_dir=temp_dir / "attachments")
                            parsed = parser.parse_file(simple_eml_file)

                            assert parsed.metadata.subject == "Simple Test Email"
                            assert len(parsed.metadata.participants) >= 1

                            # Chunk the email body
                            from mtss.parsers.chunker import DocumentChunker

                            chunker = DocumentChunker()
                            doc_id = uuid4()
                            chunks = chunker.chunk_text(
                                parsed.full_text, doc_id, str(simple_eml_file)
                            )

                            assert len(chunks) >= 1

                            # Generate embeddings
                            from mtss.processing.embeddings import EmbeddingGenerator

                            embedder = EmbeddingGenerator()
                            embedded = await embedder.embed_chunks(chunks)

                            assert len(embedded) == len(chunks)
                            for chunk in embedded:
                                assert chunk.embedding is not None

    @pytest.mark.asyncio
    async def test_error_handling_in_flow(
        self,
        temp_dir,
        comprehensive_mock_settings,
    ):
        """Test that errors in the flow are handled gracefully."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=comprehensive_mock_settings):
            from mtss.parsers.eml_parser import EMLParser

            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            # Try to parse non-existent file
            with pytest.raises(FileNotFoundError):
                parser.parse_file(temp_dir / "nonexistent.eml")

    @pytest.mark.asyncio
    async def test_attachment_processing_flow(
        self,
        temp_dir,
        sample_eml_file,
        comprehensive_mock_settings,
    ):
        """Test processing an email with attachments."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=comprehensive_mock_settings):
            from mtss.parsers.eml_parser import EMLParser

            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            parsed = parser.parse_file(sample_eml_file)

            # Should have one attachment (PDF)
            assert len(parsed.attachments) == 1
            assert parsed.attachments[0].filename == "test.pdf"
            assert parsed.attachments[0].content_type == "application/pdf"

            # Attachment should be saved
            saved_path = Path(parsed.attachments[0].saved_path)
            assert saved_path.exists()

    @pytest.mark.asyncio
    async def test_lenient_mode_continues_on_error(
        self,
        mock_console,
        mock_supabase_client,
    ):
        """Test that lenient mode continues processing despite errors."""
        from mtss.ingest.helpers import IssueTracker

        tracker = IssueTracker(console=mock_console, db=mock_supabase_client)

        # Simulate tracking errors
        await tracker.track_async(
            file_ctx="email1.eml",
            attachment="bad.pdf",
            error="Parse failed",
            severity="error",
            document_id=uuid4(),
        )

        await tracker.track_async(
            file_ctx="email2.eml",
            attachment="good.docx",
            error="Minor warning",
            severity="warning",
            document_id=uuid4(),
        )

        # In lenient mode, processing continues
        assert len(tracker) == 2
        assert tracker.error_count == 1

        # Show summary should work
        tracker.show_summary()

    @pytest.mark.asyncio
    async def test_chunking_and_context_generation_flow(
        self,
        comprehensive_mock_settings,
        sample_document,
        sample_markdown_content,
        mock_llm_completion_response,
    ):
        """Test chunking followed by context generation."""
        with patch("mtss.parsers.chunker.get_settings", return_value=comprehensive_mock_settings):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
                mock_completion.return_value = mock_llm_completion_response

                from mtss.parsers.chunker import ContextGenerator, DocumentChunker

                # Chunk the content
                chunker = DocumentChunker()
                chunks = chunker.chunk_text(
                    sample_markdown_content,
                    sample_document.id,
                    sample_document.file_path,
                )

                assert len(chunks) >= 1

                # Generate context
                context_gen = ContextGenerator()
                context = await context_gen.generate_context(
                    sample_document, sample_markdown_content
                )

                assert context is not None
                assert len(context) > 0

                # Build embedding text for each chunk
                for chunk in chunks:
                    embedding_text = context_gen.build_embedding_text(context, chunk.content)
                    assert context in embedding_text
                    assert chunk.content in embedding_text

    @pytest.mark.asyncio
    async def test_hierarchy_creation_flow(
        self,
        temp_dir,
        comprehensive_mock_settings,
        mock_supabase_client,
        sample_parsed_email,
    ):
        """Test creating document hierarchy."""
        with patch("mtss.processing.hierarchy_manager.get_settings", return_value=comprehensive_mock_settings):
            from mtss.processing.hierarchy_manager import HierarchyManager

            manager = HierarchyManager(mock_supabase_client, temp_dir)

            # Create email file
            eml_file = temp_dir / "test.eml"
            eml_file.write_text("Mock email content")

            # Create email document
            email_doc = await manager.create_email_document(eml_file, sample_parsed_email)

            assert email_doc.depth == 0
            assert email_doc.parent_id is None
            assert email_doc.root_id == email_doc.id
            mock_supabase_client.insert_document.assert_called()

    @pytest.mark.asyncio
    async def test_embedding_batch_processing(
        self,
        comprehensive_mock_settings,
        sample_chunks,
        mock_batch_embedding_response,
    ):
        """Test batch embedding processing."""
        with patch("mtss.processing.embeddings.get_settings", return_value=comprehensive_mock_settings):
            with patch("mtss.processing.embeddings.aembedding", new_callable=AsyncMock) as mock_embed:
                with patch("mtss.processing.embeddings.get_langfuse_metadata", return_value={}):
                    mock_embed.return_value = mock_batch_embedding_response

                    from mtss.processing.embeddings import EmbeddingGenerator

                    embedder = EmbeddingGenerator()
                    result = await embedder.embed_chunks(sample_chunks)

                    assert len(result) == len(sample_chunks)
                    for chunk in result:
                        assert chunk.embedding is not None
                        assert len(chunk.embedding) == 1536


class TestDataModelIntegrity:
    """Tests for data model consistency across the flow."""

    def test_document_to_chunk_metadata_flow(self, sample_document, sample_chunks):
        """Test that document metadata flows correctly to chunks."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata

        enrich_chunks_with_document_metadata(sample_chunks, sample_document)

        for chunk in sample_chunks:
            assert chunk.source_id == sample_document.source_id
            assert chunk.source_title == sample_document.source_title

    def test_chunk_id_generation_consistency(self, sample_document):
        """Test that chunk IDs are generated consistently."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata
        from mtss.models.chunk import Chunk

        # Create identical chunks twice
        def create_chunks():
            return [
                Chunk(
                    document_id=sample_document.id,
                    content="Test content",
                    chunk_index=0,
                    char_start=0,
                    char_end=100,
                    section_path=[],
                    metadata={},
                )
            ]

        chunks1 = create_chunks()
        chunks2 = create_chunks()

        enrich_chunks_with_document_metadata(chunks1, sample_document)
        enrich_chunks_with_document_metadata(chunks2, sample_document)

        # Same positions should produce same chunk IDs
        assert chunks1[0].chunk_id == chunks2[0].chunk_id

    def test_email_metadata_preservation(self, sample_document):
        """Test that email metadata is preserved through processing."""
        assert sample_document.email_metadata is not None
        assert sample_document.email_metadata.subject == "Test Email Subject"
        assert len(sample_document.email_metadata.participants) == 2
        assert sample_document.email_metadata.initiator == "sender@example.com"

    def test_attachment_metadata_preservation(self, sample_attachment_document):
        """Test that attachment metadata is preserved through processing."""
        assert sample_attachment_document.attachment_metadata is not None
        assert sample_attachment_document.attachment_metadata.content_type == "application/pdf"
        assert sample_attachment_document.attachment_metadata.size_bytes == 12345
        assert sample_attachment_document.attachment_metadata.original_filename == "report.pdf"
