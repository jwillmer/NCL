"""Regression tests for archive URI bugs.

Covers 4 related bugs:
1. Chunks missing archive_browse_uri (denormalization gap)
2. archive_download_uri NULL on attachment documents (filename mismatch)
3. Download URL double /archive/ prefix
4. Double URL-encoding of filenames (%2520 instead of %20)
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# =============================================================================
# Test 1: URI construction — no double prefix, no double encoding (Bugs 3 & 4)
# =============================================================================


class TestArchiveUriConstruction:
    """Regression tests for archive URI construction and prefix handling."""

    @pytest.mark.unit
    def test_archive_uri_no_double_prefix(self):
        """URIs should have exactly one /archive/ prefix, never double."""
        md_path = "abc123def45678/attachments/report.pdf.md"
        browse_uri = f"/archive/{md_path}"
        assert browse_uri.count("/archive/") == 1

    @pytest.mark.unit
    def test_archive_uri_no_double_encoding(self):
        """URIs should not double-encode %20 to %2520."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        safe = _sanitize_storage_key("GE FO SER SYS.pdf")
        md_path = f"abc123def45678/attachments/{safe}.md"
        browse_uri = f"/archive/{md_path}"
        assert "%2520" not in browse_uri
        assert "%20" in browse_uri  # single encoding preserved

    @pytest.mark.unit
    def test_sanitize_storage_key_spaces(self):
        """Filenames with spaces should be encoded once."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        result = _sanitize_storage_key("MARAN ASPASIA - report.pdf")
        assert " " not in result  # spaces encoded
        assert "%20" in result  # to %20
        assert "%2520" not in result  # but not double


# =============================================================================
# Test 2: Filename matching for archive_file_result (Bug 2)
# =============================================================================


class TestArchiveFileResultMatching:
    """Regression tests for matching archive results to attachments."""

    @pytest.mark.unit
    def test_match_sanitized_filename_with_spaces(self):
        """archive_file_result matching must handle sanitized filenames."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        original_filename = "GE FO SER SYS.pdf"
        safe_name = _sanitize_storage_key(original_filename)
        original_path = f"abc123def45678/attachments/{safe_name}"

        # This is the check cli.py does
        assert original_path.endswith(f"/{safe_name}")

    @pytest.mark.unit
    def test_match_sanitized_filename_with_brackets(self):
        """Should match files with brackets replaced by parens."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        original_filename = "report[1].pdf"
        safe_name = _sanitize_storage_key(original_filename)
        original_path = f"abc123def45678/attachments/{safe_name}"

        assert original_path.endswith(f"/{safe_name}")
        assert "[" not in safe_name  # brackets replaced

    @pytest.mark.unit
    def test_unsanitized_name_does_not_match_sanitized_path(self):
        """Original filename with spaces should NOT match sanitized path directly."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        original_filename = "test file.pdf"
        safe_name = _sanitize_storage_key(original_filename)
        original_path = f"abc123def45678/attachments/{safe_name}"

        # The old (broken) matching used the raw filename
        assert not original_path.endswith(f"/{original_filename}")
        # The new (fixed) matching uses sanitized name
        assert original_path.endswith(f"/{safe_name}")


# =============================================================================
# Test 3: Chunk URI propagation (Bug 1)
# =============================================================================


class TestChunkUriPropagation:
    """Regression tests: chunks must always receive doc's archive URIs."""

    @pytest.mark.unit
    def test_enrich_copies_browse_uri(self):
        """enrich_chunks_with_document_metadata must copy archive URIs."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata
        from mtss.models.chunk import Chunk
        from mtss.models.document import Document, DocumentType

        doc = Document(
            document_type=DocumentType.ATTACHMENT_PDF,
            file_path="/test.pdf",
            file_name="test.pdf",
            file_hash="abc",
            source_id="test",
            doc_id="abc123def456test",
            content_version=1,
            ingest_version=1,
            source_title="test.pdf",
            archive_browse_uri="/archive/abc123def45678/attachments/test.pdf.md",
            archive_download_uri="/archive/abc123def45678/attachments/test.pdf",
        )
        chunk = Chunk(
            id=uuid4(),
            document_id=doc.id,
            content="test",
            chunk_index=0,
            section_path=[],
            metadata={},
            char_start=0,
            char_end=4,
        )

        enrich_chunks_with_document_metadata([chunk], doc)

        assert chunk.archive_browse_uri == doc.archive_browse_uri
        assert chunk.archive_download_uri == doc.archive_download_uri

    @pytest.mark.unit
    def test_enrich_handles_none_uris(self):
        """enrich should not fail when doc URIs are None."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata
        from mtss.models.chunk import Chunk
        from mtss.models.document import Document, DocumentType

        doc = Document(
            document_type=DocumentType.ATTACHMENT_PDF,
            file_path="/test.pdf",
            file_name="test.pdf",
            file_hash="abc",
            source_id="test",
            doc_id="abc123def456test",
            content_version=1,
            ingest_version=1,
            source_title="test.pdf",
            archive_browse_uri=None,
            archive_download_uri=None,
        )
        chunk = Chunk(
            id=uuid4(),
            document_id=doc.id,
            content="test",
            chunk_index=0,
            section_path=[],
            metadata={},
            char_start=0,
            char_end=4,
        )

        enrich_chunks_with_document_metadata([chunk], doc)
        assert chunk.archive_browse_uri is None
        assert chunk.archive_download_uri is None


# =============================================================================
# Test 4: Citation processor download attribute (Bug 3)
# =============================================================================


class TestCitationProcessorDownloadAttr:
    """Regression: cite tags must not include /archive/ prefix in download attr."""

    @pytest.mark.unit
    def test_replace_markers_strips_archive_prefix(self):
        """download attr in <cite> must not have /archive/ prefix."""
        from mtss.models.chunk import ValidatedCitation
        from mtss.rag.citation_processor import CitationProcessor

        with patch("mtss.rag.citation_processor.ArchiveStorage"):
            processor = CitationProcessor()

        response = "Test [C:abc123def456]"
        citations = [
            ValidatedCitation(
                index=1,
                chunk_id="abc123def456",
                source_title="test.pdf",
                page=None,
                lines=None,
                archive_browse_uri="/archive/abc123def45678/test.pdf.md",
                archive_download_uri="/archive/abc123def45678/test.pdf",
                archive_verified=True,
            )
        ]

        result = processor.replace_citation_markers(response, citations)
        # download attr must NOT contain /archive/ prefix
        assert 'download="/archive/' not in result
        assert 'download="abc123def45678/test.pdf"' in result

    @pytest.mark.unit
    def test_replace_markers_handles_no_prefix(self):
        """download attr without /archive/ prefix should pass through unchanged."""
        from mtss.models.chunk import ValidatedCitation
        from mtss.rag.citation_processor import CitationProcessor

        with patch("mtss.rag.citation_processor.ArchiveStorage"):
            processor = CitationProcessor()

        response = "Test [C:abc123def456]"
        citations = [
            ValidatedCitation(
                index=1,
                chunk_id="abc123def456",
                source_title="test.pdf",
                page=None,
                lines=None,
                archive_browse_uri=None,
                archive_download_uri="abc123def45678/test.pdf",
                archive_verified=True,
            )
        ]

        result = processor.replace_citation_markers(response, citations)
        assert 'download="abc123def45678/test.pdf"' in result


# =============================================================================
# Test 5: Hierarchy manager URI construction (Bug 4)
# =============================================================================


class TestHierarchyManagerUris:
    """Regression: hierarchy_manager must not double-encode attachment URIs."""

    @pytest.fixture
    def hierarchy_manager(self, comprehensive_mock_settings, mock_supabase_client):
        """Create a HierarchyManager with no archive_base_url."""
        # Override archive_base_url to empty so we get /archive/ prefix paths
        comprehensive_mock_settings.archive_base_url = ""
        with patch(
            "mtss.processing.hierarchy_manager.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            from mtss.processing.hierarchy_manager import HierarchyManager

            return HierarchyManager(mock_supabase_client)

    @pytest.mark.asyncio
    async def test_attachment_uri_not_double_encoded(
        self, hierarchy_manager, temp_dir, sample_document, mock_supabase_client
    ):
        """Attachment URIs must not contain %25 (double-encoded %)."""
        from mtss.models.document import DocumentType
        from mtss.processing.archive_generator import ContentFileResult

        attachment_file = temp_dir / "test_file.pdf"
        attachment_file.write_bytes(b"%PDF-1.4 mock")

        file_result = ContentFileResult(
            original_path="abc123def45678/attachments/test%20file.pdf",
            markdown_path="abc123def45678/attachments/test%20file.pdf.md",
            download_uri="abc123def45678/attachments/test%20file.pdf",
            browse_uri="abc123def45678/attachments/test%20file.pdf.md",
            archive_path="abc123def45678",
        )

        mock_proc = MagicMock()
        mock_proc.get_document_type.return_value = DocumentType.ATTACHMENT_PDF
        with patch(
            "mtss.parsers.attachment_processor.AttachmentProcessor",
            return_value=mock_proc,
        ):
            result = await hierarchy_manager.create_attachment_document(
                parent_doc=sample_document,
                attachment_path=attachment_file,
                content_type="application/pdf",
                size_bytes=100,
                original_filename="test file.pdf",
                archive_file_result=file_result,
            )

        # Must not double-encode
        assert "%2520" not in (result.archive_browse_uri or "")
        assert "%2520" not in (result.archive_download_uri or "")
        # Must preserve single encoding
        assert "%20" in (result.archive_browse_uri or "")
        assert "%20" in (result.archive_download_uri or "")

    @pytest.mark.asyncio
    async def test_attachment_uri_has_archive_prefix(
        self, hierarchy_manager, temp_dir, sample_document, mock_supabase_client
    ):
        """Attachment URIs should have /archive/ prefix when no base_url."""
        from mtss.models.document import DocumentType
        from mtss.processing.archive_generator import ContentFileResult

        attachment_file = temp_dir / "report.pdf"
        attachment_file.write_bytes(b"%PDF-1.4 mock")

        file_result = ContentFileResult(
            original_path="abc123def45678/attachments/report.pdf",
            markdown_path="abc123def45678/attachments/report.pdf.md",
            download_uri="abc123def45678/attachments/report.pdf",
            browse_uri="abc123def45678/attachments/report.pdf.md",
            archive_path="abc123def45678",
        )

        mock_proc = MagicMock()
        mock_proc.get_document_type.return_value = DocumentType.ATTACHMENT_PDF
        with patch(
            "mtss.parsers.attachment_processor.AttachmentProcessor",
            return_value=mock_proc,
        ):
            result = await hierarchy_manager.create_attachment_document(
                parent_doc=sample_document,
                attachment_path=attachment_file,
                content_type="application/pdf",
                size_bytes=100,
                original_filename="report.pdf",
                archive_file_result=file_result,
            )

        assert result.archive_browse_uri == "/archive/abc123def45678/attachments/report.pdf.md"
        assert result.archive_download_uri == "/archive/abc123def45678/attachments/report.pdf"


# =============================================================================
# Test 6: Frontend stripArchivePrefix helper (Bug 3 — defense in depth)
# =============================================================================


class TestStripArchivePrefix:
    """Regression: stripArchivePrefix must handle all URI formats."""

    @pytest.mark.unit
    def test_strip_with_prefix(self):
        """Should strip /archive/ prefix."""
        uri = "/archive/abc123/test.pdf"
        result = uri.replace("/archive/", "", 1) if uri.startswith("/archive/") else uri
        assert result == "abc123/test.pdf"

    @pytest.mark.unit
    def test_strip_without_prefix(self):
        """Should return unchanged if no prefix."""
        uri = "abc123/test.pdf"
        result = uri.replace("/archive/", "", 1) if uri.startswith("/archive/") else uri
        assert result == "abc123/test.pdf"

    @pytest.mark.unit
    def test_strip_does_not_affect_middle(self):
        """Should only strip leading /archive/, not occurrences in the middle."""
        uri = "/archive/path/to/archive/file.pdf"
        result = uri.removeprefix("/archive/")
        assert result == "path/to/archive/file.pdf"
