"""Tests for archive generation functionality.

Tests for ArchiveGenerator class that creates browsable archive files
with markdown previews and original file downloads in Supabase Storage.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestArchiveGenerator:
    """Tests for ArchiveGenerator class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock ArchiveStorage."""
        storage = MagicMock()
        storage.upload_file = MagicMock(return_value="path/file.eml")
        storage.upload_text = MagicMock(return_value="path/file.md")
        storage.delete_folder = MagicMock()
        storage.file_exists = MagicMock(return_value=False)
        return storage

    @pytest.fixture
    def archive_generator(self, mock_storage, comprehensive_mock_settings, temp_dir):
        """Create an ArchiveGenerator with mocked storage."""
        with patch(
            "mtss.config.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            with patch(
                "mtss.processing.archive_generator.ArchiveStorage",
                return_value=mock_storage,
            ):
                from mtss.processing.archive_generator import ArchiveGenerator

                generator = ArchiveGenerator(ingest_root=temp_dir)
                generator.storage = mock_storage
                return generator

    @pytest.fixture
    def sample_parsed_email_with_attachments(self, temp_dir):
        """Create a ParsedEmail with attachments for testing."""
        from mtss.models.document import (
            EmailMessage,
            EmailMetadata,
            ParsedAttachment,
            ParsedEmail,
        )

        # Create a test attachment file
        attachment_path = temp_dir / "report.pdf"
        attachment_path.write_bytes(b"%PDF-1.4 test content")

        return ParsedEmail(
            metadata=EmailMetadata(
                subject="Test Subject",
                participants=["sender@example.com", "recipient@example.com"],
                initiator="sender@example.com",
                date_start=datetime(2024, 1, 15, 10, 30, 0),
                date_end=datetime(2024, 1, 15, 11, 0, 0),
                message_count=2,
            ),
            messages=[
                EmailMessage(
                    from_address="sender@example.com",
                    to_addresses=["recipient@example.com"],
                    date=datetime(2024, 1, 15, 10, 30, 0),
                    content="Hello, please find the report attached.",
                ),
                EmailMessage(
                    from_address="recipient@example.com",
                    to_addresses=["sender@example.com"],
                    date=datetime(2024, 1, 15, 11, 0, 0),
                    content="Thanks, received.",
                ),
            ],
            full_text="Hello, please find the report attached.\n\nThanks, received.",
            attachments=[
                ParsedAttachment(
                    filename="report.pdf",
                    content_type="application/pdf",
                    size_bytes=len(b"%PDF-1.4 test content"),
                    saved_path=str(attachment_path),
                ),
            ],
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generates_email_archive_markdown(
        self, archive_generator, sample_parsed_email, temp_dir, mock_storage
    ):
        """Should generate markdown file for email content."""
        # Create a test EML file
        eml_path = temp_dir / "test.eml"
        eml_path.write_bytes(b"Mock EML content")

        result = await archive_generator.generate_archive(
            parsed_email=sample_parsed_email,
            source_eml_path=eml_path,
        )

        # Should upload the markdown file
        upload_text_calls = mock_storage.upload_text.call_args_list
        assert len(upload_text_calls) >= 1

        # Check that markdown path is returned
        assert result.markdown_path.endswith("/email.eml.md")
        assert result.archive_path  # folder ID

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generates_attachment_archive_markdown(
        self, archive_generator, sample_parsed_email_with_attachments, temp_dir, mock_storage
    ):
        """Should generate markdown file for attachments with parsed content."""
        # Create a test EML file
        eml_path = temp_dir / "test.eml"
        eml_path.write_bytes(b"Mock EML content")

        # Provide parsed attachment content
        parsed_contents = {"report.pdf": "# Report\n\nThis is the report content."}

        result = await archive_generator.generate_archive(
            parsed_email=sample_parsed_email_with_attachments,
            source_eml_path=eml_path,
            parsed_attachment_contents=parsed_contents,
        )

        # Should have attachment files in result
        assert len(result.attachment_files) == 1
        att_file = result.attachment_files[0]
        assert "report.pdf" in att_file.original_path
        # Should have markdown since we provided parsed content
        assert att_file.markdown_path is not None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_uploads_original_files_to_bucket(
        self, archive_generator, sample_parsed_email_with_attachments, temp_dir, mock_storage
    ):
        """Should upload original EML and attachment files."""
        # Create a test EML file
        eml_path = temp_dir / "test.eml"
        eml_path.write_bytes(b"Mock EML content")

        await archive_generator.generate_archive(
            parsed_email=sample_parsed_email_with_attachments,
            source_eml_path=eml_path,
        )

        # Should upload original EML
        upload_file_calls = mock_storage.upload_file.call_args_list
        eml_upload = [c for c in upload_file_calls if "email.eml" in c[0][0]]
        assert len(eml_upload) >= 1

        # Should upload attachment original
        attachment_upload = [c for c in upload_file_calls if "report.pdf" in c[0][0]]
        assert len(attachment_upload) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_correct_uris(
        self, archive_generator, sample_parsed_email, temp_dir, mock_storage
    ):
        """Should return correct browse and download URIs."""
        # Create a test EML file
        eml_path = temp_dir / "test.eml"
        eml_path.write_bytes(b"Mock EML content")

        result = await archive_generator.generate_archive(
            parsed_email=sample_parsed_email,
            source_eml_path=eml_path,
        )

        # Archive path should be the folder ID
        assert len(result.archive_path) == 16  # doc_id[:16]

        # Markdown path should be within archive folder
        assert result.markdown_path.startswith(result.archive_path)
        assert result.markdown_path.endswith(".md")

        # Original path should be within archive folder
        assert result.original_path.startswith(result.archive_path)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cleans_existing_folder_before_upload(
        self, archive_generator, sample_parsed_email, temp_dir, mock_storage
    ):
        """Should delete existing folder before uploading new content."""
        # Create a test EML file
        eml_path = temp_dir / "test.eml"
        eml_path.write_bytes(b"Mock EML content")

        await archive_generator.generate_archive(
            parsed_email=sample_parsed_email,
            source_eml_path=eml_path,
        )

        # Should have called delete_folder
        mock_storage.delete_folder.assert_called_once()

    @pytest.mark.unit
    def test_format_size_bytes(self, archive_generator):
        """Should format file sizes correctly."""
        assert archive_generator._format_size(500) == "500 B"
        assert archive_generator._format_size(1024) == "1.0 KB"
        assert archive_generator._format_size(1536) == "1.5 KB"
        assert archive_generator._format_size(1024 * 1024) == "1.0 MB"
        assert archive_generator._format_size(1024 * 1024 * 2) == "2.0 MB"

    @pytest.mark.unit
    def test_should_skip_markdown_for_md_files(self, archive_generator):
        """Should skip markdown generation for files already in markdown format."""
        assert archive_generator._should_skip_markdown("readme.md") is True
        assert archive_generator._should_skip_markdown("NOTES.MD") is True
        assert archive_generator._should_skip_markdown("doc.markdown") is True
        assert archive_generator._should_skip_markdown("report.pdf") is False
        assert archive_generator._should_skip_markdown("data.csv") is False

    @pytest.mark.unit
    def test_get_archive_uris_when_exists(self, archive_generator, mock_storage):
        """Should return URIs when archive exists."""
        mock_storage.file_exists.return_value = True

        result = archive_generator.get_archive_uris("abc123def456789")

        assert result["browse_uri"] is not None
        assert result["download_uri"] is not None  # returns download URI when archive exists

    @pytest.mark.unit
    def test_get_archive_uris_when_not_exists(self, archive_generator, mock_storage):
        """Should return None URIs when archive doesn't exist."""
        mock_storage.file_exists.return_value = False

        result = archive_generator.get_archive_uris("nonexistent123")

        assert result["browse_uri"] is None
        assert result["download_uri"] is None


class TestSanitizeStorageKey:
    """Tests for _sanitize_storage_key function."""

    @pytest.mark.unit
    def test_replaces_brackets(self):
        """Should replace square brackets with parentheses."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        result = _sanitize_storage_key("file[1].pdf")
        assert "[" not in result
        assert "]" not in result
        assert "(1)" in result

    @pytest.mark.unit
    def test_handles_non_ascii_characters(self):
        """Should transliterate non-ASCII to ASCII."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        # Greek characters should be transliterated
        result = _sanitize_storage_key("report_Δελτα.pdf")
        assert result.isascii()

    @pytest.mark.unit
    def test_preserves_safe_characters(self):
        """Should preserve safe characters like hyphen, underscore."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        result = _sanitize_storage_key("file-name_v1.pdf")
        assert "-" in result
        assert "_" in result

    @pytest.mark.unit
    def test_replaces_leading_tilde(self):
        """Should replace leading tilde with underscore (Word temp files)."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        # ~WRD0001.jpg -> _WRD0001.jpg
        result = _sanitize_storage_key("~WRD0001.jpg")
        assert result.startswith("_")
        assert not result.startswith("~")
        assert "WRD0001.jpg" in result

    @pytest.mark.unit
    def test_preserves_non_leading_tilde(self):
        """Should preserve tilde that's not at the start."""
        from mtss.processing.archive_generator import _sanitize_storage_key

        result = _sanitize_storage_key("file~backup.txt")
        assert "~" in result


class TestContentFileResult:
    """Tests for ContentFileResult dataclass."""

    @pytest.mark.unit
    def test_content_file_result_fields(self):
        """Should have all required fields."""
        from mtss.processing.archive_generator import ContentFileResult

        result = ContentFileResult(
            original_path="abc123/attachments/file.pdf",
            markdown_path="abc123/attachments/file.pdf.md",
            download_uri="abc123/attachments/file.pdf",
            browse_uri="abc123/attachments/file.pdf.md",
            archive_path="abc123",
        )

        assert result.original_path == "abc123/attachments/file.pdf"
        assert result.markdown_path == "abc123/attachments/file.pdf.md"
        assert result.skipped is False

    @pytest.mark.unit
    def test_content_file_result_skipped(self):
        """Should support skipped flag for markdown files."""
        from mtss.processing.archive_generator import ContentFileResult

        result = ContentFileResult(
            original_path="abc123/readme.md",
            markdown_path=None,
            download_uri="abc123/readme.md",
            browse_uri=None,
            archive_path="abc123",
            skipped=True,
        )

        assert result.skipped is True
        assert result.markdown_path is None


class TestArchiveResult:
    """Tests for ArchiveResult dataclass."""

    @pytest.mark.unit
    def test_archive_result_fields(self):
        """Should have all required fields."""
        from mtss.processing.archive_generator import ArchiveResult

        result = ArchiveResult(
            archive_path="abc123def456",
            markdown_path="abc123def456/email.eml.md",
            original_path="abc123def456/email.eml",
            doc_id="abc123def456full",
            attachment_files=[],
        )

        assert result.archive_path == "abc123def456"
        assert result.doc_id == "abc123def456full"
        assert result.attachment_files == []
