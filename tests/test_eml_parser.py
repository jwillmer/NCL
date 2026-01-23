"""Tests for EML parser."""

from pathlib import Path
from unittest.mock import patch

from mtss.parsers.eml_parser import EMLParser


class TestEMLParser:
    """Tests for EMLParser class."""

    def test_parse_simple_email(self, simple_eml_file, temp_dir, mock_settings):
        """Test parsing a simple email without attachments."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(simple_eml_file)

            assert result.metadata.subject == "Simple Test Email"
            assert "sender@example.com" in result.metadata.participants
            assert "recipient@example.com" in result.metadata.participants
            assert result.full_text is not None
            assert "simple test email" in result.full_text.lower()
            assert len(result.attachments) == 0

    def test_parse_email_with_attachment(self, sample_eml_file, temp_dir, mock_settings):
        """Test parsing an email with attachments."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(sample_eml_file)

            assert result.metadata.subject == "Test Email"
            assert "sender@example.com" in result.metadata.participants
            assert len(result.attachments) == 1
            assert result.attachments[0].filename == "test.pdf"
            assert result.attachments[0].content_type == "application/pdf"
            assert Path(result.attachments[0].saved_path).exists()

    def test_html_to_plain_text(self, temp_dir, mock_settings):
        """Test HTML to plain text conversion."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            html = "<html><body><p>Hello</p><br><div>World</div></body></html>"
            result = parser.html_to_plain_text(html)

            assert "Hello" in result
            assert "World" in result

    def test_sanitize_filename(self, temp_dir, mock_settings):
        """Test filename sanitization."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            # Test various problematic filenames
            assert parser._sanitize_filename("normal.pdf") == "normal.pdf"
            assert parser._sanitize_filename("file/with/slashes.pdf") == "file_with_slashes.pdf"
            assert parser._sanitize_filename("file<>:test.pdf") == "file___test.pdf"
            assert parser._sanitize_filename("a" * 300) == "a" * 255  # Length limit

    def test_parse_address_list(self, temp_dir, mock_settings):
        """Test parsing comma-separated email addresses."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            assert parser._parse_address_list("") == []
            assert parser._parse_address_list("a@b.com") == ["a@b.com"]
            assert parser._parse_address_list("a@b.com, c@d.com") == ["a@b.com", "c@d.com"]

    def test_extract_email_address(self, temp_dir, mock_settings):
        """Test email address extraction from various formats."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            assert parser._extract_email_address("john@example.com") == "john@example.com"
            assert parser._extract_email_address("John Doe <john@example.com>") == "john@example.com"
            assert parser._extract_email_address("") is None
            assert parser._extract_email_address("invalid") is None

    def test_get_body_text_returns_full_text(self, temp_dir, mock_settings):
        """Test that get_body_text returns full_text field."""
        from mtss.models.document import EmailMetadata, ParsedEmail

        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")

            email = ParsedEmail(
                metadata=EmailMetadata(subject="Test", participants=["test@test.com"]),
                full_text="Full conversation text",
            )

            result = parser.get_body_text(email)
            assert result == "Full conversation text"

    def test_conversation_metadata(self, simple_eml_file, temp_dir, mock_settings):
        """Test that conversation metadata is extracted correctly."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(simple_eml_file)

            # Should have participants
            assert len(result.metadata.participants) >= 1
            # Should have message count
            assert result.metadata.message_count >= 1
            # Should have initiator (for single email, same as sender)
            assert result.metadata.initiator is not None


class TestRealEmailParsing:
    """Tests using the real test email with multiple attachment types."""

    def test_parse_real_email_metadata(self, real_eml_file, temp_dir, mock_settings):
        """Test parsing extracts correct metadata from real email."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(real_eml_file)

            assert result.metadata.subject is not None
            assert "TEST" in result.metadata.subject.upper() or "VESSEL" in result.metadata.subject.upper()
            assert len(result.metadata.participants) >= 2
            assert result.full_text is not None
            assert len(result.full_text) > 100  # Should have substantial body text

    def test_parse_real_email_extracts_all_attachments(self, real_eml_file, temp_dir, mock_settings):
        """Test that all 3 attachments (PDF, ZIP, PNG) are extracted."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(real_eml_file)

            assert len(result.attachments) == 3

            # Collect attachment info
            filenames = [a.filename.lower() for a in result.attachments]
            content_types = [a.content_type.lower() for a in result.attachments]

            # Check we have each type
            assert any(".pdf" in f for f in filenames), "PDF attachment not found"
            assert any(".zip" in f for f in filenames), "ZIP attachment not found"
            assert any(".png" in f for f in filenames), "PNG attachment not found"

            # Check content types
            assert any("pdf" in ct for ct in content_types), "PDF content-type not found"
            assert any("zip" in ct for ct in content_types), "ZIP content-type not found"
            assert any("png" in ct or "image" in ct for ct in content_types), "PNG content-type not found"

    def test_real_email_attachments_are_saved(self, real_eml_file, temp_dir, mock_settings):
        """Test that attachments are saved to disk and readable."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(real_eml_file)

            for attachment in result.attachments:
                saved_path = Path(attachment.saved_path)
                assert saved_path.exists(), f"Attachment not saved: {attachment.filename}"
                assert saved_path.stat().st_size > 0, f"Attachment empty: {attachment.filename}"

    def test_real_email_pdf_attachment_valid(self, real_eml_file, temp_dir, mock_settings):
        """Test that the PDF attachment is a valid PDF file."""
        with patch("mtss.parsers.eml_parser.get_settings", return_value=mock_settings):
            parser = EMLParser(attachments_dir=temp_dir / "attachments")
            result = parser.parse_file(real_eml_file)

            pdf_attachments = [a for a in result.attachments if ".pdf" in a.filename.lower()]
            assert len(pdf_attachments) >= 1

            pdf_path = Path(pdf_attachments[0].saved_path)
            content = pdf_path.read_bytes()
            assert content.startswith(b"%PDF"), "PDF file does not have valid PDF header"
