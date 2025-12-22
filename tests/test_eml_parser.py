"""Tests for EML parser."""

from pathlib import Path

import pytest

from ncl.parsers.eml_parser import EMLParser


class TestEMLParser:
    """Tests for EMLParser class."""

    def test_parse_simple_email(self, simple_eml_file, temp_dir):
        """Test parsing a simple email without attachments."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")
        result = parser.parse_file(simple_eml_file)

        assert result.metadata.subject == "Simple Test Email"
        assert "sender@example.com" in result.metadata.participants
        assert "recipient@example.com" in result.metadata.participants
        assert result.full_text is not None
        assert "simple test email" in result.full_text.lower()
        assert len(result.attachments) == 0

    def test_parse_email_with_attachment(self, sample_eml_file, temp_dir):
        """Test parsing an email with attachments."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")
        result = parser.parse_file(sample_eml_file)

        assert result.metadata.subject == "Test Email"
        assert "sender@example.com" in result.metadata.participants
        assert len(result.attachments) == 1
        assert result.attachments[0].filename == "test.pdf"
        assert result.attachments[0].content_type == "application/pdf"
        assert Path(result.attachments[0].saved_path).exists()

    def test_html_to_plain_text(self, temp_dir):
        """Test HTML to plain text conversion."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")

        html = "<html><body><p>Hello</p><br><div>World</div></body></html>"
        result = parser.html_to_plain_text(html)

        assert "Hello" in result
        assert "World" in result

    def test_sanitize_filename(self, temp_dir):
        """Test filename sanitization."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")

        # Test various problematic filenames
        assert parser._sanitize_filename("normal.pdf") == "normal.pdf"
        assert parser._sanitize_filename("file/with/slashes.pdf") == "file_with_slashes.pdf"
        assert parser._sanitize_filename("file<>:test.pdf") == "file___test.pdf"
        assert parser._sanitize_filename("a" * 300) == "a" * 255  # Length limit

    def test_parse_address_list(self, temp_dir):
        """Test parsing comma-separated email addresses."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")

        assert parser._parse_address_list("") == []
        assert parser._parse_address_list("a@b.com") == ["a@b.com"]
        assert parser._parse_address_list("a@b.com, c@d.com") == ["a@b.com", "c@d.com"]

    def test_extract_email_address(self, temp_dir):
        """Test email address extraction from various formats."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")

        assert parser._extract_email_address("john@example.com") == "john@example.com"
        assert parser._extract_email_address("John Doe <john@example.com>") == "john@example.com"
        assert parser._extract_email_address("") is None
        assert parser._extract_email_address("invalid") is None

    def test_get_body_text_returns_full_text(self, temp_dir):
        """Test that get_body_text returns full_text field."""
        from ncl.models.document import EmailMetadata, ParsedEmail

        parser = EMLParser(attachments_dir=temp_dir / "attachments")

        email = ParsedEmail(
            metadata=EmailMetadata(subject="Test", participants=["test@test.com"]),
            full_text="Full conversation text",
        )

        result = parser.get_body_text(email)
        assert result == "Full conversation text"

    def test_conversation_metadata(self, simple_eml_file, temp_dir):
        """Test that conversation metadata is extracted correctly."""
        parser = EMLParser(attachments_dir=temp_dir / "attachments")
        result = parser.parse_file(simple_eml_file)

        # Should have participants
        assert len(result.metadata.participants) >= 1
        # Should have message count
        assert result.metadata.message_count >= 1
        # Should have initiator (for single email, same as sender)
        assert result.metadata.initiator is not None
