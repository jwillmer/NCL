"""Tests for the AttachmentProcessor class."""

import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from mtss.parsers.attachment_processor import AttachmentProcessor


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.enable_ocr = False
    settings.enable_picture_description = False
    settings.embedding_model = "text-embedding-3-small"
    settings.chunk_size_tokens = 512
    settings.zip_max_depth = 3
    settings.zip_max_files = 100
    settings.zip_max_total_size_mb = 100
    return settings


@pytest.fixture
def processor(mock_settings):
    """Create an AttachmentProcessor with mocked settings."""
    with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
        proc = AttachmentProcessor()
        # Store mock_settings on processor for use in tests that need runtime patching
        proc._test_mock_settings = mock_settings
        yield proc


class TestIsZipFile:
    """Tests for is_zip_file method."""

    def test_is_zip_by_content_type(self, processor):
        """Test ZIP detection by content type."""
        assert processor.is_zip_file("test.zip", "application/zip") is True
        assert processor.is_zip_file("test.zip", "application/x-zip-compressed") is True
        assert processor.is_zip_file("test.zip", "application/x-zip") is True

    def test_is_zip_by_extension(self, processor):
        """Test ZIP detection by file extension."""
        assert processor.is_zip_file("test.zip") is True
        assert processor.is_zip_file("TEST.ZIP") is True
        assert processor.is_zip_file("archive.zip") is True

    def test_not_zip_by_extension(self, processor):
        """Test non-ZIP files are not detected as ZIP."""
        assert processor.is_zip_file("test.pdf") is False
        assert processor.is_zip_file("test.docx") is False
        assert processor.is_zip_file("test.txt") is False

    def test_is_zip_by_magic(self, processor):
        """Test ZIP detection by file magic bytes.

        Note: is_zip_file() does not check magic bytes, only extension and MIME type.
        A .bin file without ZIP extension/MIME type will not be detected as ZIP.
        """
        with TemporaryDirectory() as tmpdir:
            # Create a valid ZIP file with .zip extension
            zip_path = Path(tmpdir) / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("test.txt", "Hello World")

            assert processor.is_zip_file(str(zip_path)) is True

    def test_not_zip_invalid_file(self, processor):
        """Test non-ZIP file is not detected as ZIP."""
        with TemporaryDirectory() as tmpdir:
            # Create a non-ZIP file
            txt_path = Path(tmpdir) / "test.bin"
            txt_path.write_text("This is not a ZIP file")

            assert processor.is_zip_file(str(txt_path)) is False


class TestExtractZip:
    """Tests for extract_zip method."""

    def test_extract_simple_zip(self, processor, mock_settings):
        """Test extracting a simple ZIP with supported files."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a ZIP file with a text file (not supported by Docling but we'll test structure)
            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                # Add a PDF-like file (mock)
                zf.writestr("document.pdf", b"%PDF-1.4 mock content")
                zf.writestr("image.png", b"\x89PNG mock content")

            # Extract with patched settings
            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(zip_path)

            # Should have extracted 2 files (PDF and PNG are supported)
            assert len(extracted) == 2

            # Check extracted files exist
            for file_path, content_type in extracted:
                assert file_path.exists()

    def test_extract_zip_with_nested_zip(self, processor, mock_settings):
        """Test extracting a ZIP containing another ZIP."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create inner ZIP
            inner_zip_path = tmpdir / "inner.zip"
            with zipfile.ZipFile(inner_zip_path, "w") as zf:
                zf.writestr("nested_doc.pdf", b"%PDF-1.4 nested content")

            # Create outer ZIP containing inner ZIP
            outer_zip_path = tmpdir / "outer.zip"
            with zipfile.ZipFile(outer_zip_path, "w") as zf:
                zf.write(inner_zip_path, "inner.zip")
                zf.writestr("top_level.pdf", b"%PDF-1.4 top level content")

            # Extract with patched settings
            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(outer_zip_path)

            # Should have extracted files from both levels
            # 1 from outer (top_level.pdf) + 1 from inner (nested_doc.pdf)
            assert len(extracted) == 2

    def test_extract_zip_skips_hidden_files(self, processor, mock_settings):
        """Test that hidden files are skipped during extraction."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(".hidden_file.pdf", b"%PDF-1.4 hidden")
                zf.writestr("__MACOSX/resource_fork", b"mac stuff")
                zf.writestr("visible.pdf", b"%PDF-1.4 visible")

            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(zip_path)

            # Should only have extracted the visible file
            assert len(extracted) == 1
            assert extracted[0][0].name == "visible.pdf"

    def test_extract_zip_skips_path_traversal(self, processor, mock_settings):
        """Test that path traversal attacks are prevented."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            zip_path = tmpdir / "malicious.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                # Attempt path traversal
                zf.writestr("../../../etc/passwd", b"malicious content")
                zf.writestr("safe.pdf", b"%PDF-1.4 safe content")

            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(zip_path)

            # Should only have extracted the safe file
            assert len(extracted) == 1
            assert extracted[0][0].name == "safe.pdf"

            # Verify no files were created outside extraction directory
            extract_dir = zip_path.parent / "malicious_extracted"
            for file_path, _ in extracted:
                assert str(file_path).startswith(str(extract_dir))

    def test_extract_zip_custom_extract_dir(self, processor, mock_settings):
        """Test extracting to a custom directory."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            zip_path = tmpdir / "test.zip"
            custom_dir = tmpdir / "custom_extract"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("document.pdf", b"%PDF-1.4 content")

            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(zip_path, extract_dir=custom_dir)

            assert len(extracted) == 1
            assert str(extracted[0][0]).startswith(str(custom_dir))

    def test_extract_zip_preserves_directory_structure(self, processor, mock_settings):
        """Test that directory structure is preserved during extraction."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            zip_path = tmpdir / "structured.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("folder1/doc1.pdf", b"%PDF-1.4 doc1")
                zf.writestr("folder1/subfolder/doc2.pdf", b"%PDF-1.4 doc2")
                zf.writestr("folder2/doc3.pdf", b"%PDF-1.4 doc3")

            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                extracted = processor.extract_zip(zip_path)

            assert len(extracted) == 3

            # Check directory structure is preserved
            paths = [str(p) for p, _ in extracted]
            assert any("folder1" in p and "doc1.pdf" in p for p in paths)
            assert any("subfolder" in p and "doc2.pdf" in p for p in paths)
            assert any("folder2" in p and "doc3.pdf" in p for p in paths)

    def test_extract_zip_nonexistent_file(self, processor, mock_settings):
        """Test extracting a non-existent ZIP file raises error."""
        with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
            with pytest.raises(FileNotFoundError):
                processor.extract_zip(Path("/nonexistent/path.zip"))

    def test_extract_zip_invalid_zip(self, processor, mock_settings):
        """Test extracting an invalid ZIP file raises error."""
        with TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "not_a_zip.zip"
            invalid_path.write_text("This is not a ZIP file")

            with patch("mtss.parsers.attachment_processor.get_settings", return_value=mock_settings):
                with pytest.raises(ValueError):
                    processor.extract_zip(invalid_path)


class TestIsDangerousZipPath:
    """Tests for _is_dangerous_zip_path method."""

    def test_path_traversal(self, processor):
        """Test detection of path traversal."""
        assert processor._is_dangerous_zip_path("../etc/passwd") is True
        assert processor._is_dangerous_zip_path("folder/../../../etc/passwd") is True

    def test_absolute_unix_path(self, processor):
        """Test detection of absolute Unix paths."""
        assert processor._is_dangerous_zip_path("/etc/passwd") is True
        assert processor._is_dangerous_zip_path("/home/user/file.txt") is True

    def test_absolute_windows_path(self, processor):
        """Test detection of absolute Windows paths."""
        assert processor._is_dangerous_zip_path("C:\\Windows\\System32") is True
        assert processor._is_dangerous_zip_path("D:\\Data\\file.txt") is True

    def test_safe_paths(self, processor):
        """Test that safe paths are not flagged."""
        assert processor._is_dangerous_zip_path("folder/file.txt") is False
        assert processor._is_dangerous_zip_path("document.pdf") is False
        assert processor._is_dangerous_zip_path("deeply/nested/folder/file.pdf") is False


class TestSanitizeZipMemberPath:
    """Tests for _sanitize_zip_member_path method."""

    def test_normalize_separators(self, processor):
        """Test that backslashes are converted to forward slashes."""
        result = processor._sanitize_zip_member_path("folder\\subfolder\\file.pdf")
        assert "\\" not in result
        assert "folder" in result and "subfolder" in result

    def test_remove_leading_slashes(self, processor):
        """Test that leading slashes are removed."""
        result = processor._sanitize_zip_member_path("/folder/file.pdf")
        assert not result.startswith("/")

    def test_remove_path_traversal(self, processor):
        """Test that path traversal components are removed."""
        result = processor._sanitize_zip_member_path("folder/../other/file.pdf")
        assert ".." not in result


class TestIsSupported:
    """Tests for is_supported method.

    Note: is_supported() checks the parser registry for document parsers.
    Images are handled separately via is_image_format().
    """

    def test_supported_by_content_type(self, processor):
        """Test supported file detection by content type."""
        assert processor.is_supported("test.pdf", "application/pdf") is True
        assert processor.is_supported("test.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document") is True

    def test_supported_by_extension(self, processor):
        """Test supported file detection by extension."""
        assert processor.is_supported("document.pdf") is True
        assert processor.is_supported("document.docx") is True
        assert processor.is_supported("document.xlsx") is True

    def test_unsupported_files(self, processor):
        """Test unsupported file detection."""
        assert processor.is_supported("script.py") is False
        assert processor.is_supported("data.json") is False
        assert processor.is_supported("archive.tar.gz") is False
