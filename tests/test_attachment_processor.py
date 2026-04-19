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


class TestGeminiFallback:
    """parse_to_text should fall back to Gemini when a local parser raises
    EmptyContentError. LlamaParse is no longer the generic fallback."""

    @pytest.fixture
    def tmp_docx(self, tmp_path):
        p = tmp_path / "empty.docx"
        p.write_bytes(b"PK\x03\x04fake-docx")
        return p

    @pytest.mark.asyncio
    async def test_fallback_to_gemini_on_empty_content(self, processor, tmp_docx):
        from unittest.mock import AsyncMock

        from mtss.parsers.base import EmptyContentError

        local = MagicMock()
        local.name = "local_docx"
        local.parse = AsyncMock(side_effect=EmptyContentError("empty"))

        gemini = MagicMock()
        gemini.name = "gemini_pdf"
        gemini.model_name = "openrouter/google/gemini-2.5-flash"
        gemini.is_available = True
        gemini.parse = AsyncMock(return_value="# Extracted\n\nReal content from Gemini.")

        with patch.object(processor, "_get_tiered_parser", return_value=local):
            with patch("mtss.parsers.gemini_pdf_parser.GeminiPDFParser", return_value=gemini):
                text, parser_name, parser_model = await processor.parse_to_text(tmp_docx)

        local.parse.assert_awaited_once()
        gemini.parse.assert_awaited_once()
        assert text.startswith("# Extracted")
        assert parser_name == "gemini_pdf"
        assert parser_model == "openrouter/google/gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_no_fallback_when_gemini_unavailable(self, processor, tmp_docx):
        from unittest.mock import AsyncMock

        from mtss.parsers.base import EmptyContentError

        local = MagicMock()
        local.name = "local_docx"
        local.parse = AsyncMock(side_effect=EmptyContentError("empty"))

        gemini = MagicMock()
        gemini.is_available = False
        gemini.parse = AsyncMock()

        with patch.object(processor, "_get_tiered_parser", return_value=local):
            with patch("mtss.parsers.gemini_pdf_parser.GeminiPDFParser", return_value=gemini):
                with pytest.raises(EmptyContentError):
                    await processor.parse_to_text(tmp_docx)

        gemini.parse.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_fallback_on_non_empty_value_error(self, processor, tmp_docx):
        """Non-empty ValueError (e.g. corrupted file) should propagate, not fall back."""
        from unittest.mock import AsyncMock

        local = MagicMock()
        local.name = "local_docx"
        local.parse = AsyncMock(side_effect=ValueError("Local DOCX parsing failed: corrupt"))

        gemini = MagicMock()
        gemini.is_available = True
        gemini.parse = AsyncMock()

        with patch.object(processor, "_get_tiered_parser", return_value=local):
            with patch("mtss.parsers.gemini_pdf_parser.GeminiPDFParser", return_value=gemini):
                with pytest.raises(ValueError, match="corrupt"):
                    await processor.parse_to_text(tmp_docx)

        gemini.parse.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_is_gemini(self, processor, tmp_docx):
        """If the primary parser is Gemini itself and it raises EmptyContentError,
        don't recurse."""
        from unittest.mock import AsyncMock

        from mtss.parsers.base import EmptyContentError

        gemini_primary = MagicMock()
        gemini_primary.name = "gemini_pdf"
        gemini_primary.parse = AsyncMock(side_effect=EmptyContentError("empty"))

        with patch.object(processor, "_get_tiered_parser", return_value=gemini_primary):
            with pytest.raises(EmptyContentError):
                await processor.parse_to_text(tmp_docx)


class TestLegacyOfficeRoutesToLlamaParse:
    """Only .doc / .xls / .ppt still route to LlamaParse in the tiered router."""

    def test_doc_routes_to_llamaparse(self, processor, tmp_path):
        from mtss.parsers.llamaparse_parser import LlamaParseParser

        p = tmp_path / "legacy.doc"
        p.write_bytes(b"fake-doc")

        # Force is_available=True so the router returns the parser instance.
        with patch.object(LlamaParseParser, "is_available", True):
            parser = processor._get_tiered_parser(p, "application/msword")
        assert parser is not None
        assert parser.__class__.__name__ == "LlamaParseParser"

    def test_xls_routes_to_llamaparse(self, processor, tmp_path):
        from mtss.parsers.llamaparse_parser import LlamaParseParser

        p = tmp_path / "legacy.xls"
        p.write_bytes(b"fake-xls")
        with patch.object(LlamaParseParser, "is_available", True):
            parser = processor._get_tiered_parser(p, "application/vnd.ms-excel")
        assert parser is not None
        assert parser.__class__.__name__ == "LlamaParseParser"

    def test_ppt_routes_to_llamaparse(self, processor, tmp_path):
        from mtss.parsers.llamaparse_parser import LlamaParseParser

        p = tmp_path / "legacy.ppt"
        p.write_bytes(b"fake-ppt")
        with patch.object(LlamaParseParser, "is_available", True):
            parser = processor._get_tiered_parser(p, "application/vnd.ms-powerpoint")
        assert parser is not None
        assert parser.__class__.__name__ == "LlamaParseParser"


class TestComplexPdfRoutesToGemini:
    """PDFs classified COMPLEX should route to Gemini (not LlamaParse)."""

    def test_complex_pdf_returns_gemini(self, processor, tmp_path):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser
        from mtss.parsers.pdf_classifier import PDFComplexity

        p = tmp_path / "scanned.pdf"
        p.write_bytes(b"%PDF-1.4\n")

        with patch(
            "mtss.parsers.attachment_processor.classify_pdf",
            return_value=PDFComplexity.COMPLEX,
        ) if False else patch(
            "mtss.parsers.pdf_classifier.classify_pdf",
            return_value=PDFComplexity.COMPLEX,
        ), patch.object(GeminiPDFParser, "is_available", True):
            parser = processor._get_tiered_parser(p, "application/pdf")
        assert parser is not None
        assert parser.__class__.__name__ == "GeminiPDFParser"


class TestLlamaParseParseCallKwargs:
    """Lock the exact kwarg shape passed to ``client.parsing.parse``.

    Regression: 2026-04-17 1000-email run silently failed 916 / 919 PDFs with
    ``TypeError: AsyncParsingResource.parse() got an unexpected keyword
    argument 'cost_optimizer'`` because ``cost_optimizer`` had been passed at
    the top level. The llama-cloud 2.x SDK accepts it only inside
    ``processing_options``. None of the fallback tests noticed because they
    mock ``LlamaParseParser`` whole instead of the underlying SDK.

    These tests mock the SDK client directly and cross-check every outbound
    kwarg against the real ``AsyncLlamaCloud().parsing.parse`` signature, so
    any future SDK migration that drops a kwarg we still pass fails in CI
    instead of in production.
    """

    @pytest.fixture
    def mock_llamaparse_settings(self):
        settings = MagicMock()
        settings.llamaparse_enabled = True
        settings.llama_cloud_api_key = "test-key"
        settings.max_concurrent_llamaparse = 2
        return settings

    @pytest.fixture
    def sdk_parse_param_names(self) -> set:
        """The real parameter names accepted by the installed SDK."""
        import inspect

        from llama_cloud import AsyncLlamaCloud

        client = AsyncLlamaCloud(api_key="placeholder")
        return set(inspect.signature(client.parsing.parse).parameters)

    @pytest.mark.asyncio
    async def test_cost_optimizer_nested_in_processing_options(
        self, tmp_path, mock_llamaparse_settings
    ):
        from unittest.mock import AsyncMock

        from mtss.parsers import llamaparse_parser
        from mtss.parsers.llamaparse_parser import LlamaParseParser

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        fake_result = MagicMock()
        fake_result.markdown.pages = [MagicMock(markdown="# Title\n\nContent.")]

        fake_client = MagicMock()
        fake_client.parsing.parse = AsyncMock(return_value=fake_result)

        with patch.object(llamaparse_parser, "_get_llamaparse_client", return_value=fake_client), \
             patch.object(llamaparse_parser, "_llamaparse_semaphore", None), \
             patch.object(llamaparse_parser, "get_settings", return_value=mock_llamaparse_settings), \
             patch("mtss.parsers.llamaparse_parser.get_settings", return_value=mock_llamaparse_settings):
            parser = LlamaParseParser()
            parser.settings = mock_llamaparse_settings
            result = await parser.parse(pdf)

        assert "Content" in result
        fake_client.parsing.parse.assert_awaited_once()
        _, kwargs = fake_client.parsing.parse.call_args

        assert "cost_optimizer" not in kwargs, (
            "cost_optimizer must NOT be a top-level kwarg — llama-cloud 2.x "
            "rejects it and every PDF silently fell back to extraction_failed."
        )
        assert kwargs.get("processing_options", {}).get("cost_optimizer") == {"enable": True}, (
            "cost_optimizer must sit inside processing_options per the SDK 2.x "
            "TypedDict contract."
        )

    @pytest.mark.asyncio
    async def test_every_outbound_kwarg_is_in_sdk_signature(
        self, tmp_path, mock_llamaparse_settings, sdk_parse_param_names
    ):
        """The strongest guard: whatever we pass must exist in the installed
        SDK's signature. Drops us out of "silently-caught-as-warning" territory
        the next time the llama-cloud SDK renames or removes a parameter.
        """
        from unittest.mock import AsyncMock

        from mtss.parsers import llamaparse_parser
        from mtss.parsers.llamaparse_parser import LlamaParseParser

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        fake_result = MagicMock()
        fake_result.markdown.pages = [MagicMock(markdown="content")]
        fake_client = MagicMock()
        fake_client.parsing.parse = AsyncMock(return_value=fake_result)

        with patch.object(llamaparse_parser, "_get_llamaparse_client", return_value=fake_client), \
             patch.object(llamaparse_parser, "_llamaparse_semaphore", None), \
             patch("mtss.parsers.llamaparse_parser.get_settings", return_value=mock_llamaparse_settings):
            parser = LlamaParseParser()
            parser.settings = mock_llamaparse_settings
            await parser.parse(pdf)

        _, kwargs = fake_client.parsing.parse.call_args
        unknown = set(kwargs) - sdk_parse_param_names
        assert not unknown, (
            f"LlamaParseParser passes kwargs the installed SDK rejects: {sorted(unknown)}"
        )


class TestLocalParserEmptyContentError:
    """Local parsers must raise EmptyContentError (not plain ValueError) when they
    open the file successfully but extract no content — enables fallback routing."""

    @pytest.mark.asyncio
    async def test_docx_empty_raises_empty_content_error(self, tmp_path):
        from mtss.parsers.base import EmptyContentError
        from mtss.parsers.local_office_parser import LocalDocxParser

        docx_path = tmp_path / "empty.docx"
        docx_path.write_bytes(b"x")

        with patch("docx.Document") as mock_docx:
            fake_doc = MagicMock()
            fake_doc.paragraphs = []
            fake_doc.tables = []
            mock_docx.return_value = fake_doc

            parser = LocalDocxParser()
            with pytest.raises(EmptyContentError):
                await parser.parse(docx_path)

    @pytest.mark.asyncio
    async def test_xlsx_empty_raises_empty_content_error(self, tmp_path):
        from mtss.parsers.base import EmptyContentError
        from mtss.parsers.local_office_parser import LocalXlsxParser

        xlsx_path = tmp_path / "empty.xlsx"
        xlsx_path.write_bytes(b"x")

        with patch("openpyxl.load_workbook") as mock_wb:
            fake_wb = MagicMock()
            fake_wb.sheetnames = []
            mock_wb.return_value = fake_wb

            parser = LocalXlsxParser()
            with pytest.raises(EmptyContentError):
                await parser.parse(xlsx_path)

    @pytest.mark.asyncio
    async def test_pdf_empty_raises_empty_content_error(self, tmp_path):
        from mtss.parsers.base import EmptyContentError
        from mtss.parsers.local_pdf_parser import LocalPDFParser

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("pymupdf4llm.to_markdown", return_value=""):
            parser = LocalPDFParser()
            with pytest.raises(EmptyContentError):
                await parser.parse(pdf_path)

    @pytest.mark.asyncio
    async def test_empty_content_error_is_value_error_subclass(self):
        """EmptyContentError must be a ValueError subclass so existing handlers
        (BaseParser contract) still catch it when they don't care about fallback."""
        from mtss.parsers.base import EmptyContentError

        assert issubclass(EmptyContentError, ValueError)


class TestImageMimeDocTypeParity:
    """Every MIME type the image/vision pipeline accepts must also resolve to
    ``DocumentType.ATTACHMENT_IMAGE`` via ``AttachmentProcessor.MIME_TO_DOC_TYPE``.

    Regression: 2026-04-17 validate run flagged 2 ``.gif`` attachments as
    "text chunks missing context_summary/embedding_text". Their content was
    a valid image description — the chunks were fine, but their parent docs
    had been typed ``attachment_other`` because ``image/gif`` was absent from
    ``MIME_TO_DOC_TYPE``. Check 7 (``_check_context_summary``) excludes
    ``attachment_image`` chunks by design, so the mislabel leaked them into
    the check. WEBP had the same gap even though ``ImageProcessor`` and
    ``lane_classifier`` already supported it.

    This test anchors the three registries to the same set so a future
    contributor adding HEIC, AVIF, SVG, etc. to vision can't forget one spot.
    """

    def test_every_vision_supported_image_maps_to_attachment_image(self):
        from mtss.models.document import DocumentType
        from mtss.parsers.attachment_processor import AttachmentProcessor
        from mtss.processing.image_processor import ImageProcessor

        missing = [
            mt
            for mt in ImageProcessor.SUPPORTED_TYPES
            if AttachmentProcessor.MIME_TO_DOC_TYPE.get(mt)
            is not DocumentType.ATTACHMENT_IMAGE
        ]
        assert missing == [], (
            "ImageProcessor.SUPPORTED_TYPES must be a subset of "
            "AttachmentProcessor.MIME_TO_DOC_TYPE with ATTACHMENT_IMAGE; "
            f"drifted: {missing}"
        )

    def test_lane_classifier_image_mimetypes_match_vision_support(self):
        from mtss.ingest.lane_classifier import IMAGE_MIMETYPES
        from mtss.processing.image_processor import ImageProcessor

        assert IMAGE_MIMETYPES == ImageProcessor.SUPPORTED_TYPES, (
            "lane_classifier.IMAGE_MIMETYPES must stay in lockstep with "
            "ImageProcessor.SUPPORTED_TYPES — a drift routes images through "
            "the slow (LlamaParse) lane unnecessarily."
        )

    def test_mime_format_map_covers_every_supported_image(self):
        from mtss.ingest.helpers import MIME_FORMAT_MAP
        from mtss.processing.image_processor import ImageProcessor

        missing = [mt for mt in ImageProcessor.SUPPORTED_TYPES if mt not in MIME_FORMAT_MAP]
        assert missing == [], (
            f"MIME_FORMAT_MAP missing display names for: {missing}"
        )

    def test_get_document_type_gif_and_webp(self):
        from mtss.models.document import DocumentType
        from mtss.parsers.attachment_processor import AttachmentProcessor

        processor = AttachmentProcessor()
        assert processor.get_document_type("image/gif") is DocumentType.ATTACHMENT_IMAGE
        assert processor.get_document_type("image/webp") is DocumentType.ATTACHMENT_IMAGE


class TestPdfClassifierLoosened:
    """classify_pdf must not flag text-layer PDFs complex just because they
    contain an embedded image. Scanned PDFs and form PDFs still route to
    LlamaParse; logo-bearing invoices and reports now go local."""

    def _mock_reader(self, *, pages_text, has_fields=False, page_has_image=False):
        from unittest.mock import MagicMock

        reader = MagicMock()
        reader.get_fields.return_value = {"x": 1} if has_fields else None
        pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            if page_has_image:
                xobj = MagicMock()
                xobj.get.return_value = "/Image"
                xobjects = {"Im0": MagicMock()}
                xobjects["Im0"].get_object.return_value = xobj
                resources_xobject = MagicMock()
                resources_xobject.get_object.return_value = xobjects
                page.get.return_value = {"/XObject": resources_xobject}
            else:
                page.get.return_value = {}
            pages.append(page)
        reader.pages = pages
        return reader

    def test_text_layer_with_embedded_image_is_simple(self, tmp_path):
        """Logos/stamps inside a text-layer PDF should no longer force LlamaParse."""
        from unittest.mock import patch

        from mtss.parsers.pdf_classifier import PDFComplexity, classify_pdf

        pdf = tmp_path / "invoice_with_logo.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        reader = self._mock_reader(
            pages_text=["Invoice header with plenty of readable text for classification."] * 3,
            page_has_image=True,
        )
        with patch("pypdf.PdfReader", return_value=reader):
            assert classify_pdf(pdf) == PDFComplexity.SIMPLE

    def test_scanned_page_is_still_complex(self, tmp_path):
        from unittest.mock import patch

        from mtss.parsers.pdf_classifier import PDFComplexity, classify_pdf

        pdf = tmp_path / "scanned.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        reader = self._mock_reader(pages_text=["", ""])  # no text layer
        with patch("pypdf.PdfReader", return_value=reader):
            assert classify_pdf(pdf) == PDFComplexity.COMPLEX

    def test_form_pdf_is_still_complex(self, tmp_path):
        from unittest.mock import patch

        from mtss.parsers.pdf_classifier import PDFComplexity, classify_pdf

        pdf = tmp_path / "form.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        reader = self._mock_reader(
            pages_text=["Fillable form with plenty of static text content."],
            has_fields=True,
        )
        with patch("pypdf.PdfReader", return_value=reader):
            assert classify_pdf(pdf) == PDFComplexity.COMPLEX

    def test_estimator_classifier_agrees(self):
        """The estimator now delegates to parsers.pdf_classifier.classify_reader
        — confirm the shared helper still ranks a text-layer PDF as simple
        regardless of embedded images."""
        from mtss.parsers.pdf_classifier import PDFComplexity, classify_reader

        reader = self._mock_reader(
            pages_text=["Operational report with a plain text layer that easily clears the 50-char threshold."] * 2,
            page_has_image=True,
        )
        assert classify_reader(reader) == PDFComplexity.SIMPLE


class TestGeminiTimeoutPeekFallback:
    """When the Gemini parser blows its doc timeout, parse_to_text must fall
    back to the local PyMuPDF4LLM peek so the attachment still lands as
    something (decider will classify as full/metadata_only) rather than
    failing the whole doc. This is the safety net for dense-scanned PDFs
    whose peek happens to show prose (so the pre-Gemini bypass doesn't
    fire) but Gemini can't actually extract them."""

    @pytest.mark.asyncio
    async def test_gemini_timeout_falls_back_to_local_peek(
        self, tmp_path, comprehensive_mock_settings
    ):
        from mtss.parsers.attachment_processor import AttachmentProcessor

        pdf = tmp_path / "slow.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        peek_text = "# Scanned Report\n\nLocal peek extracted this."

        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        async def _raise_timeout(_self, _path):
            import asyncio as _asyncio
            raise _asyncio.TimeoutError("doc timeout")

        with patch(
            "mtss.parsers.attachment_processor.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch.object(
            AttachmentProcessor,
            "_get_tiered_parser",
            return_value=GeminiPDFParser(),
        ), patch.object(
            GeminiPDFParser, "parse", new=_raise_timeout
        ), patch(
            "mtss.parsers.preprocessor._peek_pdf_markdown", return_value=peek_text
        ):
            text, parser_name, _ = await AttachmentProcessor().parse_to_text(
                pdf, "application/pdf"
            )

        assert peek_text in text
        assert parser_name == "gemini_pdf_timeout_peek_fallback"


class TestPreprocessorPdfPageLimit:
    """Preprocessor diverts oversized PDFs to a cheap local peek so the decider
    can classify them as SUMMARY/METADATA_ONLY without paying to parse the
    whole doc."""

    @pytest.mark.asyncio
    async def test_oversized_pdf_uses_peek_when_pymupdf_succeeds(self, tmp_path):
        """Peek succeeds → should_process=True with oversized_pdf flag +
        preview_markdown set; the full-parse route is skipped entirely."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "huge.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        preview = "# Report\n\nFirst-page preview text."
        with patch(
            "mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=818
        ), patch(
            "mtss.parsers.preprocessor._peek_pdf_markdown", return_value=preview
        ), patch(
            "mtss.parsers.preprocessor.get_settings", return_value=fake_settings
        ):
            result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        assert result.should_process is True
        assert result.oversized_pdf is True
        assert result.preview_markdown is not None
        assert preview in result.preview_markdown
        assert "818 pages" in result.preview_markdown
        assert result.total_pages == 818
        assert result.parser_name == "oversized_pdf_peek"

    @pytest.mark.asyncio
    async def test_complex_pdf_with_no_prose_peek_bypasses_gemini(self, tmp_path):
        """COMPLEX-classified PDFs whose local peek shows no prose must
        bypass Gemini entirely. The PMC-22 class of scanned forms: image-
        dominant, classifies COMPLEX, and Gemini would burn minutes on
        hallucinated output that the decider throws away as METADATA_ONLY."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        # 3-page peek returns a tiny image-dominant noise preview.
        noise_preview = "page 1\n\npage 2\n\npage 3"
        from mtss.parsers.pdf_classifier import PDFComplexity

        with patch(
            "mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=5
        ), patch(
            "mtss.parsers.preprocessor._peek_pdf_markdown", return_value=noise_preview
        ), patch(
            "mtss.parsers.preprocessor.get_settings", return_value=fake_settings
        ), patch(
            "mtss.parsers.pdf_classifier.classify_pdf",
            return_value=PDFComplexity.COMPLEX,
        ):
            result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        assert result.should_process is True
        assert result.oversized_pdf is True
        assert result.parser_name == "oversized_pdf_peek"
        assert result.preview_markdown is not None
        assert "no prose" in result.preview_markdown.lower()

    @pytest.mark.asyncio
    async def test_complex_pdf_with_prose_peek_still_routes_to_gemini(self, tmp_path):
        """Complement: a COMPLEX PDF whose peek shows real prose must NOT
        short-circuit — route through the normal parser so Gemini can parse
        the full document."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        # Realistic prose preview with headings — clears the noise threshold
        # (>=50 tokens with healthy prose_ratio and at least one heading).
        prose_preview = (
            "# Executive Summary\n\n"
            "The vessel performance during the quarter exceeded baseline "
            "expectations across all measured metrics. Fuel efficiency "
            "improved by four percent year over year. Maintenance incidents "
            "decreased significantly compared to the prior reporting period.\n\n"
            "## Detailed Findings\n\n"
            "Detailed analysis of operational parameters follows in the "
            "sections below. Each system was evaluated independently and "
            "results have been cross-checked against the prior quarter baseline."
        )
        from mtss.parsers.pdf_classifier import PDFComplexity

        with patch(
            "mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=5
        ), patch(
            "mtss.parsers.preprocessor._peek_pdf_markdown", return_value=prose_preview
        ), patch(
            "mtss.parsers.preprocessor.get_settings", return_value=fake_settings
        ), patch(
            "mtss.parsers.pdf_classifier.classify_pdf",
            return_value=PDFComplexity.COMPLEX,
        ):
            result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        # Falls through to the normal local-parser routing, not the peek path.
        assert result.should_process is True
        assert result.oversized_pdf is False
        assert result.preview_markdown is None

    @pytest.mark.asyncio
    async def test_oversized_pdf_skips_when_peek_fails(self, tmp_path):
        """If PyMuPDF can't even extract a preview, fall back to the old skip
        behavior so we don't route a truly corrupt file to the decider."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "corrupt_huge.pdf"
        pdf.write_bytes(b"not really a pdf")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        with patch(
            "mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=818
        ), patch(
            "mtss.parsers.preprocessor._peek_pdf_markdown", return_value=None
        ), patch(
            "mtss.parsers.preprocessor.get_settings", return_value=fake_settings
        ):
            result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        assert result.should_process is False
        assert result.skip_reason is not None
        assert "pdf_too_large_unreadable" in result.skip_reason
        assert "818" in result.skip_reason

    @pytest.mark.asyncio
    async def test_pdf_under_limit_routes_to_local(self, tmp_path):
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "ok.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        with patch("mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=5):
            with patch("mtss.parsers.preprocessor.get_settings", return_value=fake_settings):
                result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        assert result.should_process is True
        assert result.parser_name == "local"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_unreadable_pdf_does_not_block(self, tmp_path):
        """If pypdf can't parse the file at all, we let the parser layer fail
        with its own error rather than pre-skipping with a misleading reason."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"not really a pdf")

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 100 * 1024 * 1024

        with patch("mtss.parsers.preprocessor._safe_count_pdf_pages", return_value=None):
            with patch("mtss.parsers.preprocessor.get_settings", return_value=fake_settings):
                result = await DocumentPreprocessor().preprocess(pdf, "application/pdf")

        assert result.should_process is True
        assert result.parser_name == "local"

    @pytest.mark.asyncio
    async def test_attachment_over_byte_cap_is_skipped(self, tmp_path):
        """The attachment_max_bytes guard fires before any parser/route work,
        protecting Gemini from base64-loading a multi-GB attachment."""
        from unittest.mock import patch

        from mtss.parsers.preprocessor import DocumentPreprocessor

        big = tmp_path / "huge.pdf"
        big.write_bytes(b"%PDF-1.4\n" + b"x" * 2_000)  # 2 KB of payload

        fake_settings = MagicMock()
        fake_settings.pdf_max_pages = 40
        fake_settings.attachment_max_bytes = 1_000  # cap below file size

        with patch("mtss.parsers.preprocessor.get_settings", return_value=fake_settings):
            result = await DocumentPreprocessor().preprocess(big, "application/pdf")

        assert result.should_process is False
        assert result.skip_reason is not None
        assert "attachment_too_large" in result.skip_reason

    @pytest.mark.asyncio
    async def test_crdownload_partial_is_rejected(self, tmp_path):
        """Chrome partial-download suffix must be rejected before parser routing.

        These files are truncated bytes, not a valid PDF/DOCX/etc. Previously
        they fell through to the LlamaParse fallback which wasted API calls.
        """
        from mtss.parsers.preprocessor import DocumentPreprocessor

        partial = tmp_path / "report.pdf.crdownload"
        partial.write_bytes(b"%PDF-1.4\npartial bytes")

        result = await DocumentPreprocessor().preprocess(partial)

        assert result.should_process is False
        assert result.skip_reason is not None
        assert "partial_download" in result.skip_reason
