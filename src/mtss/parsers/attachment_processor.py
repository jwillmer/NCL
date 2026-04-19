"""Attachment processor using parser plugins for document conversion and chunking."""

from __future__ import annotations

import logging
import mimetypes
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ..config import get_settings
from ..models.chunk import Chunk
from ..models.document import DocumentType
from ..processing.image_processor import ImageProcessor
from ..utils import sanitize_filename
from .chunker import DocumentChunker
from .preprocessor import DocumentPreprocessor, PreprocessResult
from .registry import ParserRegistry

logger = logging.getLogger(__name__)


class ZipExtractionError(Exception):
    """Raised when ZIP extraction fails due to limits or security checks."""

    pass


class ZipMemberExtractionError(Exception):
    """Raised when a ZIP member fails to extract.

    This indicates data loss from the ZIP archive and should be handled
    appropriately by the caller (either fail the document or log and continue
    in lenient mode).
    """

    def __init__(self, member: str, reason: str):
        self.member = member
        self.reason = reason
        super().__init__(f"Failed to extract ZIP member '{member}': {reason}")


class AttachmentProcessor:
    """Process attachments using parser plugins for text extraction and chunking.

    Routing lives in ``_get_tiered_parser``: simple PDFs → local PyMuPDF4LLM,
    complex PDFs → Gemini, legacy binary Office (.doc/.xls/.ppt) → LlamaParse,
    modern Office / CSV / HTML → local parsers.
    """

    # Mapping of MIME types to DocumentType.
    # Image set must mirror ImageProcessor.SUPPORTED_TYPES and
    # lane_classifier.IMAGE_MIMETYPES — otherwise a format reaches the
    # vision pipeline (producing an image_description chunk) while its
    # parent document stays classified as ATTACHMENT_OTHER, which
    # Check 7 (_check_context_summary) then flags for missing
    # context_summary/embedding_text even though image chunks skip that
    # enrichment by design.
    MIME_TO_DOC_TYPE: Dict[str, DocumentType] = {
        "application/pdf": DocumentType.ATTACHMENT_PDF,
        "image/png": DocumentType.ATTACHMENT_IMAGE,
        "image/x-png": DocumentType.ATTACHMENT_IMAGE,
        "image/jpeg": DocumentType.ATTACHMENT_IMAGE,
        "image/jpg": DocumentType.ATTACHMENT_IMAGE,
        "image/gif": DocumentType.ATTACHMENT_IMAGE,
        "image/webp": DocumentType.ATTACHMENT_IMAGE,
        "image/tiff": DocumentType.ATTACHMENT_IMAGE,
        "image/bmp": DocumentType.ATTACHMENT_IMAGE,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.ATTACHMENT_DOCX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.ATTACHMENT_PPTX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.ATTACHMENT_XLSX,
        "application/msword": DocumentType.ATTACHMENT_DOC,
        "application/vnd.ms-excel": DocumentType.ATTACHMENT_XLS,
        "application/vnd.ms-powerpoint": DocumentType.ATTACHMENT_PPT,
        "text/csv": DocumentType.ATTACHMENT_CSV,
        "application/rtf": DocumentType.ATTACHMENT_RTF,
        "text/rtf": DocumentType.ATTACHMENT_RTF,
    }

    # ZIP file MIME types
    ZIP_FORMATS: Dict[str, str] = {
        "application/zip": "ZIP",
        "application/x-zip-compressed": "ZIP",
        "application/x-zip": "ZIP",
    }

    # Office formats that are technically ZIP files but should NOT be treated as ZIPs
    OFFICE_ZIP_FORMATS: set = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "application/epub+zip",
        "application/java-archive",
    }

    # Extensions that are ZIP-based but should be processed as documents
    OFFICE_ZIP_EXTENSIONS: set = {
        ".docx",
        ".xlsx",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".epub",
        ".jar",
    }

    def __init__(self):
        """Initialize the attachment processor."""
        self._image_processor: Optional[ImageProcessor] = None
        self._chunker: Optional[DocumentChunker] = None
        self._preprocessor: Optional[DocumentPreprocessor] = None

    @property
    def image_processor(self) -> ImageProcessor:
        """Lazy initialization of ImageProcessor."""
        if self._image_processor is None:
            self._image_processor = ImageProcessor()
        return self._image_processor

    @property
    def chunker(self) -> DocumentChunker:
        """Lazy initialization of DocumentChunker."""
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def preprocessor(self) -> DocumentPreprocessor:
        """Lazy initialization of DocumentPreprocessor."""
        if self._preprocessor is None:
            self._preprocessor = DocumentPreprocessor()
        return self._preprocessor

    async def preprocess(
        self,
        file_path: Path,
        content_type: Optional[str] = None,
        classify_images: bool = True,
    ) -> PreprocessResult:
        """Preprocess a file to determine how it should be handled.

        This is the main entry point for determining file routing.
        For email-level images, set classify_images=True to filter logos/banners.
        For images from ZIPs/documents, set classify_images=False.

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type.
            classify_images: Whether to classify images (filter logos, etc.)

        Returns:
            PreprocessResult with routing decision and optional image description.
        """
        return await self.preprocessor.preprocess(file_path, content_type, classify_images)

    def is_image_format(self, content_type: Optional[str]) -> bool:
        """Check if content type is an image format supported by Vision API."""
        return self.image_processor.is_supported(content_type)

    # Extensions handled by local parsers (not in ParserRegistry)
    _LOCAL_PARSER_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".html", ".htm"}

    def is_supported(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file format is supported by any parser (registry or local).

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file (optional).

        Returns:
            True if format is supported.
        """
        path = Path(file_path)
        if path.suffix.lower() in self._LOCAL_PARSER_EXTENSIONS:
            return True
        parser = ParserRegistry.get_parser_for_file(path, content_type)
        return parser is not None

    def get_document_type(self, content_type: str) -> DocumentType:
        """Get DocumentType from MIME type.

        Args:
            content_type: MIME type string.

        Returns:
            Corresponding DocumentType enum value.
        """
        return self.MIME_TO_DOC_TYPE.get(content_type, DocumentType.ATTACHMENT_OTHER)

    async def parse_to_text(
        self,
        file_path: Path,
        content_type: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Run the tiered parser chain and return raw markdown + parser name.

        Keeps parsing and chunking separate so callers can run the embedding
        decider against the raw text before deciding how to chunk it.

        Returns:
            (markdown_text, effective_parser_name). Empty string if the parser
            successfully opened the file but extracted no content.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            EmptyContentError: If a local parser produced empty content AND no
                Gemini fallback is available.
            ValueError: If no parser available or parsing fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        parser = self._get_tiered_parser(file_path, content_type)
        if not parser:
            raise ValueError(f"No parser available for {file_path}")

        logger.info(f"Processing {file_path.name} with {parser.name} parser")
        from .base import EmptyContentError

        effective_parser_name = parser.name
        try:
            text = await parser.parse(file_path)
        except EmptyContentError:
            if not parser.name.startswith("local_"):
                raise
            from .gemini_pdf_parser import GeminiPDFParser

            fallback = GeminiPDFParser()
            if not fallback.is_available:
                raise
            logger.info(
                f"{parser.name} produced no content for {file_path.name}; falling back to Gemini"
            )
            text = await fallback.parse(file_path)
            effective_parser_name = fallback.name

        if effective_parser_name not in ("llamaparse", "gemini_pdf"):
            from ..cli._common import _service_counter
            _service_counter.add("local_parse")

        if not text or not text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return "", effective_parser_name

        text_len = len(text.strip())
        logger.info(f"Extracted {text_len} chars from {file_path.name}")
        if text_len < 50:
            logger.warning(
                f"Very little text from {file_path.name}: {repr(text.strip()[:100])}"
            )
        return text, effective_parser_name

    async def process_attachment(
        self,
        file_path: Path,
        document_id: UUID,
        content_type: Optional[str] = None,
    ) -> List[Chunk]:
        """Parse + chunk (legacy shape). Callers wanting to run the decider
        should use ``parse_to_text`` and then build chunks per-mode.
        """
        text, effective_parser_name = await self.parse_to_text(file_path, content_type)
        if not text:
            return []
        chunks = self.chunker.chunk_text(
            text=text,
            document_id=document_id,
            source_file=str(file_path),
            is_markdown=True,
            metadata={"parser": effective_parser_name},
        )
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks

    # Fallback when filename has no/mangled extension (e.g. trailing dot)
    _MIMETYPE_TO_EXT = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "text/csv": ".csv",
        "text/html": ".html",
    }

    # Legacy binary Office formats — the only remaining LlamaParse path.
    _LEGACY_OFFICE_EXTENSIONS: set[str] = {".doc", ".xls", ".ppt"}

    def _get_tiered_parser(self, file_path: Path, content_type: str | None = None):
        """Select parser using tiered routing.

        - .pdf simple  → PyMuPDF4LLM
        - .pdf complex → Gemini (OpenRouter)
        - .docx/.xlsx/.csv/.html → local parsers
        - .doc/.xls/.ppt → LlamaParse (legacy binary Office only)
        - otherwise → Gemini if available, else ParserRegistry fallback
        """
        ext = file_path.suffix.lower()
        if not ext or ext == ".":
            ext = self._MIMETYPE_TO_EXT.get(content_type or "", "")

        if ext == ".pdf":
            from .pdf_classifier import PDFComplexity, classify_pdf
            complexity = classify_pdf(file_path)
            if complexity == PDFComplexity.SIMPLE:
                from .local_pdf_parser import LocalPDFParser
                local = LocalPDFParser()
                if local.is_available:
                    return local
                logger.info(
                    f"PyMuPDF4LLM not available, escalating {file_path.name} to Gemini"
                )
            # COMPLEX PDF or local parser unavailable → Gemini.
            from .gemini_pdf_parser import GeminiPDFParser

            gemini = GeminiPDFParser()
            if gemini.is_available:
                return gemini
            logger.warning(
                f"Gemini PDF parser not available; no fallback for complex PDF {file_path.name}"
            )
            return None

        if ext == ".docx":
            from .local_office_parser import LocalDocxParser
            local = LocalDocxParser()
            if local.is_available:
                return local

        elif ext == ".xlsx":
            from .local_office_parser import LocalXlsxParser
            local = LocalXlsxParser()
            if local.is_available:
                return local

        elif ext == ".csv":
            from .local_office_parser import LocalCsvParser
            return LocalCsvParser()

        elif ext in (".html", ".htm"):
            from .local_office_parser import LocalHtmlParser
            return LocalHtmlParser()

        elif ext in self._LEGACY_OFFICE_EXTENSIONS:
            from .llamaparse_parser import LlamaParseParser

            llp = LlamaParseParser()
            if llp.is_available:
                return llp
            logger.warning(
                f"LlamaParse not available; cannot parse legacy binary {file_path.name}"
            )
            return None

        return ParserRegistry.get_parser_for_file(file_path, content_type)

    def create_image_chunk(
        self,
        file_path: Path,
        document_id: UUID,
        description: str,
        classification: Optional[str] = None,
    ) -> Chunk:
        """Create a chunk from an image description.

        Used after preprocessing has already classified and described the image.

        Args:
            file_path: Path to the image file.
            document_id: UUID of the parent document record.
            description: Pre-computed image description.
            classification: Optional classification value (e.g., "meaningful").

        Returns:
            Chunk object with image description.
        """
        metadata = {
            "source_file": str(file_path),
            "type": "image_description",
        }
        if classification:
            metadata["classification"] = classification

        return Chunk(
            document_id=document_id,
            content=description,
            chunk_index=0,
            heading_path=["Image"],
            section_title=file_path.name,
            metadata=metadata,
        )

    async def process_document_image(
        self,
        file_path: Path,
        document_id: UUID,
    ) -> List[Chunk]:
        """Process an image from inside a PDF/ZIP (no classification needed).

        This method is used for images extracted from documents where we
        assume all images are meaningful content.

        Args:
            file_path: Path to the image file.
            document_id: UUID of the parent document record.

        Returns:
            List of Chunk objects with image description.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        # Describe directly without classification
        description = await self.image_processor.describe_only(file_path)

        if not description:
            logger.warning(f"Could not get description for image {file_path}")
            return []

        chunk = Chunk(
            document_id=document_id,
            content=description,
            chunk_index=0,
            heading_path=["Image"],
            section_title=file_path.name,
            metadata={
                "source_file": str(file_path),
                "type": "image_description",
            },
        )

        return [chunk]

    def is_zip_file(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is a ZIP archive (excluding Office formats).

        DOCX, XLSX, PPTX and other Office formats are technically ZIP files
        but should be processed by LlamaParse, not extracted as ZIPs.

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file (optional).

        Returns:
            True if file is a ZIP archive that should be extracted.
        """
        # Exclude Office formats that are technically ZIPs
        if content_type and content_type in self.OFFICE_ZIP_FORMATS:
            return False

        # Check extension to exclude Office formats
        lower_path = file_path.lower()
        for ext in self.OFFICE_ZIP_EXTENSIONS:
            if lower_path.endswith(ext):
                return False

        # Now check if it's actually a ZIP
        if content_type and content_type in self.ZIP_FORMATS:
            return True

        # Check by extension
        if lower_path.endswith(".zip"):
            return True

        # Check MIME type from file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            # Exclude Office formats detected by MIME type
            if mime_type in self.OFFICE_ZIP_FORMATS:
                return False
            if mime_type in self.ZIP_FORMATS:
                return True

        return False

    def extract_zip(
        self,
        zip_path: Path,
        extract_dir: Optional[Path] = None,
        lenient: bool = False,
        _depth: int = 0,
        _file_count: int = 0,
        _total_size: int = 0,
    ) -> List[Tuple[Path, str]]:
        """Extract files from a ZIP archive with resource limits.

        Safely extracts ZIP contents, skipping dangerous paths and unsupported files.
        Returns list of extracted file paths with their MIME types.

        Args:
            zip_path: Path to the ZIP file.
            extract_dir: Directory to extract to. If None, extracts to same directory
                        as ZIP file with _extracted suffix.
            lenient: If True, log extraction errors and continue. If False (default),
                    raise ZipMemberExtractionError on any member failure.
            _depth: Internal parameter for tracking nested ZIP depth.
            _file_count: Internal parameter for tracking total extracted files.
            _total_size: Internal parameter for tracking total extracted size.

        Returns:
            List of tuples (file_path, content_type) for supported extracted files.

        Raises:
            FileNotFoundError: If ZIP file doesn't exist.
            ValueError: If ZIP file is invalid or cannot be opened.
            ZipExtractionError: If extraction limits are exceeded.
            ZipMemberExtractionError: If a member fails to extract (when lenient=False).
        """
        settings = get_settings()

        # Check depth limit
        if _depth > settings.zip_max_depth:
            raise ZipExtractionError(
                f"ZIP nesting depth {_depth} exceeds limit of {settings.zip_max_depth}"
            )

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Not a valid ZIP file: {zip_path}")

        # Create extraction directory
        if extract_dir is None:
            extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        extracted_files: List[Tuple[Path, str]] = []
        max_total_bytes = settings.zip_max_total_size_mb * 1024 * 1024

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Check file count limit
                if _file_count >= settings.zip_max_files:
                    raise ZipExtractionError(
                        f"ZIP extraction file count {_file_count} exceeds limit of {settings.zip_max_files}"
                    )

                # Security: Skip dangerous paths
                if self._is_dangerous_zip_path(member):
                    continue

                # Skip directories
                if member.endswith("/"):
                    continue

                # Skip hidden files and macOS resource forks
                basename = Path(member).name
                if basename.startswith(".") or basename.startswith("__MACOSX"):
                    continue

                # Check file info before extraction
                info = zf.getinfo(member)

                # ZIP bomb detection: check compression ratio
                # A ratio > 100:1 is suspicious (ZIP bombs can have 1000000:1)
                if info.compress_size > 0:
                    compression_ratio = info.file_size / info.compress_size
                    if compression_ratio > 100:
                        logger.warning(
                            f"Skipping {member}: suspicious compression ratio {compression_ratio:.0f}:1"
                        )
                        continue

                # Check total size limit
                if _total_size + info.file_size > max_total_bytes:
                    raise ZipExtractionError(
                        f"ZIP extraction would exceed size limit of {settings.zip_max_total_size_mb}MB"
                    )

                # Extract file
                try:
                    # Sanitize the path
                    safe_path = self._sanitize_zip_member_path(member)
                    target_path = extract_dir / safe_path

                    # Create parent directories
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

                    _file_count += 1
                    _total_size += info.file_size

                    # Get MIME type
                    mime_type, _ = mimetypes.guess_type(str(target_path))
                    content_type = mime_type or "application/octet-stream"

                    # Check if it's a nested ZIP
                    if self.is_zip_file(str(target_path), content_type):
                        # Recursively extract nested ZIPs with updated counters
                        nested_files = self.extract_zip(
                            target_path,
                            lenient=lenient,
                            _depth=_depth + 1,
                            _file_count=_file_count,
                            _total_size=_total_size,
                        )
                        extracted_files.extend(nested_files)
                        # Update counters from nested extraction
                        _file_count += len(nested_files)
                    elif self.is_supported(str(target_path), content_type):
                        extracted_files.append((target_path, content_type))
                    elif self.is_image_format(content_type):
                        # Images are handled separately
                        extracted_files.append((target_path, content_type))

                except ZipExtractionError:
                    # Re-raise limit errors
                    raise
                except (OSError, IOError, zipfile.BadZipFile) as e:
                    # Handle file extraction errors based on lenient mode
                    error_msg = f"Failed to extract {member}: {e}"
                    if lenient:
                        # Log and continue in lenient mode
                        logger.warning(error_msg)
                        continue
                    else:
                        # Raise in strict mode to prevent data loss
                        logger.error(error_msg)
                        raise ZipMemberExtractionError(member, str(e))

        return extracted_files

    def _is_dangerous_zip_path(self, path: str) -> bool:
        """Check if ZIP member path is potentially dangerous.

        Args:
            path: Path within ZIP archive.

        Returns:
            True if path is dangerous (path traversal, absolute path, etc.).
        """
        # Check for path traversal
        if ".." in path:
            return True

        # Check for absolute paths
        if path.startswith("/") or path.startswith("\\"):
            return True

        # Check for Windows absolute paths (C:, D:, etc.)
        if len(path) > 1 and path[1] == ":":
            return True

        return False

    def _sanitize_zip_member_path(self, path: str) -> str:
        """Sanitize a ZIP member path for safe extraction.

        Args:
            path: Original path from ZIP archive.

        Returns:
            Sanitized path safe for filesystem.
        """
        # Normalize separators
        path = path.replace("\\", "/")

        # Remove leading slashes
        path = path.lstrip("/")

        # Remove any remaining path traversal
        parts = []
        for part in path.split("/"):
            if part and part != "..":
                # Sanitize individual filename using shared utility
                part = sanitize_filename(part)
                if part:
                    parts.append(part)

        return "/".join(parts)
