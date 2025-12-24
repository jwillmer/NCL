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


class AttachmentProcessor:
    """Process attachments using parser plugins for text extraction and chunking.

    Uses the parser registry to find appropriate parsers for different file types.
    LlamaParse is the default parser for all document types.
    """

    # Mapping of MIME types to DocumentType
    MIME_TO_DOC_TYPE: Dict[str, DocumentType] = {
        "application/pdf": DocumentType.ATTACHMENT_PDF,
        "image/png": DocumentType.ATTACHMENT_IMAGE,
        "image/jpeg": DocumentType.ATTACHMENT_IMAGE,
        "image/jpg": DocumentType.ATTACHMENT_IMAGE,
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

    def is_supported(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file format is supported by any registered parser.

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file (optional).

        Returns:
            True if format is supported.
        """
        parser = ParserRegistry.get_parser_for_file(Path(file_path), content_type)
        return parser is not None

    def get_document_type(self, content_type: str) -> DocumentType:
        """Get DocumentType from MIME type.

        Args:
            content_type: MIME type string.

        Returns:
            Corresponding DocumentType enum value.
        """
        return self.MIME_TO_DOC_TYPE.get(content_type, DocumentType.ATTACHMENT_OTHER)

    async def process_attachment(
        self,
        file_path: Path,
        document_id: UUID,
        content_type: Optional[str] = None,
    ) -> List[Chunk]:
        """Process an attachment file and return chunks.

        Uses parser registry to find appropriate parser, then chunks result.

        Args:
            file_path: Path to the attachment file.
            document_id: UUID of the parent document record.
            content_type: MIME type of the file (optional).

        Returns:
            List of Chunk objects with text and metadata.

        Raises:
            FileNotFoundError: If the attachment file doesn't exist.
            ValueError: If no parser available or parsing fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        # Get appropriate parser from registry
        parser = ParserRegistry.get_parser_for_file(file_path, content_type)

        if not parser:
            raise ValueError(f"No parser available for {file_path}")

        # Parse document to markdown text
        logger.info(f"Processing {file_path.name} with {parser.name} parser")
        text = await parser.parse(file_path)

        if not text or not text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return []

        text_len = len(text.strip())
        logger.info(f"Extracted {text_len} chars from {file_path.name}")

        if text_len < 50:
            logger.warning(f"Very little text from {file_path.name}: {repr(text.strip()[:100])}")

        # Chunk the text using LangChain
        chunks = self.chunker.chunk_text(
            text=text,
            document_id=document_id,
            source_file=str(file_path),
            is_markdown=True,
            metadata={"parser": parser.name},
        )

        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks

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
            _depth: Internal parameter for tracking nested ZIP depth.
            _file_count: Internal parameter for tracking total extracted files.
            _total_size: Internal parameter for tracking total extracted size.

        Returns:
            List of tuples (file_path, content_type) for supported extracted files.

        Raises:
            FileNotFoundError: If ZIP file doesn't exist.
            ValueError: If ZIP file is invalid or cannot be opened.
            ZipExtractionError: If extraction limits are exceeded.
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

                # Check compressed size before extraction
                info = zf.getinfo(member)
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
                except Exception:
                    # Skip files that fail to extract for other reasons
                    continue

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
