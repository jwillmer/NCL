"""Attachment processor using Docling for document conversion and chunking."""

from __future__ import annotations

import mimetypes
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ..config import get_settings
from ..models.chunk import Chunk
from ..models.document import DocumentType


class AttachmentProcessor:
    """Process attachments using Docling for text extraction and chunking.

    Supports PDF, images (with OCR), DOCX, PPTX, XLSX, and HTML files.
    Uses HierarchicalChunker to preserve document structure.
    """

    # Mapping of MIME types to Docling InputFormat
    SUPPORTED_FORMATS: Dict[str, str] = {
        "application/pdf": "PDF",
        "image/png": "IMAGE",
        "image/jpeg": "IMAGE",
        "image/jpg": "IMAGE",
        "image/tiff": "IMAGE",
        "image/bmp": "IMAGE",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
        "text/html": "HTML",
    }

    # ZIP file MIME types
    ZIP_FORMATS: Dict[str, str] = {
        "application/zip": "ZIP",
        "application/x-zip-compressed": "ZIP",
        "application/x-zip": "ZIP",
    }

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
    }

    def __init__(self):
        """Initialize the attachment processor."""
        settings = get_settings()
        self.enable_ocr = settings.enable_ocr
        self.enable_picture_description = settings.enable_picture_description
        self._converter = None
        self._chunker = None

    @property
    def converter(self):
        """Lazy initialization of DocumentConverter."""
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                PdfPipelineOptions,
                smolvlm_picture_description,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.enable_ocr
            pipeline_options.do_table_structure = True

            if self.enable_ocr:
                pipeline_options.ocr_options = EasyOcrOptions(
                    lang=["en"],
                )

            # Enable picture description using SmolVLM model
            if self.enable_picture_description:
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = smolvlm_picture_description
                pipeline_options.picture_description_options.prompt = (
                    "Describe this image in detail. Include any text, diagrams, "
                    "charts, or visual elements. Be precise and thorough."
                )
                pipeline_options.generate_picture_images = True
                pipeline_options.images_scale = 2.0

            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.XLSX,
                    InputFormat.HTML,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                },
            )
        return self._converter

    @property
    def chunker(self):
        """Lazy initialization of HybridChunker with OpenAI tokenizer."""
        if self._chunker is None:
            from docling_core.transforms.chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

            settings = get_settings()

            # Use OpenAI tokenizer matching the embedding model
            tokenizer = OpenAITokenizer(
                model=settings.embedding_model,
                max_tokens=settings.chunk_size_tokens,
            )

            self._chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=settings.chunk_size_tokens,
                merge_peers=True,  # Merge undersized chunks with same metadata
            )
        return self._chunker

    def is_supported(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file format is supported by Docling.

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file (optional).

        Returns:
            True if format is supported.
        """
        if content_type and content_type in self.SUPPORTED_FORMATS:
            return True

        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type in self.SUPPORTED_FORMATS if mime_type else False

    def get_document_type(self, content_type: str) -> DocumentType:
        """Get DocumentType from MIME type.

        Args:
            content_type: MIME type string.

        Returns:
            Corresponding DocumentType enum value.
        """
        return self.MIME_TO_DOC_TYPE.get(content_type, DocumentType.ATTACHMENT_OTHER)

    def process_attachment(
        self,
        file_path: Path,
        document_id: UUID,
        content_type: Optional[str] = None,
    ) -> List[Chunk]:
        """Process an attachment file and return chunks.

        Args:
            file_path: Path to the attachment file.
            document_id: UUID of the parent document record.
            content_type: MIME type of the file (optional).

        Returns:
            List of Chunk objects with text and metadata.

        Raises:
            FileNotFoundError: If the attachment file doesn't exist.
            ValueError: If document conversion fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        # Convert document using Docling
        result = self.converter.convert(str(file_path))

        if not result.document:
            raise ValueError(f"Failed to convert {file_path}: no document produced")

        # Apply hierarchical chunking
        docling_chunks = list(self.chunker.chunk(result.document))

        # Convert to our Chunk model
        chunks = []
        for idx, dc in enumerate(docling_chunks):
            # Extract heading path from chunk metadata
            heading_path = []
            if hasattr(dc, "meta") and dc.meta and hasattr(dc.meta, "headings"):
                heading_path = dc.meta.headings or []

            chunk = Chunk(
                document_id=document_id,
                content=dc.text,
                chunk_index=idx,
                heading_path=heading_path,
                section_title=heading_path[-1] if heading_path else None,
                page_number=self._extract_page_number(dc),
                metadata={
                    "source_file": str(file_path),
                    "doc_items_count": (
                        len(dc.meta.doc_items)
                        if hasattr(dc, "meta") and dc.meta and hasattr(dc.meta, "doc_items")
                        else 0
                    ),
                },
            )
            chunks.append(chunk)

        return chunks

    def _extract_page_number(self, chunk) -> Optional[int]:
        """Extract page number from chunk metadata if available.

        Args:
            chunk: Docling chunk object.

        Returns:
            Page number or None.
        """
        if not hasattr(chunk, "meta") or not chunk.meta:
            return None
        if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
            return None

        for item in chunk.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page"):
                        return prov.page
        return None

    def extract_text_only(self, file_path: Path) -> str:
        """Extract full text from attachment without chunking.

        Args:
            file_path: Path to the file.

        Returns:
            Full text content as markdown.
        """
        result = self.converter.convert(str(file_path))
        if result.document:
            return result.document.export_to_markdown()
        return ""

    def is_zip_file(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is a ZIP archive.

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file (optional).

        Returns:
            True if file is a ZIP archive.
        """
        if content_type and content_type in self.ZIP_FORMATS:
            return True

        # Check by extension
        if file_path.lower().endswith(".zip"):
            return True

        # Check MIME type from file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type in self.ZIP_FORMATS:
            return True

        # Try to detect by file magic
        try:
            path = Path(file_path)
            if path.exists():
                return zipfile.is_zipfile(path)
        except Exception:
            pass

        return False

    def extract_zip(
        self,
        zip_path: Path,
        extract_dir: Optional[Path] = None,
    ) -> List[Tuple[Path, str]]:
        """Extract files from a ZIP archive.

        Safely extracts ZIP contents, skipping dangerous paths and unsupported files.
        Returns list of extracted file paths with their MIME types.

        Args:
            zip_path: Path to the ZIP file.
            extract_dir: Directory to extract to. If None, extracts to same directory
                        as ZIP file with _extracted suffix.

        Returns:
            List of tuples (file_path, content_type) for supported extracted files.

        Raises:
            ValueError: If ZIP file is invalid or cannot be opened.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Not a valid ZIP file: {zip_path}")

        # Create extraction directory
        if extract_dir is None:
            extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        extracted_files: List[Tuple[Path, str]] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
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

                    # Get MIME type
                    mime_type, _ = mimetypes.guess_type(str(target_path))
                    content_type = mime_type or "application/octet-stream"

                    # Check if it's a nested ZIP
                    if self.is_zip_file(str(target_path), content_type):
                        # Recursively extract nested ZIPs
                        nested_files = self.extract_zip(target_path)
                        extracted_files.extend(nested_files)
                    elif self.is_supported(str(target_path), content_type):
                        extracted_files.append((target_path, content_type))

                except Exception:
                    # Skip files that fail to extract
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
                # Sanitize individual filename
                part = self._sanitize_filename(part)
                if part:
                    parts.append(part)

        return "/".join(parts)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe filesystem use.

        Args:
            filename: Original filename.

        Returns:
            Sanitized filename.
        """
        import re

        # Remove null bytes and path separators
        filename = filename.replace("\x00", "").replace("/", "_").replace("\\", "_")
        # Remove other problematic characters
        filename = re.sub(r'[<>:"|?*]', "_", filename)
        # Limit length
        return filename[:255]
