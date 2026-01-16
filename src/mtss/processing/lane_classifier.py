"""Lane classifier for fast/slow email processing.

Classifies emails based on attachment types to optimize parallel processing:
- Fast lane: Emails without attachments or with only images (no LlamaParse needed)
- Slow lane: Emails with documents requiring LlamaParse (PDFs, Office files, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..parsers.eml_parser import EMLParser

logger = logging.getLogger(__name__)

# Image MIME types that don't require LlamaParse (from ImageProcessor)
IMAGE_MIMETYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/bmp",
}

# Document MIME types that require LlamaParse (from LlamaParseParser)
LLAMAPARSE_MIMETYPES = {
    # PDF
    "application/pdf",
    # Modern Office formats
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    # Legacy Office formats
    "application/msword",  # .doc
    "application/vnd.ms-excel",  # .xls
    "application/vnd.ms-powerpoint",  # .ppt
    # Other formats
    "text/csv",
    "application/rtf",
    "text/rtf",
    "application/epub+zip",
    "application/vnd.oasis.opendocument.text",  # .odt
    "application/vnd.oasis.opendocument.spreadsheet",  # .ods
    "application/vnd.oasis.opendocument.presentation",  # .odp
    # HTML
    "text/html",
}

# ZIP MIME types (may contain documents, conservative: always slow lane)
ZIP_MIMETYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/x-zip",
}

# Document extensions for fallback detection
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".doc",
    ".xls",
    ".ppt",
    ".csv",
    ".rtf",
    ".epub",
    ".odt",
    ".ods",
    ".odp",
    ".html",
    ".htm",
    ".zip",
}

# Image extensions for fallback detection
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
}


class LaneClassifier:
    """Classify emails into fast or slow processing lanes.

    Fast lane: Emails that don't need LlamaParse API calls
    - No attachments
    - Only image attachments

    Slow lane: Emails that require LlamaParse API calls
    - PDF, Office documents, etc.
    - ZIP files (may contain documents)
    - Unknown attachment types (conservative)
    """

    def __init__(self, eml_parser: "EMLParser"):
        """Initialize the lane classifier.

        Args:
            eml_parser: EMLParser instance for peeking at attachments.
        """
        self.eml_parser = eml_parser

    def classify(self, eml_path: Path) -> Literal["fast", "slow"]:
        """Classify an email for lane assignment.

        Args:
            eml_path: Path to the EML file.

        Returns:
            "fast" for emails that don't need LlamaParse
            "slow" for emails that require LlamaParse
        """
        try:
            attachments = self.eml_parser.peek_attachments(eml_path)
        except Exception as e:
            # On any error, default to slow lane (conservative)
            logger.warning(f"Failed to peek attachments for {eml_path}: {e}")
            return "slow"

        # No attachments = fast lane
        if not attachments:
            return "fast"

        # Check each attachment
        for filename, content_type in attachments:
            # ZIP files always go to slow lane (may contain documents)
            if content_type in ZIP_MIMETYPES:
                return "slow"

            # Check extension for .zip files
            if filename and filename.lower().endswith(".zip"):
                return "slow"

            # Document types go to slow lane
            if content_type in LLAMAPARSE_MIMETYPES:
                return "slow"

            # Check by extension for document types
            if filename:
                ext = Path(filename).suffix.lower()
                if ext in DOCUMENT_EXTENSIONS:
                    return "slow"

            # Images are fine for fast lane
            if content_type in IMAGE_MIMETYPES:
                continue

            # Check extension for images
            if filename:
                ext = Path(filename).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    continue

            # Unknown type with application/octet-stream - check extension
            if content_type == "application/octet-stream":
                if filename:
                    ext = Path(filename).suffix.lower()
                    if ext in IMAGE_EXTENSIONS:
                        continue
                # Unknown binary file - slow lane to be safe
                return "slow"

            # Any other unknown type - slow lane to be safe
            if content_type not in IMAGE_MIMETYPES:
                return "slow"

        # All attachments are images (or no problematic types found)
        return "fast"
