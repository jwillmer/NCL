"""Preprocessor for document routing and filtering."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..processing.image_processor import ImageProcessor
from .registry import ParserRegistry

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of preprocessing a file."""

    should_process: bool
    parser_name: Optional[str] = None
    skip_reason: Optional[str] = None
    is_image: bool = False
    is_zip: bool = False
    content_type: Optional[str] = None
    # For images that passed classification, store the description
    image_description: Optional[str] = None


class DocumentPreprocessor:
    """Preprocessor for routing files to appropriate parsers and filtering.

    Responsibilities:
    - Determine file type and select appropriate parser
    - Filter out non-content images (logos, signatures, banners)
    - Handle ZIP files specially (they need extraction, not parsing)
    - Support configuration-driven routing
    """

    # File types that are ZIP-based but should be processed as documents
    OFFICE_ZIP_EXTENSIONS = {
        ".docx",
        ".xlsx",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".epub",
        ".jar",
    }

    ZIP_MIMETYPES = {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-zip",
    }

    def __init__(self):
        """Initialize preprocessor."""
        self._image_processor: Optional[ImageProcessor] = None

    @property
    def image_processor(self) -> ImageProcessor:
        """Lazy initialization of ImageProcessor."""
        if self._image_processor is None:
            self._image_processor = ImageProcessor()
        return self._image_processor

    def get_content_type(
        self, file_path: Path, provided_type: Optional[str] = None
    ) -> str:
        """Get MIME type for a file.

        Args:
            file_path: Path to the file.
            provided_type: Optionally provided MIME type.

        Returns:
            MIME type string.
        """
        if provided_type:
            return provided_type

        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def is_zip_file(self, file_path: Path, content_type: Optional[str] = None) -> bool:
        """Check if file is a ZIP archive (not Office format).

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type.

        Returns:
            True if file is an extractable ZIP archive.
        """
        ext = file_path.suffix.lower()

        # Exclude Office formats that are technically ZIPs
        if ext in self.OFFICE_ZIP_EXTENSIONS:
            return False

        if ext == ".zip":
            return True

        actual_type = self.get_content_type(file_path, content_type)
        return actual_type in self.ZIP_MIMETYPES

    def is_image(self, content_type: Optional[str]) -> bool:
        """Check if content type is an image."""
        return self.image_processor.is_supported(content_type)

    async def preprocess(
        self,
        file_path: Path,
        content_type: Optional[str] = None,
        classify_images: bool = True,
    ) -> PreprocessResult:
        """Preprocess a file to determine how it should be handled.

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type.
            classify_images: Whether to classify images (filter logos, etc.)

        Returns:
            PreprocessResult with routing decision.
        """
        actual_type = self.get_content_type(file_path, content_type)

        # Check if it's a ZIP that needs extraction
        if self.is_zip_file(file_path, actual_type):
            return PreprocessResult(
                should_process=True,
                parser_name="zip",  # Special handling, not a parser
                is_zip=True,
                content_type=actual_type,
            )

        # Check if it's an image that needs classification
        if self.is_image(actual_type):
            if classify_images:
                result = await self.image_processor.classify_and_describe(file_path)
                if result.should_skip:
                    return PreprocessResult(
                        should_process=False,
                        skip_reason=result.skip_reason,
                        is_image=True,
                        content_type=actual_type,
                    )
                # Image is meaningful - store description for later use
                return PreprocessResult(
                    should_process=True,
                    parser_name="image",  # Special handling via ImageProcessor
                    is_image=True,
                    content_type=actual_type,
                    image_description=result.description,
                )
            else:
                # No classification - just describe
                return PreprocessResult(
                    should_process=True,
                    parser_name="image",
                    is_image=True,
                    content_type=actual_type,
                )

        # Find appropriate parser from registry
        parser = ParserRegistry.get_parser_for_file(file_path, actual_type)

        if parser:
            return PreprocessResult(
                should_process=True,
                parser_name=parser.name,
                content_type=actual_type,
            )

        # No parser found - unsupported format
        return PreprocessResult(
            should_process=False,
            skip_reason=f"unsupported_format: {actual_type}",
            content_type=actual_type,
        )
