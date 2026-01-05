"""Plain text parser plugin for .txt and text/plain files."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """Parser for plain text files.

    Handles .txt files and text/plain MIME type.
    Attempts multiple encodings for robust text extraction.
    """

    name = "text"
    supported_mimetypes = {"text/plain"}
    supported_extensions = {".txt"}

    @property
    def is_available(self) -> bool:
        """Plain text parser is always available."""
        return True

    async def parse(self, file_path: Path) -> str:
        """Parse a plain text file.

        Tries multiple common encodings to handle various text file formats.

        Args:
            file_path: Path to the text file.

        Returns:
            Text content of the file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file cannot be decoded with any supported encoding.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try common encodings in order of likelihood
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Successfully read {file_path.name} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue

        # Last resort: read with replacement characters for undecodable bytes
        logger.warning(
            f"Could not decode {file_path.name} with standard encodings, "
            "using UTF-8 with replacement characters"
        )
        return file_path.read_text(encoding="utf-8", errors="replace")
