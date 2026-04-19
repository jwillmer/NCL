"""Base parser interface for document processing plugins."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class EmptyContentError(ValueError):
    """Raised when a parser successfully opened a file but extracted no text content.

    Distinct from other ValueError cases (corrupted file, missing dependency) so
    callers can fall back to a more capable parser (e.g. LlamaParse for image-only
    DOCX/XLSX) while letting real parse failures propagate.
    """


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All parsers must implement the parse() method which takes a file path
    and returns the extracted text as markdown/text format.
    """

    # Class-level attributes parsers should override
    name: str = "base"
    supported_mimetypes: set[str] = set()
    supported_extensions: set[str] = set()

    @property
    def model_name(self) -> Optional[str]:
        """Model identifier stamped onto the ProcessingTrail ``parse`` step.

        Deterministic parsers (local_pdf, local_docx, …) return ``None`` — the
        trail records the parser name only. LLM-backed parsers override to
        return the provider/model string the ingest actually called.
        """
        return None

    @abstractmethod
    async def parse(self, file_path: Path) -> str:
        """Parse a document and return its text content.

        Args:
            file_path: Path to the document file.

        Returns:
            Extracted text content as markdown or plain text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If parsing fails.
        """
        pass

    def supports_file(self, file_path: Path, content_type: Optional[str] = None) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type.

        Returns:
            True if parser can handle this file type.
        """
        if content_type and content_type in self.supported_mimetypes:
            return True

        ext = file_path.suffix.lower()
        return ext in self.supported_extensions
