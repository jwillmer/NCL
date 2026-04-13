"""Local PDF parser using PyMuPDF4LLM for simple text PDFs."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser

logger = logging.getLogger(__name__)


class LocalPDFParser(BaseParser):
    """Parser for simple text PDFs using PyMuPDF4LLM (free, local)."""

    name = "local_pdf"

    @property
    def is_available(self) -> bool:
        try:
            import pymupdf4llm  # noqa: F401
            return True
        except ImportError:
            return False

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pymupdf4llm
            markdown = pymupdf4llm.to_markdown(str(file_path))

            if not markdown or not markdown.strip():
                raise ValueError(f"PyMuPDF4LLM produced no content for {file_path}")

            logger.info(f"Local PDF parser extracted {len(markdown)} chars from {file_path.name}")
            return markdown

        except ImportError:
            raise ValueError("pymupdf4llm is not installed. Install with: pip install pymupdf4llm")
        except Exception as e:
            raise ValueError(f"Local PDF parsing failed for {file_path}: {e}") from e
