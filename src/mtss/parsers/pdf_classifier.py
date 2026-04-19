"""PDF complexity classifier for tiered parsing."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_reader(reader) -> PDFComplexity:
    """Classify an already-opened pypdf reader. Single source of truth."""
    from pypdf.errors import PyPdfError

    if not reader.pages:
        return PDFComplexity.COMPLEX

    try:
        if reader.get_fields():
            return PDFComplexity.COMPLEX
    except (PyPdfError, KeyError, AttributeError, TypeError):
        return PDFComplexity.COMPLEX

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except (PyPdfError, KeyError, AttributeError, TypeError, ValueError):
            return PDFComplexity.COMPLEX

        if len(text.strip()) < 50:
            return PDFComplexity.COMPLEX

    return PDFComplexity.SIMPLE


def classify_pdf(file_path: Path) -> PDFComplexity:
    """Classify a PDF as simple or complex using pypdf.

    Simple: every page has an extractable text layer and there are no form fields.
    Complex: any page is scanned (no text), the PDF has form fields, or the
    reader fails to parse it.

    Embedded images (logos, stamps, diagrams embedded in a text-layer PDF) do
    NOT mark a PDF complex — routing those to a cloud parser was costing real
    money on docs PyMuPDF4LLM handles fine.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        logger.warning(f"Cannot open PDF {file_path.name} for classification: {e}")
        return PDFComplexity.COMPLEX

    return classify_reader(reader)
