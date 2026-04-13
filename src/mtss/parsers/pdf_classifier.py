"""PDF complexity classifier for tiered parsing."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_pdf(file_path: Path) -> PDFComplexity:
    """Classify a PDF as simple or complex using pypdf.

    Simple: all pages have extractable text, no images, no form fields.
    Complex: any page is scanned (no text), has images, or has form fields.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        logger.warning(f"Cannot open PDF {file_path.name} for classification: {e}")
        return PDFComplexity.COMPLEX

    if not reader.pages:
        return PDFComplexity.COMPLEX

    if reader.get_fields():
        return PDFComplexity.COMPLEX

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            return PDFComplexity.COMPLEX

        if len(text.strip()) < 50:
            return PDFComplexity.COMPLEX

        try:
            resources = page.get("/Resources") or {}
            if "/XObject" in resources:
                xobjects = resources["/XObject"].get_object()
                for obj_name in xobjects:
                    xobj = xobjects[obj_name].get_object()
                    if xobj.get("/Subtype") == "/Image":
                        return PDFComplexity.COMPLEX
        except (KeyError, AttributeError, TypeError):
            return PDFComplexity.COMPLEX

    return PDFComplexity.SIMPLE
