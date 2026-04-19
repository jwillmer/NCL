"""Preprocessor for document routing and filtering."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..image_filter import is_meaningful_image
from ..processing.image_processor import ImageProcessor
from .registry import ParserRegistry

logger = logging.getLogger(__name__)


def _safe_count_pdf_pages(file_path: Path) -> Optional[int]:
    """Return page count for a PDF, or None if it can't be determined.

    Uses PyMuPDF — page_count is a header-read (milliseconds for any size),
    versus pypdf which walks the full page tree. Returning None lets the
    parser layer surface any real unreadability as its own failure.
    """
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        with pymupdf.open(str(file_path)) as doc:
            return doc.page_count
    except Exception as e:
        logger.warning(f"Cannot count pages in {file_path.name}: {e}")
        return None


def _peek_pdf_markdown(file_path: Path, pages: int = 3) -> Optional[str]:
    """Extract the first ``pages`` pages as markdown locally via PyMuPDF4LLM.

    Used by the oversized-PDF branch so the embedding-mode decider can
    classify the doc from a cheap local preview rather than paying the full
    cloud parser. Returns None when the libs are missing or the file can't
    be opened — the caller falls back to a skip in that case.
    """
    try:
        import pymupdf4llm
    except ImportError:
        return None
    try:
        md = pymupdf4llm.to_markdown(str(file_path), pages=list(range(pages)))
    except Exception as e:
        logger.warning(f"PDF peek failed for {file_path.name}: {e}")
        return None
    return md if md and md.strip() else None


def _peek_complex_pdf_if_noise(
    file_path: Path, page_count: Optional[int]
) -> Optional[str]:
    """Short-circuit Gemini for COMPLEX PDFs whose local peek shows no prose.

    Scanned forms with no text layer (e.g. image-only regulatory circulars)
    classify COMPLEX, but Gemini burns minutes producing either junk or
    hallucinated content that the embedding-mode decider discards as
    METADATA_ONLY. Detecting this up-front costs ~100ms of PyMuPDF work.

    Returns an annotated preview string when the PDF should bypass Gemini,
    or None to let the normal parser routing run. Uses the decider's own
    rule 1 thresholds so the classification outcome is identical.
    """
    try:
        from .pdf_classifier import PDFComplexity, classify_pdf
        from ..ingest.embedding_decider import analyze
    except ImportError:
        return None

    try:
        complexity = classify_pdf(file_path)
    except Exception:
        return None
    if complexity != PDFComplexity.COMPLEX:
        return None

    preview = _peek_pdf_markdown(file_path, pages=3)
    if not preview:
        return None
    shape = analyze(preview)
    # Mirror embedding_decider rule 1: METADATA_ONLY when tiny or no-prose.
    is_noise = shape.total_tokens < 50 or (
        shape.prose_ratio < 0.15 and shape.heading_count == 0
    )
    if not is_noise:
        return None
    pages_label = f"{page_count} pages" if page_count else "unknown pages"
    return (
        f"_Complex PDF ({pages_label}) — local preview indicated no prose. "
        f"The following is the first 3 pages._\n\n"
        f"{preview}"
    )


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
    # For oversized PDFs where the preprocessor already extracted a cheap
    # first-page preview, attachment_handler uses this as parsed_content
    # instead of calling the full parser — the decider then classifies the
    # doc as SUMMARY / METADATA_ONLY from the preview alone.
    oversized_pdf: bool = False
    preview_markdown: Optional[str] = None
    total_pages: Optional[int] = None


# Extensions and MIME types handled by local parsers (tiered routing)
_LOCAL_PARSER_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".html", ".htm"}
_LOCAL_PARSER_MIMETYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
    "text/html",
}


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

        # Chrome writes ``.crdownload`` for in-progress downloads; if one ends
        # up in an email attachment it is partial bytes, not a valid file.
        # Reject before any parser wastes a call on it.
        if file_path.suffix.lower() == ".crdownload":
            return PreprocessResult(
                should_process=False,
                skip_reason=f"partial_download: {file_path.name}",
                content_type=actual_type,
            )

        # Reject anything above the configured byte ceiling before we read it.
        # Guards against accidentally pulling a multi-GB attachment into memory
        # (Gemini parser base64-encodes the whole file, so this matters).
        max_bytes = get_settings().attachment_max_bytes
        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0
        if size > max_bytes:
            return PreprocessResult(
                should_process=False,
                skip_reason=f"attachment_too_large: {size} bytes exceeds limit of {max_bytes}",
                content_type=actual_type,
            )

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
                # Local heuristic filter before Vision API call
                if not is_meaningful_image(file_path):
                    return PreprocessResult(
                        should_process=False,
                        skip_reason=f"filtered_by_heuristic: {file_path.name}",
                        is_image=True,
                        content_type=actual_type,
                    )
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

        # PDF pre-Gemini guard. Two related cases short-circuit the expensive
        # cloud parser by handing a local peek to the embedding-mode decider:
        #
        #   1. Oversized PDFs (> pdf_max_pages) — large sensor/log dumps that
        #      produce embedding noise; SUMMARY is always the right outcome.
        #   2. COMPLEX PDFs with a noise-only peek (no prose, no headings) —
        #      scanned forms where Gemini would burn minutes producing garbage
        #      that the decider would then throw away as METADATA_ONLY.
        #
        # For case 2 we run the same peek and shape-check as the decider's
        # rule 1 (prose_ratio < 0.15 AND heading_count == 0 → metadata_only).
        # If the peek clears that bar we leave routing to the normal parser
        # so genuinely prose-heavy complex PDFs still get Gemini.
        ext = file_path.suffix.lower()
        if ext == ".pdf" or actual_type == "application/pdf":
            page_count = _safe_count_pdf_pages(file_path)
            max_pages = get_settings().pdf_max_pages
            oversized = page_count is not None and page_count > max_pages
            if oversized:
                preview = _peek_pdf_markdown(file_path, pages=3)
                if preview is None:
                    return PreprocessResult(
                        should_process=False,
                        skip_reason=f"pdf_too_large_unreadable: {page_count} pages",
                        content_type=actual_type,
                    )
                annotated = (
                    f"_Oversized PDF — {page_count} pages total. "
                    f"The following is a preview of the first 3 pages._\n\n"
                    f"{preview}"
                )
                return PreprocessResult(
                    should_process=True,
                    parser_name="oversized_pdf_peek",
                    content_type=actual_type,
                    oversized_pdf=True,
                    preview_markdown=annotated,
                    total_pages=page_count,
                )

            # Not oversized — try the COMPLEX-PDF short-circuit.
            complex_peek = _peek_complex_pdf_if_noise(file_path, page_count)
            if complex_peek is not None:
                return PreprocessResult(
                    should_process=True,
                    parser_name="oversized_pdf_peek",
                    content_type=actual_type,
                    oversized_pdf=True,
                    preview_markdown=complex_peek,
                    total_pages=page_count,
                )

        # Check local parsers first (not in registry, handled by tiered routing)
        if ext in _LOCAL_PARSER_EXTENSIONS or actual_type in _LOCAL_PARSER_MIMETYPES:
            return PreprocessResult(
                should_process=True,
                parser_name="local",
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
