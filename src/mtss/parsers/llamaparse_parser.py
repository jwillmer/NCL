"""LlamaParse parser plugin for document processing."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from ..config import get_settings
from .base import BaseParser

logger = logging.getLogger(__name__)


def strip_llamaparse_image_refs(text: str) -> str:
    """Strip LlamaParse image refs, preserving alt-text.

    Defense-in-depth: new ingests pass `inline_images_in_markdown=False`,
    so refs should not appear in fresh output. This function is still applied
    after every parse to catch layout placeholders that bypass that flag, and
    is the engine behind `mtss clean-archive-md` for retroactively cleaning
    archives produced before the flag was set.

    Covers both image (`![alt](...)`) and link (`[alt](...)`) forms. Known
    target patterns:
      - page_N_image_N / page_N_chart_N / page_N_seal_N / page_N_table_N
      - page_N_layout_ocr_* (newer LlamaParse output; bounding-box suffix)
      - layout_id_not_provided / layout_* placeholders
      - the literal string "image"
    """
    text = re.sub(
        r'<img\s+[^>]*alt="([^"]*)"[^>]*/?>',
        r"\1",
        text,
    )
    # Any page-prefixed LlamaParse artifact (image, chart, seal, table, layout_ocr, ...)
    # in either image-form (`![...]`) or link-form (`[...]`).
    text = re.sub(
        r"!?\[([^\]]*)\]\(page_\d+_\w+(?:_\w+)*[^)]*\)",
        r"\1",
        text,
    )
    # Layout placeholders emitted when LlamaParse can't resolve a figure id.
    text = re.sub(
        r"!?\[([^\]]*)\]\(layout(?:_\w+)*\)",
        r"\1",
        text,
    )
    # Literal "image" placeholder.
    text = re.sub(
        r"!?\[([^\]]*)\]\(image\)",
        r"\1",
        text,
    )
    return text


# Module-level semaphore to limit concurrent LlamaParse API calls
_llamaparse_semaphore: asyncio.Semaphore | None = None


def _get_llamaparse_semaphore() -> asyncio.Semaphore:
    """Get or create the LlamaParse concurrency semaphore."""
    global _llamaparse_semaphore
    if _llamaparse_semaphore is None:
        settings = get_settings()
        _llamaparse_semaphore = asyncio.Semaphore(settings.max_concurrent_llamaparse)
    return _llamaparse_semaphore


class LlamaParseParser(BaseParser):
    """Parser using LlamaParse API for document extraction.

    Handles complex documents that need cloud-based parsing:
    - PDF documents (complex PDFs with scanned pages, images, forms)
    - Office formats (PPTX)
    - Legacy Office formats (DOC, XLS, PPT)
    - Other document formats (RTF, EPUB, ODT, ODS, ODP)

    Note: DOCX, XLSX, CSV, HTML now use local parsers via tiered routing.
    Simple PDFs also use local PyMuPDF4LLM parser.
    """

    name = "llamaparse"

    supported_mimetypes = {
        # PDF
        "application/pdf",
        # Modern Office formats
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        # Legacy Office formats
        "application/msword",  # .doc
        "application/vnd.ms-excel",  # .xls
        "application/vnd.ms-powerpoint",  # .ppt
        # Other formats
        "application/rtf",
        "text/rtf",
        "application/epub+zip",
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
    }

    supported_extensions = {
        ".pdf",
        ".pptx",
        ".doc",
        ".xls",
        ".ppt",
        ".rtf",
        ".epub",
        ".odt",
        ".ods",
        ".odp",
    }

    def __init__(self):
        """Initialize LlamaParse parser."""
        self.settings = get_settings()

    @property
    def is_available(self) -> bool:
        """Check if LlamaParse is configured."""
        return self.settings.llamaparse_enabled

    async def parse(self, file_path: Path) -> str:
        """Parse document using LlamaParse API.

        Args:
            file_path: Path to document file.

        Returns:
            Extracted text in markdown format.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If LlamaParse is not configured or parsing fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_available:
            raise ValueError("LlamaParse is not enabled (LLAMA_CLOUD_API_KEY not set)")

        from llama_cloud_services import LlamaParse

        # Hardcoded configuration as per plan.
        # inline_images_in_markdown=False: suppress `![alt](page_N_image_N.jpg)` refs
        # in the markdown output. Combined with specialized_image_parsing=True, image
        # content (charts, scanned text, diagrams) is transcribed inline as text instead
        # of left as a dangling tag. Avoids the need to strip refs downstream.
        parser = LlamaParse(
            api_key=self.settings.llama_cloud_api_key,
            tier="cost_effective",
            version="latest",
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
            precise_bounding_box=True,
            auto_mode_configuration_json='[{"trigger_mode":"or","table_in_page":true,"layout_element_in_page":"chart","full_page_image_in_page":true,"parsing_conf":{"tier":"agentic","version":"latest"}}]',
            max_pages=0,  # No limit
            specialized_image_parsing=True,
            inline_images_in_markdown=False,
        )

        sem = _get_llamaparse_semaphore()
        async with sem:
            try:
                logger.info(f"Parsing {file_path.name} with LlamaParse...")
                result = await parser.aparse(str(file_path))

                # Get markdown (combined, not split by page)
                markdown_docs = result.get_markdown_documents(split_by_page=False)
                markdown_text = "\n\n".join(doc.text for doc in markdown_docs)

                # Strip LlamaParse image refs (images not downloaded locally).
                # Preserve alt-text for semantic content: <img src="..." alt="sketch"> → sketch
                # and ![alt](page_N_image_N.jpg) → alt
                markdown_text = strip_llamaparse_image_refs(markdown_text)

                if not markdown_text or not markdown_text.strip():
                    raise ValueError(f"LlamaParse produced no content for {file_path}")

                from ..cli._common import _service_counter
                _service_counter.add("llamaparse")

                logger.info(
                    f"LlamaParse extracted {len(markdown_text)} chars from {file_path.name}"
                )
                return markdown_text

            except Exception as e:
                if "LlamaParse" in str(type(e).__name__):
                    # Strip noisy "Started parsing the file under job_id ..." lines
                    msg = "\n".join(
                        ln for ln in str(e).splitlines()
                        if not ln.startswith("Started parsing the file")
                    ).strip()
                    raise ValueError(f"LlamaParse failed: {msg}") from e
                raise
