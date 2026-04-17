"""LlamaParse parser plugin for document processing."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import get_settings
from .base import BaseParser

if TYPE_CHECKING:
    from llama_cloud import AsyncLlamaCloud

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
_llamaparse_client: "AsyncLlamaCloud | None" = None


def _get_llamaparse_semaphore() -> asyncio.Semaphore:
    """Get or create the LlamaParse concurrency semaphore."""
    global _llamaparse_semaphore
    if _llamaparse_semaphore is None:
        settings = get_settings()
        _llamaparse_semaphore = asyncio.Semaphore(settings.max_concurrent_llamaparse)
    return _llamaparse_semaphore


def _get_llamaparse_client() -> "AsyncLlamaCloud":
    """Get or create the cached AsyncLlamaCloud HTTP client."""
    global _llamaparse_client
    if _llamaparse_client is None:
        from llama_cloud import AsyncLlamaCloud

        settings = get_settings()
        _llamaparse_client = AsyncLlamaCloud(api_key=settings.llama_cloud_api_key)
    return _llamaparse_client


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

        client = _get_llamaparse_client()
        sem = _get_llamaparse_semaphore()

        async with sem:
            try:
                logger.info(f"Parsing {file_path.name} with LlamaParse...")

                # tier="agentic" + cost_optimizer routes simple pages to cost_effective
                # automatically; complex pages get true agentic processing. Cheaper than
                # blanket agentic, smarter than blanket cost_effective + auto_mode.
                # output_options.markdown.inline_images=False keeps image refs out of
                # the markdown stream; agentic tier inlines image transcriptions on its
                # own.
                result = await client.parsing.parse(
                    upload_file=str(file_path),
                    tier="agentic",
                    version="latest",
                    cost_optimizer={"enable": True},
                    expand=["markdown"],
                    output_options={
                        "markdown": {"inline_images": False},
                        "images_to_save": [],
                    },
                )

                pages = getattr(result.markdown, "pages", None) or []
                markdown_text = "\n\n".join(p.markdown for p in pages if p.markdown)

                markdown_text = strip_llamaparse_image_refs(markdown_text)

                if not markdown_text or not markdown_text.strip():
                    raise ValueError(f"LlamaParse produced no content for {file_path}")

                from ..cli._common import _service_counter
                _service_counter.add("llamaparse")

                logger.info(
                    f"LlamaParse extracted {len(markdown_text)} chars from {file_path.name}"
                )
                return markdown_text

            except (FileNotFoundError, ValueError):
                raise
            except Exception as e:
                msg = "\n".join(
                    ln for ln in str(e).splitlines()
                    if not ln.startswith("Started parsing the file")
                ).strip() or type(e).__name__
                raise ValueError(f"LlamaParse failed: {msg}") from e
