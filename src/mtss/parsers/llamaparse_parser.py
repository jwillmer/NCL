"""LlamaParse parser plugin for document processing."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import get_settings
from .base import BaseParser, EmptyContentError

if TYPE_CHECKING:
    from llama_cloud import AsyncLlamaCloud

logger = logging.getLogger(__name__)


def strip_llamaparse_image_refs(text: str) -> str:
    """Strip parser-emitted image refs, preserving alt-text.

    Name kept for backwards-compat; now covers both LlamaParse and Gemini
    PDF output. Defense-in-depth: new ingests suppress image refs in parser
    settings, so refs should not appear in fresh output. This function is
    still applied after every parse to catch layout placeholders that bypass
    those flags, and is the engine behind `mtss clean-archive-md` for
    retroactively cleaning archives produced before the flags were tightened.

    Covers both image (`![alt](...)`) and link (`[alt](...)`) forms. Known
    target patterns:
      - page_N_image_N / page_N_chart_N / page_N_seal_N / page_N_table_N  (LlamaParse)
      - page_N_layout_ocr_* (newer LlamaParse output; bounding-box suffix)
      - layout_id_not_provided / layout_* placeholders                    (LlamaParse)
      - image_N / image_N.png / image_N.jpg / image_N.jpeg                (Gemini PDF)
      - the literal string "image"
      - bare short tokens without path/extension (e.g. `doge`, `pfzo nefh`)

    Alt-text can contain nested square brackets (e.g. ``[CODE75]`` in a screen
    caption) — the inner pattern accepts any ``]`` that is *not* followed by
    ``(``, so the match only terminates at the real closing ``](``.
    """
    # Alt-text: any run of chars that isn't a `]` followed by `(`. Tolerates
    # nested `[...]` sequences, which are common in technical alt captions.
    _ALT = r"(?:[^\]]|\](?!\())*"

    text = re.sub(
        r'<img\s+[^>]*alt="([^"]*)"[^>]*/?>',
        r"\1",
        text,
    )
    # Any page-prefixed LlamaParse artifact (image, chart, seal, table, layout_ocr, ...)
    # in either image-form (`![...]`) or link-form (`[...]`).
    text = re.sub(
        rf"!?\[({_ALT})\]\(page_\d+_\w+(?:_\w+)*[^)]*\)",
        r"\1",
        text,
    )
    # Layout placeholders emitted when LlamaParse can't resolve a figure id.
    text = re.sub(
        rf"!?\[({_ALT})\]\(layout(?:_\w+)*\)",
        r"\1",
        text,
    )
    # Literal "image" placeholder.
    text = re.sub(
        rf"!?\[({_ALT})\]\(image\)",
        r"\1",
        text,
    )
    # Gemini PDF parser emits `![alt](image_N.png)` / `.jpg` refs where N is
    # a zero-based page counter. We don't extract the image bytes, so these
    # always dangle. Image-form only; the specific `image_<digits>.<ext>`
    # shape keeps real filenames like `chart.png` intact.
    text = re.sub(
        rf"!\[({_ALT})\]\(image_\d+\.(?:png|jpe?g)\)",
        r"\1",
        text,
    )
    # Image-form refs whose target has no path separator and no extension
    # (i.e. no `/` or `.`). LlamaParse occasionally emits short random tokens
    # like `doge` or `pfzo nefh` that slip through the more specific patterns.
    # Scoped to image form (`![...]`) to avoid mangling link-form references
    # to valid anchors or relative file names that happen to lack extensions.
    text = re.sub(
        rf"!\[({_ALT})\]\(([^)/.\#:?&]+)\)",
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
    # llama-cloud 2.x doesn't expose a user-facing model name — record the
    # tier instead so consumers know which vendor ran.
    _MODEL_NAME = "llamaparse:agentic"

    @property
    def model_name(self) -> str | None:
        return self._MODEL_NAME

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
                # cost_optimizer lives inside processing_options in the llama-cloud 2.x
                # SDK — passing it at top-level raised TypeError on every call and
                # silently failed 916/919 PDFs across a 1000-email run.
                # output_options.markdown.inline_images=False keeps image refs out of
                # the markdown stream; agentic tier inlines image transcriptions on its
                # own.
                result = await client.parsing.parse(
                    upload_file=str(file_path),
                    tier="agentic",
                    version="latest",
                    processing_options={"cost_optimizer": {"enable": True}},
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
                    # Use the shared empty-content sentinel so callers can
                    # distinguish "parser opened the file but extracted
                    # nothing" from "parser blew up". Other parsers
                    # (LocalCsv/LocalHtml/local docx) already raise this on
                    # the same condition; LlamaParse is now consistent.
                    raise EmptyContentError(
                        f"LlamaParse produced no content for {file_path}"
                    )

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
