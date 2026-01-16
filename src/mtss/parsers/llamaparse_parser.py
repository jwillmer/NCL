"""LlamaParse parser plugin for document processing."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_settings
from .base import BaseParser

logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """Parser using LlamaParse API for document extraction.

    Default parser for all document types including:
    - PDF documents
    - Office formats (DOCX, PPTX, XLSX)
    - Legacy Office formats (DOC, XLS, PPT)
    - Other document formats (CSV, RTF, EPUB, ODT, ODS, ODP)
    """

    name = "llamaparse"

    supported_mimetypes = {
        # PDF
        "application/pdf",
        # Modern Office formats
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        # Legacy Office formats
        "application/msword",  # .doc
        "application/vnd.ms-excel",  # .xls
        "application/vnd.ms-powerpoint",  # .ppt
        # Other formats
        "text/csv",
        "application/rtf",
        "text/rtf",
        "application/epub+zip",
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
        # HTML
        "text/html",
    }

    supported_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".xls",
        ".ppt",
        ".csv",
        ".rtf",
        ".epub",
        ".odt",
        ".ods",
        ".odp",
        ".html",
        ".htm",
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

        # Hardcoded configuration as per plan
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
        )

        try:
            logger.info(f"Parsing {file_path.name} with LlamaParse...")
            result = await parser.aparse(str(file_path))

            # Get markdown (combined, not split by page)
            markdown_docs = result.get_markdown_documents(split_by_page=False)
            markdown_text = "\n\n".join(doc.text for doc in markdown_docs)

            if not markdown_text or not markdown_text.strip():
                raise ValueError(f"LlamaParse produced no content for {file_path}")

            logger.info(
                f"LlamaParse extracted {len(markdown_text)} chars from {file_path.name}"
            )
            return markdown_text

        except Exception as e:
            if "LlamaParse" in str(type(e).__name__):
                raise ValueError(f"LlamaParse failed: {e}") from e
            raise
