"""Gemini 2.5 Flash PDF parser via OpenRouter + LiteLLM.

Used as the fallback/complex tier for PDFs and modern document formats when
PyMuPDF4LLM produces empty output or the PDF is classified COMPLEX. OpenRouter
routes PDFs natively to Gemini — no OCR preprocessing fee.

Pagination strategy:
  - Page count above ``gemini_pdf_hard_page_ceiling`` → ``TooLargeError``
    (caller forces SUMMARY mode; prevents unbounded spend).
  - Otherwise sliced into ``gemini_pdf_page_batch_size``-page batches.
  - Each batch becomes one ``litellm.acompletion`` call with a ``type:"file"``
    content block (base64-encoded PDF slice).
  - If a batch returns ``finish_reason == "length"`` the batch is split in
    half and retried recursively.
  - Concurrent batches capped by ``asyncio.Semaphore``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import List, Tuple

import litellm

from ..config import get_settings
from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY
from .base import BaseParser, EmptyContentError

litellm.drop_params = True

logger = logging.getLogger(__name__)


class TooLargeError(ValueError):
    """Raised when a PDF exceeds the configured hard page ceiling."""


_PROMPT = (
    "Convert this PDF section to clean markdown. "
    "Preserve headings, tables, and lists. "
    "Return only the markdown — no commentary, no code fences."
)

# Empirical Gemini 2.5 Flash PDF rate via OpenRouter. Used by the per-doc cost
# guard; tunable via the GEMINI_PDF_MAX_COST_USD_PER_DOC setting.
_GEMINI_PER_PAGE_COST_USD = 0.0025


class GeminiPDFParser(BaseParser):
    """Parser for complex PDFs using Gemini 2.5 Flash via OpenRouter."""

    name = "gemini_pdf"
    supported_extensions = {".pdf"}
    supported_mimetypes = {"application/pdf"}

    def __init__(self) -> None:
        # Process-wide throttle: this parser is shared across documents in the
        # ingest, so the semaphore caps OpenRouter calls globally — not per-doc.
        self._semaphore = asyncio.Semaphore(get_settings().max_concurrent_gemini_pdf)

    @property
    def is_available(self) -> bool:
        # OpenRouter routes PDFs natively to Gemini; we trust the configured
        # model rather than asking LiteLLM. Its ``supports_pdf_input`` table
        # silently returns False for ``openrouter/google/gemini-2.5-flash``,
        # which would disable this parser even though the route works.
        return bool(getattr(get_settings(), "openrouter_api_key", None))

    async def parse(self, file_path: Path) -> str:
        from pypdf import PdfReader

        settings = get_settings()
        pdf_bytes = file_path.read_bytes()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = list(reader.pages)
        page_count = len(pages)
        if page_count > settings.gemini_pdf_hard_page_ceiling:
            raise TooLargeError(
                f"PDF has {page_count} pages, exceeds hard ceiling "
                f"{settings.gemini_pdf_hard_page_ceiling}"
            )

        estimated_cost = page_count * _GEMINI_PER_PAGE_COST_USD
        if estimated_cost > settings.gemini_pdf_max_cost_usd_per_doc:
            raise TooLargeError(
                f"PDF estimated cost ${estimated_cost:.2f} exceeds per-doc cap "
                f"${settings.gemini_pdf_max_cost_usd_per_doc:.2f} "
                f"({page_count} pages * ${_GEMINI_PER_PAGE_COST_USD}/page)"
            )

        batch_size = max(1, settings.gemini_pdf_page_batch_size)
        batches: List[Tuple[int, int]] = [
            (start, min(start + batch_size, page_count))
            for start in range(0, page_count, batch_size)
        ]

        async def _run_batch(batch_range: Tuple[int, int]) -> str:
            start, end = batch_range
            async with self._semaphore:
                return await self._parse_range_adaptive(
                    pages, file_path.stem, start, end
                )

        outputs = await asyncio.gather(
            *(_run_batch(b) for b in batches), return_exceptions=False
        )

        markdown = "\n\n---\n\n".join(s for s in outputs if s)
        if not markdown.strip():
            raise EmptyContentError(
                f"Gemini parser produced no content for {file_path}"
            )
        logger.info(
            "Gemini PDF parser extracted %d chars from %s (%d pages, %d batches)",
            len(markdown),
            file_path.name,
            page_count,
            len(batches),
        )
        return markdown

    async def _parse_range_adaptive(
        self, pages, stem: str, start: int, end: int
    ) -> str:
        """Parse pages [start, end); halve on truncation until batch size = 1."""
        text, truncated = await self._call_for_range(pages, stem, start, end)
        if not truncated or end - start <= 1:
            return text

        mid = start + (end - start) // 2
        left = await self._parse_range_adaptive(pages, stem, start, mid)
        right = await self._parse_range_adaptive(pages, stem, mid, end)
        return f"{left}\n\n{right}" if left and right else (left or right)

    async def _call_for_range(
        self, pages, stem: str, start: int, end: int
    ) -> Tuple[str, bool]:
        """Run one LiteLLM call against pages [start, end); return (text, truncated)."""
        settings = get_settings()
        b64 = _slice_pages_to_base64(pages, start, end)
        content = [
            {"type": "text", "text": _PROMPT},
            {
                "type": "file",
                "file": {
                    "filename": f"{stem}_p{start + 1}-{end}.pdf",
                    "file_data": f"data:application/pdf;base64,{b64}",
                },
            },
        ]
        response = await litellm.acompletion(
            model=settings.gemini_pdf_model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            extra_body=OPENROUTER_PRIVACY_EXTRA_BODY,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        truncated = finish_reason == "length"
        return text, truncated


def _slice_pages_to_base64(pages, start: int, end: int) -> str:
    """Build a PDF from ``pages[start:end]`` and return its base64 encoding."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for idx in range(start, end):
        writer.add_page(pages[idx])
    buf = io.BytesIO()
    writer.write(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")
