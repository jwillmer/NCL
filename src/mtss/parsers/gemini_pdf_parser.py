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

from .._io import read_bytes_async
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

    # Class-level semaphore so the concurrency cap is truly process-wide.
    # Previously instantiated per-instance in __init__, and callers in
    # attachment_processor create a fresh GeminiPDFParser per attachment —
    # that defeated the throttle entirely. Lazily initialised on first use
    # because asyncio.Semaphore wants a running event loop.
    _semaphore: asyncio.Semaphore | None = None
    _semaphore_capacity: int | None = None

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        """Return the shared semaphore, (re)creating it if the configured
        capacity has changed (e.g. tests tweaking settings between runs)."""
        capacity = get_settings().max_concurrent_gemini_pdf
        if cls._semaphore is None or cls._semaphore_capacity != capacity:
            cls._semaphore = asyncio.Semaphore(capacity)
            cls._semaphore_capacity = capacity
        return cls._semaphore

    @property
    def model_name(self) -> str | None:
        # Resolved lazily so tests / runtime config changes see the current value.
        return getattr(get_settings(), "gemini_pdf_model", None)

    @property
    def is_available(self) -> bool:
        # OpenRouter routes PDFs natively to Gemini; we trust the configured
        # model rather than asking LiteLLM. Its ``supports_pdf_input`` table
        # silently returns False for ``openrouter/google/gemini-2.5-flash``,
        # which would disable this parser even though the route works.
        return bool(getattr(get_settings(), "openrouter_api_key", None))

    async def parse(self, file_path: Path) -> str:
        settings = get_settings()
        # End-to-end cap: halving + per-batch timeouts can stack if every
        # batch goes pathological. This guarantees a doc either returns
        # within budget or fails with something the caller can recover from.
        return await asyncio.wait_for(
            self._parse_impl(file_path),
            timeout=settings.gemini_pdf_doc_timeout_seconds,
        )

    async def _parse_impl(self, file_path: Path) -> str:
        from pypdf import PdfReader

        from .preprocessor import _safe_count_pdf_pages

        settings = get_settings()
        # Cheap header-only page-count peek (pymupdf) so a multi-hundred-MB
        # PDF over the ceiling can be rejected without pulling the whole file
        # into memory. `_safe_count_pdf_pages` returns None on any read error;
        # the full pypdf parse below is the authoritative count either way.
        peek_count = await asyncio.to_thread(_safe_count_pdf_pages, file_path)
        if peek_count is not None and peek_count > settings.gemini_pdf_hard_page_ceiling:
            raise TooLargeError(
                f"PDF has {peek_count} pages, exceeds hard ceiling "
                f"{settings.gemini_pdf_hard_page_ceiling}"
            )

        # File read + pypdf parse off the event loop — PDFs can be 100s of MB.
        pdf_bytes = await read_bytes_async(file_path)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
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

        # Per-doc halving counter, shared across all batches. Prevents a
        # scanned form where every batch triggers halving from running the
        # recursion into the doc timeout.
        halvings_remaining = [settings.gemini_pdf_max_halvings_per_doc]

        semaphore = self._get_semaphore()

        async def _run_batch(batch_range: Tuple[int, int]) -> str:
            start, end = batch_range
            async with semaphore:
                return await self._parse_range_adaptive(
                    pdf_bytes, file_path.stem, start, end, halvings_remaining
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
        self,
        pdf_bytes: bytes,
        stem: str,
        start: int,
        end: int,
        halvings_remaining: list,
    ) -> str:
        """Parse pages [start, end); halve on truncation OR timeout until
        batch size = 1. A single dense page that blows past the per-call
        timeout is reported as empty rather than blocking the ingest.
        ``halvings_remaining`` is a single-element list used as a mutable
        shared counter across all batches of one doc."""
        try:
            text, truncated = await self._call_for_range(pdf_bytes, stem, start, end)
        except asyncio.TimeoutError:
            if end - start <= 1:
                logger.warning(
                    "Gemini batch timed out on single page %s p%d; skipping",
                    stem,
                    start + 1,
                )
                return ""
            logger.info(
                "Gemini batch timed out on %s p%d-%d; halving",
                stem,
                start + 1,
                end,
            )
            truncated = True
            text = ""

        if not truncated or end - start <= 1:
            return text

        if halvings_remaining[0] <= 0:
            logger.warning(
                "Gemini halving budget exhausted for %s at p%d-%d; "
                "returning partial output",
                stem,
                start + 1,
                end,
            )
            return text
        halvings_remaining[0] -= 1

        mid = start + (end - start) // 2
        left = await self._parse_range_adaptive(
            pdf_bytes, stem, start, mid, halvings_remaining
        )
        right = await self._parse_range_adaptive(
            pdf_bytes, stem, mid, end, halvings_remaining
        )
        return f"{left}\n\n{right}" if left and right else (left or right)

    async def _call_for_range(
        self, pdf_bytes: bytes, stem: str, start: int, end: int
    ) -> Tuple[str, bool]:
        """Run one LiteLLM call against pages [start, end); return (text, truncated)."""
        settings = get_settings()
        # Slice + base64 on a worker thread: PdfWriter + base64.b64encode is
        # CPU-bound (tens of ms for large batches) and blocks every other
        # coroutine when run on the loop. Each call builds its own PdfReader
        # from pdf_bytes, so threads share no pypdf state.
        b64 = await asyncio.to_thread(_slice_pages_to_base64, pdf_bytes, start, end)
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
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=settings.gemini_pdf_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                extra_body=OPENROUTER_PRIVACY_EXTRA_BODY,
            ),
            timeout=settings.gemini_pdf_call_timeout_seconds,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        truncated = finish_reason == "length"

        # Output-size kill switch: on scanned forms Gemini hallucinates
        # repetitive content, blowing past any reasonable chars-per-page
        # ratio. Force the halving path so we converge on a smaller range
        # that either produces sensible output or gets skipped.
        pages_in_batch = max(1, end - start)
        max_chars = settings.gemini_pdf_max_chars_per_page * pages_in_batch
        if len(text) > max_chars:
            logger.info(
                "Gemini output for %s p%d-%d is %d chars (limit %d); "
                "treating as truncated so halving can recover",
                stem,
                start + 1,
                end,
                len(text),
                max_chars,
            )
            truncated = True

        return text, truncated


def _slice_pages_to_base64(pdf_bytes: bytes, start: int, end: int) -> str:
    """Build a PDF from ``pages[start:end]`` and return its base64 encoding.

    Takes the raw PDF bytes rather than a shared ``PdfReader`` so each caller
    (including those invoked via ``asyncio.to_thread``) works on its own
    reader/writer instances — pypdf is not documented as thread-safe for
    concurrent reads of a shared reader.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for idx in range(start, end):
        writer.add_page(reader.pages[idx])
    buf = io.BytesIO()
    writer.write(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")
