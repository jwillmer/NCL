"""Tests for the Gemini PDF parser (LiteLLM + OpenRouter + paginated upload)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_completion(text: str, finish_reason: str = "stop") -> MagicMock:
    """Build a LiteLLM-shaped response object."""
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    return resp


@pytest.fixture
def _fake_pdf(tmp_path) -> Path:
    """Minimal PDF file — just enough that pypdf can open it."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000015 00000 n \n"
        b"0000000066 00000 n \n"
        b"0000000123 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n200\n%%EOF\n"
    )
    p = tmp_path / "test.pdf"
    p.write_bytes(pdf_bytes)
    return p


# ------------------------------ Availability ------------------------------ #


class TestGeminiPDFParserAvailability:
    def test_not_available_without_openrouter_key(self, comprehensive_mock_settings):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = ""
        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            assert GeminiPDFParser().is_available is False

    def test_available_when_openrouter_key_set(self, comprehensive_mock_settings):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            assert GeminiPDFParser().is_available is True


# -------------------------------- Parse --------------------------------- #


class TestGeminiPDFParserParse:
    @pytest.mark.asyncio
    async def test_raises_file_not_found(self, tmp_path, comprehensive_mock_settings):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        missing = tmp_path / "nope.pdf"
        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            with pytest.raises(FileNotFoundError):
                await GeminiPDFParser().parse(missing)

    @pytest.mark.asyncio
    async def test_single_page_pdf_makes_one_call(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        comprehensive_mock_settings.gemini_pdf_page_batch_size = 25
        comprehensive_mock_settings.gemini_pdf_hard_page_ceiling = 200

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_completion("# Extracted\n\nHello."),
        ) as m:
            out = await GeminiPDFParser().parse(_fake_pdf)
            assert "Hello." in out
            assert m.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_response_raises_empty_content_error(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        from mtss.parsers.base import EmptyContentError
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_completion(""),
        ):
            with pytest.raises(EmptyContentError):
                await GeminiPDFParser().parse(_fake_pdf)

    @pytest.mark.asyncio
    async def test_over_page_ceiling_raises_too_large(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        from mtss.parsers.gemini_pdf_parser import (
            GeminiPDFParser,
            TooLargeError,
        )

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        comprehensive_mock_settings.gemini_pdf_hard_page_ceiling = 0

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            with pytest.raises(TooLargeError):
                await GeminiPDFParser().parse(_fake_pdf)

    @pytest.mark.asyncio
    async def test_over_per_doc_cost_cap_raises_too_large(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        """Per-doc Gemini cost guard fires before any API call when the page
        count's estimated cost exceeds the configured cap."""
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser, TooLargeError

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        comprehensive_mock_settings.gemini_pdf_hard_page_ceiling = 1000
        # 1 page * $0.0025 = $0.0025; cap at $0.001 → must trip.
        comprehensive_mock_settings.gemini_pdf_max_cost_usd_per_doc = 0.001

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            with pytest.raises(TooLargeError):
                await GeminiPDFParser().parse(_fake_pdf)

    @pytest.mark.asyncio
    async def test_request_includes_base64_file_content_block(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        """Validates the litellm-docs-specified content-block shape."""
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_completion("# ok"),
        ) as m:
            await GeminiPDFParser().parse(_fake_pdf)

        call_kwargs = m.await_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # Content should be a list with at least a text block and a file block.
        file_blocks = [c for c in content if c.get("type") == "file"]
        assert len(file_blocks) == 1
        fb = file_blocks[0]["file"]
        assert "filename" in fb
        assert fb["file_data"].startswith("data:application/pdf;base64,")

    @pytest.mark.asyncio
    async def test_privacy_body_is_applied(
        self, _fake_pdf, comprehensive_mock_settings
    ):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_completion("# ok"),
        ) as m:
            await GeminiPDFParser().parse(_fake_pdf)

        call_kwargs = m.await_args.kwargs
        extra = call_kwargs["extra_body"]
        assert extra["provider"]["data_collection"] == "deny"


# ------------------------------ Pagination ------------------------------ #


class TestGeminiPDFParserPagination:
    """Covers the page-batch splitting + adaptive halving on truncation."""

    @pytest.mark.asyncio
    async def test_multi_page_pdf_paginates_by_batch_size(
        self, tmp_path, comprehensive_mock_settings
    ):
        """A 50-page PDF at batch size 25 should trigger 2 calls."""
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        src = _build_pdf_with_pages(tmp_path / "big.pdf", page_count=50)
        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        comprehensive_mock_settings.gemini_pdf_page_batch_size = 25
        comprehensive_mock_settings.gemini_pdf_hard_page_ceiling = 200

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_completion("# batch-output"),
        ) as m:
            out = await GeminiPDFParser().parse(src)
            assert m.await_count == 2
            # Both batches concatenated.
            assert out.count("# batch-output") == 2

    @pytest.mark.asyncio
    async def test_truncated_response_triggers_halving_retry(
        self, tmp_path, comprehensive_mock_settings
    ):
        from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

        src = _build_pdf_with_pages(tmp_path / "dense.pdf", page_count=10)
        comprehensive_mock_settings.openrouter_api_key = "sk-or-test"
        comprehensive_mock_settings.gemini_pdf_page_batch_size = 10
        comprehensive_mock_settings.gemini_pdf_hard_page_ceiling = 200

        # First call truncates; subsequent halved calls succeed.
        responses = [
            _make_completion("# truncated", finish_reason="length"),
            _make_completion("# half1", finish_reason="stop"),
            _make_completion("# half2", finish_reason="stop"),
        ]

        with patch(
            "mtss.parsers.gemini_pdf_parser.get_settings",
            return_value=comprehensive_mock_settings,
        ), patch(
            "mtss.parsers.gemini_pdf_parser.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=responses,
        ) as m:
            out = await GeminiPDFParser().parse(src)
            # Original call + 2 halves = 3 calls.
            assert m.await_count == 3
            assert "half1" in out and "half2" in out


# ---------------------------- small PDF helper ---------------------------- #


def _build_pdf_with_pages(path: Path, page_count: int) -> Path:
    """Build a tiny multi-page PDF on disk using pypdf.PdfWriter."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)
    return path
