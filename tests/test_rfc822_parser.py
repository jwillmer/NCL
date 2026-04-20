"""Tests for RFC822Parser (forwarded-email attachments)."""

from __future__ import annotations

import pytest

from mtss.parsers.base import EmptyContentError
from mtss.parsers.rfc822_parser import RFC822Parser


RAW_EMAIL = (
    b"From: alice@example.com\r\n"
    b"To: bob@example.com\r\n"
    b"Cc: carol@example.com\r\n"
    b"Subject: Quarterly report\r\n"
    b"Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Please find the numbers below.\r\n"
    b"All metrics look healthy.\r\n"
)


@pytest.mark.asyncio
async def test_plain_rfc822_yields_headers_and_body(tmp_path):
    p = tmp_path / "forward.eml"
    p.write_bytes(RAW_EMAIL)

    text = await RFC822Parser().parse(p)

    assert "**From:** alice@example.com" in text
    assert "**To:** bob@example.com" in text
    assert "**Cc:** carol@example.com" in text
    assert "**Subject:** Quarterly report" in text
    assert "Please find the numbers below." in text


@pytest.mark.asyncio
async def test_html_only_body_is_stripped(tmp_path):
    raw = (
        b"From: a@x\r\n"
        b"Subject: HTML only\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b"<p>Hello <b>world</b></p>"
    )
    p = tmp_path / "html.eml"
    p.write_bytes(raw)

    text = await RFC822Parser().parse(p)

    assert "Hello world" in text
    assert "<p>" not in text


@pytest.mark.asyncio
async def test_utf8_bom_is_stripped(tmp_path):
    p = tmp_path / "bom.eml"
    p.write_bytes(b"\xef\xbb\xbf" + RAW_EMAIL)

    text = await RFC822Parser().parse(p)

    # First header must parse despite the BOM prefix.
    assert "**From:** alice@example.com" in text


@pytest.mark.asyncio
async def test_inner_attachment_names_listed(tmp_path):
    raw = (
        b'From: a@x\r\n'
        b'Subject: with attach\r\n'
        b'MIME-Version: 1.0\r\n'
        b'Content-Type: multipart/mixed; boundary="BOUND"\r\n'
        b'\r\n'
        b'--BOUND\r\n'
        b'Content-Type: text/plain\r\n'
        b'\r\n'
        b'See attached.\r\n'
        b'--BOUND\r\n'
        b'Content-Type: application/pdf; name="report.pdf"\r\n'
        b'Content-Disposition: attachment; filename="report.pdf"\r\n'
        b'\r\n'
        b'%PDF-1.4\r\n'
        b'--BOUND--\r\n'
    )
    p = tmp_path / "multi.eml"
    p.write_bytes(raw)

    text = await RFC822Parser().parse(p)

    assert "See attached." in text
    assert "report.pdf" in text


@pytest.mark.asyncio
async def test_empty_body_raises_empty_content_error(tmp_path):
    p = tmp_path / "empty.eml"
    p.write_bytes(b"")
    with pytest.raises(EmptyContentError):
        await RFC822Parser().parse(p)


def test_registered_for_message_rfc822():
    """Parser must be discoverable by MIME so the ingest routes to it."""
    from pathlib import Path

    from mtss.parsers.registry import ParserRegistry

    # Re-register to ensure the registry sees the class (tests may run in
    # any order; registration happens at import time).
    import mtss.parsers  # noqa: F401

    parser = ParserRegistry.get_parser_for_file(
        Path("inner.eml"), "message/rfc822"
    )
    assert parser is not None
    assert parser.name == "rfc822"
