"""Tests for LocalCsvParser — including the binary-trailer regression."""

from __future__ import annotations

import pytest

from mtss.parsers.local_office_parser import LocalCsvParser


@pytest.mark.asyncio
async def test_parses_plain_csv(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_bytes(b'"name","value"\r\n"a",1\r\n"b",2\r\n')
    out = await LocalCsvParser().parse(p)
    assert out.splitlines() == ["name | value", "a | 1", "b | 2"]


@pytest.mark.asyncio
async def test_recovers_from_binary_trailer(tmp_path):
    """CSV body + NUL + binary junk must not crash the parser.

    Some exports (seen with BWTS OpLog files) append a binary trailer after
    the CSV body. That trailer contains unquoted bare ``\\r`` which trips
    ``csv.reader`` with: "new-line character seen in unquoted field - do you
    need to open the file with newline=''?". The parser truncates at the
    first NUL and normalizes line endings so the intact rows still parse.
    """
    body = b'"name","value"\r\n"a",1\r\n"b",2\r\n'
    # Simulate the corrupt trailer: NULs followed by a bare \r in high-ASCII.
    trailer = b"\x00\x00\x00garbage\r\x80\x81\r\x82\x83"
    p = tmp_path / "with_trailer.csv"
    p.write_bytes(body + trailer)

    out = await LocalCsvParser().parse(p)

    assert "a | 1" in out
    assert "b | 2" in out
    assert "garbage" not in out  # trailer was dropped


@pytest.mark.asyncio
async def test_empty_csv_raises_empty_content_error(tmp_path):
    """Empty CSVs must raise ``EmptyContentError`` so the Gemini fallback
    triggers — they previously raised generic ``ValueError`` and silently
    failed the attachment."""
    from mtss.parsers.base import EmptyContentError

    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    with pytest.raises(EmptyContentError):
        await LocalCsvParser().parse(p)


@pytest.mark.asyncio
async def test_whitespace_only_csv_raises_empty_content_error(tmp_path):
    """A CSV whose rows are all blank should also be classified as empty."""
    from mtss.parsers.base import EmptyContentError

    p = tmp_path / "blank.csv"
    p.write_bytes(b"\n\n,,,\n\n")
    with pytest.raises(EmptyContentError):
        await LocalCsvParser().parse(p)


@pytest.mark.asyncio
async def test_empty_html_raises_empty_content_error(tmp_path):
    """Empty HTML bodies must raise ``EmptyContentError`` to trigger the
    Gemini fallback — previously silent failures hid them from validate."""
    from mtss.parsers.base import EmptyContentError
    from mtss.parsers.local_office_parser import LocalHtmlParser

    p = tmp_path / "empty.html"
    p.write_bytes(b"<html><body></body></html>")
    with pytest.raises(EmptyContentError):
        await LocalHtmlParser().parse(p)
