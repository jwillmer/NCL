"""Tests for 7z extraction — mirrors the ZIP path coverage."""

from __future__ import annotations

from pathlib import Path

import py7zr
import pytest

from mtss.parsers.attachment_processor import AttachmentProcessor


def _make_7z(dir_path: Path, members: dict[str, bytes]) -> Path:
    """Write a 7z archive with the given {arcname: bytes} members."""
    staging = dir_path / "_staging"
    staging.mkdir()
    archive_path = dir_path / "test.7z"
    with py7zr.SevenZipFile(archive_path, "w") as arc:
        for arcname, data in members.items():
            # Write the member through a real file so py7zr records paths the
            # same way production 7z tools do.
            real = staging / arcname.replace("/", "_")
            real.write_bytes(data)
            arc.write(real, arcname)
    return archive_path


def test_is_zip_file_recognises_7z(tmp_path):
    ap = AttachmentProcessor()
    archive = _make_7z(tmp_path, {"a.txt": b"hello"})
    assert ap.is_zip_file(str(archive)) is True
    assert ap.is_zip_file(str(archive), "application/x-7z-compressed") is True


def test_extract_zip_dispatches_to_7z_backend(tmp_path):
    """The public ``extract_zip`` must also handle 7z archives."""
    ap = AttachmentProcessor()
    archive = _make_7z(
        tmp_path,
        {
            "hello.txt": b"hello from 7z",
            "nested/note.xml": b"<r>1</r>",
        },
    )

    extract_dir = tmp_path / "out"
    results = ap.extract_zip(archive, extract_dir=extract_dir)

    files_by_name = {p.name: p for p, _ in results}
    assert "hello.txt" in files_by_name
    assert "note.xml" in files_by_name
    assert files_by_name["hello.txt"].read_bytes() == b"hello from 7z"


def test_seven_z_preprocess_marks_as_archive(tmp_path):
    """Preprocessor must classify .7z as is_zip so the ingest extracts it."""
    import asyncio

    from mtss.parsers.preprocessor import DocumentPreprocessor

    archive = _make_7z(tmp_path, {"a.txt": b"hi"})
    result = asyncio.run(DocumentPreprocessor().preprocess(archive))
    assert result.should_process is True
    assert result.is_zip is True


def test_count_zip_members_covers_7z(tmp_path):
    from mtss.ingest.attachment_handler import _count_zip_members

    archive = _make_7z(
        tmp_path,
        {
            "a.txt": b"a",
            "b.txt": b"b",
            "c/d.txt": b"c",
        },
    )
    assert _count_zip_members(archive) == 3


@pytest.mark.asyncio
async def test_text_parser_handles_xml_and_ini(tmp_path):
    """TextParser must accept .xml and .ini so the ingest stops logging
    them as ``unsupported_format`` and actually indexes their content."""
    from mtss.parsers.registry import ParserRegistry

    xml = tmp_path / "data.xml"
    xml.write_bytes(b"<root><child>value</child></root>")
    ini = tmp_path / "config.ini"
    ini.write_bytes(b"[Section]\nkey=value\n")

    xml_parser = ParserRegistry.get_parser_for_file(xml, "application/xml")
    ini_parser = ParserRegistry.get_parser_for_file(
        ini, "application/x-wine-extension-ini"
    )
    assert xml_parser is not None and xml_parser.name == "text"
    assert ini_parser is not None and ini_parser.name == "text"

    assert "<root>" in await xml_parser.parse(xml)
    assert "[Section]" in await ini_parser.parse(ini)
