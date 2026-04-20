"""Tests for MsgParser (Outlook .msg attachments)."""

from __future__ import annotations

import pytest

from mtss.parsers.msg_parser import MsgParser


class _FakeAttachment:
    def __init__(self, name: str):
        self.longFilename = name
        self.shortFilename = None
        self.name = name


class _FakeMessage:
    def __init__(self, **fields):
        self.sender = fields.get("sender")
        self.to = fields.get("to")
        self.cc = fields.get("cc")
        self.subject = fields.get("subject")
        self.date = fields.get("date")
        self.body = fields.get("body")
        self.htmlBody = fields.get("html")
        self.attachments = [
            _FakeAttachment(name) for name in fields.get("attachments", [])
        ]
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_msg_parser_assembles_headers_and_body(tmp_path, monkeypatch):
    """MsgParser formats MSG fields into the same shape as RFC822Parser."""
    p = tmp_path / "mail.msg"
    p.write_bytes(b"\xd0\xcf\x11\xe0")  # OLE magic; content isn't parsed here

    fake = _FakeMessage(
        sender="alice@example.com",
        to="bob@example.com",
        subject="Update",
        date="2024-01-01T12:00:00Z",
        body="Inline body text.",
        attachments=["spec.pdf", "chart.png"],
    )

    import mtss.parsers.msg_parser as module

    class _FakeExtractMsg:
        Message = lambda self, path: fake  # noqa: E731

    monkeypatch.setitem(
        __import__("sys").modules, "extract_msg", _FakeExtractMsg()
    )

    text = await MsgParser().parse(p)

    assert "**From:** alice@example.com" in text
    assert "**Subject:** Update" in text
    assert "Inline body text." in text
    assert "spec.pdf" in text and "chart.png" in text
    assert fake.closed is True


@pytest.mark.asyncio
async def test_msg_parser_falls_back_to_html_when_body_empty(
    tmp_path, monkeypatch
):
    p = tmp_path / "mail.msg"
    p.write_bytes(b"\xd0\xcf\x11\xe0")

    fake = _FakeMessage(
        sender="a@x",
        subject="HTML only",
        body="",
        html="<p>Hello <b>world</b></p>",
    )

    class _FakeExtractMsg:
        Message = lambda self, path: fake  # noqa: E731

    monkeypatch.setitem(
        __import__("sys").modules, "extract_msg", _FakeExtractMsg()
    )

    text = await MsgParser().parse(p)

    assert "Hello world" in text
    assert "<p>" not in text


def test_msg_parser_registered_for_outlook_mime():
    from pathlib import Path

    import mtss.parsers  # noqa: F401
    from mtss.parsers.registry import ParserRegistry

    parser = ParserRegistry.get_parser_for_file(
        Path("inner.msg"), "application/vnd.ms-outlook"
    )
    assert parser is not None
    assert parser.name == "msg"
