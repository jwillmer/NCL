"""Parser for forwarded emails delivered as message/rfc822 attachments."""

from __future__ import annotations

import email
import logging
from email import policy
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from .base import BaseParser, EmptyContentError

logger = logging.getLogger(__name__)


_HEADER_FIELDS = ("From", "To", "Cc", "Subject", "Date")


class RFC822Parser(BaseParser):
    """Extract a forwarded-email attachment to a headers + body text block.

    Handles the common case where an MUA attaches a prior message as
    ``message/rfc822`` (typical filename: ``.eml``). We emit a compact
    markdown-ish string so the downstream chunker/decider can treat it
    like any other text document. Nested attachments inside the forwarded
    message are listed by filename only — recursing would duplicate the
    extraction the outer email's own attachment loop already performs.
    """

    name = "rfc822"
    supported_mimetypes = {"message/rfc822"}
    supported_extensions = {".eml"}

    @property
    def is_available(self) -> bool:
        return True

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw = file_path.read_bytes()
        # Some clients prefix a UTF-8 BOM that corrupts the first header line;
        # mirror EMLParser's handling.
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]

        try:
            msg = email.message_from_bytes(raw, policy=policy.default)
        except Exception as e:
            raise ValueError(f"Failed to parse RFC822 message {file_path.name}: {e}") from e

        parts: list[str] = []
        for field in _HEADER_FIELDS:
            value = msg.get(field)
            if value:
                parts.append(f"**{field}:** {value}")

        body = _extract_body(msg)
        if body:
            parts.append("")
            parts.append(body)

        inner_attachments = _list_inner_attachments(msg)
        if inner_attachments:
            parts.append("")
            parts.append("**Attachments referenced:** " + ", ".join(inner_attachments))

        content = "\n".join(parts).strip()
        if not content:
            raise EmptyContentError(
                f"RFC822 parser produced no content for {file_path}"
            )

        logger.info(
            f"RFC822 parser extracted {len(content)} chars from {file_path.name}"
        )
        return content


def _extract_body(msg: EmailMessage) -> Optional[str]:
    """Prefer text/plain; fall back to stripped HTML."""
    plain: Optional[str] = None
    html: Optional[str] = None

    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition.lower():
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain" and plain is None:
                plain = _decode_part(part)
            elif ctype == "text/html" and html is None:
                html = _decode_part(part)
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            plain = _decode_part(msg)
        elif ctype == "text/html":
            html = _decode_part(msg)

    if plain:
        return plain.strip()
    if html:
        from .eml_parser import EMLParser

        converter = EMLParser.__new__(EMLParser)
        return converter.html_to_plain_text(html).strip()
    return None


def _decode_part(part: EmailMessage) -> Optional[str]:
    try:
        payload = part.get_payload(decode=True)
    except Exception:
        return None
    if not payload:
        return None

    declared = part.get_content_charset()
    for charset in (declared, "utf-8", "iso-8859-1", "cp1252"):
        if not charset:
            continue
        try:
            return payload.decode(charset)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace")


def _list_inner_attachments(msg: EmailMessage) -> list[str]:
    names: list[str] = []
    if not msg.is_multipart():
        return names
    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", "")).lower()
        filename = part.get_filename()
        if "attachment" in disposition or filename:
            if filename:
                names.append(filename)
    return names
