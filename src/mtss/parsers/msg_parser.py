"""Parser for Outlook ``.msg`` (application/vnd.ms-outlook) attachments."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser, EmptyContentError

logger = logging.getLogger(__name__)


class MsgParser(BaseParser):
    """Extract text from Outlook ``.msg`` compound-document files.

    Uses ``extract-msg`` — a pure-Python OLE parser — so we do not depend
    on Outlook/Win32. Emits a headers + body block analogous to the
    RFC822 parser so the chunker sees a uniform shape across email
    attachment formats.
    """

    name = "msg"
    supported_mimetypes = {"application/vnd.ms-outlook"}
    supported_extensions = {".msg"}

    @property
    def is_available(self) -> bool:
        try:
            import extract_msg  # noqa: F401
            return True
        except ImportError:
            return False

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import extract_msg
        except ImportError:
            raise ValueError(
                "extract-msg is not installed. Install with: pip install extract-msg"
            )

        try:
            msg = extract_msg.Message(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to open .msg file {file_path.name}: {e}") from e

        try:
            parts: list[str] = []
            for label, value in (
                ("From", getattr(msg, "sender", None)),
                ("To", getattr(msg, "to", None)),
                ("Cc", getattr(msg, "cc", None)),
                ("Subject", getattr(msg, "subject", None)),
                ("Date", getattr(msg, "date", None)),
            ):
                if value:
                    parts.append(f"**{label}:** {value}")

            body = getattr(msg, "body", None) or ""
            if not body:
                html = getattr(msg, "htmlBody", None)
                if html:
                    if isinstance(html, bytes):
                        html = html.decode("utf-8", errors="replace")
                    from .eml_parser import EMLParser

                    converter = EMLParser.__new__(EMLParser)
                    body = converter.html_to_plain_text(html)

            body = (body or "").strip()
            if body:
                parts.append("")
                parts.append(body)

            attachment_names = []
            for att in getattr(msg, "attachments", None) or []:
                name = (
                    getattr(att, "longFilename", None)
                    or getattr(att, "shortFilename", None)
                    or getattr(att, "name", None)
                )
                if name:
                    attachment_names.append(str(name))
            if attachment_names:
                parts.append("")
                parts.append(
                    "**Attachments referenced:** " + ", ".join(attachment_names)
                )

            content = "\n".join(parts).strip()
        finally:
            close = getattr(msg, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        if not content:
            raise EmptyContentError(
                f"MSG parser produced no content for {file_path}"
            )

        logger.info(
            f"MSG parser extracted {len(content)} chars from {file_path.name}"
        )
        return content
