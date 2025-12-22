"""Parser for EML email files using Python's email library."""

from __future__ import annotations

import email
import hashlib
import re
from email import policy
from email.message import EmailMessage as EmailMsg
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ..config import get_settings
from ..models.document import EmailMessage, EmailMetadata, ParsedAttachment, ParsedEmail


class EMLParser:
    """Parser for EML email files and conversations.

    Handles both single emails and threaded conversations.
    Extracts participants, messages, and attachments.
    """

    def __init__(self, attachments_dir: Optional[Path] = None):
        """Initialize the EML parser.

        Args:
            attachments_dir: Directory to save extracted attachments.
                            Defaults to settings.attachments_dir.
        """
        settings = get_settings()
        self.attachments_dir = attachments_dir or settings.attachments_dir
        self.attachments_dir.mkdir(parents=True, exist_ok=True)

    def parse_file(self, eml_path: Path) -> ParsedEmail:
        """Parse an EML file and extract all content.

        Args:
            eml_path: Path to the EML file.

        Returns:
            ParsedEmail containing metadata, messages, and attachments.
        """
        with open(eml_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        return self._parse_message(msg, eml_path)

    def _parse_message(self, msg: EmailMsg, source_path: Path) -> ParsedEmail:
        """Parse an EmailMessage object.

        Args:
            msg: The email message to parse.
            source_path: Original file path (used for attachment directory naming).

        Returns:
            ParsedEmail with conversation data.
        """
        # Extract body content
        body_plain, body_html = self._extract_body(msg)
        body_text = body_plain or ""
        if not body_text and body_html:
            body_text = self.html_to_plain_text(body_html)

        # Parse conversation - try to split into individual messages
        messages = self._parse_conversation(body_text, msg)

        # Build conversation metadata
        metadata = self._build_conversation_metadata(msg, messages)

        # Extract attachments
        attachments = self._extract_attachments(msg, source_path)

        return ParsedEmail(
            metadata=metadata,
            messages=messages,
            full_text=body_text,
            attachments=attachments,
        )

    def _build_conversation_metadata(
        self, msg: EmailMsg, messages: List[EmailMessage]
    ) -> EmailMetadata:
        """Build conversation-level metadata from parsed messages.

        Args:
            msg: Original email message for headers.
            messages: Parsed individual messages.

        Returns:
            EmailMetadata with conversation info.
        """
        # Collect all participants
        participants: Set[str] = set()

        # Add participants from headers
        from_addr = self._extract_email_address(msg.get("From", ""))
        if from_addr:
            participants.add(from_addr)

        for addr in self._parse_address_list(msg.get("To", "")):
            email_addr = self._extract_email_address(addr)
            if email_addr:
                participants.add(email_addr)

        for addr in self._parse_address_list(msg.get("Cc", "")):
            email_addr = self._extract_email_address(addr)
            if email_addr:
                participants.add(email_addr)

        # Add participants from parsed messages
        for message in messages:
            email_addr = self._extract_email_address(message.from_address)
            if email_addr:
                participants.add(email_addr)
            for addr in message.to_addresses:
                email_addr = self._extract_email_address(addr)
                if email_addr:
                    participants.add(email_addr)

        # Determine initiator (first sender in thread, or the from address)
        initiator = None
        if messages:
            # Messages are typically newest-first in EML, so last is initiator
            initiator = self._extract_email_address(messages[-1].from_address)
        if not initiator:
            initiator = from_addr

        # Get dates
        dates = [m.date for m in messages if m.date]
        date_start = min(dates) if dates else None
        date_end = max(dates) if dates else None

        # If no dates from messages, try header
        if not date_start:
            date_str = msg.get("Date", "")
            if date_str:
                try:
                    date_start = parsedate_to_datetime(date_str)
                    date_end = date_start
                except (ValueError, TypeError):
                    pass

        # Get message IDs and references
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        references_str = msg.get("References", "")
        references = references_str.split() if references_str else []

        return EmailMetadata(
            subject=msg.get("Subject", ""),
            participants=sorted(list(participants)),
            initiator=initiator,
            date_start=date_start,
            date_end=date_end,
            message_count=len(messages) if messages else 1,
            message_ids=[message_id] if message_id else [],
            in_reply_to=in_reply_to or None,
            references=references,
        )

    def _parse_conversation(self, body_text: str, msg: EmailMsg) -> List[EmailMessage]:
        """Parse email body into individual messages in conversation.

        Handles common email thread formats with quoted replies.

        Args:
            body_text: Full email body text.
            msg: Original message for header info.

        Returns:
            List of EmailMessage objects representing the conversation.
        """
        messages: List[EmailMessage] = []

        # Common patterns for message separators in email threads
        # Pattern: "From: ... Sent: ... To: ... Subject: ..."
        # Pattern: "On <date>, <person> wrote:"
        # Pattern: "-----Original Message-----"

        # Split on common separators
        separator_patterns = [
            r"(?=^-{3,}\s*Original Message\s*-{3,})",  # -----Original Message-----
            r"(?=^From:\s+.+?\nSent:\s+.+?\nTo:\s+.+?\n)",  # Outlook style
            r"(?=^On\s+.+?,\s+.+?\s+wrote:)",  # Gmail style: "On Mon, Jan 1, John wrote:"
            r"(?=^>{1,}\s*On\s+.+?,\s+.+?\s+wrote:)",  # Quoted gmail style
        ]

        # Try to split the conversation
        parts = [body_text]
        for pattern in separator_patterns:
            new_parts = []
            for part in parts:
                split_parts = re.split(pattern, part, flags=re.MULTILINE | re.IGNORECASE)
                new_parts.extend([p.strip() for p in split_parts if p.strip()])
            parts = new_parts

        # Parse each part into an EmailMessage
        for part in parts:
            message = self._parse_single_message(part)
            if message and message.content.strip():
                messages.append(message)

        # If no messages parsed, create one from the full text
        if not messages and body_text.strip():
            from_addr = msg.get("From", "")
            date_str = msg.get("Date", "")
            date = None
            if date_str:
                try:
                    date = parsedate_to_datetime(date_str)
                except (ValueError, TypeError):
                    pass

            messages.append(
                EmailMessage(
                    from_address=from_addr,
                    to_addresses=self._parse_address_list(msg.get("To", "")),
                    cc_addresses=self._parse_address_list(msg.get("Cc", "")),
                    date=date,
                    content=body_text,
                )
            )

        return messages

    def _parse_single_message(self, text: str) -> Optional[EmailMessage]:
        """Parse a single message block from conversation.

        Args:
            text: Text block representing one message.

        Returns:
            EmailMessage or None if parsing fails.
        """
        from_address = ""
        to_addresses: List[str] = []
        date = None
        content = text

        # Try to extract "From:" line
        from_match = re.search(r"^From:\s*(.+?)$", text, re.MULTILINE | re.IGNORECASE)
        if from_match:
            from_address = from_match.group(1).strip()

        # Try to extract "To:" line
        to_match = re.search(r"^To:\s*(.+?)$", text, re.MULTILINE | re.IGNORECASE)
        if to_match:
            to_addresses = self._parse_address_list(to_match.group(1))

        # Try to extract date from various formats
        # "Sent: Monday, January 1, 2024 12:00 PM"
        sent_match = re.search(r"^Sent:\s*(.+?)$", text, re.MULTILINE | re.IGNORECASE)
        if sent_match:
            try:
                date = parsedate_to_datetime(sent_match.group(1).strip())
            except (ValueError, TypeError):
                pass

        # "On Mon, Jan 1, 2024 at 12:00 PM, John <john@example.com> wrote:"
        on_match = re.search(
            r"^On\s+(.+?),\s+(.+?)\s+wrote:", text, re.MULTILINE | re.IGNORECASE
        )
        if on_match and not from_address:
            from_address = on_match.group(2).strip()
            try:
                date = parsedate_to_datetime(on_match.group(1).strip())
            except (ValueError, TypeError):
                pass

        # Remove header lines from content
        content = re.sub(
            r"^(From|To|Cc|Sent|Subject|Date):\s*.+?$",
            "",
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        content = re.sub(r"^On\s+.+?,\s+.+?\s+wrote:\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^-{3,}\s*Original Message\s*-{3,}\s*$", "", content, flags=re.MULTILINE)

        # Remove leading ">" quote markers
        content = re.sub(r"^>+\s?", "", content, flags=re.MULTILINE)

        content = content.strip()

        if not content:
            return None

        return EmailMessage(
            from_address=from_address,
            to_addresses=to_addresses,
            date=date,
            content=content,
        )

    def _extract_email_address(self, addr_string: str) -> Optional[str]:
        """Extract just the email address from a string like 'John Doe <john@example.com>'.

        Args:
            addr_string: Address string possibly with name.

        Returns:
            Email address or None.
        """
        if not addr_string:
            return None

        # Try to extract from angle brackets
        match = re.search(r"<([^>]+@[^>]+)>", addr_string)
        if match:
            return match.group(1).lower().strip()

        # Check if it's already just an email
        if "@" in addr_string and " " not in addr_string.strip():
            return addr_string.lower().strip()

        # Try to find email pattern
        match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", addr_string)
        if match:
            return match.group(0).lower().strip()

        return None

    def _parse_address_list(self, addr_string: str) -> List[str]:
        """Parse comma-separated email addresses.

        Args:
            addr_string: Comma-separated address string.

        Returns:
            List of individual addresses.
        """
        if not addr_string:
            return []
        return [addr.strip() for addr in addr_string.split(",") if addr.strip()]

    def _extract_body(self, msg: EmailMsg) -> Tuple[Optional[str], Optional[str]]:
        """Extract plain text and HTML body from email.

        Args:
            msg: The email message.

        Returns:
            Tuple of (plain_text_body, html_body).
        """
        body_plain = None
        body_html = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain" and body_plain is None:
                    body_plain = self._decode_payload(part)
                elif content_type == "text/html" and body_html is None:
                    body_html = self._decode_payload(part)
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                body_plain = self._decode_payload(msg)
            elif content_type == "text/html":
                body_html = self._decode_payload(msg)

        return body_plain, body_html

    def _decode_payload(self, part: EmailMsg) -> Optional[str]:
        """Decode email part payload to string.

        Args:
            part: Email part to decode.

        Returns:
            Decoded string content or None.
        """
        try:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        except Exception:
            pass
        return None

    def _extract_attachments(
        self, msg: EmailMsg, source_path: Path
    ) -> List[ParsedAttachment]:
        """Extract and save attachments from email.

        Args:
            msg: The email message.
            source_path: Path to original EML file.

        Returns:
            List of ParsedAttachment objects.
        """
        attachments = []

        if not msg.is_multipart():
            return attachments

        # Create subdirectory for this email's attachments based on file hash
        email_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:12]
        email_attach_dir = self.attachments_dir / email_hash
        email_attach_dir.mkdir(parents=True, exist_ok=True)

        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))

            # Check for attachments (explicit or inline with filename)
            filename = part.get_filename()
            if "attachment" not in content_disposition and not filename:
                continue

            filename = filename or "unnamed_attachment"
            filename = self._sanitize_filename(filename)
            content_type = part.get_content_type()

            payload = part.get_payload(decode=True)
            if not payload:
                continue

            # Save attachment to disk with unique name if needed
            saved_path = email_attach_dir / filename
            counter = 1
            while saved_path.exists():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                saved_path = email_attach_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            with open(saved_path, "wb") as f:
                f.write(payload)

            attachments.append(
                ParsedAttachment(
                    filename=filename,
                    content_type=content_type,
                    size_bytes=len(payload),
                    saved_path=str(saved_path),
                )
            )

        return attachments

    def _sanitize_filename(self, filename: str) -> str:
        """Remove or replace invalid filename characters.

        Args:
            filename: Original filename.

        Returns:
            Sanitized filename safe for filesystem.
        """
        # Remove null bytes and path separators
        filename = filename.replace("\x00", "").replace("/", "_").replace("\\", "_")
        # Remove other problematic characters
        filename = re.sub(r'[<>:"|?*]', "_", filename)
        # Limit length
        return filename[:255]

    def html_to_plain_text(self, html: str) -> str:
        """Convert HTML body to plain text (basic conversion).

        Args:
            html: HTML content.

        Returns:
            Plain text extracted from HTML.
        """
        # Remove script and style elements
        html = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Replace common block elements with newlines
        html = re.sub(r"<(br|p|div|h[1-6]|li)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Remove remaining HTML tags
        html = re.sub(r"<[^>]+>", "", html)
        # Decode HTML entities
        html = unescape(html)
        # Clean up whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    def get_body_text(self, parsed_email: ParsedEmail) -> str:
        """Get the full conversation text from a parsed email.

        Args:
            parsed_email: Parsed email object.

        Returns:
            Full conversation text.
        """
        return parsed_email.full_text
