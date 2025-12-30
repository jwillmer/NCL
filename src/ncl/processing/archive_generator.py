"""Archive generator for browsable email and attachment content.

Creates a structured archive with markdown previews and original files
for download, enabling source citations with clickable links.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

from ..models.document import ParsedEmail, EmailMessage


# File extensions that are already markdown - skip conversion
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}


@dataclass
class ContentFileResult:
    """Result of generating browsable content file."""

    original_path: str  # Relative path to original file in archive
    markdown_path: Optional[str]  # Relative path to markdown preview
    download_uri: str  # URI for download
    browse_uri: Optional[str]  # URI for viewing
    archive_path: str  # Base archive folder path
    skipped: bool = False  # True if already markdown


@dataclass
class ArchiveResult:
    """Result of generating archive for an email."""

    archive_path: str  # Relative path to archive folder (e.g., "abc123def456")
    markdown_path: str  # Relative path to email markdown (e.g., "abc123def456/email.eml.md")
    original_path: str  # Relative path to original email (e.g., "abc123def456/email.eml")
    doc_id: str
    attachment_files: List[ContentFileResult]  # List of attachment file results


class ArchiveGenerator:
    """Generate browsable archive from parsed emails and attachments.

    Creates a folder structure for each email that enables:
    - Direct linking from citations
    - Browsable markdown previews
    - Download links to original files
    """

    def __init__(self, archive_dir: Optional[Path] = None, ingest_root: Optional[Path] = None):
        """Initialize archive generator.

        Args:
            archive_dir: Root directory for generated archives. If None, uses settings.
            ingest_root: Root directory for ingestion (for stable source IDs). If None, uses settings.
        """
        from ..config import get_settings
        settings = get_settings()

        if archive_dir is None:
            archive_dir = settings.archive_dir
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.ingest_root = ingest_root or settings.eml_source_dir

    async def generate_archive(
        self,
        parsed_email: ParsedEmail,
        source_eml_path: Path,
        parsed_attachment_contents: Optional[Dict[str, str]] = None,
    ) -> ArchiveResult:
        """Generate complete archive folder for an email.

        Args:
            parsed_email: Parsed email with metadata and messages.
            source_eml_path: Path to original EML file.
            parsed_attachment_contents: Optional mapping of filename -> parsed text content.

        Returns:
            ArchiveResult with paths and URIs for all generated files.
        """
        from ..utils import compute_doc_id, normalize_source_id

        parsed_attachment_contents = parsed_attachment_contents or {}

        # Compute doc_id from source path and file hash
        import hashlib
        sha256 = hashlib.sha256()
        with open(source_eml_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        source_id = normalize_source_id(str(source_eml_path), self.ingest_root)
        doc_id = compute_doc_id(source_id, file_hash)

        # Create archive folder using truncated doc_id
        # Clear existing folder to ensure clean re-ingest (removes stale attachments)
        folder_id = doc_id[:16]
        folder = self.archive_dir / folder_id
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Copy original EML file
        email_original = folder / "email.eml"
        shutil.copy2(source_eml_path, email_original)

        # Process attachments to copy originals and create .md previews
        attachments_dir = folder / "attachments"
        attachment_files: List[ContentFileResult] = []

        if parsed_email.attachments:
            attachments_dir.mkdir(exist_ok=True)
            attachment_files = self._process_attachments_to_files(
                parsed_email,
                attachments_dir,
                parsed_attachment_contents,
                folder_id,
            )

        # Generate email markdown now (without [View] links initially)
        # Attachment .md files and [View] links are added later via update_attachment_markdown()
        email_md = folder / "email.eml.md"
        self._write_email_markdown(parsed_email, email_md, attachment_files, folder_id)

        # Write metadata JSON for programmatic access
        self._write_metadata(
            folder,
            doc_id,
            parsed_email,
            folder_id,
            {f.original_path: {"download_uri": f.download_uri, "browse_uri": f.browse_uri}
             for f in attachment_files},
        )

        return ArchiveResult(
            archive_path=folder_id,
            markdown_path=f"{folder_id}/email.eml.md",
            original_path=f"{folder_id}/email.eml",
            doc_id=doc_id,
            attachment_files=attachment_files,
        )

    def _write_email_markdown(
        self,
        parsed_email: ParsedEmail,
        output_path: Path,
        attachment_files: Optional[List[ContentFileResult]] = None,
        folder_id: Optional[str] = None,
    ) -> None:
        """Generate email markdown file with full conversation.

        Args:
            parsed_email: Parsed email with metadata and messages.
            output_path: Path to write the markdown file.
            attachment_files: List of attachment file results (used to determine
                which attachments have .md files for [View] links).
            folder_id: Archive folder ID for full path links.
        """
        meta = parsed_email.metadata
        lines: List[str] = []

        # Header
        subject = meta.subject or "(No Subject)"
        lines.append(f"# {subject}")
        lines.append("")
        lines.append("**Type:** Email Conversation")

        if meta.participants:
            lines.append(f"**Participants:** {', '.join(meta.participants)}")

        if meta.date_start and meta.date_end:
            start = self._format_date(meta.date_start)
            end = self._format_date(meta.date_end)
            if start == end:
                lines.append(f"**Date:** {start}")
            else:
                lines.append(f"**Date Range:** {start} to {end}")
        elif meta.date_start:
            lines.append(f"**Date:** {self._format_date(meta.date_start)}")

        lines.append(f"**Messages:** {meta.message_count}")
        lines.append("")
        # Full path for download link
        if folder_id:
            lines.append(f"[Download Original]({folder_id}/email.eml)")
        else:
            lines.append("[Download Original](email.eml)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Individual messages
        for i, msg in enumerate(parsed_email.messages, 1):
            lines.extend(self._format_message(msg, i))

        # If no individual messages, use full_text
        if not parsed_email.messages and parsed_email.full_text:
            lines.append("## Content")
            lines.append("")
            lines.append(parsed_email.full_text)
            lines.append("")

        # Attachments section
        if parsed_email.attachments:
            lines.append("---")
            lines.append("")
            lines.append("## Attachments")
            lines.append("")

            # Build lookup: filename -> has_markdown
            has_md: Dict[str, bool] = {}
            if attachment_files:
                for f in attachment_files:
                    # original_path is like "abc123/attachments/file.pdf"
                    filename = Path(f.original_path).name
                    has_md[filename] = f.markdown_path is not None

            # Build attachment path prefix
            att_prefix = f"{folder_id}/attachments" if folder_id else "attachments"

            for att in parsed_email.attachments:
                att_name = att.filename
                # URL-encode filename for markdown links (spaces break markdown parsing)
                att_name_encoded = quote(att_name)

                # Only add [View] link if .md file was created
                if has_md.get(att_name, False):
                    lines.append(
                        f"- [{att_name}]({att_prefix}/{att_name_encoded}) "
                        f"([View]({att_prefix}/{att_name_encoded}.md))"
                    )
                else:
                    lines.append(f"- [{att_name}]({att_prefix}/{att_name_encoded})")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _format_message(self, msg: EmailMessage, index: int) -> List[str]:
        """Format a single email message as markdown."""
        lines: List[str] = []
        lines.append(f"## Message {index}")
        lines.append(f"**From:** {msg.from_address}")

        if msg.to_addresses:
            lines.append(f"**To:** {', '.join(msg.to_addresses)}")

        if msg.cc_addresses:
            lines.append(f"**Cc:** {', '.join(msg.cc_addresses)}")

        if msg.date:
            lines.append(f"**Date:** {self._format_date(msg.date)}")

        lines.append("")
        lines.append(msg.content)
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for display."""
        return dt.strftime("%Y-%m-%d %H:%M")

    def _process_attachments_to_files(
        self,
        parsed_email: ParsedEmail,
        attachments_dir: Path,
        parsed_contents: Dict[str, str],
        folder_id: str,
    ) -> List[ContentFileResult]:
        """Process all attachments, creating originals and markdown previews."""
        results: List[ContentFileResult] = []

        for att in parsed_email.attachments:
            filename = att.filename
            source_path = Path(att.saved_path)

            if not source_path.exists():
                continue

            # Copy original file
            dest_original = attachments_dir / filename
            shutil.copy2(source_path, dest_original)

            # URIs relative to archive root
            original_path = f"{folder_id}/attachments/{filename}"
            download_uri = original_path

            markdown_path: Optional[str] = None
            browse_uri: Optional[str] = None
            skipped = self._should_skip_markdown(filename)

            # Generate markdown preview if we have parsed content
            if not skipped:
                parsed_content = parsed_contents.get(filename, "")
                if parsed_content:
                    md_file = attachments_dir / f"{filename}.md"
                    self._write_content_markdown(
                        md_file,
                        filename,
                        att.content_type,
                        att.size_bytes,
                        parsed_content,
                        folder_id,
                    )
                    markdown_path = f"{folder_id}/attachments/{filename}.md"
                    browse_uri = markdown_path

            results.append(ContentFileResult(
                original_path=original_path,
                markdown_path=markdown_path,
                download_uri=download_uri,
                browse_uri=browse_uri,
                archive_path=folder_id,
                skipped=skipped,
            ))

        return results

    def _should_skip_markdown(self, filename: str) -> bool:
        """Check if file is already markdown format."""
        return Path(filename).suffix.lower() in MARKDOWN_EXTENSIONS

    def _write_content_markdown(
        self,
        output_path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
        parsed_content: str,
        folder_id: Optional[str] = None,
    ) -> None:
        """Write markdown preview for an attachment."""
        lines: List[str] = []

        lines.append(f"# {filename}")
        lines.append("")
        lines.append(f"**Type:** {content_type}")
        lines.append(f"**Size:** {self._format_size(size_bytes)}")
        lines.append(f"**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        # Full path from archive root for download link
        if folder_id:
            download_path = f"{folder_id}/attachments/{quote(filename)}"
        else:
            download_path = quote(filename)
        lines.append(f"[Download Original]({download_path})")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Content")
        lines.append("")
        lines.append(parsed_content)
        lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def update_attachment_markdown(
        self,
        doc_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        parsed_content: str,
    ) -> Optional[str]:
        """Update or create an attachment's .md file with parsed content.

        Call this after attachment content has been extracted to add the
        .md file and update the email markdown to show [View] link.

        Args:
            doc_id: Document ID (truncated to 16 chars for folder).
            filename: Original attachment filename.
            content_type: MIME type of the attachment.
            size_bytes: File size in bytes.
            parsed_content: Extracted text content.

        Returns:
            Relative path to the created .md file, or None if failed.
        """
        logger.debug(f"update_attachment_markdown: doc_id={doc_id[:16]}, filename={filename}")

        if self._should_skip_markdown(filename):
            logger.debug(f"  Skipping - file is already markdown")
            return None

        if not parsed_content or not parsed_content.strip():
            logger.debug(f"  Skipping - no parsed_content (len={len(parsed_content) if parsed_content else 0})")
            return None

        folder_id = doc_id[:16]
        folder = self.archive_dir / folder_id
        attachments_dir = folder / "attachments"

        logger.debug(f"  attachments_dir={attachments_dir}, exists={attachments_dir.exists()}")

        if not attachments_dir.exists():
            logger.debug(f"  Skipping - attachments_dir does not exist")
            return None

        # Write the .md file
        md_file = attachments_dir / f"{filename}.md"
        logger.debug(f"  Writing to {md_file}")
        self._write_content_markdown(
            md_file,
            filename,
            content_type,
            size_bytes,
            parsed_content,
            folder_id,
        )

        return f"{folder_id}/attachments/{filename}.md"

    def regenerate_email_markdown(
        self,
        doc_id: str,
        parsed_email: ParsedEmail,
    ) -> None:
        """Regenerate the email markdown with updated [View] links.

        Call this after all attachments have been processed to update
        the email markdown with correct [View] links based on which
        attachments now have .md files.

        Args:
            doc_id: Document ID (truncated to 16 chars for folder).
            parsed_email: Parsed email with metadata and messages.
        """
        folder_id = doc_id[:16]
        folder = self.archive_dir / folder_id
        attachments_dir = folder / "attachments"

        if not folder.exists():
            return

        # Build list of attachments that have .md files
        attachment_files: List[ContentFileResult] = []
        if parsed_email.attachments and attachments_dir.exists():
            for att in parsed_email.attachments:
                filename = att.filename
                md_file = attachments_dir / f"{filename}.md"
                markdown_path = f"{folder_id}/attachments/{filename}.md" if md_file.exists() else None

                attachment_files.append(ContentFileResult(
                    original_path=f"{folder_id}/attachments/{filename}",
                    markdown_path=markdown_path,
                    download_uri=f"{folder_id}/attachments/{filename}",
                    browse_uri=markdown_path,
                    archive_path=folder_id,
                    skipped=False,
                ))

        # Regenerate email markdown with updated attachment info
        email_md = folder / "email.eml.md"
        self._write_email_markdown(parsed_email, email_md, attachment_files, folder_id)

    def _write_metadata(
        self,
        folder: Path,
        doc_id: str,
        parsed_email: ParsedEmail,
        folder_id: str,
        attachment_map: Dict[str, Dict[str, str]],
    ) -> None:
        """Write metadata.json for programmatic access."""
        meta = parsed_email.metadata
        metadata: Dict[str, Any] = {
            "doc_id": doc_id,
            "folder_id": folder_id,
            "subject": meta.subject,
            "participants": meta.participants,
            "initiator": meta.initiator,
            "date_start": meta.date_start.isoformat() if meta.date_start else None,
            "date_end": meta.date_end.isoformat() if meta.date_end else None,
            "message_count": meta.message_count,
            "email_browse_uri": f"{folder_id}/email.eml.md",
            "email_download_uri": f"{folder_id}/email.eml",
            "attachments": attachment_map,
        }

        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def get_archive_uris(self, doc_id: str) -> Dict[str, Optional[str]]:
        """Get archive URIs for a document if it exists.

        Args:
            doc_id: Document ID to look up.

        Returns:
            Dict with browse_uri and download_uri, or None if not found.
        """
        folder_id = doc_id[:16]
        folder = self.archive_dir / folder_id

        if not folder.exists():
            return {"browse_uri": None, "download_uri": None}

        return {
            "browse_uri": f"{folder_id}/email.eml.md",
            "download_uri": f"{folder_id}/email.eml",
        }
