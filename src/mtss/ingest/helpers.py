"""Helper functions and classes for ingest operations.

Extracted from cli.py for better maintainability and reusability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from rich.console import Console
from rich.table import Table

from ..utils import compute_chunk_id

if TYPE_CHECKING:
    from ..models.chunk import Chunk
    from ..models.document import Document
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


# MIME type to human-readable format name mapping
MIME_FORMAT_MAP: dict[str, str] = {
    "application/pdf": "PDF",
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/tiff": "TIFF",
    "image/bmp": "BMP",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/msword": "DOC",
    "application/vnd.ms-excel": "XLS",
    "application/vnd.ms-powerpoint": "PPT",
    "text/csv": "CSV",
    "application/rtf": "RTF",
    "text/rtf": "RTF",
    "text/html": "HTML",
    "application/zip": "ZIP",
    "application/x-zip-compressed": "ZIP",
}


def get_format_name(content_type: str) -> str:
    """Get human-readable format name from MIME type.

    Args:
        content_type: MIME type string (e.g., "application/pdf").

    Returns:
        Human-readable format name (e.g., "PDF").
    """
    return MIME_FORMAT_MAP.get(content_type, content_type.split("/")[-1].upper())


def enrich_chunks_with_document_metadata(
    chunks: list[Chunk],
    doc: Document,
) -> None:
    """Enrich chunks with citation metadata from their document.

    Sets chunk_id, source_id, source_title, and archive URIs on each chunk.
    Modifies chunks in-place.

    Args:
        chunks: List of chunks to enrich.
        doc: Parent document with citation metadata.
    """
    for chunk in chunks:
        # Generate stable chunk_id if not already set
        if not chunk.chunk_id and doc.doc_id:
            char_start = chunk.char_start or (chunk.chunk_index * 1000)
            char_end = chunk.char_end or (char_start + len(chunk.content))
            chunk.chunk_id = compute_chunk_id(doc.doc_id, char_start, char_end)
        chunk.source_id = doc.source_id
        chunk.source_title = doc.source_title
        chunk.archive_browse_uri = doc.archive_browse_uri
        chunk.archive_download_uri = doc.archive_download_uri


class IssueTracker:
    """Track processing issues for end-of-run summary.

    Collects parsing/processing issues during ingest and displays
    a summary table at the end. Optionally persists issues to database
    for monitoring and debugging.
    """

    def __init__(
        self,
        console: Console | None = None,
        db: Optional["SupabaseClient"] = None,
    ):
        """Initialize the issue tracker.

        Args:
            console: Rich console for output. Creates new one if not provided.
            db: Optional database client for persisting issues to ingest_events table.
        """
        self._issues: list[dict[str, str]] = []
        self._console = console or Console()
        self._db = db

    async def track_async(
        self,
        file_ctx: str,
        attachment: str,
        error: str,
        severity: str = "warning",
        event_type: str = "parse_failure",
        document_id: Optional[UUID] = None,
    ) -> None:
        """Track a parsing/processing issue asynchronously with optional DB persistence.

        Args:
            file_ctx: Email file context (e.g., filename).
            attachment: Attachment name that had the issue.
            error: Error message describing the issue.
            severity: Event severity ('error', 'warning', 'info').
            event_type: Type of event for categorization.
            document_id: Optional parent document UUID for DB linking.
        """
        self._issues.append({
            "email": file_ctx,
            "attachment": attachment,
            "error": error,
            "severity": severity,
            "event_type": event_type,
        })

        # Print to console with appropriate styling
        if severity == "error":
            self._console.print(f"[red][{file_ctx}] ✗ {attachment}: {error}[/red]")
        else:
            self._console.print(f"[yellow][{file_ctx}] ⚠ {attachment}: {error}[/yellow]")

        # Persist to database if client is available (sync call)
        if self._db and document_id:
            try:
                truncated = error[:197] + "..." if len(error) > 200 else error
                self._db.log_ingest_event(
                    document_id=document_id,
                    event_type=event_type,
                    severity=severity,
                    message=truncated,
                    file_path=attachment,
                )
            except Exception as e:
                logger.warning(f"Failed to persist ingest event to DB: {e}")

    def track(self, file_ctx: str, attachment: str, error: str) -> None:
        """Track a parsing/processing issue and print warning (sync version).

        For backward compatibility. Does not persist to database.

        Args:
            file_ctx: Email file context (e.g., filename).
            attachment: Attachment name that had the issue.
            error: Error message describing the issue.
        """
        self._issues.append({
            "email": file_ctx,
            "attachment": attachment,
            "error": error,
            "severity": "warning",
            "event_type": "parse_failure",
        })
        self._console.print(f"[yellow][{file_ctx}] ⚠ {attachment}: {error}[/yellow]")

    def show_summary(self) -> None:
        """Display summary table of all processing issues."""
        if not self._issues:
            return

        self._console.print()

        # Count by severity
        errors = sum(1 for i in self._issues if i.get("severity") == "error")
        warnings = len(self._issues) - errors

        title = f"Processing Issues ({len(self._issues)} total"
        if errors:
            title += f", {errors} errors"
        title += ")"

        table = Table(title=f"⚠ {title}")
        table.add_column("Email", style="cyan")
        table.add_column("Attachment", style="yellow")
        table.add_column("Error", style="red")
        table.add_column("Type", style="dim")

        for issue in self._issues:
            error_text = issue["error"]
            if len(error_text) > 60:
                error_text = error_text[:60] + "..."
            severity_marker = "✗" if issue.get("severity") == "error" else "⚠"
            table.add_row(
                issue["email"],
                issue["attachment"],
                f"{severity_marker} {error_text}",
                issue.get("event_type", "unknown"),
            )

        self._console.print(table)

    def clear(self) -> None:
        """Clear all tracked issues."""
        self._issues.clear()

    @property
    def issues(self) -> list[dict[str, str]]:
        """Get the list of tracked issues."""
        return self._issues.copy()

    @property
    def error_count(self) -> int:
        """Get the count of error-severity issues."""
        return sum(1 for i in self._issues if i.get("severity") == "error")

    def __len__(self) -> int:
        """Return the number of tracked issues."""
        return len(self._issues)
