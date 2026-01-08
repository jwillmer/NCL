"""Helper functions and classes for ingest operations.

Extracted from cli.py for better maintainability and reusability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from ..utils import compute_chunk_id

if TYPE_CHECKING:
    from ..models.chunk import Chunk
    from ..models.document import Document


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
    a summary table at the end.
    """

    def __init__(self, console: Console | None = None):
        """Initialize the issue tracker.

        Args:
            console: Rich console for output. Creates new one if not provided.
        """
        self._issues: list[dict[str, str]] = []
        self._console = console or Console()

    def track(self, file_ctx: str, attachment: str, error: str) -> None:
        """Track a parsing/processing issue and print warning.

        Args:
            file_ctx: Email file context (e.g., filename).
            attachment: Attachment name that had the issue.
            error: Error message describing the issue.
        """
        self._issues.append({
            "email": file_ctx,
            "attachment": attachment,
            "error": error,
        })
        self._console.print(f"[yellow][{file_ctx}] ⚠ {attachment}: {error}[/yellow]")

    def show_summary(self) -> None:
        """Display summary table of all processing issues."""
        if not self._issues:
            return

        self._console.print()
        table = Table(title=f"⚠ Processing Issues ({len(self._issues)} total)")
        table.add_column("Email", style="cyan")
        table.add_column("Attachment", style="yellow")
        table.add_column("Error", style="red")

        for issue in self._issues:
            error_text = issue["error"]
            if len(error_text) > 60:
                error_text = error_text[:60] + "..."
            table.add_row(issue["email"], issue["attachment"], error_text)

        self._console.print(table)

    def clear(self) -> None:
        """Clear all tracked issues."""
        self._issues.clear()

    @property
    def issues(self) -> list[dict[str, str]]:
        """Get the list of tracked issues."""
        return self._issues.copy()

    def __len__(self) -> int:
        """Return the number of tracked issues."""
        return len(self._issues)
