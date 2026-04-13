"""Ingest module for email processing helpers."""

from .attachment_handler import process_attachment, process_zip_attachment
from .components import (
    IngestComponents,
    create_ingest_components,
    create_local_ingest_components,
)
from .estimator import IngestEstimator
from .helpers import (
    MIME_FORMAT_MAP,
    IssueTracker,
    enrich_chunks_with_document_metadata,
    get_format_name,
)
from .pipeline import EmailResult, process_email
from .repair import (
    IssueRecord,
    check_document_issues,
    find_orphaned_documents,
    fix_document_issues,
    fix_missing_archives,
    fix_missing_context,
    fix_missing_lines,
    fix_missing_topics,
    scan_ingest_issues,
)

__all__ = [
    "EmailResult",
    "IngestComponents",
    "IngestEstimator",
    "IssueRecord",
    "IssueTracker",
    "MIME_FORMAT_MAP",
    "check_document_issues",
    "create_ingest_components",
    "create_local_ingest_components",
    "enrich_chunks_with_document_metadata",
    "find_orphaned_documents",
    "fix_document_issues",
    "fix_missing_archives",
    "fix_missing_context",
    "fix_missing_lines",
    "fix_missing_topics",
    "get_format_name",
    "process_attachment",
    "process_email",
    "process_zip_attachment",
    "scan_ingest_issues",
]
