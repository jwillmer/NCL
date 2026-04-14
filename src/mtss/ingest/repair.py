"""Ingest repair/update logic for scanning and fixing document issues.

With atomic persistence (Phase 4), crash-recovery repairs for missing
archives, lines, and context are no longer needed — partial documents
cannot exist. Only orphan detection and topic backfill remain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import UUID

if TYPE_CHECKING:
    from .components import IngestComponents
    from ..models.document import Document
    from ..storage.supabase_client import SupabaseClient


@dataclass
class IssueRecord:
    """Record of issues found for a document."""
    eml_path: Path
    doc: "Document"
    child_docs: list["Document"]
    issues: list[str]
    cached_chunks: dict = None  # UUID -> list[Chunk], populated during scan

    def __post_init__(self):
        if self.cached_chunks is None:
            self.cached_chunks = {}


async def find_orphaned_documents(source_dir: Path, db: "SupabaseClient") -> list[UUID]:
    """Find documents in DB whose source files no longer exist.

    Args:
        source_dir: Directory containing source .eml files.
        db: Database client.

    Returns:
        List of document UUIDs to delete.
    """
    from ..utils import normalize_source_id

    # Get all existing .eml files
    existing_files = set()
    for eml_path in source_dir.rglob("*.eml"):
        source_id = normalize_source_id(str(eml_path), source_dir)
        existing_files.add(source_id)

    # Get all root documents from DB
    db_source_ids = await db.get_all_root_source_ids()

    # Find orphans (in DB but file doesn't exist)
    orphan_ids = []
    for source_id, doc_id in db_source_ids.items():
        if source_id not in existing_files:
            orphan_ids.append(doc_id)

    return orphan_ids


async def scan_ingest_issues(
    source_dir: Path,
    components: "IngestComponents",
    checks: set[str],
    limit: int,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> list[IssueRecord]:
    """Scan .eml files and identify documents with issues.

    Args:
        source_dir: Directory containing .eml files.
        components: Shared ingest components.
        checks: Set of issue types to check for (only "topics" is valid).
        limit: Maximum documents to return (0 = unlimited).
        on_progress: Optional callback (description, current, total).
        on_verbose: Optional callback (message, file_context).

    Returns:
        List of IssueRecord objects for documents with issues.
    """
    from ..utils import normalize_source_id

    issues: list[IssueRecord] = []

    eml_files = list(source_dir.rglob("*.eml"))
    if on_progress:
        on_progress(f"Found {len(eml_files)} .eml files", 0, len(eml_files))

    for i, eml_path in enumerate(eml_files):
        if limit > 0 and len(issues) >= limit:
            break

        if on_progress:
            on_progress(f"Scanning {eml_path.name}...", i, len(eml_files))

        # Compute source_id the same way as regular ingest
        source_id = normalize_source_id(str(eml_path), source_dir)

        # Find document by source_id
        doc = await components.db.get_document_by_source_id(source_id)
        if not doc:
            # Not ingested - skip
            continue

        # Get child documents (attachments)
        child_docs = await components.db.get_document_children(doc.id)

        # Check for issues (returns cached chunks for reuse in fix phase)
        doc_issues, cached_chunks = await check_document_issues(doc, child_docs, components, checks)

        if doc_issues:
            issues.append(IssueRecord(
                eml_path=eml_path,
                doc=doc,
                child_docs=child_docs,
                issues=doc_issues,
                cached_chunks=cached_chunks,
            ))
            if on_verbose:
                on_verbose(f"Found issues: {', '.join(doc_issues)}", eml_path.name)

    if on_progress:
        on_progress("Scan complete", len(eml_files), len(eml_files))

    return issues


async def check_document_issues(
    doc: "Document",
    child_docs: list["Document"],
    components: "IngestComponents",
    checks: set[str],
) -> tuple[list[str], dict]:
    """Check a document for issues.

    Args:
        doc: Root document to check.
        child_docs: Child documents (attachments).
        components: Shared ingest components.
        checks: Set of issue types to check for (only "topics" is valid).

    Returns:
        Tuple of (issue list, cached_chunks dict).
        cached_chunks maps UUID -> list[Chunk] for reuse in fix phase.
    """
    issues = []
    cached_chunks: dict = {}

    if "topics" in checks:
        # Fetch chunks for topic check
        doc_chunks = await components.db.get_chunks_by_document(doc.id)
        if doc_chunks:
            cached_chunks[doc.id] = doc_chunks

            # Check if any chunks are missing topic_ids
            has_topics_or_checked = any(
                (c.metadata and c.metadata.get("topic_ids")) or
                (c.metadata and c.metadata.get("topics_checked"))
                for c in doc_chunks
            )
            if not has_topics_or_checked:
                issues.append("missing_topics")

    return issues, cached_chunks


async def fix_document_issues(
    record: IssueRecord,
    components: "IngestComponents",
    checks: set[str],
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> int:
    """Fix all issues for a document.

    Args:
        record: Issue record with document and issues.
        components: Shared ingest components.
        checks: Set of issue types being fixed.
        on_verbose: Optional callback (message, file_context).

    Returns:
        Number of chunks created/updated.
    """
    # Topics (doesn't depend on others, just needs content)
    if "topics" in checks and "missing_topics" in record.issues:
        await fix_missing_topics(record, components, on_verbose=on_verbose)

    return 0


async def fix_missing_topics(
    record: IssueRecord,
    components: "IngestComponents",
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> None:
    """Extract topics and update chunk metadata for documents missing topics.

    This is a lightweight fix that:
    1. Downloads archived markdown (or parses email if no archive)
    2. Extracts topics using LLM
    3. Updates chunk metadata with topic_ids (no re-chunking/re-embedding)

    Args:
        record: Issue record with document info.
        components: Shared ingest components.
        on_verbose: Optional callback (message, file_context).
    """
    from uuid import UUID as UUIDType

    from ..parsers.email_cleaner import split_into_messages

    def _verbose(msg: str) -> None:
        if on_verbose:
            on_verbose(msg, record.eml_path.name)

    # Check if topic extraction is enabled
    if not components.topic_extractor or not components.topic_matcher:
        _verbose("Topic extraction not enabled")
        return

    # Get content for topic extraction
    content = None

    # Try to get content from archive first
    if record.doc.archive_browse_uri:
        try:
            relative_path = record.doc.archive_browse_uri
            if relative_path.startswith("/archive/"):
                relative_path = relative_path[len("/archive/"):]
            content_bytes = components.archive_storage.download_file(relative_path)
            content = content_bytes.decode("utf-8")
            _verbose(f"Using archived content ({len(content)} chars)")
        except Exception as e:
            _verbose(f"Failed to download archive: {e}")

    # Fallback: parse email directly
    if not content:
        parsed = components.eml_parser.parse_file(record.eml_path)
        if parsed and parsed.body_text:
            content = parsed.body_text
            _verbose(f"Using parsed email body ({len(content)} chars)")

    if not content:
        _verbose("No content available for topic extraction")
        return

    # Get subject from email_metadata or fall back to source_title
    subject = ""
    if record.doc.email_metadata and record.doc.email_metadata.subject:
        subject = record.doc.email_metadata.subject
    elif record.doc.source_title:
        subject = record.doc.source_title

    # Check if content is markdown (archived) or raw email text
    is_markdown = content.strip().startswith("#")

    if is_markdown:
        # For markdown content, it's already clean - use directly
        # Skip the metadata header (first ~500 chars typically) to get to message content
        # Find the first "## Message" or "## Content" section
        msg_start = content.find("## Message")
        if msg_start == -1:
            msg_start = content.find("## Content")
        if msg_start == -1:
            msg_start = content.find("---")  # After metadata
            if msg_start != -1:
                msg_start = content.find("\n", msg_start + 3)  # After the ---

        message_content = content[msg_start:] if msg_start > 0 else content

        structured_input = f"""Subject: {subject}

Content:
{message_content[:4000]}"""
    else:
        # For raw email text, split into messages to find original problem
        thread_messages = split_into_messages(content)
        if len(thread_messages) > 1:
            original_msg = thread_messages[-1][:1500]  # Bottom = original problem
        else:
            original_msg = thread_messages[0][:1500] if thread_messages else ""

        structured_input = f"""Subject: {subject}

Original Message:
{original_msg}

Summary:
{content[:2000]}"""

    # Extract topics
    try:
        extracted = await components.topic_extractor.extract_topics(structured_input)
        if not extracted:
            _verbose("No topics extracted")
            # Mark as checked so we don't retry (e.g., marketing emails with no problems)
            await components.db.update_chunks_topics_checked(record.doc.id)
            return

        # Get or create topic IDs
        topic_ids: list[str] = []
        for topic in extracted:
            topic_id = await components.topic_matcher.get_or_create_topic(
                topic.name, topic.description
            )
            topic_ids.append(str(topic_id))

        _verbose(f"Extracted {len(topic_ids)} topics")

        # Update chunk metadata
        updated = await components.db.update_chunks_topic_ids(record.doc.id, topic_ids)
        _verbose(f"Updated {updated} chunks with topic_ids")

        # Update topic counts
        await components.db.increment_topic_counts(
            [UUIDType(tid) for tid in topic_ids],
            chunk_delta=updated,
            document_delta=1,
        )

    except Exception as e:
        _verbose(f"Topic extraction failed: {e}")
