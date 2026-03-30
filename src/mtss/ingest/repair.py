"""Ingest repair/update logic for scanning and fixing document issues.

Extracted from cli.py to allow reuse and testability. All functions accept
callback parameters instead of using console/progress directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import UUID

from .helpers import enrich_chunks_with_document_metadata

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
        checks: Set of issue types to check for.
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
        checks: Set of issue types to check for.

    Returns:
        Tuple of (issue list, cached_chunks dict).
        cached_chunks maps UUID -> list[Chunk] for reuse in fix phase.
    """
    from ..models.document import DocumentType

    issues = []
    cached_chunks: dict = {}

    if "archives" in checks:
        # Check if root doc is missing archive
        if not doc.archive_browse_uri:
            issues.append("missing_archive")
        # Check children (except images which don't have archives)
        for child in child_docs:
            if not child.archive_browse_uri and child.document_type != DocumentType.ATTACHMENT_IMAGE:
                issues.append("missing_child_archive")
                break

    # Fetch chunks once for all checks (avoid duplicate DB calls)
    need_chunks = "chunks" in checks or "context" in checks or "topics" in checks

    if need_chunks:
        doc_chunks = await components.db.get_chunks_by_document(doc.id)
        if doc_chunks:
            cached_chunks[doc.id] = doc_chunks

        for child in child_docs:
            if child.document_type != DocumentType.ATTACHMENT_IMAGE:
                child_chunks = await components.db.get_chunks_by_document(child.id)
                if child_chunks:
                    cached_chunks[child.id] = child_chunks

    if "chunks" in checks:
        # Check if any chunks are missing line numbers
        doc_chunks = cached_chunks.get(doc.id, [])
        if doc_chunks and any(c.line_from is None for c in doc_chunks):
            issues.append("missing_lines")
        # Check child docs
        for child in child_docs:
            child_chunks = cached_chunks.get(child.id, [])
            if child_chunks and any(c.line_from is None for c in child_chunks):
                issues.append("missing_child_lines")
                break

    if "context" in checks:
        # Check if any chunks are missing context summaries
        doc_chunks = cached_chunks.get(doc.id, [])
        if doc_chunks and any(c.context_summary is None for c in doc_chunks):
            issues.append("missing_context")
        # Check child docs
        for child in child_docs:
            child_chunks = cached_chunks.get(child.id, [])
            if child_chunks and any(c.context_summary is None for c in child_chunks):
                issues.append("missing_child_context")
                break

    if "topics" in checks:
        # Check if any chunks are missing topic_ids
        doc_chunks = cached_chunks.get(doc.id, [])
        if doc_chunks:
            # Check if topics exist OR topics were checked (but none found)
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
    chunks_created = 0

    # Parse email once (needed for archive fixes)
    parsed_email = None
    if "missing_archive" in record.issues or "missing_child_archive" in record.issues:
        parsed_email = components.eml_parser.parse_file(record.eml_path)

    # Fix in order of dependency:
    # 1. Archives first (chunks depend on archive content)
    if "archives" in checks and ("missing_archive" in record.issues or "missing_child_archive" in record.issues):
        await fix_missing_archives(record, components, parsed_email, on_verbose=on_verbose)

    # 2. Chunks second (context depends on chunk structure)
    if "chunks" in checks and ("missing_lines" in record.issues or "missing_child_lines" in record.issues):
        chunks_created += await fix_missing_lines(record, components, on_verbose=on_verbose)

    # 3. Context third
    if "context" in checks and ("missing_context" in record.issues or "missing_child_context" in record.issues):
        chunks_created += await fix_missing_context(record, components, on_verbose=on_verbose)

    # 4. Topics last (doesn't depend on others, just needs content)
    if "topics" in checks and "missing_topics" in record.issues:
        await fix_missing_topics(record, components, on_verbose=on_verbose)

    return chunks_created


async def fix_missing_archives(
    record: IssueRecord,
    components: "IngestComponents",
    parsed_email,
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> None:
    """Generate missing archive files using same function as regular ingest.

    First checks if files exist in bucket but DB is just missing the URI.
    If files exist in bucket, updates DB only. Otherwise regenerates archives.

    Args:
        record: Issue record with document info.
        components: Shared ingest components.
        parsed_email: Parsed email object.
        on_verbose: Optional callback (message, file_context).
    """
    from ..models.document import DocumentType

    def _verbose(msg: str) -> None:
        if on_verbose:
            on_verbose(msg, record.eml_path.name)

    if not parsed_email:
        _verbose("Cannot fix archives: email not parsed")
        return

    # Get parent folder_id for bucket lookups
    folder_id = record.doc.doc_id[:16] if record.doc.doc_id else None

    # Fix root document archive if missing
    if not record.doc.archive_browse_uri:
        # Check if archive exists in bucket first
        bucket_path = f"{folder_id}/email.eml.md"
        if folder_id and components.archive_storage.file_exists(bucket_path):
            browse_uri = f"/archive/{bucket_path}"
            await components.db.update_document_archive_uris(record.doc.id, browse_uri)
            record.doc.archive_browse_uri = browse_uri
            _verbose(f"Archive found in bucket, DB updated: {browse_uri}")
        else:
            _verbose("Generating archive...")
            # Use IDENTICAL archive generation as regular ingest
            archive_result = await components.archive_generator.generate_archive(
                parsed_email=parsed_email,
                source_eml_path=record.eml_path,
            )
            # Use same format as hierarchy_manager: /archive/{path}
            browse_uri = f"/archive/{archive_result.markdown_path}"
            await components.db.update_document_archive_uris(
                record.doc.id,
                browse_uri,
            )
            record.doc.archive_browse_uri = browse_uri
            _verbose(f"Archive created: {browse_uri}")

    # Fix child document archives if missing
    if not folder_id:
        return

    for child in record.child_docs:
        # Skip images - they don't have archives
        if child.document_type == DocumentType.ATTACHMENT_IMAGE:
            continue

        if child.archive_browse_uri:
            continue

        # Check if .md file exists in bucket
        expected_path = f"{folder_id}/attachments/{child.file_name}.md"
        if components.archive_storage.file_exists(expected_path):
            # File exists in bucket - just update DB
            browse_uri = f"/archive/{expected_path}"
            await components.db.update_document_archive_uris(child.id, browse_uri)
            child.archive_browse_uri = browse_uri
            _verbose(f"Archive found in bucket for {child.file_name}, DB updated")
        else:
            # File doesn't exist - upload original from email and regenerate .md from chunks
            from pathlib import Path as PathLib

            from .archive_generator import _sanitize_storage_key

            # Find matching attachment in parsed email and upload original
            matching_att = None
            if parsed_email and parsed_email.attachments:
                for att in parsed_email.attachments:
                    if att.filename == child.file_name:
                        matching_att = att
                        break

            if not matching_att:
                _verbose(f"Attachment not found in parsed email: {child.file_name}")
                _verbose("Deleting email data for clean re-ingest...")
                # Delete from bucket first
                if folder_id:
                    components.archive_storage.delete_folder(folder_id)
                # Delete from DB (cascades to children and chunks)
                components.db.delete_document_for_reprocess(record.doc.id)
                raise Exception(f"Deleted for re-ingest: attachment '{child.file_name}' not found in parsed email")

            if not PathLib(matching_att.saved_path).exists():
                _verbose(f"Attachment file missing on disk: {matching_att.saved_path}")
                continue

            # Upload the original attachment file first
            safe_filename = _sanitize_storage_key(child.file_name)
            original_path = f"{folder_id}/attachments/{safe_filename}"
            with open(matching_att.saved_path, "rb") as f:
                file_content = f.read()
            content_type = matching_att.content_type or "application/octet-stream"
            components.archive_storage.upload_file(original_path, file_content, content_type)
            _verbose(f"  Uploaded original: {child.file_name}")

            # Original uploaded successfully - now regenerate .md from chunk content
            chunks = await components.db.get_chunks_by_document(child.id)
            if not chunks:
                _verbose(f"No chunks found to regenerate archive for {child.file_name}")
                continue

            # Combine all chunk content to get full document text
            full_content = "\n\n".join(c.content for c in sorted(chunks, key=lambda x: x.chunk_index or 0))
            # Get content type and size from attachment_metadata
            size_bytes = 0
            if child.attachment_metadata:
                content_type = child.attachment_metadata.content_type
                size_bytes = child.attachment_metadata.size_bytes
            md_path = components.archive_generator.update_attachment_markdown(
                doc_id=folder_id,
                filename=child.file_name,
                content_type=content_type,
                size_bytes=size_bytes,
                parsed_content=full_content,
            )
            if md_path:
                # md_path is already URL-encoded from _sanitize_storage_key, don't quote again
                browse_uri = f"/archive/{md_path}"
                download_path = md_path.removesuffix('.md')
                download_uri = f"/archive/{download_path}"
                await components.db.update_document_archive_uris(
                    child.id, browse_uri, download_uri
                )
                child.archive_browse_uri = browse_uri
                child.archive_download_uri = download_uri
                _verbose(f"Archive regenerated for {child.file_name}")
            else:
                _verbose(f"Failed to regenerate archive for {child.file_name}")


async def fix_missing_lines(
    record: IssueRecord,
    components: "IngestComponents",
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> int:
    """Regenerate chunks with line numbers using same chunker as regular ingest.

    Uses atomic replace to ensure no data loss if insert fails.

    Args:
        record: Issue record with document info and cached chunks.
        components: Shared ingest components.
        on_verbose: Optional callback (message, file_context).

    Returns:
        Number of chunks created.
    """
    from ..models.document import DocumentType

    def _verbose(msg: str) -> None:
        if on_verbose:
            on_verbose(msg, record.eml_path.name)

    chunks_created = 0

    # Process root document and all children
    all_docs = [record.doc] + record.child_docs

    for target_doc in all_docs:
        # Skip image attachments - they have single chunks without line tracking
        if target_doc.document_type == DocumentType.ATTACHMENT_IMAGE:
            continue

        # Use cached chunks if available, otherwise fetch
        existing_chunks = record.cached_chunks.get(target_doc.id)
        if existing_chunks is None:
            existing_chunks = await components.db.get_chunks_by_document(target_doc.id)
        if not existing_chunks:
            continue
        if all(c.line_from is not None for c in existing_chunks):
            continue

        # Get markdown content from archive
        if not target_doc.archive_browse_uri:
            _verbose(f"Skipping {target_doc.file_name}: no archive URI")
            continue

        try:
            relative_path = target_doc.archive_browse_uri
            if relative_path.startswith("/archive/"):
                relative_path = relative_path[len("/archive/"):]

            content_bytes = components.archive_storage.download_file(relative_path)
            markdown = content_bytes.decode("utf-8")
        except Exception as e:
            _verbose(f"Failed to download archive: {e}")
            continue

        # Re-chunk using IDENTICAL chunker as regular ingest
        chunks = components.chunker.chunk_text(
            text=markdown,
            document_id=target_doc.id,
            source_file=target_doc.archive_browse_uri or "",
            is_markdown=True,
        )

        if not chunks:
            _verbose(f"No chunks created for {target_doc.file_name}")
            continue

        # Generate context using IDENTICAL context generator
        context = await components.context_generator.generate_context(
            target_doc, markdown[:4000]
        )

        # Enrich chunks using IDENTICAL helper from ingest
        enrich_chunks_with_document_metadata(chunks, target_doc)

        # Apply context to all chunks (same as regular ingest)
        for chunk in chunks:
            if context:
                chunk.context_summary = context
                chunk.embedding_text = components.context_generator.build_embedding_text(
                    context, chunk.content
                )

        # Embed chunks
        chunks = await components.embeddings.embed_chunks(chunks)

        # Atomic replace: delete + insert in single transaction
        count = await components.db.replace_chunks_atomic(target_doc.id, chunks)

        chunks_created += count
        _verbose(f"Replaced with {count} chunks for {target_doc.file_name}")

    return chunks_created


async def fix_missing_context(
    record: IssueRecord,
    components: "IngestComponents",
    on_verbose: Callable[[str, str | None], None] | None = None,
) -> int:
    """Add context summaries using same generator as regular ingest.

    Args:
        record: Issue record with document info and cached chunks.
        components: Shared ingest components.
        on_verbose: Optional callback (message, file_context).

    Returns:
        Number of chunks updated.
    """
    from ..models.document import DocumentType

    def _verbose(msg: str) -> None:
        if on_verbose:
            on_verbose(msg, record.eml_path.name)

    chunks_updated = 0

    # Process root document and all children
    all_docs = [record.doc] + record.child_docs

    for target_doc in all_docs:
        # Skip image attachments - they use image descriptions, not context summaries
        if target_doc.document_type == DocumentType.ATTACHMENT_IMAGE:
            continue

        # Use cached chunks if available, otherwise fetch
        chunks = record.cached_chunks.get(target_doc.id)
        if chunks is None:
            chunks = await components.db.get_chunks_by_document(target_doc.id)
        if not chunks:
            continue
        # Skip if already has context
        if all(c.context_summary is not None for c in chunks):
            continue

        # Get content for context generation
        content = None
        if target_doc.archive_browse_uri:
            try:
                relative_path = target_doc.archive_browse_uri
                if relative_path.startswith("/archive/"):
                    relative_path = relative_path[len("/archive/"):]
                content_bytes = components.archive_storage.download_file(relative_path)
                content = content_bytes.decode("utf-8")
            except Exception as e:
                _verbose(f"Archive download failed: {e}")

        if not content:
            # Fallback to concatenating chunk content
            content = "\n\n".join(c.content for c in chunks)

        if not content.strip():
            _verbose(f"No content available for {target_doc.file_name}")
            continue

        # Generate context using IDENTICAL generator as regular ingest
        try:
            context = await components.context_generator.generate_context(
                target_doc, content[:4000]
            )
        except Exception as e:
            _verbose(f"Context generation error for {target_doc.file_name}: {e}")
            continue

        if not context:
            _verbose(f"Context generation returned empty for {target_doc.file_name}")
            continue

        # Update chunks with context (same embedding text format)
        for chunk in chunks:
            chunk.context_summary = context
            chunk.embedding_text = components.context_generator.build_embedding_text(
                context, chunk.content
            )

        # Re-embed using IDENTICAL embedding generator
        chunks = await components.embeddings.embed_chunks(chunks)

        # Update in database
        for chunk in chunks:
            await components.db.update_chunk_context(chunk)

        chunks_updated += len(chunks)
        _verbose(f"Updated context for {len(chunks)} chunks in {target_doc.file_name}")

    return chunks_updated


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
