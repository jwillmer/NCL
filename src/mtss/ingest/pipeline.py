"""Single-email processing pipeline for ingest.

Handles parsing a single EML file, extracting body chunks,
processing attachments, generating embeddings, and storing results.
Extracted from cli.py for better modularity.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
from typing import TYPE_CHECKING, Callable, Optional

from ..config import get_settings
from ..ingest.archive_generator import _sanitize_storage_key
from ..models.document import ProcessingStatus
from ..parsers.email_cleaner import (
    clean_email_body,
    remove_boilerplate_from_message,
    split_into_messages,
)
from ..utils import compute_chunk_id, compute_doc_id, normalize_source_id
from .attachment_handler import process_attachment
from .helpers import noop_verbose

if TYPE_CHECKING:
    from ..models.chunk import Chunk
    from ..models.document import Document
    from ..storage.progress_tracker import ProgressTracker
    from ..storage.unsupported_file_logger import UnsupportedFileLogger
    from .components import IngestComponents
    from .helpers import IssueTracker
    from .version_manager import VersionManager


@dataclass
class EmailResult:
    """Result of processing a single email."""

    chunks_created: int = 0
    attachment_count: int = 0
    skipped: bool = False
    skip_reason: str = ""


_noop_verbose = noop_verbose  # backward compat alias


async def _resolve_existing_document(
    source_id: str,
    file_hash: str,
    components: "IngestComponents",
    version_manager: "VersionManager | None",
    vprint,
    file_ctx: str,
) -> tuple[str, "UUID | None"]:
    """Check if document already exists and determine action.

    Returns:
        (action, doc_id_to_cleanup) where action is:
        - 'skip': document already processed, skip it
        - 'cleanup_and_proceed': delete old document, then process
        - 'proceed': no existing document, process normally
    """
    if version_manager:
        decision = await version_manager.check_document(source_id, file_hash)

        if decision.action == "skip":
            if decision.existing_doc_id:
                existing = await components.db.get_document_by_id(decision.existing_doc_id)
                if existing and existing.status != ProcessingStatus.COMPLETED:
                    vprint(f"Cleaning up partial document for retry", file_ctx)
                    return ("cleanup_and_proceed", decision.existing_doc_id)
            return ("skip", None)

        if decision.action in ("reprocess", "update"):
            vprint(f"{'Reprocessing' if decision.action == 'reprocess' else 'Updating'}: {decision.reason}", file_ctx)
            if decision.existing_doc_id:
                return ("cleanup_and_proceed", decision.existing_doc_id)
            # No existing doc to clean up, fall through to orphan check

    else:
        # Legacy fallback: check by hash
        existing = await components.db.get_document_by_hash(file_hash)
        if existing:
            if existing.status == ProcessingStatus.COMPLETED:
                return ("skip", None)
            vprint("Cleaning up partial document for retry", file_ctx)
            return ("cleanup_and_proceed", existing.id)

    # Orphan safety check: clean up any orphaned document with same doc_id
    target_doc_id = compute_doc_id(source_id, file_hash)
    orphaned = await components.db.get_document_by_doc_id(target_doc_id)
    if orphaned:
        if orphaned.status == ProcessingStatus.COMPLETED:
            vprint("Skipping (found completed by doc_id)", file_ctx)
            return ("skip", None)
        vprint(f"Cleaning up orphaned document {target_doc_id}", file_ctx)
        return ("cleanup_and_proceed", orphaned.id)

    return ("proceed", None)


async def _extract_topics(
    components: "IngestComponents",
    parsed_email,
    body_text: str,
    archive_result,
    context_summary: str | None,
    vprint,
    file_ctx: str,
) -> list[str]:
    """Extract topic IDs from email content. Returns empty list on failure."""
    if not components.topic_extractor or not components.topic_matcher:
        return []

    try:
        # Prefer archived markdown content (banners/signatures already removed)
        # Fall back to raw body_text if archive not available
        topic_content = None
        if archive_result and archive_result.markdown_content:
            topic_content = archive_result.markdown_content
            vprint(f"Using archived content for topics ({len(topic_content)} chars)", file_ctx)
        elif body_text:
            topic_content = body_text

        if not topic_content:
            return []

        # Build topic extraction input from most relevant sources
        topic_input_parts = []

        # 1. Subject line - usually contains the core topic
        subject = parsed_email.metadata.subject or ""
        if subject:
            topic_input_parts.append(f"Subject: {subject}")

        # Check if content is markdown (archived) or raw email text
        is_markdown = topic_content.strip().startswith("#")

        if is_markdown:
            # For markdown content, it's already clean
            # Skip header to get to message content
            msg_start = topic_content.find("## Message")
            if msg_start == -1:
                msg_start = topic_content.find("## Content")
            if msg_start == -1:
                msg_start = topic_content.find("---")
                if msg_start != -1:
                    msg_start = topic_content.find("\n", msg_start + 3)

            message_content = topic_content[msg_start:] if msg_start > 0 else topic_content
            topic_input_parts.append(f"Content:\n{message_content[:3000]}")
        else:
            # For raw email text, split into messages to find original problem
            thread_messages = split_into_messages(topic_content)
            if thread_messages:
                # Get the original message (bottom of thread = last in list)
                original_msg = thread_messages[-1] if len(thread_messages) > 1 else thread_messages[0]
                # Clean boilerplate
                original_msg = remove_boilerplate_from_message(original_msg)
                if original_msg.strip():
                    topic_input_parts.append(f"Original message:\n{original_msg[:2000]}")

        # 3. Context summary if available (semantic-rich)
        if context_summary:
            topic_input_parts.append(f"Summary: {context_summary}")

        topic_input = "\n\n".join(topic_input_parts)

        if not topic_input.strip():
            return []

        topic_ids: list[str] = []
        extracted_topics = await components.topic_extractor.extract_topics(topic_input)
        if extracted_topics:
            batch_ids = await components.topic_matcher.get_or_create_topics_batch(
                [(t.name, t.description) for t in extracted_topics]
            )
            topic_ids.extend(str(tid) for tid in batch_ids)
        if topic_ids:
            vprint(f"Topics extracted: {len(topic_ids)}", file_ctx)
        return topic_ids

    except Exception as e:
        vprint(f"Topic extraction failed (continuing): {e}", file_ctx)
        return []


def _create_body_chunks(
    email_doc: "Document",
    body_text: str,
    context_summary: str | None,
    vessel_ids: list[str],
    vessel_types: list[str],
    vessel_classes: list[str],
    topic_ids: list[str],
    components: "IngestComponents",
) -> list["Chunk"]:
    """Split email body into message chunks with enrichment metadata."""
    from ..models.chunk import Chunk

    chunks: list["Chunk"] = []
    messages = split_into_messages(body_text)

    char_offset = 0
    for msg_idx, message in enumerate(messages):
        # Clean boilerplate from this message
        cleaned_message = remove_boilerplate_from_message(message)
        if not cleaned_message.strip():
            components.db.log_ingest_event(
                document_id=email_doc.id,
                event_type="message_filtered",
                severity="info",
                message=f"Message {msg_idx} empty after boilerplate removal ({len(message.split())} words raw)",
                source_eml_path=email_doc.source_id,
            )
            continue
        # Skip very short messages (auto-replies, signatures only)
        if len(cleaned_message.split()) < 20:
            components.db.log_ingest_event(
                document_id=email_doc.id,
                event_type="message_filtered",
                severity="info",
                message=f"Message {msg_idx} too short after cleaning: {len(cleaned_message.split())} words",
                source_eml_path=email_doc.source_id,
            )
            continue

        # Find char positions in original body
        msg_start = body_text.find(message[:50], char_offset)
        if msg_start == -1:
            msg_start = char_offset
        msg_end = msg_start + len(message)
        char_offset = msg_end

        # Compute stable chunk_id
        chunk_id = compute_chunk_id(email_doc.doc_id or "", msg_start, msg_end)

        # Build embedding text with context (using cleaned message)
        # Apply context summary to ALL chunks for better search relevance
        embedding_text = cleaned_message
        if context_summary:
            embedding_text = components.context_generator.build_embedding_text(context_summary, cleaned_message)

        # Build chunk metadata including vessel and topic info for filtering
        chunk_metadata: dict = {"type": "email_body", "message_index": msg_idx}
        if vessel_ids:
            chunk_metadata["vessel_ids"] = vessel_ids
        if vessel_types:
            chunk_metadata["vessel_types"] = vessel_types
        if vessel_classes:
            chunk_metadata["vessel_classes"] = vessel_classes
        if topic_ids:
            chunk_metadata["topic_ids"] = topic_ids

        chunks.append(
            Chunk(
                document_id=email_doc.id,
                content=message,
                chunk_index=msg_idx,
                heading_path=["Email Body", f"Message {msg_idx + 1}"],
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                context_summary=context_summary,
                embedding_text=embedding_text,
                char_start=msg_start,
                char_end=msg_end,
                source_id=email_doc.source_id,
                source_title=email_doc.source_title,
                archive_browse_uri=email_doc.archive_browse_uri,
                archive_download_uri=email_doc.archive_download_uri,
            )
        )

    return chunks


async def process_email(
    eml_path: Path,
    components: "IngestComponents",
    tracker: "ProgressTracker",
    unsupported_logger: "UnsupportedFileLogger",
    version_manager: "VersionManager | None" = None,
    force_reparse: bool = False,
    lenient: bool = False,
    on_verbose: Callable[[str, str | None], None] | None = None,
    issue_tracker: "IssueTracker | None" = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> EmailResult:
    """Process a single EML file with all its attachments.

    Args:
        eml_path: Path to the .eml file.
        components: Shared ingest components (db, parsers, etc.).
        tracker: Progress tracker for file status.
        unsupported_logger: Logger for unsupported file types.
        version_manager: Optional version manager for skip/reprocess decisions.
        force_reparse: If True, ignore cached content and re-parse everything.
        lenient: If True, continue on errors instead of failing.
        on_verbose: Optional callback for verbose logging (msg, file_ctx).
        issue_tracker: Optional issue tracker for recording processing problems.
        on_progress: Optional callback for progress updates (current, total, description).

    Returns:
        EmailResult with processing statistics.
    """
    vprint = on_verbose or _noop_verbose
    result = EmailResult()

    file_hash = tracker.compute_file_hash(eml_path)
    source_eml_path = str(eml_path)
    file_ctx = eml_path.name  # Short filename for logging
    settings = get_settings()

    # Use hierarchy_manager's ingest_root to ensure consistent doc_id computation
    source_id = normalize_source_id(source_eml_path, components.hierarchy_manager.ingest_root)

    # Step 1: Dedup check
    action, cleanup_doc_id = await _resolve_existing_document(
        source_id, file_hash, components, version_manager, vprint, file_ctx
    )
    if action == "skip":
        result.skipped = True
        result.skip_reason = "already_processed"
        return result
    if action == "cleanup_and_proceed" and cleanup_doc_id:
        components.db.delete_document_for_reprocess(cleanup_doc_id)

    await tracker.mark_started(eml_path, file_hash)
    vprint(f"Processing: {eml_path}", file_ctx)

    # Step 2: Parse email
    parsed_email = components.eml_parser.parse_file(eml_path)
    vprint(f"Parsed: \"{parsed_email.metadata.subject}\" - {len(parsed_email.attachments)} attachments", file_ctx)

    # Notify progress callback with attachment count
    attachment_count = len(parsed_email.attachments)
    result.attachment_count = attachment_count
    if on_progress:
        on_progress(0, max(1, attachment_count), file_ctx)

    # Generate archive if generator is available
    archive_result = None
    if components.archive_generator:
        try:
            archive_result = await components.archive_generator.generate_archive(
                parsed_email=parsed_email,
                source_eml_path=eml_path,
                preserve_md=not force_reparse,  # Keep cached .md files unless force reparsing
            )
            vprint(f"Archive generated: {archive_result.archive_path} ({len(archive_result.attachment_files)} attachments)", file_ctx)
        except Exception as e:
            vprint(f"Archive generation failed: {e}", file_ctx)

    # Build email document in memory (deferred DB insert for atomic persist)
    email_doc = components.hierarchy_manager.build_email_document(
        eml_path, parsed_email, archive_result=archive_result
    )

    # Get email body text (used for vessel matching, context, topics, and chunking)
    email_chunks: list["Chunk"] = []
    body_text = components.eml_parser.get_body_text(parsed_email)

    # Match vessels in email content
    vessel_ids: list[str] = []
    vessel_types: list[str] = []
    vessel_classes: list[str] = []
    if components.vessel_matcher:
        matched_vessels = components.vessel_matcher.find_vessels_in_email(
            subject=parsed_email.metadata.subject,
            body=body_text,
        )
        vessel_ids = [str(v) for v in matched_vessels]
        if vessel_ids:
            vessel_types = components.vessel_matcher.get_types_for_ids(matched_vessels)
            vessel_classes = components.vessel_matcher.get_classes_for_ids(matched_vessels)
            vprint(f"Matched {len(vessel_ids)} vessel(s)", file_ctx)

    # Generate context summary for the email if context generator is available
    context_summary = None
    if components.context_generator and body_text:
        try:
            context_summary = await components.context_generator.generate_context(
                email_doc, body_text[:4000]
            )
            vprint(f"Context generated: {len(context_summary)} chars", file_ctx)
        except Exception as e:
            vprint(f"Context generation failed: {e}", file_ctx)

    # Step 3: Topics (extracted)
    topic_ids = await _extract_topics(
        components, parsed_email, body_text, archive_result, context_summary, vprint, file_ctx
    )

    # Step 4: Body chunks (extracted)
    if body_text:
        # Clean email body using LLM boundary detection + regex boilerplate removal
        settings = get_settings()
        cleaner_model = settings.get_model(settings.email_cleaner_model)
        cleaned_body = await clean_email_body(body_text, model=cleaner_model)

        body_chunks_list = _create_body_chunks(
            email_doc, cleaned_body, context_summary,
            vessel_ids, vessel_types, vessel_classes, topic_ids, components,
        )
        if not body_chunks_list:
            components.db.log_ingest_event(
                document_id=email_doc.id,
                event_type="no_body_chunks",
                severity="info",
                message="Email body produced 0 chunks (forwarding/cover email)",
                source_eml_path=email_doc.source_id,
            )
        vprint(f"Email body: {len(body_chunks_list)} chunk(s)", file_ctx)
        email_chunks.extend(body_chunks_list)

    # Process attachments with progress updates
    attachment_chunk_count = 0
    attachment_docs: list["Document"] = []
    # Build lookup dict for archive file results (avoids O(N^2) scan)
    archive_file_map: dict[str, object] = {}
    if archive_result and archive_result.attachment_files:
        for fr in archive_result.attachment_files:
            key = fr.original_path.rsplit("/", 1)[-1]
            archive_file_map[key] = fr

    # Process attachments concurrently (LlamaParse calls run in parallel)
    _att_completed = 0

    async def _run_attachment(i: int, attachment):
        nonlocal _att_completed
        safe_name = _sanitize_storage_key(attachment.filename)
        archive_file_result = archive_file_map.get(safe_name)

        chunks = await process_attachment(
            attachment=attachment,
            email_doc=email_doc,
            source_eml_path=source_eml_path,
            file_ctx=file_ctx,
            components=components,
            unsupported_logger=unsupported_logger,
            archive_file_result=archive_file_result,
            vessel_ids=vessel_ids,
            vessel_types=vessel_types,
            vessel_classes=vessel_classes,
            force_reparse=force_reparse,
            lenient=lenient,
            on_verbose=on_verbose,
            issue_tracker=issue_tracker,
            email_context_summary=context_summary,
            collect_docs=attachment_docs,
        )
        _att_completed += 1
        if on_progress:
            on_progress(_att_completed, max(1, attachment_count), file_ctx)
        return chunks

    att_results = await asyncio.gather(
        *[_run_attachment(i, att) for i, att in enumerate(parsed_email.attachments)],
        return_exceptions=True,
    )

    for i, att_result in enumerate(att_results):
        if isinstance(att_result, BaseException):
            att_name = parsed_email.attachments[i].filename
            vprint(f"Attachment failed: {att_name}: {att_result}", file_ctx)
            if not lenient:
                raise att_result
        else:
            attachment_chunk_count += len(att_result)
            email_chunks.extend(att_result)

    # Summary of attachments processed
    if parsed_email.attachments:
        body_chunks = 1 if body_text else 0
        vprint(
            f"Summary: {len(parsed_email.attachments)} attachments -> "
            f"{attachment_chunk_count} chunks (+ {body_chunks} email body)",
            file_ctx,
        )

    # Generate email markdown (always run - this is the single source of truth)
    # generate_archive() sets up the folder structure but defers markdown generation
    # so we can include correct [View] links based on which attachments have .md files
    if components.archive_generator and email_doc.doc_id:
        components.archive_generator.regenerate_email_markdown(email_doc.doc_id, parsed_email)
        if parsed_email.attachments:
            vprint("Email markdown generated with [View] links", file_ctx)
        else:
            vprint("Email markdown generated", file_ctx)

    # Generate embeddings for all chunks
    if email_chunks:
        vprint(f"Generating embeddings for {len(email_chunks)} chunks...", file_ctx)
        email_chunks = await components.embeddings.embed_chunks(email_chunks)

    # Atomic persist: all documents + chunks + topic counts in one operation
    email_doc.status = ProcessingStatus.COMPLETED
    topic_uuids = [UUID(tid) for tid in topic_ids] if topic_ids else None
    await components.db.persist_ingest_result(
        email_doc=email_doc,
        attachment_docs=attachment_docs,
        chunks=email_chunks,
        topic_ids=topic_uuids,
        chunk_delta=len(email_chunks),
    )
    vprint(f"Persisted {len(email_chunks)} chunks + {len(attachment_docs)} attachment docs", file_ctx)
    await tracker.mark_completed(eml_path)

    # Clean up attachment folder after successful processing
    if parsed_email.attachments:
        attachment_folder = Path(parsed_email.attachments[0].saved_path).parent
        # Safety: only delete if under managed attachments dir
        if (attachment_folder.exists()
            and attachment_folder != settings.attachments_dir
            and settings.attachments_dir in attachment_folder.parents):
            shutil.rmtree(attachment_folder, ignore_errors=True)
            vprint(f"Cleaned up: {attachment_folder.name}", file_ctx)

    result.chunks_created = len(email_chunks)
    return result
