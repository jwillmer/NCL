"""CLI entry point for MTSS - Email RAG Pipeline."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from dataclasses import dataclass

import nest_asyncio

# Apply nest_asyncio to allow nested event loops (required for LlamaParse)
nest_asyncio.apply()

# Disable Langfuse tracing for CLI operations (only used for API)
# This prevents "[non-fatal] Tracing: server error 503" messages during ingest
import litellm

litellm.success_callback = [cb for cb in litellm.success_callback if "langfuse" not in cb]
litellm.failure_callback = [cb for cb in litellm.failure_callback if "langfuse" not in cb]
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from .config import get_settings
from .ingest.helpers import (
    IssueTracker,
    enrich_chunks_with_document_metadata,
    get_format_name,
)
from .models.chunk import Chunk
from .models.document import ProcessingStatus
from .parsers.attachment_processor import AttachmentProcessor
from .parsers.chunker import ContextGenerator, DocumentChunker
from .parsers.email_cleaner import (
    remove_boilerplate_from_message,
    split_into_messages,
)
from .parsers.eml_parser import EMLParser
from .processing.archive_generator import ArchiveGenerator, _sanitize_storage_key
from .processing.embeddings import EmbeddingGenerator
from .processing.hierarchy_manager import HierarchyManager
from .processing.lane_classifier import LaneClassifier
from .processing.version_manager import VersionManager
from .processing.topics import TopicExtractor, TopicMatcher
from .processing.vessel_matcher import VesselMatcher
from .rag.query_engine import RAGQueryEngine, format_response_with_sources
from .storage.archive_storage import ArchiveStorage
from .storage.progress_tracker import ProgressTracker
from .storage.supabase_client import SupabaseClient
from .storage.unsupported_file_logger import UnsupportedFileLogger
from .utils import compute_chunk_id

app = typer.Typer(
    name="MTSS",
    help="MTSS - Email RAG Pipeline for processing EML files with attachments",
)
vessels_app = typer.Typer(help="Vessel registry management")
app.add_typer(vessels_app, name="vessels")
topics_app = typer.Typer(help="Topic management for categorization and filtering")
app.add_typer(topics_app, name="topics")
console = Console()

# Module-level verbose flag
_verbose = False

# Flag to track if shutdown was requested
_shutdown_requested = False

# Files in PROCESSING state longer than this are considered stale and reset on --retry-failed
STALE_PROCESSING_THRESHOLD_MINUTES = 5


def _handle_interrupt(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second Ctrl+C - force exit
        console.print("\n[red]Force exiting...[/red]")
        sys.exit(1)
    _shutdown_requested = True
    console.print("\n[yellow]Graceful shutdown requested - completing in-progress files... (press Ctrl+C again to force exit)[/yellow]")
    # Don't raise KeyboardInterrupt - let the loop finish gracefully


def vprint(msg: str, file_ctx: str | None = None):
    """Print verbose output if enabled, with optional file context for concurrent logs."""
    if _verbose:
        if file_ctx:
            console.print(f"[dim][{file_ctx}] {msg}[/dim]")
        else:
            console.print(f"[dim]{msg}[/dim]")


# Module-level issue tracker instance (replaces _processing_issues list)
_issue_tracker = IssueTracker(console)


def track_issue(file_ctx: str, attachment: str, error: str):
    """Track a parsing/processing issue for end summary and print warning.

    Wrapper function for backward compatibility with existing code.
    """
    _issue_tracker.track(file_ctx, attachment, error)


@app.command()
def ingest(
    source_dir: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Directory containing EML files",
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        "-b",
        help="Number of files to process in each batch",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from previous progress",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry previously failed files",
    ),
    reprocess_outdated: bool = typer.Option(
        False,
        "--reprocess-outdated",
        help="Reprocess files that were ingested with an older version",
    ),
    lenient: bool = typer.Option(
        False,
        "--lenient",
        help="Continue processing on errors instead of failing (logs to ingest_events)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with detailed processing info",
    ),
):
    """Ingest EML files and their attachments into the RAG system."""
    global _verbose, _shutdown_requested
    _verbose = verbose
    _shutdown_requested = False

    # Set up signal handler for Ctrl+C
    original_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        asyncio.run(_ingest(source_dir, batch_size, resume, retry_failed, reprocess_outdated, lenient))
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        # Force exit to clean up any lingering async resources (e.g., httpx clients from Agents SDK)
        os._exit(0)


async def _ingest(
    source_dir: Optional[Path],
    batch_size: int,
    resume: bool,
    retry_failed: bool,
    reprocess_outdated: bool = False,
    lenient: bool = False,
):
    """Async implementation of ingest command."""
    global _shutdown_requested

    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    # Initialize components
    db = SupabaseClient()
    tracker = ProgressTracker(db)
    unsupported_logger = UnsupportedFileLogger(db)
    eml_parser = EMLParser()
    attachment_processor = AttachmentProcessor()
    hierarchy_manager = HierarchyManager(db, ingest_root=source_dir)
    embeddings = EmbeddingGenerator()
    archive_generator = ArchiveGenerator(ingest_root=source_dir)
    context_generator = ContextGenerator()
    version_manager = VersionManager(db)

    # Load vessel registry for matching
    vessels = await db.get_all_vessels()
    vessel_matcher = VesselMatcher(vessels) if vessels else None
    if vessel_matcher:
        vprint(f"Loaded {vessel_matcher.vessel_count} vessels ({vessel_matcher.name_count} names)")
    else:
        vprint("No vessels in registry - vessel tagging disabled")

    # Initialize topic extraction for categorization
    topic_extractor = TopicExtractor()
    topic_matcher = TopicMatcher(db, embeddings)
    vprint("Topic extraction enabled")

    # Initialize continuous report writer
    from .storage.failure_report import IngestReportWriter
    report_writer = IngestReportWriter(source_dir=str(source_dir))
    console.print(f"[dim]Report: {report_writer.get_path()}[/dim]")

    try:
        # Get files to process
        if reprocess_outdated:
            files = await tracker.get_outdated_files(source_dir, settings.current_ingest_version)
            console.print(f"[yellow]Reprocessing {len(files)} outdated files (ingest_version < {settings.current_ingest_version})[/yellow]")
        elif retry_failed:
            # Reset stale files stuck in "processing" state before getting failed files
            stale_count = await tracker.reset_stale_processing(
                max_age_minutes=STALE_PROCESSING_THRESHOLD_MINUTES
            )
            if stale_count:
                console.print(f"[yellow]Reset {stale_count} stale processing files[/yellow]")
            files = await tracker.get_failed_files()
            console.print(f"[yellow]Retrying {len(files)} failed files[/yellow]")
        elif resume:
            files = await tracker.get_pending_files(source_dir)
            console.print(f"[green]Found {len(files)} pending files[/green]")
        else:
            files = list(source_dir.glob("**/*.eml"))
            console.print(f"[green]Found {len(files)} total EML files[/green]")

        vprint(f"Source directory: {source_dir}")

        if not files:
            console.print("[green]No files to process![/green]")
            return

        processed_count = 0

        # Reset issues list for this run
        _issue_tracker.clear()

        # Classify files into fast/slow queues
        # Fast: no attachments or only images (no LlamaParse needed)
        # Slow: documents requiring LlamaParse (PDFs, Office files, ZIPs)
        lane_classifier = LaneClassifier(eml_parser)
        fast_queue: asyncio.Queue[Path] = asyncio.Queue()
        slow_queue: asyncio.Queue[Path] = asyncio.Queue()

        for f in files:
            lane = lane_classifier.classify(f)
            if lane == "fast":
                fast_queue.put_nowait(f)
            else:
                slow_queue.put_nowait(f)

        fast_count = fast_queue.qsize()
        slow_count = slow_queue.qsize()
        vprint(f"Lane assignment: {fast_count} fast, {slow_count} slow")

        # Split workers: 60% fast, 40% slow (minimum 1 each if both queues have items)
        total_workers = settings.max_concurrent_files
        if fast_count > 0 and slow_count > 0:
            fast_worker_count = max(1, round(total_workers * 0.6))
            slow_worker_count = max(1, total_workers - fast_worker_count)
        elif fast_count > 0:
            fast_worker_count = total_workers
            slow_worker_count = 0
        else:
            fast_worker_count = 0
            slow_worker_count = total_workers

        vprint(f"Worker split: {fast_worker_count} fast workers, {slow_worker_count} slow workers")

        # Process files with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            # Create worker slot tasks (one progress bar per concurrent worker)
            worker_tasks: list[TaskID] = []
            for i in range(settings.max_concurrent_files):
                task_id = progress.add_task(
                    f"[dim][{i + 1}] (idle)[/dim]", total=1, completed=0
                )
                worker_tasks.append(task_id)

            # Slot queue for dynamic worker assignment
            slot_queue: asyncio.Queue[int] = asyncio.Queue()
            for i in range(settings.max_concurrent_files):
                slot_queue.put_nowait(i)

            # Total progress bar
            total_task = progress.add_task("[bold]Total[/bold]", total=len(files))

            async def process_one(file_path: Path) -> None:
                """Process a single file, acquiring a display slot."""
                nonlocal processed_count

                if _shutdown_requested:
                    return

                slot_idx = await slot_queue.get()
                worker_task = worker_tasks[slot_idx]

                try:
                    progress.update(
                        worker_task,
                        description=f"[{slot_idx + 1}] {file_path.name}",
                        total=1,
                        completed=0,
                    )

                    await _process_single_email(
                        file_path,
                        eml_parser,
                        attachment_processor,
                        hierarchy_manager,
                        embeddings,
                        db,
                        tracker,
                        unsupported_logger,
                        archive_generator=archive_generator,
                        context_generator=context_generator,
                        version_manager=version_manager,
                        vessel_matcher=vessel_matcher,
                        topic_extractor=topic_extractor,
                        topic_matcher=topic_matcher,
                        progress=progress,
                        worker_task=worker_task,
                        slot_idx=slot_idx,
                        force_reparse=reprocess_outdated,
                        lenient=lenient,
                    )
                    processed_count += 1
                except Exception as e:
                    await tracker.mark_failed(file_path, str(e))
                    report_writer.add_eml_failure(str(file_path), str(e))
                    console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                finally:
                    progress.update(total_task, advance=1)
                    progress.update(
                        worker_task,
                        description=f"[dim][{slot_idx + 1}] (idle)[/dim]",
                        total=1,
                        completed=0,
                    )
                    slot_queue.put_nowait(slot_idx)

            async def fast_worker() -> None:
                """Worker for fast queue. Helps slow queue when fast is empty."""
                while True:
                    if _shutdown_requested:
                        return
                    try:
                        file_path = fast_queue.get_nowait()
                        await process_one(file_path)
                    except asyncio.QueueEmpty:
                        # Fast queue empty - help with slow queue
                        try:
                            file_path = slow_queue.get_nowait()
                            await process_one(file_path)
                        except asyncio.QueueEmpty:
                            return  # Both queues empty

            async def slow_worker() -> None:
                """Worker dedicated to slow queue."""
                while True:
                    if _shutdown_requested:
                        return
                    try:
                        file_path = slow_queue.get_nowait()
                        await process_one(file_path)
                    except asyncio.QueueEmpty:
                        return  # Slow queue empty

            # Start both worker pools in parallel
            all_workers = (
                [fast_worker() for _ in range(fast_worker_count)] +
                [slow_worker() for _ in range(slow_worker_count)]
            )
            await asyncio.gather(*all_workers)

            if _shutdown_requested:
                # Mark remaining queued files as failed so they can be retried
                remaining_files: list[Path] = []
                while True:
                    try:
                        remaining_files.append(fast_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                while True:
                    try:
                        remaining_files.append(slow_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                for file_path in remaining_files:
                    await tracker.mark_failed(file_path, "interrupted_during_shutdown")

                console.print(f"[yellow]Stopped after {processed_count} files ({len(remaining_files)} marked for retry)[/yellow]")

        # Show final stats and issue summary
        stats = await tracker.get_processing_stats()
        _show_stats(stats)
        _issue_tracker.show_summary()

        # Finalize report with stats and cleanup old reports
        report_writer.update_stats(len(files), processed_count)
        report_writer.cleanup_old_reports()
        console.print(f"\nReport: {report_writer.get_path()}")

    finally:
        await db.close()


async def _process_single_email(
    eml_path: Path,
    eml_parser: EMLParser,
    attachment_processor: AttachmentProcessor,
    hierarchy_manager: HierarchyManager,
    embeddings: EmbeddingGenerator,
    db: SupabaseClient,
    tracker: ProgressTracker,
    unsupported_logger: UnsupportedFileLogger,
    archive_generator: ArchiveGenerator | None = None,
    context_generator: ContextGenerator | None = None,
    version_manager: VersionManager | None = None,
    vessel_matcher: VesselMatcher | None = None,
    topic_extractor: TopicExtractor | None = None,
    topic_matcher: TopicMatcher | None = None,
    progress: Progress | None = None,
    worker_task: TaskID | None = None,
    slot_idx: int | None = None,
    force_reparse: bool = False,
    lenient: bool = False,
):
    """Process a single EML file with all its attachments."""
    file_hash = tracker.compute_file_hash(eml_path)
    source_eml_path = str(eml_path)
    file_ctx = eml_path.name  # Short filename for logging
    settings = get_settings()

    # Check using version manager if available
    # Use hierarchy_manager's ingest_root to ensure consistent doc_id computation
    from .utils import compute_doc_id, normalize_source_id
    source_id = normalize_source_id(source_eml_path, hierarchy_manager.ingest_root)

    if version_manager:
        decision = await version_manager.check_document(source_id, file_hash)

        if decision.action == "skip":
            # Check if it's actually completed - partial failures need cleanup
            if decision.existing_doc_id:
                existing = await db.get_document_by_id(decision.existing_doc_id)
                if existing and existing.status != ProcessingStatus.COMPLETED:
                    vprint(f"Cleaning up partial document for retry: {eml_path}", file_ctx)
                    db.delete_document_for_reprocess(decision.existing_doc_id)
                else:
                    vprint(f"Skipping (already processed): {eml_path}", file_ctx)
                    return
            else:
                vprint(f"Skipping (already processed): {eml_path}", file_ctx)
                return
        elif decision.action == "reprocess":
            vprint(f"Reprocessing: {decision.reason}", file_ctx)
            # Delete old document (cascades to child docs and chunks)
            if decision.existing_doc_id:
                db.delete_document_for_reprocess(decision.existing_doc_id)
        elif decision.action == "update":
            vprint(f"Updating: {decision.reason}", file_ctx)
            # Delete old document before inserting updated version
            if decision.existing_doc_id:
                db.delete_document_for_reprocess(decision.existing_doc_id)
    else:
        # Fallback to legacy check
        existing = await db.get_document_by_hash(file_hash)
        if existing:
            if existing.status == ProcessingStatus.COMPLETED:
                vprint(f"Skipping (already processed): {eml_path}", file_ctx)
                return
            else:
                # Partial failure - clean up old document before retry
                vprint(f"Cleaning up partial document for retry: {eml_path}", file_ctx)
                db.delete_document_for_reprocess(existing.id)

    # Final safety check: clean up any orphaned document with same doc_id
    # This catches edge cases where version_manager didn't find it but it exists
    target_doc_id = compute_doc_id(source_id, file_hash)
    orphaned = await db.get_document_by_doc_id(target_doc_id)
    if orphaned:
        if orphaned.status == ProcessingStatus.COMPLETED:
            vprint(f"Skipping (found completed by doc_id): {eml_path}", file_ctx)
            return
        vprint(f"Cleaning up orphaned document {target_doc_id}: {eml_path}", file_ctx)
        db.delete_document_for_reprocess(orphaned.id)

    await tracker.mark_started(eml_path, file_hash)
    vprint(f"Processing: {eml_path}", file_ctx)

    # Parse email
    parsed_email = eml_parser.parse_file(eml_path)
    vprint(f"Parsed: \"{parsed_email.metadata.subject}\" - {len(parsed_email.attachments)} attachments", file_ctx)

    # Update progress bar with attachment count
    attachment_count = len(parsed_email.attachments)
    if progress and worker_task is not None and slot_idx is not None:
        # Set total to attachment count (minimum 1 to show progress for emails without attachments)
        total = max(1, attachment_count)
        progress.update(
            worker_task,
            description=f"[{slot_idx + 1}] {file_ctx}",
            total=total,
            completed=0,
        )

    # Generate archive if generator is available
    archive_result = None
    if archive_generator:
        try:
            archive_result = await archive_generator.generate_archive(
                parsed_email=parsed_email,
                source_eml_path=eml_path,
                preserve_md=not force_reparse,  # Keep cached .md files unless force reparsing
            )
            vprint(f"Archive generated: {archive_result.archive_path} ({len(archive_result.attachment_files)} attachments)", file_ctx)
        except Exception as e:
            vprint(f"Archive generation failed: {e}", file_ctx)

    # Create email document in hierarchy
    email_doc = await hierarchy_manager.create_email_document(
        eml_path, parsed_email, archive_result=archive_result
    )

    # Match vessels in email content
    vessel_ids: list[str] = []
    vessel_types: list[str] = []
    vessel_classes: list[str] = []
    if vessel_matcher:
        body_text_for_matching = eml_parser.get_body_text(parsed_email)
        matched_vessels = vessel_matcher.find_vessels_in_email(
            subject=parsed_email.metadata.subject,
            body=body_text_for_matching,
        )
        vessel_ids = [str(v) for v in matched_vessels]
        if vessel_ids:
            vessel_types = vessel_matcher.get_types_for_ids(matched_vessels)
            vessel_classes = vessel_matcher.get_classes_for_ids(matched_vessels)
            vprint(f"Matched {len(vessel_ids)} vessel(s)", file_ctx)

    # Create chunks from email body
    email_chunks: list[Chunk] = []
    body_text = eml_parser.get_body_text(parsed_email)

    # Generate context summary for the email if context generator is available
    context_summary = None
    if context_generator and body_text:
        try:
            context_summary = await context_generator.generate_context(
                email_doc, body_text[:4000]
            )
            vprint(f"Context generated: {len(context_summary)} chars", file_ctx)
        except Exception as e:
            vprint(f"Context generation failed: {e}", file_ctx)

    # Extract topics from email content for categorization and pre-filtering
    # Strategy: Use archived markdown (cleaner) + subject + original message + context summary
    # In email threads, the ORIGINAL message (at bottom) contains the problem description,
    # while recent messages (at top) often just contain solutions/acknowledgments.
    topic_ids: list[str] = []
    if topic_extractor and topic_matcher:
        try:
            # Prefer archived markdown content (banners/signatures already removed)
            # Fall back to raw body_text if archive not available
            topic_content = None
            if archive_result and archive_result.markdown_content:
                topic_content = archive_result.markdown_content
                vprint(f"Using archived content for topics ({len(topic_content)} chars)", file_ctx)
            elif body_text:
                topic_content = body_text

            if topic_content:
                # Build topic extraction input from most relevant sources
                topic_input_parts = []

                # 1. Subject line - usually contains the core topic
                subject = parsed_email.metadata.get("subject", "")
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

                if topic_input.strip():
                    extracted_topics = await topic_extractor.extract_topics(topic_input)
                    for topic in extracted_topics:
                        topic_id = await topic_matcher.get_or_create_topic(
                            topic.name, topic.description
                        )
                        topic_ids.append(str(topic_id))
                    if topic_ids:
                        vprint(f"Topics extracted: {len(topic_ids)}", file_ctx)
        except Exception as e:
            vprint(f"Topic extraction failed (continuing): {e}", file_ctx)
            # Don't fail ingest on topic extraction failure

    if body_text:
        # Split email thread into individual messages for better embedding quality
        # Each message becomes a separate chunk to improve semantic search relevance
        messages = split_into_messages(body_text)
        vprint(f"Email body: {len(messages)} message(s) -> chunks", file_ctx)

        char_offset = 0
        for msg_idx, message in enumerate(messages):
            # Clean boilerplate from this message
            cleaned_message = remove_boilerplate_from_message(message)
            if not cleaned_message.strip():
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
                embedding_text = context_generator.build_embedding_text(context_summary, cleaned_message)

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

            email_chunks.append(
                Chunk(
                    document_id=email_doc.id,
                    content=message,  # Store original message content
                    chunk_index=msg_idx,
                    heading_path=["Email Body", f"Message {msg_idx + 1}"],
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    context_summary=context_summary,  # Apply to all chunks
                    embedding_text=embedding_text,  # Cleaned for better embeddings
                    char_start=msg_start,
                    char_end=msg_end,
                    source_id=email_doc.source_id,
                    source_title=email_doc.source_title,
                    archive_browse_uri=email_doc.archive_browse_uri,
                    archive_download_uri=email_doc.archive_download_uri,
                )
            )

    # Process attachments with progress updates
    attachment_chunk_count = 0
    for i, attachment in enumerate(parsed_email.attachments):
        # Get archive file result for this attachment if available
        archive_file_result = None
        if archive_result and archive_result.attachment_files:
            # Look for matching attachment in archive result by sanitized filename
            # Storage keys are sanitized (spaces→%20, brackets→parens), so match accordingly
            from .processing.archive_generator import _sanitize_storage_key

            safe_name = _sanitize_storage_key(attachment.filename)
            for file_result in archive_result.attachment_files:
                # original_path is like "abc123/attachments/sanitized_file.pdf"
                if file_result.original_path.endswith(f"/{safe_name}"):
                    archive_file_result = file_result
                    break

        attachment_chunks = await _process_attachment(
            attachment=attachment,
            email_doc=email_doc,
            source_eml_path=source_eml_path,
            file_ctx=file_ctx,
            attachment_processor=attachment_processor,
            hierarchy_manager=hierarchy_manager,
            db=db,
            unsupported_logger=unsupported_logger,
            archive_file_result=archive_file_result,
            context_generator=context_generator,
            archive_generator=archive_generator,
            vessel_ids=vessel_ids,
            vessel_types=vessel_types,
            vessel_classes=vessel_classes,
            force_reparse=force_reparse,
            lenient=lenient,
        )
        attachment_chunk_count += len(attachment_chunks)
        email_chunks.extend(attachment_chunks)

        # Update progress after each attachment
        if progress and worker_task is not None:
            progress.update(worker_task, completed=i + 1)

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
    if archive_generator and email_doc.doc_id:
        archive_generator.regenerate_email_markdown(email_doc.doc_id, parsed_email)
        if parsed_email.attachments:
            vprint("Email markdown generated with [View] links", file_ctx)
        else:
            vprint("Email markdown generated", file_ctx)

    # Generate embeddings for all chunks
    if email_chunks:
        # Update progress to show embeddings stage
        if progress and worker_task is not None and slot_idx is not None:
            progress.update(
                worker_task,
                description=f"[{slot_idx + 1}] {file_ctx} [dim](embeddings)[/dim]",
            )

        vprint(f"Generating embeddings for {len(email_chunks)} chunks...", file_ctx)
        email_chunks = await embeddings.embed_chunks(email_chunks)
        await db.insert_chunks(email_chunks)
        vprint(f"Inserted {len(email_chunks)} chunks to database", file_ctx)

        # Update topic counts for accurate pre-filtering
        if topic_ids:
            from uuid import UUID
            await db.increment_topic_counts(
                [UUID(tid) for tid in topic_ids],
                chunk_delta=len(email_chunks),
                document_delta=1,
            )

    # Mark email as completed
    await db.update_document_status(email_doc.id, ProcessingStatus.COMPLETED)
    await tracker.mark_completed(eml_path)

    # Clean up attachment folder after successful processing
    if parsed_email.attachments:
        import shutil
        attachment_folder = Path(parsed_email.attachments[0].saved_path).parent
        # Safety: only delete if under managed attachments dir
        if (attachment_folder.exists()
            and attachment_folder != settings.attachments_dir
            and settings.attachments_dir in attachment_folder.parents):
            shutil.rmtree(attachment_folder, ignore_errors=True)
            vprint(f"Cleaned up: {attachment_folder.name}", file_ctx)


def _extract_content_from_cached_markdown(cached_md: str) -> str | None:
    """Extract content section from cached attachment markdown.

    The cached markdown format has a header followed by "## Content" section.
    This extracts just the content portion for re-chunking.

    Args:
        cached_md: Full markdown content from cache.

    Returns:
        Extracted content section, or None if not found.
    """
    content_marker = "## Content\n"
    idx = cached_md.find(content_marker)
    if idx == -1:
        return None
    content = cached_md[idx + len(content_marker):].strip()
    return content if content else None


async def _process_attachment(
    attachment,
    email_doc,
    source_eml_path: str,
    file_ctx: str,
    attachment_processor: AttachmentProcessor,
    hierarchy_manager: HierarchyManager,
    db: SupabaseClient,
    unsupported_logger: UnsupportedFileLogger,
    archive_file_result=None,
    context_generator: ContextGenerator | None = None,
    archive_generator: ArchiveGenerator | None = None,
    vessel_ids: list[str] | None = None,
    vessel_types: list[str] | None = None,
    vessel_classes: list[str] | None = None,
    force_reparse: bool = False,
    lenient: bool = False,
) -> list[Chunk]:
    """Process a single attachment and return its chunks.

    Uses preprocessor for routing decisions.
    Also updates the archive with .md files for parsed attachments.
    Checks for cached parsed content before calling LlamaParse (unless force_reparse=True).
    """
    chunks: list[Chunk] = []
    vessel_ids = vessel_ids or []
    vessel_types = vessel_types or []
    vessel_classes = vessel_classes or []
    file_path = Path(attachment.saved_path)

    # Format size for display
    size_kb = attachment.size_bytes / 1024 if attachment.size_bytes else 0
    size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
    format_name = get_format_name(attachment.content_type or "unknown")

    # Preprocess to get routing decision (classify_images=True for email-level attachments)
    result = await attachment_processor.preprocess(
        file_path, attachment.content_type, classify_images=True
    )

    # Determine processor type for display
    if result.is_zip:
        processor = "ZIP"
    elif result.is_image:
        processor = "Vision"
    elif result.parser_name:
        processor = result.parser_name.capitalize()
    else:
        processor = "unsupported"

    vprint(f"Attachment: {attachment.filename} ({format_name}, {size_str}) [{processor}]", file_ctx)

    # Handle based on preprocess result
    if not result.should_process:
        # Log skipped file
        reason = result.skip_reason or "unsupported_format"
        if "non_content" in reason.lower() or result.is_image:
            vprint("  -> Skipped: non-content image (logo/banner/signature)", file_ctx)
            reason = "classified_as_non_content"
        else:
            vprint(f"  -> Skipped: {reason}", file_ctx)
        await unsupported_logger.log_unsupported_file(
            file_path=file_path,
            reason=reason,
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )
        return chunks

    # Handle ZIP files
    if result.is_zip:
        chunks.extend(
            await _process_zip_attachment(
                attachment=attachment,
                email_doc=email_doc,
                source_eml_path=source_eml_path,
                file_ctx=file_ctx,
                attachment_processor=attachment_processor,
                hierarchy_manager=hierarchy_manager,
                db=db,
                unsupported_logger=unsupported_logger,
                archive_generator=archive_generator,
                vessel_ids=vessel_ids,
                vessel_types=vessel_types,
                vessel_classes=vessel_classes,
                force_reparse=force_reparse,
                lenient=lenient,
            )
        )
        return chunks

    # Create attachment document for non-ZIP files
    attach_doc = await hierarchy_manager.create_attachment_document(
        parent_doc=email_doc,
        attachment_path=file_path,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        original_filename=attachment.filename,
        archive_file_result=archive_file_result,
    )

    try:
        parsed_content: str | None = None

        if result.is_image and result.image_description:
            # Image was already classified and described during preprocessing
            chunk = attachment_processor.create_image_chunk(
                file_path, attach_doc.id, result.image_description, "meaningful"
            )
            chunks.append(chunk)
            parsed_content = result.image_description
            vprint("  -> 1 chunk created (image described)", file_ctx)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
        elif result.is_image:
            # Image needs description (preprocessing didn't provide one)
            attach_chunks = await attachment_processor.process_document_image(
                file_path, attach_doc.id
            )
            chunks.extend(attach_chunks)
            if attach_chunks:
                parsed_content = attach_chunks[0].content
            vprint(f"  -> {len(attach_chunks)} chunks created (image described)", file_ctx)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
        else:
            # Document - check for cached parsed content before calling LlamaParse
            cached_content: str | None = None
            if (
                not force_reparse
                and archive_generator
                and email_doc.doc_id
                and result.parser_name  # Only for documents that use parsers
            ):
                folder_id = email_doc.doc_id[:16]
                safe_filename = _sanitize_storage_key(attachment.filename)
                cached_md_path = f"{folder_id}/attachments/{safe_filename}.md"
                try:
                    if archive_generator.storage.file_exists(cached_md_path):
                        cached_md = archive_generator.storage.download_text(cached_md_path)
                        cached_content = _extract_content_from_cached_markdown(cached_md)
                        if cached_content:
                            vprint(f"  -> Using cached content ({len(cached_content)} chars)", file_ctx)
                except Exception as e:
                    vprint(f"  -> Cache check failed: {e}", file_ctx)
                    cached_content = None

            if cached_content:
                # Use cached content - create chunks directly
                attach_chunks = attachment_processor.chunker.chunk_text(
                    text=cached_content,
                    document_id=attach_doc.id,
                    source_file=str(file_path),
                    is_markdown=True,
                )
                parsed_content = cached_content
                vprint(f"  -> {len(attach_chunks)} chunks created (from cache)", file_ctx)
            else:
                # Parse with LlamaParse (or other parser)
                attach_chunks = await attachment_processor.process_attachment(
                    file_path, attach_doc.id, attachment.content_type
                )
                if attach_chunks:
                    # Combine all chunks for the .md file
                    parsed_content = "\n\n".join(c.content for c in attach_chunks if c.content)
                    vprint(f"  -> {len(attach_chunks)} chunks created", file_ctx)
                else:
                    vprint("  -> 0 chunks (document has no extractable text)", file_ctx)

            # Generate context summary for better search relevance
            if attach_chunks and context_generator and parsed_content:
                try:
                    attach_context = await context_generator.generate_context(
                        attach_doc, parsed_content[:4000]
                    )
                    if attach_context:
                        vprint(f"  -> Context generated: {len(attach_context)} chars", file_ctx)
                        # Apply context summary to ALL chunks
                        for chunk in attach_chunks:
                            chunk.context_summary = attach_context
                            chunk.embedding_text = context_generator.build_embedding_text(
                                attach_context, chunk.content
                            )
                except Exception as e:
                    vprint(f"  -> Context generation failed: {e}", file_ctx)

            chunks.extend(attach_chunks)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)

        # Update archive with .md file for this attachment
        vprint(f"  -> MD check: archive_gen={archive_generator is not None}, parsed_content={len(parsed_content) if parsed_content else 0} chars, doc_id={email_doc.doc_id[:16] if email_doc.doc_id else None}", file_ctx)
        if archive_generator and parsed_content and email_doc.doc_id:
            md_path = archive_generator.update_attachment_markdown(
                doc_id=email_doc.doc_id,
                filename=attachment.filename,
                content_type=attachment.content_type or "application/octet-stream",
                size_bytes=attachment.size_bytes or 0,
                parsed_content=parsed_content,
            )
            if md_path:
                vprint(f"  -> Archive updated: {md_path}", file_ctx)
                # Update document and chunks with archive URIs
                # md_path is already sanitized by _sanitize_storage_key, don't quote again
                browse_uri = f"/archive/{md_path}"
                download_path = md_path.removesuffix('.md')
                download_uri = f"/archive/{download_path}"
                attach_doc.archive_browse_uri = browse_uri
                attach_doc.archive_download_uri = download_uri
                await db.update_document_archive_uris(
                    attach_doc.id, browse_uri, download_uri
                )
            else:
                vprint("  -> update_attachment_markdown returned None", file_ctx)
        else:
            vprint("  -> Skipping .md creation", file_ctx)

    except Exception as e:
        track_issue(file_ctx, attachment.filename, str(e))
        await db.update_document_status(attach_doc.id, ProcessingStatus.FAILED, str(e))
        await unsupported_logger.log_unsupported_file(
            file_path=file_path,
            reason="extraction_failed",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )

    # Enrich chunks with document citation metadata (source_id, source_title, archive URIs)
    enrich_chunks_with_document_metadata(chunks, attach_doc)

    # Add vessel metadata to chunk metadata for filtering
    if vessel_ids or vessel_types or vessel_classes:
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            if vessel_ids:
                chunk.metadata["vessel_ids"] = vessel_ids
            if vessel_types:
                chunk.metadata["vessel_types"] = vessel_types
            if vessel_classes:
                chunk.metadata["vessel_classes"] = vessel_classes

    return chunks


async def _process_zip_attachment(
    attachment,
    email_doc,
    source_eml_path: str,
    file_ctx: str,
    attachment_processor: AttachmentProcessor,
    hierarchy_manager: HierarchyManager,
    db: SupabaseClient,
    unsupported_logger: UnsupportedFileLogger,
    archive_generator: ArchiveGenerator | None = None,
    vessel_ids: list[str] | None = None,
    vessel_types: list[str] | None = None,
    vessel_classes: list[str] | None = None,
    force_reparse: bool = False,
    lenient: bool = False,
) -> list[Chunk] :
    """Extract and process files from a ZIP attachment.

    Note: ZIP contents don't use the archive cache since they don't have
    pre-existing archive paths. They are always parsed fresh.
    """
    chunks: list[Chunk] = []
    vessel_ids = vessel_ids or []
    vessel_types = vessel_types or []
    vessel_classes = vessel_classes or []

    try:
        extracted_files = attachment_processor.extract_zip(
            Path(attachment.saved_path), lenient=lenient
        )
        for extracted_path, extracted_content_type in extracted_files:
            # Preprocess extracted file (classify_images=False - trust ZIP contents)
            result = await attachment_processor.preprocess(
                extracted_path, extracted_content_type, classify_images=False
            )

            if not result.should_process:
                await unsupported_logger.log_unsupported_file(
                    file_path=extracted_path,
                    reason=result.skip_reason or "unsupported_format",
                    source_eml_path=source_eml_path,
                    source_zip_path=attachment.saved_path,
                    parent_document_id=email_doc.id,
                )
                continue

            # Create document for each extracted file
            attach_doc = await hierarchy_manager.create_attachment_document(
                parent_doc=email_doc,
                attachment_path=extracted_path,
                content_type=extracted_content_type,
                size_bytes=extracted_path.stat().st_size,
                original_filename=extracted_path.name,
            )

            # Process extracted file based on preprocess result
            try:
                parsed_content = None
                if result.is_image:
                    # Images from ZIPs - describe without classification
                    attach_chunks = await attachment_processor.process_document_image(
                        extracted_path, attach_doc.id
                    )
                    if attach_chunks:
                        parsed_content = attach_chunks[0].content
                else:
                    # Documents - use parser registry
                    attach_chunks = await attachment_processor.process_attachment(
                        extracted_path, attach_doc.id, extracted_content_type
                    )
                    if attach_chunks:
                        parsed_content = "\n\n".join(c.content for c in attach_chunks if c.content)
                # Enrich chunks with document citation metadata
                enrich_chunks_with_document_metadata(attach_chunks, attach_doc)
                # Add vessel metadata to chunk metadata for filtering
                if vessel_ids or vessel_types or vessel_classes:
                    for chunk in attach_chunks:
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        if vessel_ids:
                            chunk.metadata["vessel_ids"] = vessel_ids
                        if vessel_types:
                            chunk.metadata["vessel_types"] = vessel_types
                        if vessel_classes:
                            chunk.metadata["vessel_classes"] = vessel_classes
                chunks.extend(attach_chunks)
                await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)

                # Update archive with .md file for this extracted file
                if archive_generator and parsed_content and email_doc.doc_id:
                    archive_generator.update_attachment_markdown(
                        doc_id=email_doc.doc_id,
                        filename=extracted_path.name,
                        content_type=extracted_content_type,
                        size_bytes=extracted_path.stat().st_size,
                        parsed_content=parsed_content,
                    )
            except Exception as e:
                track_issue(file_ctx, f"{attachment.filename}/{extracted_path.name}", str(e))
                await db.update_document_status(
                    attach_doc.id, ProcessingStatus.FAILED, str(e)
                )
                await unsupported_logger.log_unsupported_file(
                    file_path=extracted_path,
                    reason="extraction_failed",
                    source_eml_path=source_eml_path,
                    source_zip_path=attachment.saved_path,
                    parent_document_id=email_doc.id,
                )

    except Exception as e:
        track_issue(file_ctx, attachment.filename, f"ZIP extraction failed: {e}")
        await unsupported_logger.log_unsupported_file(
            file_path=Path(attachment.saved_path),
            reason="corrupted",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )

    return chunks


def _show_stats(stats: dict):
    """Display processing statistics."""
    table = Table(title="Processing Statistics")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for status, count in stats.items():
        table.add_row(status.capitalize(), str(count))

    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates for reranking"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
    rerank_top_n: int = typer.Option(5, "--rerank-top-n", "-n", help="Final results after reranking"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
):
    """Query the RAG system with a question.

    Uses two-stage retrieval: vector search + reranking for 20-35% better accuracy.
    """
    asyncio.run(_query(question, top_k, threshold, rerank_top_n, not no_rerank))


async def _query(
    question: str,
    top_k: int,
    threshold: float,
    rerank_top_n: int,
    use_rerank: bool,
):
    """Async implementation of query command."""
    engine = RAGQueryEngine()

    try:
        status_msg = "Searching, reranking, and generating answer..." if use_rerank else "Searching and generating answer..."
        with console.status(status_msg):
            response = await engine.query(
                question=question,
                top_k=top_k,
                similarity_threshold=threshold,
                rerank_top_n=rerank_top_n,
                use_rerank=use_rerank,
            )

        formatted = format_response_with_sources(response)
        console.print(formatted)

    finally:
        await engine.close()


@app.command()
def search(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates for reranking"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
    rerank_top_n: int = typer.Option(10, "--rerank-top-n", "-n", help="Final results after reranking"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
):
    """Search for relevant documents without generating an answer.

    Uses two-stage retrieval: vector search + reranking for better results.
    """
    asyncio.run(_search(query_text, top_k, threshold, rerank_top_n, not no_rerank))


async def _search(
    query_text: str,
    top_k: int,
    threshold: float,
    rerank_top_n: int,
    use_rerank: bool,
):
    """Async implementation of search command."""
    engine = RAGQueryEngine()

    try:
        status_msg = "Searching and reranking..." if use_rerank else "Searching..."
        with console.status(status_msg):
            sources = await engine.search_only(
                question=query_text,
                top_k=top_k,
                similarity_threshold=threshold,
                rerank_top_n=rerank_top_n,
                use_rerank=use_rerank,
            )

        if not sources:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title=f"Search Results for: {query_text}")
        table.add_column("#", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Subject", style="green")
        table.add_column("Relevance", justify="right")
        table.add_column("Preview", max_width=50)

        for i, source in enumerate(sources, 1):
            # Show rerank score if available, otherwise similarity score
            relevance = source.rerank_score if source.rerank_score is not None else source.score
            relevance_label = f"{relevance:.1%}" + (" ✓" if source.rerank_score is not None else "")
            file_name = Path(source.file_path).name if source.file_path else "-"
            table.add_row(
                str(i),
                file_name,
                source.email_subject or "-",
                relevance_label,
                source.text[:100] + "..." if source.text else "-",
            )

        console.print(table)

    finally:
        await engine.close()


@app.command()
def stats():
    """Show processing statistics."""
    asyncio.run(_show_processing_stats())


async def _show_processing_stats():
    """Async implementation of stats command."""
    db = SupabaseClient()
    tracker = ProgressTracker(db)
    unsupported_logger = UnsupportedFileLogger(db)

    try:
        # Show processing stats
        stats = await tracker.get_processing_stats()
        _show_stats(stats)

        # Show unsupported files stats
        unsupported_stats = await unsupported_logger.get_unsupported_files_stats()
        if unsupported_stats.get("total", 0) > 0:
            console.print()  # Empty line separator

            # Show by reason
            by_reason = unsupported_stats.get("by_reason", {})
            if by_reason:
                table = Table(title=f"Unsupported Files ({unsupported_stats['total']} total)")
                table.add_column("Reason", style="yellow")
                table.add_column("Count", justify="right", style="red")

                for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
                    table.add_row(reason.replace("_", " ").title(), str(count))

                console.print(table)

            # Show by MIME type
            by_mime = unsupported_stats.get("by_mime_type", {})
            if by_mime:
                console.print()
                table = Table(title="Unsupported Files by MIME Type")
                table.add_column("MIME Type", style="cyan")
                table.add_column("Count", justify="right", style="red")

                for mime_type, count in sorted(by_mime.items(), key=lambda x: -x[1])[:10]:
                    table.add_row(mime_type or "unknown", str(count))

                console.print(table)
    finally:
        await db.close()


@app.command()
def failures(
    latest: bool = typer.Option(
        False, "--latest", "-l", help="Show details of latest report"
    ),
    export: bool = typer.Option(
        False, "--export", "-e", help="Export current failures from database"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of reports to list"),
):
    """View ingest reports from past runs."""
    asyncio.run(_failures(latest, export, limit))


async def _failures(latest: bool, export: bool, limit: int):
    """Async implementation of failures command."""
    import json

    from .storage.failure_report import FailureReportGenerator

    db = SupabaseClient()

    try:
        generator = FailureReportGenerator(db)

        if export:
            # Generate fresh report from current database state
            report = await generator.generate_report()
            paths = generator.export_report(report)
            console.print(f"Report: {paths['json']}")
            return

        reports = generator.list_reports()

        if not reports:
            console.print("[dim]No ingest reports found[/dim]")
            return

        if latest:
            # Show details of latest report
            latest_report = reports[0]
            console.print("\n[bold]Latest Ingest Report[/bold]")
            console.print(f"Timestamp: {latest_report['timestamp']}")
            console.print(f"Issues: {latest_report['total_failures']}")
            console.print(f"File: {latest_report['json_path']}")

            # Show failure breakdown from JSON
            with open(latest_report["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("failures"):
                console.print()
                table = Table(title=f"Failures ({len(data['failures'])} total)")
                table.add_column("Type", style="yellow")
                table.add_column("File", style="cyan")
                table.add_column("Parent EML", style="dim")
                table.add_column("Reason/Error", style="red", max_width=40)

                for failure in data["failures"][:20]:  # Limit display
                    error_text = failure.get("error") or failure.get("reason") or ""
                    if len(error_text) > 40:
                        error_text = error_text[:40] + "..."
                    table.add_row(
                        failure["type"],
                        failure["file_name"],
                        failure.get("parent_eml") or "-",
                        error_text,
                    )

                console.print(table)

                if len(data["failures"]) > 20:
                    console.print(
                        f"[dim]... and {len(data['failures']) - 20} more[/dim]"
                    )
        else:
            # List recent reports
            table = Table(title="Ingest Reports")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Issues", justify="right")
            table.add_column("File")

            for report in reports[:limit]:
                timestamp_str = report["timestamp"][:19] if report["timestamp"] else "-"
                table.add_row(
                    timestamp_str,
                    str(report["total_failures"]),
                    Path(report["json_path"]).name,
                )

            console.print(table)
    finally:
        await db.close()


@app.command()
def reset_stale(
    max_age: int = typer.Option(60, "--max-age", "-m", help="Max age in minutes"),
):
    """Reset files stuck in 'processing' state."""
    asyncio.run(_reset_stale(max_age))


async def _reset_stale(max_age: int):
    """Async implementation of reset-stale command."""
    db = SupabaseClient()
    tracker = ProgressTracker(db)

    try:
        await tracker.reset_stale_processing(max_age)
        console.print(f"[green]Reset stale processing entries older than {max_age} minutes[/green]")
    finally:
        await db.close()


@app.command()
def reset_failures(
    report_file: Path = typer.Argument(
        ...,
        help="Path to JSON report file containing failed documents",
    ),
    eml_only: bool = typer.Option(
        False,
        "--eml-only",
        "-e",
        help="Only reset EML file failures (skip attachment failures)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be reset without making changes",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Reset failed documents from a JSON report file for reprocessing.

    Reads a failure report JSON file (generated during ingest) and removes
    the failed EML documents from the database, allowing them to be
    reprocessed on the next ingest run.

    This command deletes:
    - processing_log entries (file processing state)
    - documents and their children (attachments, chunks)
    - unsupported_files records
    - archive folders from Supabase Storage

    Workflow:
        1. MTSS failures --latest          # View latest failure report
        2. MTSS reset-failures <file> -n   # Preview what will be reset
        3. MTSS reset-failures <file> -y   # Reset the failed documents
        4. MTSS ingest                     # Reprocess the files

    Examples:
        MTSS reset-failures data/reports/ingest_20260105_170109.json
        MTSS reset-failures data/reports/ingest_20260105_170109.json --dry-run
        MTSS reset-failures data/reports/ingest_20260105_170109.json -y
    """
    asyncio.run(_reset_failures(report_file, eml_only, dry_run, yes))


async def _reset_failures(report_file: Path, eml_only: bool, dry_run: bool, yes: bool):
    """Async implementation of reset-failures command."""
    import json

    if not report_file.exists():
        console.print(f"[red]Report file not found: {report_file}[/red]")
        raise typer.Exit(1)

    # Load the report
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON file: {e}[/red]")
        raise typer.Exit(1)

    failures = report_data.get("failures", [])
    if not failures:
        console.print("[yellow]No failures found in report[/yellow]")
        return

    # Filter to EML failures only if requested
    if eml_only:
        failures = [f for f in failures if f.get("type") == "eml_file"]

    # Extract unique file paths for EML files
    eml_failures = [f for f in failures if f.get("type") == "eml_file"]
    eml_paths = list(set(f.get("file_path") for f in eml_failures if f.get("file_path")))

    if not eml_paths:
        console.print("[yellow]No EML file failures to reset[/yellow]")
        return

    # Show what will be reset
    console.print(f"\n[bold]Files to reset ({len(eml_paths)} EML files):[/bold]")
    table = Table()
    table.add_column("File", style="cyan")
    table.add_column("Error", style="red", max_width=60)

    for failure in eml_failures:
        file_path = failure.get("file_path", "")
        error = failure.get("error") or ""
        if len(error) > 60:
            error = error[:60] + "..."
        table.add_row(Path(file_path).name, error)

    console.print(table)

    if dry_run:
        console.print("\n[dim]Dry run - no changes made[/dim]")
        return

    if not yes:
        console.print()
        confirm = typer.confirm(f"Reset {len(eml_paths)} failed documents from database?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    # Reset the documents
    db = SupabaseClient()
    try:
        counts = await db.reset_failed_documents(eml_paths)
        console.print("\n[green]Reset complete:[/green]")
        console.print(f"  Documents deleted: {counts['documents']}")
        console.print(f"  Processing log entries deleted: {counts['processing_log']}")
        console.print(f"  Archive folders deleted: {counts['archives']}")
        console.print("\n[dim]Run 'MTSS ingest' to reprocess these files[/dim]")
    finally:
        await db.close()


@app.command()
def clean(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed deletion info",
    ),
):
    """Delete all data from database and local processed files. Useful for testing."""
    global _verbose
    _verbose = verbose

    console.print("[yellow]⚠️  This will delete ALL data from the database and processed files![/yellow]")

    if not yes:
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    asyncio.run(_clean())


async def _clean():
    """Async implementation of clean command."""
    import shutil

    from .storage.archive_storage import ArchiveStorage, ArchiveStorageError

    settings = get_settings()
    db = SupabaseClient()

    try:
        # Delete from database
        vprint("Deleting database records...")
        counts = await db.delete_all_data()

        # Show database deletion results
        total_db_records = sum(counts.values())
        if total_db_records > 0:
            console.print(f"[green]Deleted {total_db_records} database records:[/green]")
            for table, count in counts.items():
                if count > 0:
                    vprint(f"  {table}: {count}")
        else:
            console.print("[dim]No database records to delete[/dim]")

        # Delete local processed files
        processed_dir = settings.data_processed_dir
        files_deleted = 0

        if processed_dir.exists():
            vprint(f"Cleaning {processed_dir}...")
            for item in processed_dir.iterdir():
                if item.is_dir():
                    file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                    files_deleted += file_count
                    shutil.rmtree(item)
                    vprint(f"  Deleted {item.name}/ ({file_count} files)")
                elif item.is_file():
                    item.unlink()
                    files_deleted += 1
                    vprint(f"  Deleted {item.name}")

            if files_deleted > 0:
                console.print(f"[green]Deleted {files_deleted} processed files[/green]")
            else:
                console.print("[dim]No processed files to delete[/dim]")
        else:
            console.print("[dim]Processed directory does not exist[/dim]")

        # Delete archive files from Supabase Storage bucket
        vprint("Cleaning Supabase Storage bucket...")
        try:
            storage = ArchiveStorage()
            archive_files_deleted = storage.delete_all()
            if archive_files_deleted > 0:
                console.print(f"[green]Deleted {archive_files_deleted} archive files from storage bucket[/green]")
            else:
                console.print("[dim]No archive files to delete from storage bucket[/dim]")
        except ArchiveStorageError as e:
            console.print(f"[yellow]Warning: Failed to clean storage bucket: {e}[/yellow]")

        console.print("[green]✓ Clean complete[/green]")

    finally:
        await db.close()


@app.command()
def reprocess(
    target_version: int = typer.Option(
        None,
        "--target-version",
        "-t",
        help="Re-process documents below this ingest version (default: current version)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of documents to process",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be processed without making changes",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Re-process documents that were ingested with an older ingest version.

    Use this after upgrading the ingest logic to re-process existing documents
    with the new logic. Documents will be identified by their stable source_id
    and updated in place.
    """
    global _verbose
    _verbose = verbose
    asyncio.run(_reprocess(target_version, limit, dry_run))


async def _reprocess(target_version: int | None, limit: int, dry_run: bool):
    """Async implementation of reprocess command."""
    settings = get_settings()
    db = SupabaseClient()
    version_manager = VersionManager(db)

    try:
        # Use current version if not specified
        version = target_version or settings.current_ingest_version

        # Get count of documents needing reprocessing
        count = await version_manager.count_reprocess_candidates(version)

        if count == 0:
            console.print(f"[green]No documents need reprocessing (all at version {version} or higher)[/green]")
            return

        console.print(f"[yellow]Found {count} documents below ingest version {version}[/yellow]")

        if dry_run:
            # Show sample of documents that would be processed
            candidates = await version_manager.get_reprocess_candidates(version, min(limit, 20))

            table = Table(title="Documents to Reprocess (sample)")
            table.add_column("Source ID", style="cyan", max_width=20)
            table.add_column("Doc ID", style="green", max_width=20)
            table.add_column("Current Version", justify="right")
            table.add_column("File Path", max_width=40)

            for doc in candidates:
                table.add_row(
                    (doc.source_id or "")[:20],
                    (doc.doc_id or "")[:20],
                    str(doc.ingest_version),
                    Path(doc.file_path).name if doc.file_path else "-",
                )

            console.print(table)
            console.print("\n[dim]Run without --dry-run to process these documents[/dim]")
            return

        # Get documents to reprocess
        candidates = await version_manager.get_reprocess_candidates(version, limit)
        console.print(f"[yellow]Processing {len(candidates)} documents...[/yellow]")

        # TODO: Implement actual reprocessing logic
        # This would involve:
        # 1. Deleting old chunks for each document
        # 2. Re-parsing the source file
        # 3. Regenerating chunks with new context
        # 4. Updating the document's ingest_version
        console.print("[red]Reprocessing logic not yet implemented - use 'MTSS clean' and re-ingest[/red]")

    finally:
        await db.close()


# ==================== Re-index Commands ====================


@app.command("reindex-chunks")
def reindex_chunks(
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Re-index a specific document by UUID",
    ),
    missing_lines: bool = typer.Option(
        False,
        "--missing-lines",
        "-m",
        help="Re-index all documents with chunks missing line numbers",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be processed without making changes",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of documents to process",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Re-index chunks from archived markdown files.

    This command re-creates chunks for documents that have archived .md files,
    adding line numbers for citation highlighting and context summaries for
    better search relevance.

    Use --missing-lines to process all documents with chunks that lack line
    number information (needed for citation highlighting).

    Examples:
        MTSS reindex-chunks --missing-lines --dry-run
        MTSS reindex-chunks --doc-id abc123...
    """
    global _verbose
    _verbose = verbose

    if not doc_id and not missing_lines:
        console.print("[red]Error: Specify --doc-id or --missing-lines[/red]")
        raise typer.Exit(1)

    asyncio.run(_reindex_chunks(doc_id, missing_lines, dry_run, limit))


async def _reindex_chunks(
    doc_id: Optional[str],
    missing_lines: bool,
    dry_run: bool,
    limit: int,
):
    """Async implementation of reindex-chunks command."""
    from uuid import UUID

    db = SupabaseClient()
    storage = ArchiveStorage()
    chunker = DocumentChunker()
    context_generator = ContextGenerator()
    embedding_generator = EmbeddingGenerator()

    try:
        # Find documents to re-index
        if doc_id:
            # Single document by ID
            docs = await _get_documents_by_id(db, doc_id)
        else:
            # Documents with chunks missing line numbers
            docs = await _get_documents_missing_lines(db, limit)

        if not docs:
            console.print("[yellow]No documents found to re-index[/yellow]")
            return

        console.print(f"Found {len(docs)} document(s) to re-index")

        if dry_run:
            console.print("[yellow]DRY RUN - No changes will be made[/yellow]")
            for doc in docs:
                console.print(f"  • {doc['source_title'] or doc['id']}")
            return

        # Process each document
        stats = {"success": 0, "failed": 0, "chunks_created": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Re-indexing documents...", total=len(docs))

            for doc in docs:
                doc_uuid = UUID(doc["id"])
                source_title = doc.get("source_title") or str(doc_uuid)[:8]
                archive_uri = doc.get("archive_browse_uri")

                progress.update(task, description=f"Processing {source_title}...")

                if not archive_uri:
                    vprint(f"Skipping {source_title}: no archive URI", "")
                    stats["failed"] += 1
                    progress.advance(task)
                    continue

                try:
                    # Download archived markdown
                    relative_path = archive_uri
                    if relative_path.startswith("/archive/"):
                        relative_path = relative_path[len("/archive/"):]

                    content_bytes = storage.download_file(relative_path)
                    markdown_text = content_bytes.decode("utf-8")
                    vprint(f"Downloaded {len(markdown_text)} chars from {relative_path}", "")

                    # Delete existing chunks for this document
                    deleted = await db.delete_chunks_by_document(doc_uuid)
                    vprint(f"Deleted {deleted} existing chunks", "")

                    # Re-chunk the markdown (with line tracking)
                    chunks = chunker.chunk_text(
                        text=markdown_text,
                        document_id=doc_uuid,
                        source_file=relative_path,
                        is_markdown=True,
                    )

                    if not chunks:
                        vprint(f"No chunks created from {source_title}", "")
                        stats["failed"] += 1
                        progress.advance(task)
                        continue

                    # Generate context summary
                    try:
                        # Create a minimal document object for context generation
                        from .models.document import Document, DocumentType
                        temp_doc = Document(
                            id=doc_uuid,
                            document_type=DocumentType.ATTACHMENT_PDF,
                            source_title=doc.get("source_title"),
                        )
                        context_summary = await context_generator.generate_context(
                            temp_doc, markdown_text[:4000]
                        )
                        vprint(f"Generated context: {len(context_summary)} chars", "")
                    except Exception as e:
                        vprint(f"Context generation failed: {e}", "")
                        context_summary = None

                    # Apply metadata and context to all chunks
                    for chunk in chunks:
                        chunk.chunk_id = compute_chunk_id(
                            doc.get("doc_id") or str(doc_uuid),
                            chunk.char_start or 0,
                            chunk.char_end or 0,
                        )
                        chunk.source_id = doc.get("source_id")
                        chunk.source_title = doc.get("source_title")
                        chunk.archive_browse_uri = doc.get("archive_browse_uri")
                        chunk.archive_download_uri = doc.get("archive_download_uri")

                        # Apply context summary to ALL chunks
                        if context_summary:
                            chunk.context_summary = context_summary
                            chunk.embedding_text = context_generator.build_embedding_text(
                                context_summary, chunk.content
                            )

                    # Generate embeddings
                    chunks = await embedding_generator.embed_chunks(chunks)
                    vprint(f"Generated embeddings for {len(chunks)} chunks", "")

                    # Insert new chunks
                    await db.insert_chunks(chunks)
                    vprint(f"Inserted {len(chunks)} chunks", "")

                    stats["success"] += 1
                    stats["chunks_created"] += len(chunks)

                except Exception as e:
                    console.print(f"[red]Failed to re-index {source_title}: {e}[/red]")
                    stats["failed"] += 1

                progress.advance(task)

        # Print summary
        console.print("\n[green]✓ Re-index complete[/green]")
        console.print(f"  Documents: {stats['success']} success, {stats['failed']} failed")
        console.print(f"  Chunks created: {stats['chunks_created']}")

    finally:
        await db.close()


async def _get_documents_by_id(db: SupabaseClient, doc_id: str) -> list[dict]:
    """Get a single document by ID."""
    result = db.client.table("documents").select(
        "id, doc_id, source_id, source_title, archive_browse_uri, archive_download_uri"
    ).eq("id", doc_id).execute()
    return result.data or []


async def _get_documents_missing_lines(db: SupabaseClient, limit: int) -> list[dict]:
    """Get documents with chunks that are missing line numbers."""
    # Direct query to find chunks missing line numbers
    result = db.client.table("chunks").select(
        "document_id"
    ).is_("line_from", "null").limit(limit * 10).execute()

    if not result.data:
        return []

    # Get unique document IDs
    doc_ids = list(set(row["document_id"] for row in result.data))[:limit]

    # Fetch document details (only those with archive URIs)
    docs_result = db.client.table("documents").select(
        "id, doc_id, source_id, source_title, archive_browse_uri, archive_download_uri"
    ).in_("id", doc_ids).not_.is_("archive_browse_uri", "null").execute()

    return docs_result.data or []


# ==================== Ingest Update Command ====================


@app.command("ingest-update")
def ingest_update(
    source_dir: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Directory containing EML files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Scan and report issues without fixing",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        "-l",
        help="Max documents to process (0 = all)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
    ),
):
    """Validate and repair ingested data.

    Scans ingested emails and automatically fixes:

    - Orphaned documents (source file deleted, removes from DB)
    - Missing/mislinked archives (checks bucket, updates DB or regenerates)
    - Missing chunk line numbers (re-chunks from archive, atomic replace)
    - Missing context summaries (regenerates with LLM, includes retry)

    Uses identical processing as regular ingest for consistency.

    Examples:
        MTSS ingest-update --dry-run          # Scan only, show issues
        MTSS ingest-update                    # Fix all issues
        MTSS ingest-update --limit 10 -v      # Fix 10 docs with details
    """
    global _verbose
    _verbose = verbose

    asyncio.run(_ingest_update(source_dir, dry_run, limit))


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


async def _ingest_update(
    source_dir: Optional[Path],
    dry_run: bool,
    limit: int,
):
    """Async implementation of ingest-update command."""
    from .ingest.components import create_ingest_components

    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Scanning: {source_dir}[/dim]")

    # Initialize database and components
    db = SupabaseClient()

    try:
        # Load vessels for component initialization
        vessels = await db.get_all_vessels()

        # Create shared components (same as regular ingest)
        components = create_ingest_components(db, source_dir, vessels)

        # Always check all issue types (now includes topics)
        checks = {"archives", "chunks", "context", "topics"}

        # Phase 1a: Scan for issues in existing documents
        issues = await _scan_ingest_issues(source_dir, components, checks, limit)

        # Phase 1b: Find orphaned documents (in DB but source file deleted)
        orphan_ids = await _find_orphaned_documents(source_dir, db)

        has_issues = bool(issues) or bool(orphan_ids)

        if not has_issues:
            console.print("[green]No issues found - all data is up to date![/green]")
            return

        # Display summary
        if issues:
            console.print(f"\n[yellow]Found {len(issues)} document(s) with issues:[/yellow]")
            issue_counts: dict[str, int] = {}
            for record in issues:
                for issue in record.issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

            table = Table(title="Issues Found")
            table.add_column("Issue Type", style="cyan")
            table.add_column("Count", justify="right", style="yellow")
            for issue_type, count in sorted(issue_counts.items()):
                table.add_row(issue_type, str(count))
            console.print(table)

        if orphan_ids:
            console.print(f"\n[yellow]Found {len(orphan_ids)} orphaned document(s) (source file deleted)[/yellow]")

        if dry_run:
            console.print("\n[dim]DRY RUN - No changes will be made[/dim]")
            if issues:
                console.print("\n[dim]Sample of affected documents:[/dim]")
                for record in issues[:10]:
                    console.print(f"  • {record.eml_path.name}: {', '.join(record.issues)}")
                if len(issues) > 10:
                    console.print(f"  ... and {len(issues) - 10} more")
            return

        stats = {"fixed": 0, "failed": 0, "chunks_created": 0, "orphans_removed": 0}

        # Phase 2a: Remove orphaned documents first
        if orphan_ids:
            console.print(f"\n[yellow]Removing {len(orphan_ids)} orphaned document(s)...[/yellow]")
            stats["orphans_removed"] = await db.delete_orphaned_documents(orphan_ids)

        # Phase 2b: Fix document issues
        if issues:
            console.print(f"\n[yellow]Fixing {len(issues)} document(s)...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Fixing issues...", total=len(issues))

                for record in issues:
                    try:
                        progress.update(task, description=f"Fixing {record.eml_path.name}...")
                        chunks_created = await _fix_document_issues(record, components, checks)
                        stats["fixed"] += 1
                        stats["chunks_created"] += chunks_created
                    except Exception as e:
                        console.print(f"[red]Failed to fix {record.eml_path.name}: {e}[/red]")
                        stats["failed"] += 1

                    progress.advance(task)

        # Print summary
        console.print("\n[green]✓ Ingest update complete[/green]")
        console.print(f"  Documents fixed: {stats['fixed']}")
        console.print(f"  Documents failed: {stats['failed']}")
        console.print(f"  Chunks created/updated: {stats['chunks_created']}")
        if stats["orphans_removed"]:
            console.print(f"  Orphans removed: {stats['orphans_removed']}")

    finally:
        await db.close()


async def _find_orphaned_documents(source_dir: Path, db: SupabaseClient) -> List[UUID]:
    """Find documents in DB whose source files no longer exist.

    Args:
        source_dir: Directory containing source .eml files.
        db: Database client.

    Returns:
        List of document UUIDs to delete.
    """
    from .utils import normalize_source_id

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


async def _scan_ingest_issues(
    source_dir: Path,
    components: "IngestComponents",
    checks: set[str],
    limit: int,
) -> list[IssueRecord]:
    """Scan .eml files and identify documents with issues.

    Args:
        source_dir: Directory containing .eml files.
        components: Shared ingest components.
        checks: Set of issue types to check for.
        limit: Maximum documents to return (0 = unlimited).

    Returns:
        List of IssueRecord objects for documents with issues.
    """
    from .utils import normalize_source_id

    issues: list[IssueRecord] = []

    eml_files = list(source_dir.rglob("*.eml"))
    console.print(f"[dim]Found {len(eml_files)} .eml files[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning for issues...", total=len(eml_files))

        for eml_path in eml_files:
            if limit > 0 and len(issues) >= limit:
                break

            progress.update(task, description=f"Scanning {eml_path.name}...")

            # Compute source_id the same way as regular ingest
            source_id = normalize_source_id(str(eml_path), source_dir)

            # Find document by source_id
            doc = await components.db.get_document_by_source_id(source_id)
            if not doc:
                # Not ingested - skip
                progress.advance(task)
                continue

            # Get child documents (attachments)
            child_docs = await components.db.get_document_children(doc.id)

            # Check for issues (returns cached chunks for reuse in fix phase)
            doc_issues, cached_chunks = await _check_document_issues(doc, child_docs, components, checks)

            if doc_issues:
                issues.append(IssueRecord(
                    eml_path=eml_path,
                    doc=doc,
                    child_docs=child_docs,
                    issues=doc_issues,
                    cached_chunks=cached_chunks,
                ))
                vprint(f"Found issues: {', '.join(doc_issues)}", eml_path.name)

            progress.advance(task)

    return issues


async def _check_document_issues(
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
    from .models.document import DocumentType

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


async def _fix_document_issues(
    record: IssueRecord,
    components: "IngestComponents",
    checks: set[str],
) -> int:
    """Fix all issues for a document.

    Args:
        record: Issue record with document and issues.
        components: Shared ingest components.
        checks: Set of issue types being fixed.

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
        await _fix_missing_archives(record, components, parsed_email)

    # 2. Chunks second (context depends on chunk structure)
    if "chunks" in checks and ("missing_lines" in record.issues or "missing_child_lines" in record.issues):
        chunks_created += await _fix_missing_lines(record, components)

    # 3. Context third
    if "context" in checks and ("missing_context" in record.issues or "missing_child_context" in record.issues):
        chunks_created += await _fix_missing_context(record, components)

    # 4. Topics last (doesn't depend on others, just needs content)
    if "topics" in checks and "missing_topics" in record.issues:
        await _fix_missing_topics(record, components)

    return chunks_created


async def _fix_missing_archives(
    record: IssueRecord,
    components: "IngestComponents",
    parsed_email,
) -> None:
    """Generate missing archive files using same function as regular ingest.

    First checks if files exist in bucket but DB is just missing the URI.
    If files exist in bucket, updates DB only. Otherwise regenerates archives.

    Args:
        record: Issue record with document info.
        components: Shared ingest components.
        parsed_email: Parsed email object.
    """
    from .models.document import DocumentType

    if not parsed_email:
        vprint("Cannot fix archives: email not parsed", record.eml_path.name)
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
            vprint(f"Archive found in bucket, DB updated: {browse_uri}", record.eml_path.name)
        else:
            vprint("Generating archive...", record.eml_path.name)
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
            vprint(f"Archive created: {browse_uri}", record.eml_path.name)

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
            vprint(f"Archive found in bucket for {child.file_name}, DB updated", record.eml_path.name)
        else:
            # File doesn't exist - upload original from email and regenerate .md from chunks
            from pathlib import Path

            from .processing.archive_generator import _sanitize_storage_key

            # Find matching attachment in parsed email and upload original
            matching_att = None
            if parsed_email and parsed_email.attachments:
                for att in parsed_email.attachments:
                    if att.filename == child.file_name:
                        matching_att = att
                        break

            if not matching_att:
                vprint(f"Attachment not found in parsed email: {child.file_name}", record.eml_path.name)
                vprint("Deleting email data for clean re-ingest...", record.eml_path.name)
                # Delete from bucket first
                if folder_id:
                    components.archive_storage.delete_folder(folder_id)
                # Delete from DB (cascades to children and chunks)
                components.db.delete_document_for_reprocess(record.doc.id)
                raise Exception(f"Deleted for re-ingest: attachment '{child.file_name}' not found in parsed email")

            if not Path(matching_att.saved_path).exists():
                vprint(f"Attachment file missing on disk: {matching_att.saved_path}", record.eml_path.name)
                continue

            # Upload the original attachment file first
            safe_filename = _sanitize_storage_key(child.file_name)
            original_path = f"{folder_id}/attachments/{safe_filename}"
            with open(matching_att.saved_path, "rb") as f:
                file_content = f.read()
            content_type = matching_att.content_type or "application/octet-stream"
            components.archive_storage.upload_file(original_path, file_content, content_type)
            vprint(f"  Uploaded original: {child.file_name}", record.eml_path.name)

            # Original uploaded successfully - now regenerate .md from chunk content
            chunks = await components.db.get_chunks_by_document(child.id)
            if not chunks:
                vprint(f"No chunks found to regenerate archive for {child.file_name}", record.eml_path.name)
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
                vprint(f"Archive regenerated for {child.file_name}", record.eml_path.name)
            else:
                vprint(f"Failed to regenerate archive for {child.file_name}", record.eml_path.name)


async def _fix_missing_lines(
    record: IssueRecord,
    components: "IngestComponents",
) -> int:
    """Regenerate chunks with line numbers using same chunker as regular ingest.

    Uses atomic replace to ensure no data loss if insert fails.

    Args:
        record: Issue record with document info and cached chunks.
        components: Shared ingest components.

    Returns:
        Number of chunks created.
    """
    from .models.document import DocumentType

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
            vprint(f"Skipping {target_doc.file_name}: no archive URI", record.eml_path.name)
            continue

        try:
            relative_path = target_doc.archive_browse_uri
            if relative_path.startswith("/archive/"):
                relative_path = relative_path[len("/archive/"):]

            content_bytes = components.archive_storage.download_file(relative_path)
            markdown = content_bytes.decode("utf-8")
        except Exception as e:
            vprint(f"Failed to download archive: {e}", record.eml_path.name)
            continue

        # Re-chunk using IDENTICAL chunker as regular ingest
        chunks = components.chunker.chunk_text(
            text=markdown,
            document_id=target_doc.id,
            source_file=target_doc.archive_browse_uri or "",
            is_markdown=True,
        )

        if not chunks:
            vprint(f"No chunks created for {target_doc.file_name}", record.eml_path.name)
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
        vprint(f"Replaced with {count} chunks for {target_doc.file_name}", record.eml_path.name)

    return chunks_created


async def _fix_missing_context(
    record: IssueRecord,
    components: "IngestComponents",
) -> int:
    """Add context summaries using same generator as regular ingest.

    Args:
        record: Issue record with document info and cached chunks.
        components: Shared ingest components.

    Returns:
        Number of chunks updated.
    """
    from .models.document import DocumentType

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
                vprint(f"Archive download failed: {e}", record.eml_path.name)

        if not content:
            # Fallback to concatenating chunk content
            content = "\n\n".join(c.content for c in chunks)

        if not content.strip():
            vprint(f"No content available for {target_doc.file_name}", record.eml_path.name)
            continue

        # Generate context using IDENTICAL generator as regular ingest
        try:
            context = await components.context_generator.generate_context(
                target_doc, content[:4000]
            )
        except Exception as e:
            vprint(f"Context generation error for {target_doc.file_name}: {e}", record.eml_path.name)
            continue

        if not context:
            vprint(f"Context generation returned empty for {target_doc.file_name}", record.eml_path.name)
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
        vprint(f"Updated context for {len(chunks)} chunks in {target_doc.file_name}", record.eml_path.name)

    return chunks_updated


async def _fix_missing_topics(
    record: IssueRecord,
    components: "IngestComponents",
) -> None:
    """Extract topics and update chunk metadata for documents missing topics.

    This is a lightweight fix that:
    1. Downloads archived markdown (or parses email if no archive)
    2. Extracts topics using LLM
    3. Updates chunk metadata with topic_ids (no re-chunking/re-embedding)

    Args:
        record: Issue record with document info.
        components: Shared ingest components.
    """
    from uuid import UUID as UUIDType

    from .parsers.email_cleaner import split_into_messages

    # Check if topic extraction is enabled
    if not components.topic_extractor or not components.topic_matcher:
        vprint("Topic extraction not enabled", record.eml_path.name)
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
            vprint(f"Using archived content ({len(content)} chars)", record.eml_path.name)
        except Exception as e:
            vprint(f"Failed to download archive: {e}", record.eml_path.name)

    # Fallback: parse email directly
    if not content:
        parsed = components.eml_parser.parse_file(record.eml_path)
        if parsed and parsed.body_text:
            content = parsed.body_text
            vprint(f"Using parsed email body ({len(content)} chars)", record.eml_path.name)

    if not content:
        vprint("No content available for topic extraction", record.eml_path.name)
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
            vprint("No topics extracted", record.eml_path.name)
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

        vprint(f"Extracted {len(topic_ids)} topics", record.eml_path.name)

        # Update chunk metadata
        updated = await components.db.update_chunks_topic_ids(record.doc.id, topic_ids)
        vprint(f"Updated {updated} chunks with topic_ids", record.eml_path.name)

        # Update topic counts
        await components.db.increment_topic_counts(
            [UUIDType(tid) for tid in topic_ids],
            chunk_delta=updated,
            document_delta=1,
        )

    except Exception as e:
        vprint(f"Topic extraction failed: {e}", record.eml_path.name)


# ==================== Vessel Commands ====================


@vessels_app.command("import")
def vessels_import(
    csv_file: Optional[Path] = typer.Argument(
        None,
        help="Path to vessel CSV file (semicolon-delimited)",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        "-c",
        help="Clear existing vessels before import",
    ),
):
    """Import vessel register from CSV file.

    Default file: data/vessel-list.csv

    CSV format (semicolon-delimited, 3 required columns):
        NAME;TYPE;CLASS
        MARAN THALEIA;VLCC;Canopus Class
    """
    asyncio.run(_vessels_import(csv_file, clear))


async def _vessels_import(csv_file: Optional[Path], clear: bool):
    """Async implementation of vessels import command."""
    import csv

    from .models.vessel import Vessel

    settings = get_settings()

    # Default CSV path
    if csv_file is None:
        csv_file = settings.data_dir / "vessel-list.csv"

    if not csv_file.exists():
        console.print(f"[red]CSV file not found: {csv_file}[/red]")
        raise typer.Exit(1)

    db = SupabaseClient()

    try:
        # Clear existing vessels if requested
        if clear:
            deleted = await db.delete_all_vessels()
            console.print(f"[yellow]Cleared {deleted} existing vessels[/yellow]")

        # Read CSV file
        vessels_to_import: list[Vessel] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            # Detect delimiter (semicolon or comma)
            sample = f.read(1024)
            f.seek(0)
            delimiter = ";" if ";" in sample else ","

            reader = csv.DictReader(f, delimiter=delimiter)

            # Validate required columns are present
            required_columns = {"NAME", "TYPE", "CLASS"}
            if reader.fieldnames:
                actual_columns = set(reader.fieldnames)
                missing = required_columns - actual_columns
                if missing:
                    console.print(f"[red]Missing required columns: {', '.join(missing)}[/red]")
                    console.print("[dim]Required columns: NAME, TYPE, CLASS[/dim]")
                    raise typer.Exit(1)

            for row in reader:
                name = row.get("NAME", "").strip()
                vessel_type = row.get("TYPE", "").strip()
                vessel_class = row.get("CLASS", "").strip()

                if not name:
                    continue  # Skip rows without vessel name

                if not vessel_type or not vessel_class:
                    console.print(f"[yellow]Warning: Skipping {name} - missing TYPE or CLASS[/yellow]")
                    continue

                vessel = Vessel(
                    name=name,
                    vessel_type=vessel_type,
                    vessel_class=vessel_class,
                )
                vessels_to_import.append(vessel)

        if not vessels_to_import:
            console.print("[yellow]No vessels found in CSV file[/yellow]")
            return

        # Import vessels with progress
        imported_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Importing vessels...", total=len(vessels_to_import))

            for vessel in vessels_to_import:
                try:
                    await db.upsert_vessel(vessel)
                    imported_count += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to import {vessel.name}: {e}[/yellow]")
                progress.update(task, advance=1)

        console.print(f"[green]Imported {imported_count} vessels from {csv_file.name}[/green]")

    finally:
        await db.close()


@vessels_app.command("retag")
def vessels_retag(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview changes without updating the database",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output per document",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of documents to process",
    ),
):
    """Re-tag existing chunks with vessel IDs from archive content.

    Scans the markdown files in the archive bucket, matches vessel names
    using the current vessel registry, and updates chunk metadata.

    Use this after importing new vessels to tag existing documents
    without re-ingesting from scratch.

    Examples:
        MTSS vessels retag              # Re-tag all documents
        MTSS vessels retag --dry-run    # Preview changes
        MTSS vessels retag --limit 100  # Process first 100 documents
    """
    global _verbose
    _verbose = verbose
    asyncio.run(_vessels_retag(dry_run, limit))


async def _vessels_retag(dry_run: bool, limit: int | None):
    """Async implementation of vessels retag command."""
    from uuid import UUID as UUIDType

    from .storage.archive_storage import ArchiveStorage, ArchiveStorageError

    db = SupabaseClient()

    try:
        # Load vessel registry
        vessels = await db.get_all_vessels()
        if not vessels:
            console.print("[yellow]No vessels in registry - nothing to tag[/yellow]")
            return

        vessel_matcher = VesselMatcher(vessels)
        console.print(f"Loaded {vessel_matcher.vessel_count} vessels ({vessel_matcher.name_count} names)")

        # Initialize archive storage
        try:
            storage = ArchiveStorage()
        except ArchiveStorageError as e:
            console.print(f"[red]Failed to initialize archive storage: {e}[/red]")
            return

        # Get documents to process
        documents = await db.get_root_documents_for_retagging(limit)
        if not documents:
            console.print("[yellow]No documents found to retag[/yellow]")
            return

        console.print(f"Found {len(documents)} documents to scan")
        if dry_run:
            console.print("[dim]Dry run mode - no changes will be made[/dim]")

        # Track statistics
        docs_scanned = 0
        docs_updated = 0
        chunks_updated = 0
        vessels_added = 0
        vessels_removed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning documents...", total=len(documents))

            for doc in documents:
                doc_id = doc.get("doc_id")
                document_uuid = UUIDType(doc["id"])
                file_name = doc.get("file_name", "unknown")

                if not doc_id:
                    progress.update(task, advance=1)
                    continue

                # Download markdown content from archive
                content_parts: list[str] = []

                # Get email markdown
                try:
                    email_md_path = f"{doc_id}/email.eml.md"
                    email_content = storage.download_file(email_md_path)
                    content_parts.append(email_content.decode("utf-8", errors="ignore"))
                except ArchiveStorageError:
                    vprint(f"No email.md found for {doc_id}", file_name)

                # Get attachment markdowns
                try:
                    files = storage.list_files(doc_id)
                    for f in files:
                        fname = f.get("name", "")
                        if fname.endswith(".md") and fname != "email.eml.md":
                            try:
                                # Files in attachments subfolder
                                att_path = f"{doc_id}/attachments/{fname}"
                                att_content = storage.download_file(att_path)
                                content_parts.append(att_content.decode("utf-8", errors="ignore"))
                            except ArchiveStorageError:
                                pass
                except ArchiveStorageError:
                    pass

                if not content_parts:
                    vprint(f"No content found for {doc_id}", file_name)
                    progress.update(task, advance=1)
                    continue

                # Match vessels in combined content
                combined_content = "\n\n".join(content_parts)
                matched_vessels = vessel_matcher.find_vessels(combined_content)
                new_vessel_ids = sorted([str(v) for v in matched_vessels])
                new_vessel_types = vessel_matcher.get_types_for_ids(matched_vessels)
                new_vessel_classes = vessel_matcher.get_classes_for_ids(matched_vessels)

                # Get current vessel IDs
                current_vessel_ids = await db.get_current_vessel_ids(document_uuid)

                # Check if update is needed (always update to add types/classes for backfill)
                if set(new_vessel_ids) != set(current_vessel_ids) or new_vessel_types or new_vessel_classes:
                    added = set(new_vessel_ids) - set(current_vessel_ids)
                    removed = set(current_vessel_ids) - set(new_vessel_ids)

                    if added or removed or new_vessel_types or new_vessel_classes:
                        vprint(
                            f"{file_name}: +{len(added)} -{len(removed)} vessels, "
                            f"{len(new_vessel_types)} types, {len(new_vessel_classes)} classes",
                            doc_id[:8],
                        )

                        if not dry_run:
                            updated = await db.update_chunks_vessel_metadata(
                                document_uuid, new_vessel_ids, new_vessel_types, new_vessel_classes
                            )
                            chunks_updated += updated

                        docs_updated += 1
                        vessels_added += len(added)
                        vessels_removed += len(removed)

                docs_scanned += 1
                progress.update(task, advance=1)

        # Show summary
        console.print()
        table = Table(title="Vessel Retagging Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Documents scanned", str(docs_scanned))
        table.add_row("Documents updated", str(docs_updated))
        if not dry_run:
            table.add_row("Chunks updated", str(chunks_updated))
        table.add_row("Vessel tags added", str(vessels_added))
        table.add_row("Vessel tags removed", str(vessels_removed))

        console.print(table)

        if dry_run and docs_updated > 0:
            console.print("\n[dim]Run without --dry-run to apply changes[/dim]")

    finally:
        await db.close()


@vessels_app.command("list")
def vessels_list():
    """List all vessels in the registry."""
    asyncio.run(_vessels_list())


async def _vessels_list():
    """Async implementation of vessels list command."""
    db = SupabaseClient()

    try:
        vessels = await db.get_all_vessels()

        if not vessels:
            console.print("[dim]No vessels in registry[/dim]")
            return

        table = Table(title=f"Vessel Registry ({len(vessels)} vessels)")
        table.add_column("Name", style="cyan")
        table.add_column("IMO", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("DWT", justify="right")

        for vessel in vessels:
            dwt_str = f"{vessel.dwt:,}" if vessel.dwt else "-"
            table.add_row(
                vessel.name,
                vessel.imo or "-",
                vessel.vessel_type or "-",
                dwt_str,
            )

        console.print(table)

    finally:
        await db.close()


# ==================== Topics Commands ====================


@topics_app.command("list")
def topics_list():
    """List all topics with document counts."""
    asyncio.run(_topics_list())


async def _topics_list():
    """Async implementation of topics list command."""
    db = SupabaseClient()

    try:
        topics = await db.get_all_topics()

        if not topics:
            console.print("[dim]No topics in database[/dim]")
            return

        table = Table(title=f"Topics ({len(topics)} total)")
        table.add_column("Name", style="cyan")
        table.add_column("Chunks", justify="right", style="green")

        for topic in topics:
            table.add_row(
                topic.display_name,
                str(topic.chunk_count),
            )

        console.print(table)

    finally:
        await db.close()


@app.command()
def estimate(
    source_dir: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Directory containing EML files",
    ),
    page_cost: float = typer.Option(
        0.00625,
        "--page-cost",
        help="LlamaParse cost per page USD (default: 5 avg credits × $0.00125/credit)",
    ),
    vision_cost: float = typer.Option(
        0.01,
        "--vision-cost",
        help="Vision API cost per image USD",
    ),
    text_cost: float = typer.Option(
        0.001,
        "--text-cost",
        help="LLM text processing cost per file USD",
    ),
    embedding_cost: float = typer.Option(
        0.00002,
        "--embedding-cost",
        help="Embedding cost per chunk USD (text-embedding-3-small)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show per-file details, errors, and files with issues",
    ),
):
    """Estimate ingest cost by extracting attachments and counting pages.

    Extracts all EML attachments (including ZIP contents) into a persistent
    folder, counts actual pages, and produces a cost breakdown.
    Subsequent runs use cached results and are instant.
    """
    from .ingest.estimator import (
        IngestEstimator,
        PAGE_COUNT_CATEGORIES,
        TEXT_CATEGORIES,
        VISION_CATEGORIES,
    )

    settings = get_settings()
    source = source_dir or settings.eml_source_dir

    if not source.exists():
        console.print(f"[red]Source directory not found: {source}[/red]")
        raise typer.Exit(1)

    with console.status("Scanning EML files and counting pages..."):
        estimator = IngestEstimator(source_dir=source)
        result = estimator.scan()

    if result.eml_count == 0:
        console.print("[dim]No EML files found in source directory.[/dim]")
        raise typer.Exit(0)

    _show_estimate(result, page_cost, vision_cost, text_cost, embedding_cost, verbose)


def _show_estimate(
    result,
    page_cost: float,
    vision_cost: float,
    text_cost: float,
    embedding_cost: float,
    verbose: bool,
):
    """Display estimate results as Rich tables."""
    from .ingest.estimator import (
        PAGE_COUNT_CATEGORIES,
        TEXT_CATEGORIES,
        VISION_CATEGORIES,
    )

    # ── Table 1: File Inventory ────────────────────────────────────────
    inv_table = Table(title="Ingest File Inventory")
    inv_table.add_column("Category", style="cyan")
    inv_table.add_column("Files", justify="right", style="green")
    inv_table.add_column("Pages", justify="right")
    inv_table.add_column("Unknown", justify="right", style="yellow")

    # Display order
    display_order = [
        "PDF", "DOCX", "PPTX", "XLSX", "DOC", "PPT", "XLS",
        "Other Docs", "Images", "Text/Markdown", "Other",
    ]

    total_files = 0
    total_pages = 0
    total_unknown = 0

    for cat_name in display_order:
        stats = result.categories.get(cat_name)
        if not stats or stats.file_count == 0:
            continue

        total_files += stats.file_count
        has_pages = cat_name in PAGE_COUNT_CATEGORIES

        if has_pages:
            total_pages += stats.page_count
            total_unknown += stats.pages_unknown
            pages_str = str(stats.page_count)
            if stats.pages_unknown > 0:
                pct = round(100 * stats.pages_unknown / stats.file_count)
                unknown_str = f"{stats.pages_unknown} ({pct}%)"
            else:
                unknown_str = "0"
        elif cat_name in VISION_CATEGORIES and stats.images_meaningful > 0:
            # Show meaningful/skipped split for images
            pages_str = f"~{stats.images_meaningful} meaningful"
            skipped = stats.images_skipped
            unknown_str = f"{skipped} skipped" if skipped else "0"
        else:
            pages_str = "—"
            unknown_str = "—"

        inv_table.add_row(cat_name, str(stats.file_count), pages_str, unknown_str)

    # Total row
    inv_table.add_section()
    if total_unknown > 0 and total_files > 0:
        pct = round(100 * total_unknown / total_files)
        total_unknown_str = f"{total_unknown} ({pct}%)"
    else:
        total_unknown_str = "0"
    inv_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_files}[/bold]",
        f"[bold]{total_pages}[/bold]",
        f"[bold]{total_unknown_str}[/bold]",
    )

    console.print(inv_table)
    console.print()

    # ── Table 2: Cost Estimate ─────────────────────────────────────────
    cost_table = Table(title="Estimated Ingest Cost")
    cost_table.add_column("Service", style="cyan")
    cost_table.add_column("Units", justify="right")
    cost_table.add_column("Unit Cost", justify="right")
    cost_table.add_column("Cost", justify="right", style="green")

    # LlamaParse (documents) — uses total_pages from page-counted categories
    llama_pages = total_pages
    llama_total = llama_pages * page_cost
    cost_table.add_row(
        "LlamaParse (documents)",
        f"{llama_pages} pages",
        f"${page_cost:.5f}",
        f"${llama_total:.2f}",
    )

    # Vision API (images) — only meaningful images get described
    image_meaningful = 0
    image_skipped = 0
    for cat_name in VISION_CATEGORIES:
        stats = result.categories.get(cat_name)
        if stats:
            image_meaningful += stats.images_meaningful
            image_skipped += stats.images_skipped
    # Fall back to total file_count if heuristic data not available (old cache)
    if image_meaningful == 0 and image_skipped == 0:
        for cat_name in VISION_CATEGORIES:
            stats = result.categories.get(cat_name)
            if stats:
                image_meaningful = stats.file_count
    vision_total = image_meaningful * vision_cost
    image_label = f"~{image_meaningful} images"
    if image_skipped > 0:
        image_label += f" ({image_skipped} skipped)"
    cost_table.add_row(
        "Vision API (images)",
        image_label,
        f"${vision_cost:.5f}",
        f"${vision_total:.2f}",
    )

    # LLM text (text files)
    text_count = 0
    for cat_name in TEXT_CATEGORIES:
        stats = result.categories.get(cat_name)
        if stats:
            text_count += stats.file_count
    text_total = text_count * text_cost
    cost_table.add_row(
        "LLM text (text files)",
        f"{text_count} files",
        f"${text_cost:.5f}",
        f"${text_total:.2f}",
    )

    # Embeddings (~1.5x pages)
    estimated_chunks = round(total_pages * 1.5)
    embed_total = estimated_chunks * embedding_cost
    cost_table.add_row(
        "Embeddings (~1.5x pages)",
        f"~{estimated_chunks} chunks",
        f"${embedding_cost:.5f}",
        f"${embed_total:.2f}",
    )

    # Total row
    grand_total = llama_total + vision_total + text_total + embed_total
    cost_table.add_section()
    cost_table.add_row(
        "[bold]TOTAL ESTIMATED[/bold]",
        "",
        "",
        f"[bold]${grand_total:.2f}[/bold]",
    )

    console.print(cost_table)

    # ── Footnotes ──────────────────────────────────────────────────────
    console.print()
    console.print(
        f"  [dim]LlamaParse: 5 avg credits/page × $0.00125/credit ($50 / 40k credits)[/dim]"
    )
    console.print(
        f"  [dim]Embeddings: ~1.5 chunks/page × text-embedding-3-small ($0.02/M tokens)[/dim]"
    )
    if image_skipped > 0:
        console.print(
            f"  [dim]Images: {image_skipped} likely non-content (logos, icons, banners) "
            f"excluded by size/dimension heuristic[/dim]"
        )

    if total_unknown > 0:
        console.print(
            f"  [dim]{total_unknown} files had unknown page counts (counted as 1 page) "
            f"— use --verbose to see details[/dim]"
        )

    if result.scan_errors:
        console.print(
            f"  [yellow]{len(result.scan_errors)} EML files failed to process[/yellow]"
        )

    # ── Footer ─────────────────────────────────────────────────────────
    console.print()
    console.print(
        f"[dim]Scanned {result.eml_count} EML files "
        f"({result.cached_count} cached, {result.extracted_count} newly extracted) "
        f"in {result.elapsed_seconds:.1f}s[/dim]"
    )

    # ── Verbose output ─────────────────────────────────────────────────
    if verbose and (result.all_issues or result.scan_errors):
        console.print()

        if result.all_issues:
            issue_table = Table(title="Files with Issues")
            issue_table.add_column("File", style="cyan")
            issue_table.add_column("Issue", style="yellow")
            issue_table.add_column("Detail", style="dim")

            for issue in result.all_issues:
                issue_table.add_row(issue.file, issue.issue, issue.detail)

            console.print(issue_table)

        if result.scan_errors:
            console.print()
            console.print("[bold yellow]Scan Errors:[/bold yellow]")
            for err in result.scan_errors:
                console.print(f"  [red]• {err}[/red]")


if __name__ == "__main__":
    app()
