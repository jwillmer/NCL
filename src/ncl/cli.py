"""CLI entry point for NCL - Email RAG Pipeline."""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import nest_asyncio

# Apply nest_asyncio to allow nested event loops (required for LlamaParse)
nest_asyncio.apply()
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
from .models.chunk import Chunk
from .models.document import ProcessingStatus
from .parsers.attachment_processor import AttachmentProcessor
from .parsers.chunker import ContextGenerator
from .parsers.email_cleaner import (
    clean_email_body,
    remove_boilerplate_from_message,
    split_into_messages,
)
from .parsers.eml_parser import EMLParser
from .processing.archive_generator import ArchiveGenerator
from .processing.embeddings import EmbeddingGenerator
from .processing.hierarchy_manager import HierarchyManager
from .processing.version_manager import VersionManager
from .rag.query_engine import RAGQueryEngine, format_response_with_sources
from .storage.unsupported_file_logger import UnsupportedFileLogger
from .storage.progress_tracker import ProgressTracker
from .storage.supabase_client import SupabaseClient
from .utils import compute_chunk_id

app = typer.Typer(
    name="ncl",
    help="NCL - Email RAG Pipeline for processing EML files with attachments",
)
console = Console()

# Module-level verbose flag
_verbose = False

# Flag to track if shutdown was requested
_shutdown_requested = False


def _handle_interrupt(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second Ctrl+C - force exit
        console.print("\n[red]Force exiting...[/red]")
        sys.exit(1)
    _shutdown_requested = True
    console.print("\n[yellow]Interrupted! Cleaning up... (press Ctrl+C again to force exit)[/yellow]")
    raise KeyboardInterrupt


def vprint(msg: str, file_ctx: str | None = None):
    """Print verbose output if enabled, with optional file context for concurrent logs."""
    if _verbose:
        if file_ctx:
            console.print(f"[dim][{file_ctx}] {msg}[/dim]")
        else:
            console.print(f"[dim]{msg}[/dim]")


# Track processing issues for end-of-run summary
_processing_issues: list[dict] = []


def track_issue(file_ctx: str, attachment: str, error: str):
    """Track a parsing/processing issue for end summary and print warning."""
    _processing_issues.append({
        "email": file_ctx,
        "attachment": attachment,
        "error": error
    })
    console.print(f"[yellow][{file_ctx}] ⚠ {attachment}: {error}[/yellow]")


def _enrich_chunks_with_document_metadata(
    chunks: list[Chunk],
    doc: "Document",
) -> None:
    """Enrich chunks with citation metadata from their document.

    Sets source_id, source_title, and archive URIs on each chunk.
    Modifies chunks in-place.

    Args:
        chunks: List of chunks to enrich.
        doc: Parent document with citation metadata.
    """
    from .models.document import Document  # Avoid circular import at top level
    for chunk in chunks:
        chunk.source_id = doc.source_id
        chunk.source_title = doc.source_title
        chunk.archive_browse_uri = doc.archive_browse_uri
        chunk.archive_download_uri = doc.archive_download_uri


def _show_issue_summary(issues: list[dict]):
    """Display summary table of all parsing/processing issues."""
    if not issues:
        return

    console.print()
    table = Table(title=f"⚠ Processing Issues ({len(issues)} total)")
    table.add_column("Email", style="cyan")
    table.add_column("Attachment", style="yellow")
    table.add_column("Error", style="red")

    for issue in issues:
        error_text = issue["error"]
        if len(error_text) > 60:
            error_text = error_text[:60] + "..."
        table.add_row(issue["email"], issue["attachment"], error_text)

    console.print(table)


def _get_format_name(content_type: str) -> str:
    """Get human-readable format name from MIME type."""
    format_map = {
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
    return format_map.get(content_type, content_type.split("/")[-1].upper())


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
        asyncio.run(_ingest(source_dir, batch_size, resume, retry_failed, reprocess_outdated))
    except KeyboardInterrupt:
        console.print("[yellow]Processing stopped by user[/yellow]")
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
    archive_generator = ArchiveGenerator()
    context_generator = ContextGenerator()
    version_manager = VersionManager(db)

    try:
        # Get files to process
        if reprocess_outdated:
            files = await tracker.get_outdated_files(source_dir, settings.current_ingest_version)
            console.print(f"[yellow]Reprocessing {len(files)} outdated files (ingest_version < {settings.current_ingest_version})[/yellow]")
        elif retry_failed:
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
        _processing_issues.clear()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(settings.max_concurrent_files)

        # Create a queue of available slot indices for worker progress bars
        slot_queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(settings.max_concurrent_files):
            slot_queue.put_nowait(i)

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

            # Total progress bar
            total_task = progress.add_task("[bold]Total[/bold]", total=len(files))

            async def process_with_limit(file_path: Path) -> tuple[Path, bool, str | None]:
                """Process a file with concurrency limiting and real-time progress."""
                nonlocal processed_count
                async with semaphore:
                    # Acquire a worker slot from queue
                    slot_idx = await slot_queue.get()
                    worker_task = worker_tasks[slot_idx]

                    try:
                        if _shutdown_requested:
                            progress.update(total_task, advance=1)
                            return (file_path, False, "shutdown")

                        # Show starting state
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
                            progress=progress,
                            worker_task=worker_task,
                            slot_idx=slot_idx,
                        )
                        processed_count += 1
                        progress.update(total_task, advance=1)
                        return (file_path, True, None)
                    except Exception as e:
                        await tracker.mark_failed(file_path, str(e))
                        console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                        progress.update(total_task, advance=1)
                        return (file_path, False, str(e))
                    finally:
                        # Release slot - mark as idle
                        progress.update(
                            worker_task,
                            description=f"[dim][{slot_idx + 1}] (idle)[/dim]",
                            total=1,
                            completed=0,
                        )
                        slot_queue.put_nowait(slot_idx)

            # Process all files concurrently (semaphore limits to max_concurrent_files at a time)
            tasks = [process_with_limit(f) for f in files]
            await asyncio.gather(*tasks, return_exceptions=True)

            if _shutdown_requested:
                console.print(f"[yellow]Stopped after {processed_count} files[/yellow]")

        # Show final stats and issue summary
        stats = await tracker.get_processing_stats()
        _show_stats(stats)
        _show_issue_summary(_processing_issues)

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
    progress: Progress | None = None,
    worker_task: TaskID | None = None,
    slot_idx: int | None = None,
):
    """Process a single EML file with all its attachments."""
    file_hash = tracker.compute_file_hash(eml_path)
    source_eml_path = str(eml_path)
    file_ctx = eml_path.name  # Short filename for logging

    # Check using version manager if available
    if version_manager:
        from .utils import normalize_source_id
        settings = get_settings()
        source_id = normalize_source_id(source_eml_path, settings.eml_source_dir)
        decision = await version_manager.check_document(source_id, file_hash)

        if decision.action == "skip":
            vprint(f"Skipping (already processed): {eml_path}", file_ctx)
            return
        elif decision.action == "reprocess":
            vprint(f"Reprocessing: {decision.reason}", file_ctx)
            # Delete old document (cascades to child docs and chunks)
            if decision.existing_doc_id:
                db.delete_document_for_reprocess(decision.existing_doc_id)
        elif decision.action == "update":
            vprint(f"Updating: {decision.reason}", file_ctx)
    else:
        # Fallback to legacy check
        existing = await db.get_document_by_hash(file_hash)
        if existing and existing.status == ProcessingStatus.COMPLETED:
            vprint(f"Skipping (already processed): {eml_path}", file_ctx)
            return

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
            )
            vprint(f"Archive generated: {archive_result.archive_path} ({len(archive_result.attachment_files)} attachments)", file_ctx)
        except Exception as e:
            vprint(f"Archive generation failed: {e}", file_ctx)

    # Create email document in hierarchy
    email_doc = await hierarchy_manager.create_email_document(
        eml_path, parsed_email, archive_result=archive_result
    )

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
            embedding_text = cleaned_message
            if context_summary and msg_idx == 0:
                # Only add context summary to the first message
                embedding_text = context_generator.build_embedding_text(context_summary, cleaned_message)

            email_chunks.append(
                Chunk(
                    document_id=email_doc.id,
                    content=message,  # Store original message content
                    chunk_index=msg_idx,
                    heading_path=["Email Body", f"Message {msg_idx + 1}"],
                    metadata={"type": "email_body", "message_index": msg_idx},
                    chunk_id=chunk_id,
                    context_summary=context_summary if msg_idx == 0 else None,
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
            # Look for matching attachment in archive result by filename
            for file_result in archive_result.attachment_files:
                # original_path is like "abc123/attachments/file.pdf"
                if file_result.original_path.endswith(f"/{attachment.filename}"):
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

    # Mark email as completed
    await db.update_document_status(email_doc.id, ProcessingStatus.COMPLETED)
    await tracker.mark_completed(eml_path)

    # Clean up processed attachments (archive has the originals for download)
    if archive_result and parsed_email.attachments:
        for attachment in parsed_email.attachments:
            try:
                processed_path = Path(attachment.saved_path)
                if processed_path.exists():
                    processed_path.unlink()
                # Clean up parent directory if empty
                parent = processed_path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception as e:
                vprint(f"Warning: Failed to clean up {attachment.saved_path}: {e}", file_ctx)


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
) -> list[Chunk]:
    """Process a single attachment and return its chunks.

    Uses preprocessor for routing decisions.
    """
    chunks: list[Chunk] = []
    file_path = Path(attachment.saved_path)

    # Format size for display
    size_kb = attachment.size_bytes / 1024 if attachment.size_bytes else 0
    size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
    format_name = _get_format_name(attachment.content_type or "unknown")

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
            vprint(f"  -> Skipped: non-content image (logo/banner/signature)", file_ctx)
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
        if result.is_image and result.image_description:
            # Image was already classified and described during preprocessing
            chunk = attachment_processor.create_image_chunk(
                file_path, attach_doc.id, result.image_description, "meaningful"
            )
            chunks.append(chunk)
            vprint(f"  -> 1 chunk created (image described)", file_ctx)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
        elif result.is_image:
            # Image needs description (preprocessing didn't provide one)
            attach_chunks = await attachment_processor.process_document_image(
                file_path, attach_doc.id
            )
            chunks.extend(attach_chunks)
            vprint(f"  -> {len(attach_chunks)} chunks created (image described)", file_ctx)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
        else:
            # Document - use parser registry
            attach_chunks = await attachment_processor.process_attachment(
                file_path, attach_doc.id, attachment.content_type
            )
            if attach_chunks:
                vprint(f"  -> {len(attach_chunks)} chunks created", file_ctx)
            else:
                vprint(f"  -> 0 chunks (document has no extractable text)", file_ctx)
            chunks.extend(attach_chunks)
            await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
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
    _enrich_chunks_with_document_metadata(chunks, attach_doc)

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
) -> list[Chunk]:
    """Extract and process files from a ZIP attachment."""
    chunks: list[Chunk] = []

    try:
        extracted_files = attachment_processor.extract_zip(Path(attachment.saved_path))
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
                if result.is_image:
                    # Images from ZIPs - describe without classification
                    attach_chunks = await attachment_processor.process_document_image(
                        extracted_path, attach_doc.id
                    )
                else:
                    # Documents - use parser registry
                    attach_chunks = await attachment_processor.process_attachment(
                        extracted_path, attach_doc.id, extracted_content_type
                    )
                # Enrich chunks with document citation metadata
                _enrich_chunks_with_document_metadata(attach_chunks, attach_doc)
                chunks.extend(attach_chunks)
                await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
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
            relevance = source.rerank_score if source.rerank_score is not None else source.similarity_score
            relevance_label = f"{relevance:.1%}" + (" ✓" if source.rerank_score is not None else "")
            table.add_row(
                str(i),
                Path(source.file_path).name,
                source.email_subject or "-",
                relevance_label,
                source.chunk_content[:100] + "...",
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

        # Delete archive files
        archive_dir = settings.archive_dir
        archive_files_deleted = 0

        if archive_dir.exists():
            vprint(f"Cleaning {archive_dir}...")
            for item in archive_dir.iterdir():
                if item.is_dir():
                    file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                    archive_files_deleted += file_count
                    shutil.rmtree(item)
                    vprint(f"  Deleted {item.name}/ ({file_count} files)")
                elif item.is_file():
                    item.unlink()
                    archive_files_deleted += 1
                    vprint(f"  Deleted {item.name}")

            if archive_files_deleted > 0:
                console.print(f"[green]Deleted {archive_files_deleted} archive files[/green]")
            else:
                console.print("[dim]No archive files to delete[/dim]")
        else:
            console.print("[dim]Archive directory does not exist[/dim]")

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
            console.print(f"\n[dim]Run without --dry-run to process these documents[/dim]")
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
        console.print("[red]Reprocessing logic not yet implemented - use 'ncl clean' and re-ingest[/red]")

    finally:
        await db.close()


if __name__ == "__main__":
    app()
