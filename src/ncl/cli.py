"""CLI entry point for NCL - Email RAG Pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from .config import get_settings
from .models.chunk import Chunk
from .models.document import ProcessingStatus
from .parsers.attachment_processor import AttachmentProcessor
from .parsers.eml_parser import EMLParser
from .processing.embeddings import EmbeddingGenerator
from .processing.hierarchy_manager import HierarchyManager
from .rag.query_engine import RAGQueryEngine, format_response_with_sources
from .storage.progress_tracker import ProgressTracker
from .storage.supabase_client import SupabaseClient

app = typer.Typer(
    name="ncl",
    help="NCL - Email RAG Pipeline for processing EML files with attachments",
)
console = Console()


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
):
    """Ingest EML files and their attachments into the RAG system."""
    asyncio.run(_ingest(source_dir, batch_size, resume, retry_failed))


async def _ingest(
    source_dir: Optional[Path],
    batch_size: int,
    resume: bool,
    retry_failed: bool,
):
    """Async implementation of ingest command."""
    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    # Initialize components
    db = SupabaseClient()
    tracker = ProgressTracker(db)
    eml_parser = EMLParser()
    attachment_processor = AttachmentProcessor()
    hierarchy_manager = HierarchyManager(db)
    embeddings = EmbeddingGenerator()

    try:
        # Get files to process
        if retry_failed:
            files = await tracker.get_failed_files()
            console.print(f"[yellow]Retrying {len(files)} failed files[/yellow]")
        elif resume:
            files = await tracker.get_pending_files(source_dir)
            console.print(f"[green]Found {len(files)} pending files[/green]")
        else:
            files = list(source_dir.glob("**/*.eml"))
            console.print(f"[green]Found {len(files)} total EML files[/green]")

        if not files:
            console.print("[green]No files to process![/green]")
            return

        # Estimate cost
        console.print(f"[dim]Estimated embedding cost: ~${len(files) * 0.001:.2f}[/dim]")

        # Process files with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing emails...", total=len(files))

            for file_path in files:
                try:
                    await _process_single_email(
                        file_path,
                        eml_parser,
                        attachment_processor,
                        hierarchy_manager,
                        embeddings,
                        db,
                        tracker,
                    )
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Error processing {file_path}: {e}[/red]")
                    await tracker.mark_failed(file_path, str(e))
                    progress.update(task, advance=1)

        # Show final stats
        stats = await tracker.get_processing_stats()
        _show_stats(stats)

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
):
    """Process a single EML file with all its attachments."""
    file_hash = tracker.compute_file_hash(eml_path)

    # Check if already processed
    existing = await db.get_document_by_hash(file_hash)
    if existing and existing.status == ProcessingStatus.COMPLETED:
        return

    await tracker.mark_started(eml_path, file_hash)

    # Parse email
    parsed_email = eml_parser.parse_file(eml_path)

    # Create email document in hierarchy
    email_doc = await hierarchy_manager.create_email_document(eml_path, parsed_email)

    # Create chunks from email body
    email_chunks: list[Chunk] = []
    body_text = eml_parser.get_body_text(parsed_email)

    if body_text:
        email_chunks.append(
            Chunk(
                document_id=email_doc.id,
                content=body_text,
                chunk_index=0,
                heading_path=["Email Body"],
                metadata={"type": "email_body"},
            )
        )

    # Process attachments
    for attachment in parsed_email.attachments:
        if attachment_processor.is_supported(attachment.saved_path, attachment.content_type):
            # Create attachment document
            attach_doc = await hierarchy_manager.create_attachment_document(
                parent_doc=email_doc,
                attachment_path=Path(attachment.saved_path),
                content_type=attachment.content_type,
                size_bytes=attachment.size_bytes,
                original_filename=attachment.filename,
            )

            # Process and chunk attachment
            try:
                attach_chunks = attachment_processor.process_attachment(
                    Path(attachment.saved_path),
                    attach_doc.id,
                    attachment.content_type,
                )
                email_chunks.extend(attach_chunks)

                await db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
            except Exception as e:
                await db.update_document_status(
                    attach_doc.id,
                    ProcessingStatus.FAILED,
                    str(e),
                )

    # Generate embeddings for all chunks
    if email_chunks:
        email_chunks = await embeddings.embed_chunks(email_chunks)
        await db.insert_chunks(email_chunks)

    # Mark email as completed
    await db.update_document_status(email_doc.id, ProcessingStatus.COMPLETED)
    await tracker.mark_completed(eml_path)


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
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold"),
):
    """Query the RAG system with a question."""
    asyncio.run(_query(question, top_k, threshold))


async def _query(question: str, top_k: int, threshold: float):
    """Async implementation of query command."""
    engine = RAGQueryEngine()

    try:
        with console.status("Searching and generating answer..."):
            response = await engine.query(
                question=question,
                top_k=top_k,
                similarity_threshold=threshold,
            )

        formatted = format_response_with_sources(response)
        console.print(formatted)

    finally:
        await engine.close()


@app.command()
def search(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
):
    """Search for relevant documents without generating an answer."""
    asyncio.run(_search(query_text, top_k, threshold))


async def _search(query_text: str, top_k: int, threshold: float):
    """Async implementation of search command."""
    engine = RAGQueryEngine()

    try:
        with console.status("Searching..."):
            sources = await engine.search_only(
                question=query_text,
                top_k=top_k,
                similarity_threshold=threshold,
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
            table.add_row(
                str(i),
                Path(source.file_path).name,
                source.email_subject or "-",
                f"{source.similarity_score:.1%}",
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

    try:
        stats = await tracker.get_processing_stats()
        _show_stats(stats)
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


if __name__ == "__main__":
    app()
