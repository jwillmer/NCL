"""Maintenance CLI commands: ingest-update, reprocess, reindex-chunks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from . import _common
from ._common import console, vprint


def register(app: typer.Typer):
    """Register maintenance commands on the app."""

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
        _common._verbose = verbose
        asyncio.run(_reprocess(target_version, limit, dry_run))

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
        _common._verbose = verbose

        if not doc_id and not missing_lines:
            console.print("[red]Error: Specify --doc-id or --missing-lines[/red]")
            raise typer.Exit(1)

        asyncio.run(_reindex_chunks(doc_id, missing_lines, dry_run, limit))

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
        - Missing topics (backfills topic extraction for older documents)

        With atomic persistence, crash-recovery repairs (missing archives,
        lines, context) are no longer needed.

        Examples:
            MTSS ingest-update --dry-run          # Scan only, show issues
            MTSS ingest-update                    # Fix all issues
            MTSS ingest-update --limit 10 -v      # Fix 10 docs with details
        """
        _common._verbose = verbose

        asyncio.run(_ingest_update(source_dir, dry_run, limit))


async def _reprocess(target_version: int | None, limit: int, dry_run: bool):
    """Async implementation of reprocess command."""
    from ..config import get_settings
    from ..ingest.version_manager import VersionManager
    from ..storage.supabase_client import SupabaseClient

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


async def _reindex_chunks(
    doc_id: Optional[str],
    missing_lines: bool,
    dry_run: bool,
    limit: int,
):
    """Async implementation of reindex-chunks command."""
    from uuid import UUID

    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..processing.embeddings import EmbeddingGenerator
    from ..storage.archive_storage import ArchiveStorage
    from ..storage.supabase_client import SupabaseClient
    from ..utils import compute_chunk_id

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
                console.print(f"  \u2022 {doc['source_title'] or doc['id']}")
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
                        from ..models.document import Document, DocumentType
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
        console.print("\n[green]\u2713 Re-index complete[/green]")
        console.print(f"  Documents: {stats['success']} success, {stats['failed']} failed")
        console.print(f"  Chunks created: {stats['chunks_created']}")

    finally:
        await db.close()


async def _get_documents_by_id(db, doc_id: str) -> list[dict]:
    """Get a single document by ID."""
    result = db.client.table("documents").select(
        "id, doc_id, source_id, source_title, archive_browse_uri, archive_download_uri"
    ).eq("id", doc_id).execute()
    return result.data or []


async def _get_documents_missing_lines(db, limit: int) -> list[dict]:
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


async def _ingest_update(
    source_dir: Optional[Path],
    dry_run: bool,
    limit: int,
):
    """Async implementation of ingest-update command."""
    from ..config import get_settings
    from ..ingest.components import create_ingest_components
    from ..ingest.repair import (
        find_orphaned_documents,
        fix_document_issues,
        scan_ingest_issues,
    )
    from ..storage.supabase_client import SupabaseClient

    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Scanning: {source_dir}[/dim]")

    # Initialize database and components
    db = SupabaseClient()

    # Verbose callback for repair functions
    def _on_verbose(msg: str, file_ctx: str | None = None) -> None:
        vprint(msg, file_ctx)

    try:
        # Load vessels for component initialization
        vessels = await db.get_all_vessels()

        # Create shared components (same as regular ingest)
        components = create_ingest_components(db, source_dir, vessels)

        # With atomic persistence, only topic backfill remains
        checks = {"topics"}

        # Progress callback for scan phase
        scan_progress = None
        scan_task = None

        def _on_scan_progress(description: str, current: int, total: int) -> None:
            nonlocal scan_progress, scan_task
            if scan_progress is None:
                return
            if scan_task is None:
                scan_task = scan_progress.add_task(description, total=total)
            else:
                scan_progress.update(scan_task, description=description, completed=current)

        # Phase 1a: Scan for issues in existing documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as scan_progress:
            issues = await scan_ingest_issues(
                source_dir, components, checks, limit,
                on_progress=_on_scan_progress,
                on_verbose=_on_verbose,
            )

        # Phase 1b: Find orphaned documents (in DB but source file deleted)
        orphan_ids = await find_orphaned_documents(source_dir, db)

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
                    console.print(f"  {record.eml_path.name}: {', '.join(record.issues)}")
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
                        chunks_created = await fix_document_issues(
                            record, components, checks, on_verbose=_on_verbose,
                        )
                        stats["fixed"] += 1
                        stats["chunks_created"] += chunks_created
                    except Exception as e:
                        console.print(f"[red]Failed to fix {record.eml_path.name}: {e}[/red]")
                        stats["failed"] += 1

                    progress.advance(task)

        # Print summary
        console.print("\n[green]Ingest update complete[/green]")
        console.print(f"  Documents fixed: {stats['fixed']}")
        console.print(f"  Documents failed: {stats['failed']}")
        console.print(f"  Chunks created/updated: {stats['chunks_created']}")
        if stats["orphans_removed"]:
            console.print(f"  Orphans removed: {stats['orphans_removed']}")

    finally:
        await db.close()
