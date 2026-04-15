"""Vessels and topics CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID as UUIDType

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


def register(app: typer.Typer, vessels_app: typer.Typer, topics_app: typer.Typer):
    """Register vessel and topic commands on their sub-apps."""

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
        _common._verbose = verbose
        asyncio.run(_vessels_retag(dry_run, limit))

    @vessels_app.command("list")
    def vessels_list():
        """List all vessels in the registry."""
        asyncio.run(_vessels_list())

    # ==================== Topics Commands ====================

    @topics_app.command("list")
    def topics_list():
        """List all topics with document counts."""
        asyncio.run(_topics_list())


async def _vessels_import(csv_file: Optional[Path], clear: bool):
    """Async implementation of vessels import command."""
    import csv

    from ..config import get_settings
    from ..models.vessel import Vessel
    from ..storage.supabase_client import SupabaseClient

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
                aliases_str = (row.get("ALIASES") or "").strip()
                aliases = [a.strip() for a in aliases_str.split(",") if a.strip()] if aliases_str else []

                if not name:
                    continue  # Skip rows without vessel name

                if not vessel_type or not vessel_class:
                    console.print(f"[yellow]Warning: Skipping {name} - missing TYPE or CLASS[/yellow]")
                    continue

                vessel = Vessel(
                    name=name,
                    vessel_type=vessel_type,
                    vessel_class=vessel_class,
                    aliases=aliases,
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


async def _vessels_retag(dry_run: bool, limit: int | None):
    """Async implementation of vessels retag command."""
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.archive_storage import ArchiveStorage, ArchiveStorageError
    from ..storage.supabase_client import SupabaseClient

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


async def _vessels_list():
    """Async implementation of vessels list command."""
    from ..storage.supabase_client import SupabaseClient

    db = SupabaseClient()

    try:
        vessels = await db.get_all_vessels()

        if not vessels:
            console.print("[dim]No vessels in registry[/dim]")
            return

        table = Table(title=f"Vessel Registry ({len(vessels)} vessels)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Class", style="green")
        table.add_column("Aliases", style="dim")

        for vessel in vessels:
            aliases_str = ", ".join(vessel.aliases) if vessel.aliases else "-"
            table.add_row(
                vessel.name,
                vessel.vessel_type or "-",
                vessel.vessel_class or "-",
                aliases_str,
            )

        console.print(table)

    finally:
        await db.close()


async def _topics_list():
    """Async implementation of topics list command."""
    from ..storage.supabase_client import SupabaseClient

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
