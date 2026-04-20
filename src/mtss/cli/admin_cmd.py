"""Admin CLI commands: stats, failures, reset-stale, reset-failures, clean."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.table import Table

from . import _common
from ._common import console, vprint, _show_stats


def register(app: typer.Typer):
    """Register admin commands on the app."""

    @app.command()
    def stats():
        """Show processing statistics."""
        asyncio.run(_show_processing_stats())

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

    @app.command()
    def reset_stale(
        max_age: int = typer.Option(60, "--max-age", "-m", help="Max age in minutes"),
    ):
        """Reset files stuck in 'processing' state."""
        asyncio.run(_reset_stale(max_age))

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
        _common._verbose = verbose

        console.print("[yellow]\u26a0\ufe0f  This will delete ALL data from the database and processed files![/yellow]")

        if not yes:
            confirm = typer.confirm("Are you sure you want to continue?")
            if not confirm:
                console.print("[dim]Cancelled[/dim]")
                raise typer.Exit(0)

        asyncio.run(_clean())


async def _show_processing_stats():
    """Async implementation of stats command."""
    from ..storage.progress_tracker import ProgressTracker
    from ..storage.supabase_client import SupabaseClient
    from ..storage.unsupported_file_logger import UnsupportedFileLogger

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


async def _failures(latest: bool, export: bool, limit: int):
    """Async implementation of failures command."""
    import json

    from ..storage.failure_report import FailureReportGenerator
    from ..storage.supabase_client import SupabaseClient

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


async def _reset_stale(max_age: int):
    """Async implementation of reset-stale command."""
    from ..storage.progress_tracker import ProgressTracker
    from ..storage.supabase_client import SupabaseClient

    db = SupabaseClient()
    tracker = ProgressTracker(db)

    try:
        await tracker.reset_stale_processing(max_age)
        console.print(f"[green]Reset stale processing entries older than {max_age} minutes[/green]")
    finally:
        await db.close()


async def _reset_failures(report_file: Path, eml_only: bool, dry_run: bool, yes: bool):
    """Async implementation of reset-failures command."""
    import json

    from ..storage.supabase_client import SupabaseClient

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


async def _clean():
    """Async implementation of clean command."""
    import shutil

    from ..config import get_settings
    from ..storage.archive_storage import ArchiveStorage, ArchiveStorageError
    from ..storage.supabase_client import SupabaseClient

    settings = get_settings()
    db = SupabaseClient()

    try:
        # Delete from database — live spinner so the user can see which
        # table is draining; a full clean can run for minutes on a real
        # corpus and without this feedback the CLI looks frozen.
        with console.status(
            "[cyan]Deleting database records...[/cyan]", spinner="dots"
        ) as _status:
            counts = await db.delete_all_data(
                status=lambda msg: _status.update(f"[cyan]{msg}[/cyan]")
            )

        # Show database deletion results
        total_db_records = sum(counts.values())
        if total_db_records > 0:
            console.print(f"[green]Deleted {total_db_records} database records:[/green]")
            for table_name, count in counts.items():
                if count > 0:
                    vprint(f"  {table_name}: {count}")
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

        console.print("[green]\u2713 Clean complete[/green]")

    finally:
        await db.close()
