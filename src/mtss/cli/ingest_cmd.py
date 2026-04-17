"""Ingest and estimate CLI commands."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
)

from . import _common
from .ingest_reporting import show_estimate as _show_estimate
from .ingest_reporting import write_run_summary as _write_run_summary
from ._common import (
    STALE_PROCESSING_THRESHOLD_MINUTES,
    _issue_tracker,
    _service_counter,
    _show_stats,
    console,
    vprint,
)


def register(app: typer.Typer):
    """Register ingest and estimate commands on the app."""

    @app.command()
    def ingest(
        source_dir: Optional[Path] = typer.Option(
            None,
            "--source",
            "-s",
            help="Directory containing EML files",
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
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Output directory for processed data (default: <source>/../output)",
        ),
        limit: Optional[int] = typer.Option(
            None,
            "--limit",
            "-n",
            help="Max number of files to process (for validation batches)",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output with detailed processing info",
        ),
    ):
        """Ingest EML files and their attachments into the RAG system."""
        _common._verbose = verbose
        _common._shutdown_requested = False

        # Validate output_dir for path traversal
        if output_dir and ".." in str(output_dir):
            console.print("[red]--output-dir must not contain '..' (path traversal)[/red]")
            raise typer.Exit(1)

        # Set up signal handler for Ctrl+C
        from ._common import _handle_interrupt
        original_handler = signal.signal(signal.SIGINT, _handle_interrupt)

        try:
            asyncio.run(_ingest(
                source_dir, resume, retry_failed,
                reprocess_outdated, lenient, output_dir, limit,
            ))
        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            signal.signal(signal.SIGINT, original_handler)

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
            help="LlamaParse cost per page USD (default: 5 avg credits * $0.00125/credit)",
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
        from ..config import get_settings
        from ..ingest.estimator import IngestEstimator

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


async def _ingest(
    source_dir: Optional[Path],
    resume: bool,
    retry_failed: bool,
    reprocess_outdated: bool = False,
    lenient: bool = False,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
):
    """Async implementation of ingest command."""
    from ..config import get_settings
    from ..ingest.components import create_local_ingest_components
    from ..ingest.lane_classifier import LaneClassifier
    from ..ingest.pipeline import process_email
    from ..ingest.version_manager import VersionManager
    from ..models.vessel import load_vessels_from_csv
    from ..storage.failure_report import IngestReportWriter
    from ..storage.local_client import LocalStorageClient
    from ..storage.local_progress_tracker import LocalProgressTracker

    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    resolved_output = (output_dir or source_dir.parent / "output").resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Output: {resolved_output}[/green]")

    console.print("[dim]Loading prior data...[/dim]", end="\r")
    db = LocalStorageClient(output_dir=resolved_output)
    tracker = LocalProgressTracker(output_dir=resolved_output)
    unsupported_logger = db
    version_manager = VersionManager(db)
    vessels = load_vessels_from_csv()

    components = create_local_ingest_components(
        db=db,
        output_dir=resolved_output,
        source_dir=source_dir,
        vessels=vessels,
        enable_topics=True,
    )

    if components.vessel_matcher:
        vprint(f"Loaded {components.vessel_matcher.vessel_count} vessels ({components.vessel_matcher.name_count} names)")
    else:
        vprint("No vessels in registry - vessel tagging disabled")
    # Keep reference for lane classifier (needs eml_parser)
    eml_parser = components.eml_parser

    # Initialize continuous report writer
    report_writer = IngestReportWriter(source_dir=str(source_dir))
    vprint(f"Report: {report_writer.get_path()}")

    run_start = time.monotonic()
    run_start_time = datetime.now(timezone.utc).isoformat()

    try:
        # Get files to process
        console.print("[dim]Scanning files...[/dim]", end="\r")
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

        if limit and len(files) > limit:
            files = files[:limit]
            console.print(f"[yellow]Limited to {limit} files[/yellow]")

        vprint(f"Source directory: {source_dir}")

        if not files:
            console.print("[green]No files to process![/green]")
            return

        processed_count = 0

        # Reset trackers for this run
        _issue_tracker.clear()
        _service_counter.reset()

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

        # Split workers: slow workers capped at LlamaParse concurrency,
        # rest go to fast lane. Both worker types help the other queue
        # when their primary queue empties. The LlamaParse semaphore
        # caps concurrent API calls regardless of worker count.
        total_workers = settings.max_concurrent_files
        if fast_count > 0 and slow_count > 0:
            slow_worker_count = min(slow_count, settings.max_concurrent_llamaparse, total_workers - 1)
            fast_worker_count = total_workers - slow_worker_count
        elif fast_count > 0:
            fast_worker_count = total_workers
            slow_worker_count = 0
        else:
            fast_worker_count = 0
            slow_worker_count = total_workers

        vprint(f"Worker split: {fast_worker_count} fast, {slow_worker_count} slow (LlamaParse limit: {settings.max_concurrent_llamaparse})")

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

                if _common._shutdown_requested:
                    return

                slot_idx = await slot_queue.get()
                worker_task = worker_tasks[slot_idx]

                def _on_progress(current: int, total: int, desc: str) -> None:
                    """Callback for progress bar updates from pipeline."""
                    progress.update(
                        worker_task,
                        description=f"[{slot_idx + 1}] {desc}",
                        total=total,
                        completed=current,
                    )

                try:
                    progress.update(
                        worker_task,
                        description=f"[{slot_idx + 1}] {file_path.name}",
                        total=1,
                        completed=0,
                    )

                    email_result = await asyncio.wait_for(
                        process_email(
                            eml_path=file_path,
                            components=components,
                            tracker=tracker,
                            unsupported_logger=unsupported_logger,
                            version_manager=version_manager,
                            force_reparse=reprocess_outdated or retry_failed,
                            lenient=lenient,
                            on_verbose=vprint,
                            issue_tracker=_issue_tracker,
                            on_progress=_on_progress,
                        ),
                        timeout=settings.per_file_timeout_seconds,
                    )
                    if not email_result.skipped:
                        processed_count += 1
                except asyncio.TimeoutError:
                    error_msg = f"Timed out after {settings.per_file_timeout_seconds}s"
                    await tracker.mark_failed(file_path, error_msg)
                    report_writer.add_eml_failure(str(file_path), error_msg)
                    console.print(f"[red]Timeout: {file_path.name} ({error_msg})[/red]")
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
                """Worker for fast queue. Helps slow queue when idle."""
                while True:
                    if _common._shutdown_requested:
                        return
                    try:
                        file_path = fast_queue.get_nowait()
                        await process_one(file_path)
                    except asyncio.QueueEmpty:
                        # Fast queue empty — help with slow queue
                        try:
                            file_path = slow_queue.get_nowait()
                            await process_one(file_path)
                        except asyncio.QueueEmpty:
                            return  # Both queues empty

            async def slow_worker() -> None:
                """Worker for slow queue. Helps fast queue when idle."""
                while True:
                    if _common._shutdown_requested:
                        return
                    try:
                        file_path = slow_queue.get_nowait()
                        await process_one(file_path)
                    except asyncio.QueueEmpty:
                        # Slow queue empty — help with fast queue
                        try:
                            file_path = fast_queue.get_nowait()
                            await process_one(file_path)
                        except asyncio.QueueEmpty:
                            return  # Both queues empty

            # Start both worker pools in parallel
            all_workers = (
                [fast_worker() for _ in range(fast_worker_count)] +
                [slow_worker() for _ in range(slow_worker_count)]
            )
            await asyncio.gather(*all_workers)

            if _common._shutdown_requested:
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

        # Write run summary
        _write_run_summary(resolved_output, run_start, run_start_time, len(files), processed_count, stats, _service_counter)

        # Finalize report with stats and cleanup old reports
        report_writer.update_stats(len(files), processed_count)
        report_writer.cleanup_old_reports()
        vprint(f"\nReport: {report_writer.get_path()}")

    finally:
        try:
            tracker.compact()
        except Exception:
            pass

        # Merge near-duplicate topics before flushing
        console.print("[dim]Merging similar topics…[/dim]")
        try:
            merges = db.merge_similar_topics(threshold=0.78)
            if merges:
                vprint(f"\nMerged {len(merges)} similar topics:")
                for absorbed, kept, sim in merges:
                    vprint(f"  {absorbed} → {kept} ({sim})")
        except Exception as e:
            console.print(f"[yellow]Topic merge failed (non-fatal): {e}[/yellow]")

        console.print("[dim]Flushing documents/chunks/topics to disk…[/dim]")
        db.flush()
        db.write_manifest()
        console.print(f"[green]Manifest written to {resolved_output / 'manifest.json'}[/green]")
        await db.close()
