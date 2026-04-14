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
from ._common import (
    STALE_PROCESSING_THRESHOLD_MINUTES,
    _issue_tracker,
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
        local_only: bool = typer.Option(
            False,
            "--local-only",
            help="Write to local JSONL files instead of Supabase",
        ),
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Output directory for local-only mode (default: <source>/../output)",
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
                source_dir, batch_size, resume, retry_failed,
                reprocess_outdated, lenient, local_only, output_dir, limit,
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
    batch_size: int,
    resume: bool,
    retry_failed: bool,
    reprocess_outdated: bool = False,
    lenient: bool = False,
    local_only: bool = False,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
):
    """Async implementation of ingest command."""
    from ..config import get_settings
    from ..ingest.lane_classifier import LaneClassifier
    from ..ingest.pipeline import process_email
    from ..ingest.version_manager import VersionManager
    from ..storage.failure_report import IngestReportWriter

    settings = get_settings()
    source_dir = source_dir or settings.eml_source_dir

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    # Initialize components based on mode
    if local_only:
        from ..ingest.components import create_local_ingest_components
        from ..storage.local_client import LocalStorageClient
        from ..storage.local_progress_tracker import LocalProgressTracker

        resolved_output = (output_dir or source_dir.parent / "output").resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Local-only mode: output -> {resolved_output}[/green]")

        db = LocalStorageClient(output_dir=resolved_output)
        tracker = LocalProgressTracker(output_dir=resolved_output)
        unsupported_logger = db  # LocalStorageClient has log_unsupported_file()
        version_manager = VersionManager(db)
        vessels = await db.get_all_vessels()

        components = create_local_ingest_components(
            db=db,
            output_dir=resolved_output,
            source_dir=source_dir,
            vessels=vessels,
            enable_topics=True,
        )
    else:
        from ..ingest.components import create_ingest_components
        from ..storage.progress_tracker import ProgressTracker
        from ..storage.supabase_client import SupabaseClient
        from ..storage.unsupported_file_logger import UnsupportedFileLogger

        db = SupabaseClient()
        tracker = ProgressTracker(db)
        unsupported_logger = UnsupportedFileLogger(db)
        version_manager = VersionManager(db)
        vessels = await db.get_all_vessels()

        components = create_ingest_components(
            db=db,
            source_dir=source_dir,
            vessels=vessels,
            enable_topics=True,
        )

    if components.vessel_matcher:
        vprint(f"Loaded {components.vessel_matcher.vessel_count} vessels ({components.vessel_matcher.name_count} names)")
    else:
        vprint("No vessels in registry - vessel tagging disabled")
    vprint("Topic extraction enabled")

    # Keep reference for lane classifier (needs eml_parser)
    eml_parser = components.eml_parser

    # Initialize continuous report writer
    report_writer = IngestReportWriter(source_dir=str(source_dir))
    console.print(f"[dim]Report: {report_writer.get_path()}[/dim]")

    run_start = time.monotonic()
    run_start_time = datetime.now(timezone.utc).isoformat()

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

        if limit and len(files) > limit:
            files = files[:limit]
            console.print(f"[yellow]Limited to {limit} files[/yellow]")

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
                            force_reparse=reprocess_outdated,
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
                """Worker for fast queue. Helps slow queue when fast is empty."""
                while True:
                    if _common._shutdown_requested:
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
                    if _common._shutdown_requested:
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

        # Write run summary (local-only mode)
        if local_only:
            _write_run_summary(resolved_output, run_start, run_start_time, len(files), processed_count, stats)

        # Finalize report with stats and cleanup old reports
        report_writer.update_stats(len(files), processed_count)
        report_writer.cleanup_old_reports()
        console.print(f"\nReport: {report_writer.get_path()}")

    finally:
        # Flush and write manifest for local-only mode
        if local_only:
            db.flush()
            db.write_manifest()
            console.print(f"[green]Manifest written to {resolved_output / 'manifest.json'}[/green]")
        await db.close()


def _write_run_summary(
    output_dir: Path,
    run_start: float,
    run_start_time: str,
    files_attempted: int,
    processed_count: int,
    stats: dict,
):
    """Write run summary to run_history.jsonl and print to console."""
    from rich.table import Table

    elapsed = time.monotonic() - run_start
    now = datetime.now(timezone.utc).isoformat()

    # Count documents created during THIS run (by timestamp)
    run_docs: dict[str, int] = {}  # doc_type -> count
    total_docs: dict[str, int] = {}
    docs_path = output_dir / "documents.jsonl"
    if docs_path.exists():
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    dt = d.get("document_type", "unknown")
                    total_docs[dt] = total_docs.get(dt, 0) + 1
                except json.JSONDecodeError:
                    pass

    # Count chunks and topics (total in output dir)
    chunk_count = topic_count = 0
    for name, var_name in [("chunks.jsonl", "chunk"), ("topics.jsonl", "topic")]:
        path = output_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            if var_name == "chunk":
                chunk_count = count
            else:
                topic_count = count

    # Count events by reason
    events_by_reason: dict[str, int] = {}
    events_path = output_dir / "ingest_events.jsonl"
    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    reason = e.get("reason", "unknown")
                    events_by_reason[reason] = events_by_reason.get(reason, 0) + 1
                except json.JSONDecodeError:
                    pass

    doc_count = sum(total_docs.values())
    vision_images = total_docs.get("attachment_image", 0)

    # Build summary
    summary = {
        "timestamp": now,
        "elapsed_seconds": round(elapsed, 1),
        "files_attempted": files_attempted,
        "files_processed": processed_count,
        "files_failed": stats.get("failed", 0),
        "cumulative": {
            "documents": doc_count,
            "chunks": chunk_count,
            "topics": topic_count,
            "doc_types": total_docs,
        },
        "services": {
            "vision_images": vision_images,
            "skipped_events": events_by_reason,
        },
    }

    # Append to run_history.jsonl
    history_path = output_dir / "run_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    # Print concise summary table
    table = Table(title="Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("This Run", justify="right", style="green")
    table.add_column("Cumulative", justify="right", style="dim")

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    table.add_row("Duration", f"{mins}m {secs}s", "")
    table.add_row("Files processed", str(processed_count), str(stats.get("completed", 0)))
    table.add_row("Failed", str(stats.get("failed", 0)), "")
    table.add_row("Documents", f"+{processed_count}", str(doc_count))
    table.add_row("Chunks", "", str(chunk_count))
    table.add_row("Topics", "", str(topic_count))
    table.add_section()
    table.add_row("Vision API (images)", "", str(vision_images))
    for dt, count in sorted(total_docs.items()):
        if dt != "email" and dt != "attachment_image":
            table.add_row(f"  {dt}", "", str(count))
    if events_by_reason:
        skipped = sum(events_by_reason.values())
        table.add_row("Skipped (non-content)", "", str(skipped))

    console.print(table)
    console.print(f"[dim]Run history: {history_path}[/dim]")


def _show_estimate(
    result,
    page_cost: float,
    vision_cost: float,
    text_cost: float,
    embedding_cost: float,
    verbose: bool,
):
    """Display estimate results as Rich tables."""
    from rich.table import Table

    from ..ingest.estimator import (
        LLAMAPARSE_ONLY_CATEGORIES,
        LOCAL_PARSE_CATEGORIES,
        PAGE_COUNT_CATEGORIES,
        TEXT_CATEGORIES,
        VISION_CATEGORIES,
    )

    # -- Table 1: File Inventory --
    inv_table = Table(title="Ingest File Inventory")
    inv_table.add_column("Category", style="cyan")
    inv_table.add_column("Files", justify="right", style="green")
    inv_table.add_column("Pages", justify="right")
    inv_table.add_column("Unknown", justify="right", style="yellow")
    inv_table.add_column("Parse Method", style="dim")

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
            pages_str = "\u2014"
            unknown_str = "\u2014"

        # Determine parse method for this category
        if cat_name in LOCAL_PARSE_CATEGORIES:
            method = "local (free)"
        elif cat_name in LLAMAPARSE_ONLY_CATEGORIES:
            method = "LlamaParse"
        elif cat_name == "PDF":
            sp = stats.pdf_simple_pages
            cp = stats.pdf_complex_pages
            if sp > 0 and cp > 0:
                method = f"local:{sp}pp / LP:{cp}pp"
            elif sp > 0:
                method = "local (free)"
            elif cp > 0:
                method = "LlamaParse"
            else:
                method = "LlamaParse*"
        elif cat_name in VISION_CATEGORIES:
            method = "Vision API"
        elif cat_name in TEXT_CATEGORIES:
            method = "local (free)"
        else:
            method = "--"

        inv_table.add_row(cat_name, str(stats.file_count), pages_str, unknown_str, method)

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
        "",
    )

    console.print(inv_table)
    console.print()

    # -- Table 2: Cost Estimate --
    cost_table = Table(title="Estimated Ingest Cost")
    cost_table.add_column("Service", style="cyan")
    cost_table.add_column("Units", justify="right")
    cost_table.add_column("Unit Cost", justify="right")
    cost_table.add_column("Cost", justify="right", style="green")

    # Compute local vs LlamaParse page split
    local_pages = 0
    llama_pages = 0
    for cat_name in PAGE_COUNT_CATEGORIES:
        stats = result.categories.get(cat_name)
        if not stats:
            continue
        if cat_name in LOCAL_PARSE_CATEGORIES:
            local_pages += stats.page_count
        elif cat_name == "PDF":
            if stats.pdf_simple_pages or stats.pdf_complex_pages:
                local_pages += stats.pdf_simple_pages
                llama_pages += stats.pdf_complex_pages
            else:
                llama_pages += stats.page_count  # legacy cache
        else:
            llama_pages += stats.page_count

    # Local parsing (free)
    if local_pages > 0:
        cost_table.add_row(
            "Local parsing (free)",
            f"{local_pages} pages",
            "$0.00000",
            "$0.00",
        )

    # LlamaParse (complex/legacy docs)
    llama_total = llama_pages * page_cost
    cost_table.add_row(
        "LlamaParse (complex docs)",
        f"{llama_pages} pages",
        f"${page_cost:.5f}",
        f"${llama_total:.2f}",
    )

    # Vision API (images) -- only meaningful images get described
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

    # -- Footnotes --
    console.print()
    console.print(
        "  [dim]Local: simple PDFs (PyMuPDF), DOCX (python-docx), XLSX (openpyxl) -- no API cost[/dim]"
    )
    console.print(
        "  [dim]LlamaParse: complex/scanned PDFs, PPTX, legacy Office -- "
        "5 avg credits/page x $0.00125/credit[/dim]"
    )
    console.print(
        "  [dim]Embeddings: ~1.5 chunks/page x text-embedding-3-small ($0.02/M tokens)[/dim]"
    )
    if image_skipped > 0:
        console.print(
            f"  [dim]Images: {image_skipped} likely non-content (logos, icons, banners) "
            f"excluded by size/dimension heuristic[/dim]"
        )

    if total_unknown > 0:
        console.print(
            f"  [dim]{total_unknown} files had unknown page counts (counted as 1 page) "
            f"\u2014 use --verbose to see details[/dim]"
        )

    if result.scan_errors:
        console.print(
            f"  [yellow]{len(result.scan_errors)} EML files failed to process[/yellow]"
        )

    # -- Footer --
    console.print()
    console.print(
        f"[dim]Scanned {result.eml_count} EML files "
        f"({result.cached_count} cached, {result.extracted_count} newly extracted) "
        f"in {result.elapsed_seconds:.1f}s[/dim]"
    )

    # -- Verbose output --
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
                console.print(f"  [red]\u2022 {err}[/red]")
