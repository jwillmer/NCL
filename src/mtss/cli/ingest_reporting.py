"""Ingest reporting and cost estimation display logic.

Extracted from ingest_cmd.py — pure display/reporting with no side effects
on ingest state.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ._common import console


def write_run_summary(
    output_dir: Path,
    run_start: float,
    run_start_time: str,
    files_attempted: int,
    processed_count: int,
    stats: dict,
    service_counter=None,
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
    service_data = service_counter.to_dict() if service_counter else None
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
    if service_data:
        summary["services"].update(service_data)

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
    duration_str = f"{mins}m {secs}s"
    if processed_count > 0:
        avg = elapsed / processed_count
        duration_str += f" (~{avg:.0f}s per mail)"
    table.add_row("Duration", duration_str, "")
    table.add_row("Files processed", str(processed_count), str(stats.get("completed", 0)))
    table.add_row("Failed", str(stats.get("failed", 0)), "")
    table.add_row("Documents", f"+{processed_count}", str(doc_count))
    table.add_row("Chunks", "", str(chunk_count))
    table.add_row("Topics", "", str(topic_count))

    # Service call breakdown
    svc = service_data.get("service_calls", {}) if service_data else {}
    if svc:
        table.add_section()
        # Display order: local first, then paid services
        display_order = [
            ("local_parse", "Parsed locally (free)"),
            ("llamaparse", "LlamaParse"),
            ("embedding", "Embeddings"),
            ("vision", "Vision API"),
            ("llm_context", "LLM context gen"),
            ("llm_topics", "LLM topic extraction"),
            ("llm_digest", "LLM thread digest"),
            ("llm_cleaner", "LLM email cleaner"),
        ]
        for key, label in display_order:
            count = svc.get(key, 0)
            if count:
                table.add_row(label, str(count), "")
        # Show any unlisted services
        shown = {k for k, _ in display_order}
        for key, count in sorted(svc.items()):
            if key not in shown and count:
                table.add_row(key, str(count), "")
    else:
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


def show_estimate(
    result,
    page_cost: float,
    gemini_page_cost: float,
    vision_cost: float,
    text_cost: float,
    embedding_cost: float,
    verbose: bool,
):
    """Display estimate results as Rich tables."""
    from rich.table import Table

    from ..ingest.estimator import (
        GEMINI_PARSE_CATEGORIES,
        LEGACY_OFFICE_CATEGORIES,
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
        elif cat_name in LEGACY_OFFICE_CATEGORIES:
            method = "LlamaParse"
        elif cat_name == "PDF":
            sp = stats.pdf_simple_pages
            cp = stats.pdf_complex_pages
            if sp > 0 and cp > 0:
                method = f"local:{sp}pp / Gemini:{cp}pp"
            elif sp > 0:
                method = "local (free)"
            elif cp > 0:
                method = "Gemini"
            else:
                method = "Gemini*"
        elif cat_name in GEMINI_PARSE_CATEGORIES:
            method = "Gemini"
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

    # Compute local vs Gemini vs LlamaParse page split
    local_pages = 0
    gemini_pages = 0
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
                gemini_pages += stats.pdf_complex_pages
            else:
                gemini_pages += stats.page_count  # legacy cache, assume complex → Gemini
        elif cat_name in LEGACY_OFFICE_CATEGORIES:
            llama_pages += stats.page_count
        else:
            # PPTX + Other Docs route to Gemini
            gemini_pages += stats.page_count

    # Local parsing (free)
    if local_pages > 0:
        cost_table.add_row(
            "Local parsing (free)",
            f"{local_pages} pages",
            "$0.00000",
            "$0.00",
        )

    # Gemini (complex PDFs + modern formats Gemini handles)
    gemini_total = gemini_pages * gemini_page_cost
    if gemini_pages > 0:
        cost_table.add_row(
            "Gemini 2.5 Flash (complex docs)",
            f"{gemini_pages} pages",
            f"${gemini_page_cost:.5f}",
            f"${gemini_total:.2f}",
        )

    # LlamaParse (legacy binary Office only)
    llama_total = llama_pages * page_cost
    if llama_pages > 0:
        cost_table.add_row(
            "LlamaParse (legacy Office)",
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
    grand_total = gemini_total + llama_total + vision_total + text_total + embed_total
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
        "  [dim]Gemini: complex/scanned PDFs + PPTX + other docs via OpenRouter "
        "(~$0.0025/page)[/dim]"
    )
    console.print(
        "  [dim]LlamaParse: legacy binary .doc/.xls/.ppt only -- "
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
