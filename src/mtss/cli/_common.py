"""Shared helpers for CLI command modules."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from ..ingest.helpers import IssueTracker

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


def _show_stats(stats: dict):
    """Display processing statistics."""
    table = Table(title="Processing Statistics")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for status, count in stats.items():
        table.add_row(status.capitalize(), str(count))

    console.print(table)


def make_progress() -> Progress:
    """Create a standard progress bar with spinner, text, bar, and percentage."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
