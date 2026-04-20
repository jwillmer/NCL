"""Backup CLI command: snapshot ingest.db + archive/ into data/backup/<ts>/."""

from __future__ import annotations

import shutil
import sqlite3
import time
import zipfile
from datetime import datetime
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


DB_FILENAME = "ingest.db"
ARCHIVE_DIRNAME = "archive"
ARCHIVE_ZIP_NAME = "archive.zip"

# Nextcloud-synced share is the preferred default destination so snapshots replicate
# off-machine automatically. Local data/backup/ is the fallback only.
PREFERRED_DEST = Path(
    r"C:\Users\mail\Nextcloud2\DNV\Client projects\Marantankers\AI Bot\Backups"
)


def register(app: typer.Typer):
    """Register the backup command on the app."""

    @app.command()
    def backup(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Source directory containing ingest.db + archive/ (default: data/output)",
        ),
        dest: Optional[Path] = typer.Option(
            None,
            "--dest",
            "-d",
            help="Backup root; snapshot goes in <dest>/<timestamp>/ (default: data/backup)",
        ),
        compression_level: int = typer.Option(
            1,
            "--compression-level",
            "-c",
            min=0,
            max=9,
            help="Zip compression level 0-9 (0=store, 1=fast, 9=max). Default 1.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            "-n",
            help="Show what would be backed up without writing anything",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Verbose output (per-file zip entries)",
        ),
    ):
        """Snapshot the SQLite DB + archive folder.

        Creates ``<dest>/<YYYYMMDD-HHMMSS>/`` containing:
          - ``ingest.db``     — VACUUM INTO copy (safe during concurrent reads)
          - ``archive.zip``   — zipped archive/ folder (deflate, level 1 by default)

        The DB alone is useless without archive/ (chunk re-derivation, URI targets),
        so both are captured atomically-per-file.
        """
        _common._verbose = verbose
        _run_backup(output_dir, dest, compression_level, dry_run)


def _resolve_paths(output_dir: Optional[Path], dest: Optional[Path]) -> tuple[Path, Path]:
    """Resolve source + backup root using the same convention as other commands.

    When ``dest`` is not provided, prefer ``PREFERRED_DEST`` (Nextcloud) if it exists
    so backups replicate off-machine. If it doesn't, prompt the user for a path
    rather than silently writing to a local fallback.
    """
    from ..config import get_settings

    settings = get_settings()
    resolved_output = (output_dir or settings.eml_source_dir.parent / "output").resolve()

    if dest is not None:
        resolved_dest = dest.resolve()
    elif PREFERRED_DEST.exists():
        resolved_dest = PREFERRED_DEST.resolve()
        console.print(f"[dim]Using preferred backup location: {resolved_dest}[/dim]")
    else:
        console.print(
            f"[yellow]Preferred backup location not found: {PREFERRED_DEST}[/yellow]"
        )
        local_fallback = settings.eml_source_dir.parent / "backup"
        answer = typer.prompt(
            "Enter backup destination path",
            default=str(local_fallback),
        )
        resolved_dest = Path(answer).expanduser().resolve()

    return resolved_output, resolved_dest


def _human_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _run_backup(
    output_dir: Optional[Path],
    dest: Optional[Path],
    compression_level: int,
    dry_run: bool,
) -> None:
    src, backup_root = _resolve_paths(output_dir, dest)
    db_path = src / DB_FILENAME
    archive_dir = src / ARCHIVE_DIRNAME

    if not db_path.exists():
        console.print(f"[red]ingest.db not found at {db_path}[/red]")
        raise typer.Exit(1)

    has_archive = archive_dir.is_dir()
    if not has_archive:
        console.print(
            f"[yellow]Warning: archive dir missing at {archive_dir}. "
            "DB will still be backed up, but the snapshot will be useless without archive content.[/yellow]"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_dir = backup_root / timestamp
    db_dest = snapshot_dir / DB_FILENAME
    zip_dest = snapshot_dir / ARCHIVE_ZIP_NAME

    console.print(f"[cyan]Source:[/cyan]   {src}")
    console.print(f"[cyan]Snapshot:[/cyan] {snapshot_dir}")

    if dry_run:
        db_size = db_path.stat().st_size
        console.print(f"[dim]  would VACUUM INTO → {db_dest} (source: {_human_bytes(db_size)})[/dim]")
        if has_archive:
            files, total = _scan_archive(archive_dir)
            console.print(
                f"[dim]  would zip {files} files ({_human_bytes(total)}) → {zip_dest}[/dim]"
            )
        console.print("[yellow]Dry-run: nothing written.[/yellow]")
        return

    if snapshot_dir.exists():
        console.print(f"[red]Refusing to overwrite existing snapshot dir: {snapshot_dir}[/red]")
        raise typer.Exit(1)

    snapshot_dir.mkdir(parents=True, exist_ok=False)

    db_bytes, db_seconds = _backup_database(db_path, db_dest)
    zip_bytes, zip_files, zip_seconds = (0, 0, 0.0)
    if has_archive:
        zip_bytes, zip_files, zip_seconds = _zip_archive(archive_dir, zip_dest, compression_level)

    _print_summary(
        snapshot_dir=snapshot_dir,
        db_bytes=db_bytes,
        db_seconds=db_seconds,
        zip_files=zip_files,
        zip_bytes=zip_bytes,
        zip_seconds=zip_seconds,
        has_archive=has_archive,
    )


def _backup_database(src_db: Path, dest_db: Path) -> tuple[int, float]:
    """Run ``VACUUM INTO`` — online, consistent, produces a compact copy."""
    start = time.monotonic()
    conn = sqlite3.connect(src_db)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        # VACUUM INTO is atomic and concurrency-safe for readers; brief writer lock.
        # Parameter binding doesn't work for file paths in VACUUM INTO, so escape quotes manually.
        escaped = str(dest_db).replace("'", "''")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"VACUUM INTO {dest_db.name}", total=None)
            conn.execute(f"VACUUM INTO '{escaped}'")
    finally:
        conn.close()
    elapsed = time.monotonic() - start
    size = dest_db.stat().st_size
    console.print(
        f"[green]\u2713[/green] DB backed up: {_human_bytes(size)} in {elapsed:.1f}s"
    )
    return size, elapsed


def _scan_archive(archive_dir: Path) -> tuple[int, int]:
    """Count files + total bytes under archive_dir (for dry-run + progress totals)."""
    files = 0
    total = 0
    for p in archive_dir.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return files, total


def _zip_archive(
    archive_dir: Path, zip_dest: Path, compression_level: int
) -> tuple[int, int, float]:
    """Zip archive_dir → zip_dest. Returns (bytes_written, file_count, seconds)."""
    files, total_bytes = _scan_archive(archive_dir)
    if files == 0:
        console.print("[yellow]archive/ is empty — writing empty zip[/yellow]")

    compression = zipfile.ZIP_STORED if compression_level == 0 else zipfile.ZIP_DEFLATED
    start = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Zipping {files} files", total=max(total_bytes, 1))
        with zipfile.ZipFile(
            zip_dest,
            mode="w",
            compression=compression,
            compresslevel=compression_level if compression == zipfile.ZIP_DEFLATED else None,
            allowZip64=True,
        ) as zf:
            for path in archive_dir.rglob("*"):
                if not path.is_file():
                    continue
                arcname = path.relative_to(archive_dir).as_posix()
                vprint(f"  + {arcname}")
                zf.write(path, arcname=arcname)
                progress.update(task, advance=path.stat().st_size)

    elapsed = time.monotonic() - start
    out_size = zip_dest.stat().st_size
    ratio = (out_size / total_bytes) if total_bytes else 1.0
    console.print(
        f"[green]\u2713[/green] archive zipped: {files} files, "
        f"{_human_bytes(total_bytes)} \u2192 {_human_bytes(out_size)} "
        f"({ratio:.0%}) in {elapsed:.1f}s"
    )
    return out_size, files, elapsed


def _print_summary(
    *,
    snapshot_dir: Path,
    db_bytes: int,
    db_seconds: float,
    zip_files: int,
    zip_bytes: int,
    zip_seconds: float,
    has_archive: bool,
) -> None:
    table = Table(title="Backup Summary")
    table.add_column("Artifact", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Seconds", justify="right", style="dim")
    table.add_row(DB_FILENAME, _human_bytes(db_bytes), f"{db_seconds:.1f}")
    if has_archive:
        table.add_row(
            f"{ARCHIVE_ZIP_NAME} ({zip_files} files)",
            _human_bytes(zip_bytes),
            f"{zip_seconds:.1f}",
        )
    else:
        table.add_row(ARCHIVE_ZIP_NAME, "(skipped — archive/ missing)", "-")
    console.print(table)
    console.print(f"[green]Snapshot written to: {snapshot_dir}[/green]")
