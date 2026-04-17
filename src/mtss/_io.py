"""Crash-safe I/O helpers for canonical JSONL output.

These helpers are shared by local storage and progress-tracking code that
must not corrupt on crash mid-write. A crashed flush of ``documents.jsonl``
or ``chunks.jsonl`` destroys data that costs hours and real money to
regenerate, so writes to canonical files MUST use ``atomic_write_text``.
Append-only logs (e.g. ``processing_log.jsonl``) use ``fsync_append_line``
to ensure each record is durable before we move on.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to ``{path}.tmp`` in the same directory, flushes and fsyncs the
    file descriptor, then atomically replaces the target via ``os.replace``.
    On any exception before the replace, the tmp file is removed so it does
    not accumulate. The original target at ``path`` is untouched until the
    replace succeeds.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            # Best-effort cleanup; re-raise the original error below.
            pass
        raise


def fsync_append_line(path: Path, line: str, *, encoding: str = "utf-8") -> None:
    """Append ``line`` (plus a trailing newline) to ``path`` durably.

    Opens ``path`` in append mode, writes ``line + "\\n"``, flushes, and
    fsyncs the file descriptor before closing. Mirrors the pattern used in
    ``LocalProgressTracker._save_entry``.
    """
    with open(path, "a", encoding=encoding) as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
