"""Crash-safe IO helpers + shared low-level utilities.

These helpers are shared by local storage, progress-tracking, and retry code
that must not corrupt on crash mid-write or give up silently on transient
failures. A crashed flush of ``documents.jsonl`` or ``chunks.jsonl`` destroys
data that costs hours and real money to regenerate, so writes to canonical
files MUST use ``atomic_write_text``. Append-only logs (e.g.
``processing_log.jsonl``) use ``fsync_append_line`` to ensure each record is
durable before we move on. ``retry_with_backoff`` is the single place
exponential-backoff retry logic lives — archive listings, uploads, and
anything else with transient-failure exposure should route through it.

Kept dependency-free (standard library only) so it can be imported from
anywhere in the package without cycles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


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


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    retriable: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn()`` with exponential-backoff retry on ``retriable`` exceptions.

    Delays between attempts are ``backoff_base * 2**0, 2**1, ...`` seconds.
    The last exception is re-raised if all attempts fail. If a non-``retriable``
    exception is raised it propagates immediately without retry.

    Args:
        fn: Zero-argument callable to invoke. Callers should close over any
            needed state with a lambda or nested function.
        max_attempts: Total number of attempts (including the initial call).
            Must be >= 1.
        backoff_base: Base delay in seconds. Actual delay for attempt ``i``
            (0-indexed, before the next try) is ``backoff_base * 2**i``.
        retriable: Tuple of exception types that trigger a retry. Defaults to
            ``(Exception,)`` for the broad-catch behavior legacy call sites
            relied on. Pass a narrower tuple to let unexpected errors surface.
        on_retry: Optional callback invoked as ``on_retry(attempt, exc, delay)``
            before each sleep, where ``attempt`` is 1-indexed and ``delay`` is
            the pending sleep duration. Useful for logging hooks.
        sleep: Injected sleep function; defaults to ``time.sleep``. Tests pass
            a mock to avoid real waits without patching ``time`` globally.

    Returns:
        The value returned by ``fn()`` on the first successful call.

    Raises:
        The last exception raised by ``fn()`` if all attempts fail, or any
        non-retriable exception raised by ``fn()``.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retriable as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = backoff_base * (2 ** attempt)
                if on_retry is not None:
                    on_retry(attempt + 1, exc, delay)
                sleep(delay)
    # Exhausted — re-raise the last captured exception.
    assert last_exc is not None  # loop ran at least once
    raise last_exc
