"""Tests for concurrent archive upload behaviour in ``_import_archives``.

Fix #4 from the code-health report converted the archive-upload phase of
``mtss import`` from a serial ``for`` loop into a bounded ``asyncio.gather``
fan-out. These tests cover three properties:

  * the ``asyncio.Semaphore`` actually caps in-flight uploads;
  * every file is eventually attempted and counted;
  * both soft failures (``_upload_with_retry`` returns ``False``) and hard
    failures (it raises) are counted as ``changes["failed"]`` without
    bubbling out of ``gather`` (which uses ``return_exceptions=True``).

All tests mock ``ArchiveStorage`` and patch ``_upload_with_retry`` so they
never touch Supabase. ``asyncio.to_thread`` still dispatches the real
(patched) callable onto a worker thread, which is what we want to assert
about concurrency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mtss.cli import import_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_archive_tree(root: Path, folder_id: str, count: int) -> list[Path]:
    """Populate ``root/<folder_id>/`` with ``count`` tiny files on disk.

    ``_import_archives`` walks the filesystem itself, so the files have to
    actually exist — mocking ``Path.rglob`` would bypass the security
    checks the function performs on each path.
    """
    folder = root / folder_id
    folder.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for i in range(count):
        p = folder / f"file_{i:03d}.txt"
        p.write_bytes(f"content-{i}".encode("utf-8"))
        files.append(p)
    return files


def _make_mock_archive_storage() -> MagicMock:
    """Mock of ``ArchiveStorage`` that pretends the bucket is empty."""
    storage = MagicMock()
    storage.list_folder.return_value = []  # no existing remote files / folders
    storage.bucket = MagicMock()
    storage.bucket.remove = MagicMock()
    storage.delete_folder = MagicMock()
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportArchivesConcurrency:
    """Concurrency + counting behaviour for ``_import_archives``."""

    async def test_parallel_uploads_respect_semaphore_limit(
        self, tmp_path, monkeypatch
    ):
        """With the concurrency cap at 3, no more than 3 uploads are in
        flight at once even when 20 files are queued."""
        folder_id = "a" * 16
        _make_archive_tree(tmp_path, folder_id, count=20)

        # Drop the cap; the test gets to drive how many concurrent uploads
        # are permitted without rebuilding the loop.
        monkeypatch.setattr(import_cmd, "_ARCHIVE_UPLOAD_CONCURRENCY", 3)

        # Track live concurrency. We use an asyncio.Lock to safely update
        # the counter from within the upload worker threads that bounce
        # results back to the event loop through asyncio.to_thread.
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()
        release = asyncio.Event()
        observed = asyncio.Event()

        # Capture the running loop so the worker threads can schedule
        # coroutines back onto it.
        loop = asyncio.get_running_loop()

        def fake_upload(storage, rel_key, payload, content_type):
            nonlocal in_flight, peak

            async def _enter():
                nonlocal in_flight, peak
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                    if in_flight >= 3:
                        observed.set()

            async def _exit():
                nonlocal in_flight
                async with lock:
                    in_flight -= 1

            # Enter the "upload" critical section on the loop.
            asyncio.run_coroutine_threadsafe(_enter(), loop).result()
            # Block this worker thread until the test has observed that
            # the semaphore allowed exactly as many uploads as configured.
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            asyncio.run_coroutine_threadsafe(_exit(), loop).result()
            return True

        async def release_once_observed():
            await observed.wait()
            # Give the loop one more tick so any stragglers that raced
            # past the semaphore would be caught by ``peak``.
            await asyncio.sleep(0.05)
            release.set()

        monkeypatch.setattr(import_cmd, "_upload_with_retry", fake_upload)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            totals: dict = {}
            changes: dict = {"new_archive files": 0, "failed": 0, "orphans_removed": 0}
            releaser = asyncio.create_task(release_once_observed())
            await import_cmd._import_archives(
                tmp_path, {folder_id}, totals, changes, dry_run=False, verbose=False
            )
            await releaser

        assert peak <= 3, f"semaphore breach: saw {peak} concurrent uploads"
        assert peak == 3, f"expected to saturate cap of 3, saw {peak}"
        assert changes["new_archive files"] == 20
        assert changes["failed"] == 0

    async def test_all_uploads_execute(self, tmp_path, monkeypatch):
        """Every queued file is attempted and counted as success when the
        upload mock returns True."""
        folder_id = "b" * 16
        _make_archive_tree(tmp_path, folder_id, count=10)

        call_count = 0

        def fake_upload(storage, rel_key, payload, content_type):
            nonlocal call_count
            call_count += 1
            return True

        monkeypatch.setattr(import_cmd, "_upload_with_retry", fake_upload)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            totals: dict = {}
            changes: dict = {"new_archive files": 0, "failed": 0, "orphans_removed": 0}
            await import_cmd._import_archives(
                tmp_path, {folder_id}, totals, changes, dry_run=False, verbose=False
            )

        assert call_count == 10
        assert changes["new_archive files"] == 10
        assert changes["failed"] == 0

    async def test_upload_failure_counted_as_failed(self, tmp_path, monkeypatch):
        """A soft failure (``_upload_with_retry`` returning False) for one
        of five files produces exactly one ``failed`` increment and four
        ``new_archive files``."""
        folder_id = "c" * 16
        _make_archive_tree(tmp_path, folder_id, count=5)

        call_index = 0
        call_lock = __import__("threading").Lock()

        def fake_upload(storage, rel_key, payload, content_type):
            nonlocal call_index
            with call_lock:
                my_index = call_index
                call_index += 1
            # The 3rd invocation (index 2) fails; order isn't guaranteed
            # because of concurrency, but exactly one False is returned.
            return my_index != 2

        monkeypatch.setattr(import_cmd, "_upload_with_retry", fake_upload)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            totals: dict = {}
            changes: dict = {"new_archive files": 0, "failed": 0, "orphans_removed": 0}
            await import_cmd._import_archives(
                tmp_path, {folder_id}, totals, changes, dry_run=False, verbose=False
            )

        assert changes["new_archive files"] == 4
        assert changes["failed"] == 1

    async def test_upload_exception_counted_as_failed(self, tmp_path, monkeypatch):
        """If ``_upload_with_retry`` raises, ``gather(..., return_exceptions=
        True)`` captures the exception and we count it as ``failed`` without
        propagating."""
        folder_id = "d" * 16
        _make_archive_tree(tmp_path, folder_id, count=4)

        call_index = 0
        call_lock = __import__("threading").Lock()

        def fake_upload(storage, rel_key, payload, content_type):
            nonlocal call_index
            with call_lock:
                my_index = call_index
                call_index += 1
            if my_index == 1:
                raise RuntimeError("transient gateway boom")
            return True

        monkeypatch.setattr(import_cmd, "_upload_with_retry", fake_upload)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            totals: dict = {}
            changes: dict = {"new_archive files": 0, "failed": 0, "orphans_removed": 0}
            # Must not raise — the whole point of return_exceptions=True.
            await import_cmd._import_archives(
                tmp_path, {folder_id}, totals, changes, dry_run=False, verbose=False
            )

        assert changes["new_archive files"] == 3
        assert changes["failed"] == 1

    async def test_semaphore_constant_is_module_level(self):
        """Smoke-check that the tunable constant lives where operators and
        tests expect (and has a sane default)."""
        assert hasattr(import_cmd, "_ARCHIVE_UPLOAD_CONCURRENCY")
        assert isinstance(import_cmd._ARCHIVE_UPLOAD_CONCURRENCY, int)
        assert import_cmd._ARCHIVE_UPLOAD_CONCURRENCY >= 1
