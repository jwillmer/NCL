"""Tests for the archive-upload phase of ``mtss import``.

The upload fan-out used to dispatch sync Supabase calls via
``asyncio.to_thread``; on Windows that surfaced ``WSAEWOULDBLOCK``
errors under concurrent large-body POSTs. The current implementation
drives everything through ``storage.async_upload.upload_many`` (one
shared ``httpx.AsyncClient`` with a tuned ``Limits`` pool + IOCP
backpressure). These tests cover:

  * ``_import_archives`` delegates the upload phase to ``upload_many``
    with the configured concurrency / retry budget;
  * ``on_progress`` successes feed ``changes["new_archive files"]``
    and failures feed ``changes["failed"]``;
  * the module-level tunables exist with sane defaults so operators
    can tweak them at runtime.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mtss.cli import import_cmd
from mtss.storage.async_upload import UploadItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_archive_tree(root: Path, folder_id: str, count: int) -> list[Path]:
    """Populate ``root/<folder_id>/`` with ``count`` tiny files on disk.

    ``_import_archives`` walks the filesystem itself, so the files have
    to actually exist — mocking ``Path.rglob`` would bypass the security
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
    """Mock ``ArchiveStorage`` that pretends the bucket is empty."""
    storage = MagicMock()
    storage.list_folder.return_value = []  # no existing remote files / folders
    storage.bucket = MagicMock()
    storage.bucket.remove = MagicMock()
    storage.delete_folder = MagicMock()
    storage.bucket_name = "test-archive-bucket"
    return storage


def _install_fake_upload_many(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcomes: list[bool] | None = None,
    raise_for_keys: set[str] | None = None,
):
    """Replace ``import_cmd.upload_many`` with a controllable fake.

    The fake honours ``on_progress`` the same way the real implementation
    does (one call per item, with the boolean outcome) so the caller's
    counters get the right totals.
    """
    calls: list[dict] = []

    async def fake_upload_many(
        items,
        *,
        bucket_name,
        max_concurrency,
        max_attempts,
        backoff_base,
        jitter,
        on_retry=None,
        on_progress=None,
    ):
        calls.append(
            {
                "items": list(items),
                "bucket_name": bucket_name,
                "max_concurrency": max_concurrency,
                "max_attempts": max_attempts,
                "backoff_base": backoff_base,
                "jitter": jitter,
            }
        )
        results: list[bool] = []
        for idx, item in enumerate(items):
            if raise_for_keys and item.remote_key in raise_for_keys:
                # A genuine exception bubbling out of upload_many would
                # kill the entire import. The real helper catches per-
                # item failures and returns False — simulate that here.
                success = False
            elif outcomes is not None:
                success = outcomes[idx]
            else:
                success = True
            if on_progress is not None:
                on_progress(item, success)
            results.append(success)
        return results

    monkeypatch.setattr(import_cmd, "upload_many", fake_upload_many)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportArchivesAsyncFanout:
    """``_import_archives`` → ``upload_many`` delegation + accounting."""

    async def test_delegates_to_upload_many_with_tuned_knobs(
        self, tmp_path, monkeypatch
    ):
        """Every file becomes an ``UploadItem`` and the configured tunables
        (concurrency, retry budget, backoff) are forwarded intact."""
        folder_id = "a" * 16
        _make_archive_tree(tmp_path, folder_id, count=12)

        calls = _install_fake_upload_many(monkeypatch)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            totals: dict = {}
            changes: dict = {
                "new_archive files": 0,
                "failed": 0,
                "orphans_removed": 0,
            }
            await import_cmd._import_archives(
                tmp_path, {folder_id}, totals, changes, dry_run=False, verbose=False
            )

        assert len(calls) == 1
        call = calls[0]
        # Every file queued, exactly once.
        assert len(call["items"]) == 12
        assert all(isinstance(it, UploadItem) for it in call["items"])
        # The tunables come straight from the module-level constants so
        # operators can tweak one place at runtime.
        assert call["max_concurrency"] == import_cmd._ARCHIVE_UPLOAD_CONCURRENCY
        assert call["max_attempts"] == import_cmd._UPLOAD_MAX_ATTEMPTS
        assert call["backoff_base"] == import_cmd._UPLOAD_BACKOFF_BASE
        assert call["bucket_name"] == "test-archive-bucket"
        # All succeeded → exactly 12 new files, no failures.
        assert changes["new_archive files"] == 12
        assert changes["failed"] == 0

    async def test_content_type_guessed_per_file(self, tmp_path, monkeypatch):
        """mimetypes.guess_type runs before dispatch; unknowns fall back
        to ``application/octet-stream`` so the upload never goes untyped."""
        folder_id = "b" * 16
        folder = tmp_path / folder_id
        folder.mkdir(parents=True)
        (folder / "email.eml").write_bytes(b"x")
        (folder / "report.pdf").write_bytes(b"y")
        (folder / "weird.xyzzy").write_bytes(b"z")

        calls = _install_fake_upload_many(monkeypatch)

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            changes: dict = {
                "new_archive files": 0, "failed": 0, "orphans_removed": 0,
            }
            await import_cmd._import_archives(
                tmp_path, {folder_id}, {}, changes, dry_run=False, verbose=False
            )

        items = {it.remote_key: it for it in calls[0]["items"]}
        assert items[f"{folder_id}/report.pdf"].content_type == "application/pdf"
        # Python's mimetypes knows .eml → message/rfc822; just verify it
        # isn't the fallback. (The table can shift slightly across Python
        # versions, so we don't hardcode the string.)
        assert items[f"{folder_id}/email.eml"].content_type != "application/octet-stream"
        # Unknown extension falls back.
        assert (
            items[f"{folder_id}/weird.xyzzy"].content_type == "application/octet-stream"
        )

    async def test_failure_count_matches_failed_outcomes(
        self, tmp_path, monkeypatch
    ):
        """If 2 of 5 uploads come back False, ``changes["failed"]`` is 2
        and ``changes["new_archive files"]`` is 3."""
        folder_id = "c" * 16
        _make_archive_tree(tmp_path, folder_id, count=5)

        # Outcomes aligned with the file-discovery order. ``_import_archives``
        # doesn't guarantee order, but since every item funnels through the
        # same fake_upload_many loop we just need the right *count* of
        # Falses in the outcome vector.
        _install_fake_upload_many(
            monkeypatch, outcomes=[True, False, True, False, True]
        )

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            changes: dict = {
                "new_archive files": 0, "failed": 0, "orphans_removed": 0,
            }
            await import_cmd._import_archives(
                tmp_path, {folder_id}, {}, changes, dry_run=False, verbose=False
            )

        assert changes["new_archive files"] == 3
        assert changes["failed"] == 2

    async def test_no_files_to_upload_short_circuits(
        self, tmp_path, monkeypatch
    ):
        """When every local file already exists remotely, ``upload_many``
        isn't invoked at all — no point in spinning up an AsyncClient."""
        folder_id = "d" * 16
        files = _make_archive_tree(tmp_path, folder_id, count=3)

        calls = _install_fake_upload_many(monkeypatch)

        mock_storage = _make_mock_archive_storage()

        # Mark folder as present on remote AND every file as already there.
        def fake_list_folder(folder, *args, **kwargs):
            if folder == "":
                return [{"name": folder_id, "id": None}]  # folder entry
            if folder == folder_id:
                return [
                    {"name": p.name, "id": f"id-{p.name}"} for p in files
                ]
            return []

        mock_storage.list_folder.side_effect = fake_list_folder

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=mock_storage,
        ):
            changes: dict = {
                "new_archive files": 0, "failed": 0, "orphans_removed": 0,
            }
            await import_cmd._import_archives(
                tmp_path, {folder_id}, {}, changes, dry_run=False, verbose=False
            )

        assert len(calls) == 0
        assert changes["new_archive files"] == 0
        assert changes["failed"] == 0


class TestImportArchivesTunables:
    """Smoke-checks for the module-level constants operators rely on."""

    def test_upload_concurrency_constant_is_module_level(self):
        assert hasattr(import_cmd, "_ARCHIVE_UPLOAD_CONCURRENCY")
        assert isinstance(import_cmd._ARCHIVE_UPLOAD_CONCURRENCY, int)
        assert import_cmd._ARCHIVE_UPLOAD_CONCURRENCY >= 1

    def test_upload_retry_budget_is_module_level(self):
        assert hasattr(import_cmd, "_UPLOAD_MAX_ATTEMPTS")
        assert isinstance(import_cmd._UPLOAD_MAX_ATTEMPTS, int)
        assert import_cmd._UPLOAD_MAX_ATTEMPTS >= 1

    def test_upload_backoff_base_is_module_level(self):
        assert hasattr(import_cmd, "_UPLOAD_BACKOFF_BASE")
        assert isinstance(import_cmd._UPLOAD_BACKOFF_BASE, (int, float))
        assert import_cmd._UPLOAD_BACKOFF_BASE > 0
