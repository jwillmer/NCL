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
from mtss.utils import compute_folder_id


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

    async def test_sanitizes_unsafe_keys_and_repoints_uris(
        self, tmp_path, monkeypatch
    ):
        """Files with ``%`` or other chars Supabase's key validator rejects
        must be uploaded under a sanitized key, and the persisted URIs
        for those docs must be rewritten to match."""
        folder_id = "e" * 16
        folder = tmp_path / folder_id / "attachments"
        folder.mkdir(parents=True)
        (folder / "safe.pdf").write_bytes(b"a")
        (folder / "2%_sds.pdf").write_bytes(b"b")
        (folder / "report#draft.pdf").write_bytes(b"c")

        calls = _install_fake_upload_many(monkeypatch)

        # Capture UPDATE params for later assertion.
        update_calls: list[tuple] = []

        class _FakeConn:
            async def execute(self, sql, *args):
                if "UPDATE documents" in sql or "UPDATE chunks" in sql:
                    update_calls.append((sql, args))

            def transaction(self):
                class _Ctx:
                    async def __aenter__(self_inner):
                        return None

                    async def __aexit__(self_inner, *a):
                        return None

                return _Ctx()

        class _FakeAcquire:
            async def __aenter__(self_inner):
                return _FakeConn()

            async def __aexit__(self_inner, *a):
                return None

        fake_pool = MagicMock()
        fake_pool.acquire = MagicMock(return_value=_FakeAcquire())

        fake_db = MagicMock()

        async def _get_pool():
            return fake_pool

        fake_db.get_pool = _get_pool

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=_make_mock_archive_storage(),
        ):
            changes: dict = {
                "new_archive files": 0, "failed": 0, "orphans_removed": 0,
            }
            await import_cmd._import_archives(
                tmp_path, {folder_id}, {}, changes, dry_run=False, verbose=False,
                db=fake_db,
            )

        remote_keys = {it.remote_key for it in calls[0]["items"]}
        # ``%`` and ``#`` replaced by ``_``, slash preserved.
        assert f"{folder_id}/attachments/safe.pdf" in remote_keys
        assert f"{folder_id}/attachments/2__sds.pdf" in remote_keys
        assert f"{folder_id}/attachments/report_draft.pdf" in remote_keys
        # No raw invalid chars ever reach the upload call.
        assert all("%" not in k and "#" not in k for k in remote_keys)

        # Two UPDATE pairs (documents + chunks) per sanitized file = 4.
        assert len(update_calls) == 4
        # Every UPDATE carries the original /archive/... suffix as $1
        # and the safe suffix as $2, so citations don't break.
        update_pairs = {(c[1][0], c[1][1]) for c in update_calls}
        assert (
            f"/archive/{folder_id}/attachments/2%_sds.pdf",
            f"/archive/{folder_id}/attachments/2__sds.pdf",
        ) in update_pairs
        assert (
            f"/archive/{folder_id}/attachments/report#draft.pdf",
            f"/archive/{folder_id}/attachments/report_draft.pdf",
        ) in update_pairs

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


class TestImportArchivesSanitizedKeyMatching:
    """Remote keys for files with unsafe chars are stored sanitized. Both
    the "already uploaded?" check and orphan-file detection must compare
    against the sanitized form, not the raw on-disk rel_key — otherwise
    every import re-uploads the same files AND flags the valid remote
    copies as orphan."""

    async def test_already_uploaded_sanitized_file_is_not_requeued(
        self, tmp_path, monkeypatch
    ):
        folder_id = "f" * 32
        folder = tmp_path / folder_id / "attachments"
        folder.mkdir(parents=True)
        raw_name = "2%_sds.pdf"
        safe_name = "2__sds.pdf"
        (folder / raw_name).write_bytes(b"payload")

        calls = _install_fake_upload_many(monkeypatch)

        mock_storage = _make_mock_archive_storage()

        def fake_list_folder(folder_arg, *args, **kwargs):
            if folder_arg == "":
                return [{"name": folder_id, "id": None}]
            if folder_arg == f"{folder_id}/attachments":
                # Remote holds the SANITIZED name — previous run uploaded
                # under safe_name.
                return [{"name": safe_name, "id": f"id-{safe_name}"}]
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
                tmp_path,
                {folder_id},
                {},
                changes,
                dry_run=False,
                verbose=False,
            )

        # Must NOT re-upload the file.
        assert len(calls) == 0 or len(calls[0]["items"]) == 0
        assert changes["new_archive files"] == 0
        mock_storage.bucket.remove.assert_not_called()

    async def test_sanitized_remote_file_not_flagged_orphan(
        self, tmp_path, monkeypatch
    ):
        """The sanitized remote file matches the local raw file after
        sanitization → must not trigger ``bucket.remove``."""
        folder_id = "e" * 32
        folder = tmp_path / folder_id / "attachments"
        folder.mkdir(parents=True)
        (folder / "2%_sds.pdf").write_bytes(b"payload")

        _install_fake_upload_many(monkeypatch)

        mock_storage = _make_mock_archive_storage()

        def fake_list_folder(folder_arg, *args, **kwargs):
            if folder_arg == "":
                return [{"name": folder_id, "id": None}]
            if folder_arg == f"{folder_id}/attachments":
                return [{"name": "2__sds.pdf", "id": "id-1"}]
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
                tmp_path,
                {folder_id},
                {},
                changes,
                dry_run=False,
                verbose=False,
            )

        # No orphan deletion — sanitize-aware matching found it.
        mock_storage.bucket.remove.assert_not_called()
        assert changes["orphans_removed"] == 0


class TestImportArchivesOrphanFolderKeyspace:
    """Regression: archive folders live under ``compute_folder_id(doc_id)``
    (32-char), not ``doc_id[:16]``. Passing the wrong keyspace in
    ``local_doc_folder_ids`` makes every remote folder look orphan and
    queued for deletion — catastrophic on a fresh full-run after prior
    wave imports. Keep this test as the guardrail."""

    async def test_matching_folder_is_not_orphan(self, tmp_path, monkeypatch):
        doc_id = "abcdef0123456789"  # 16-char doc_id
        folder_id = compute_folder_id(doc_id)  # 32-char archive folder
        assert len(folder_id) == 32

        # Local tree uses the 32-char compute_folder_id name — this is
        # how archive_generator writes folders on disk.
        _make_archive_tree(tmp_path, folder_id, count=2)

        _install_fake_upload_many(monkeypatch)

        mock_storage = _make_mock_archive_storage()
        # Remote bucket already has the folder and its files (prior wave).
        def fake_list_folder(folder, *args, **kwargs):
            if folder == "":
                return [{"name": folder_id, "id": None}]
            if folder == folder_id:
                return [
                    {"name": f"file_{i:03d}.txt", "id": f"id-{i}"}
                    for i in range(2)
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
            # Caller hands the correct 32-char folder_id (as fixed
            # ``_import_data`` now does via compute_folder_id).
            await import_cmd._import_archives(
                tmp_path,
                {folder_id},
                {},
                changes,
                dry_run=False,
                verbose=False,
            )

        mock_storage.delete_folder.assert_not_called()
        assert changes["orphans_removed"] == 0

    async def test_16char_doc_id_is_flagged_orphan_regression(
        self, tmp_path, monkeypatch
    ):
        """Guards the historical bug direction: if a caller passes
        ``doc_id[:16]`` (as ``_import_data`` used to), the legitimate
        32-char folder is wrongly treated as orphan. This test locks in
        the contract: the caller MUST produce 32-char folder IDs."""
        doc_id = "abcdef0123456789"
        folder_id = compute_folder_id(doc_id)

        _make_archive_tree(tmp_path, folder_id, count=1)
        _install_fake_upload_many(monkeypatch)

        mock_storage = _make_mock_archive_storage()

        def fake_list_folder(folder, *args, **kwargs):
            if folder == "":
                return [{"name": folder_id, "id": None}]
            if folder == folder_id:
                return [{"name": "file_000.txt", "id": "id-0"}]
            return []

        mock_storage.list_folder.side_effect = fake_list_folder

        with patch(
            "mtss.storage.archive_storage.ArchiveStorage",
            return_value=mock_storage,
        ):
            changes: dict = {
                "new_archive files": 0, "failed": 0, "orphans_removed": 0,
            }
            # Simulate the old buggy caller: passes 16-char doc_id instead
            # of compute_folder_id. Should flag the legitimate folder as
            # orphan — this is the data-destroying pathway.
            await import_cmd._import_archives(
                tmp_path,
                {doc_id[:16]},
                {},
                changes,
                dry_run=False,
                verbose=False,
            )

        mock_storage.delete_folder.assert_called_once_with(folder_id)

    def test_caller_folder_id_helper_uses_compute_folder_id(self):
        """The fix is a one-line switch from ``doc_id[:16]`` to
        ``compute_folder_id(doc_id)`` in ``_import_data``. Lock in the
        module-level import of ``compute_folder_id`` so a future refactor
        can't silently revive the 16-char path."""
        import inspect

        src = inspect.getsource(import_cmd)
        # Must import + use the 32-char helper.
        assert "from ..utils import compute_folder_id" in src
        # The specific orphan-folder site must call compute_folder_id.
        marker = "local_doc_folder_ids = {"
        i = src.index(marker)
        # Look ahead at the set-comprehension body (next ~400 chars).
        block = src[i : i + 400]
        assert "compute_folder_id(d[" in block, block
        # And the old buggy form must be gone from that block.
        assert 'd["doc_id"][:16]' not in block, block


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
