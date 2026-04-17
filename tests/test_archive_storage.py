"""Tests for ArchiveStorage.list_folder — pagination + retry behavior.

Covers two real-world bugs fixed in this module:
    1. Silent truncation at 100 items (default storage3 list limit).
    2. Transient gateway errors (JSONDecodeError) being swallowed and
       making remote folders look partially empty.
"""

from __future__ import annotations

from json import JSONDecodeError
from unittest.mock import MagicMock, patch

import pytest

from mtss.storage import archive_storage as archive_storage_module
from mtss.storage.archive_storage import ArchiveStorage, ArchiveStorageError


def _make_storage_with_bucket(bucket: MagicMock) -> ArchiveStorage:
    """Build an ArchiveStorage instance with mocked infrastructure."""
    storage = ArchiveStorage.__new__(ArchiveStorage)
    storage.client = MagicMock()
    storage.bucket_name = "test-bucket"
    storage.bucket = bucket
    return storage


def _file_entry(name: str) -> dict:
    return {"name": name, "id": f"id-{name}"}


def _folder_entry(name: str) -> dict:
    return {"name": name, "id": None}


class TestListFolderPagination:
    def test_paginates_past_100_item_default(self):
        """Folder with 104 files must return all 104, not 100."""
        full = [_file_entry(f"file-{i:03d}.jpg") for i in range(104)]
        bucket = MagicMock()
        bucket.list = MagicMock(side_effect=[full[:100], full[100:]])
        storage = _make_storage_with_bucket(bucket)

        result = storage.list_folder("doc-id/attachments")

        assert len(result) == 104
        assert bucket.list.call_count == 2
        assert bucket.list.call_args_list[0].args[1] == {"limit": 100, "offset": 0}
        assert bucket.list.call_args_list[1].args[1] == {"limit": 100, "offset": 100}

    def test_single_page_stops_without_second_call(self):
        bucket = MagicMock()
        bucket.list = MagicMock(return_value=[_file_entry("a"), _file_entry("b")])
        storage = _make_storage_with_bucket(bucket)

        result = storage.list_folder("doc-id")

        assert len(result) == 2
        assert bucket.list.call_count == 1

    def test_empty_folder_returns_empty_list(self):
        bucket = MagicMock()
        bucket.list = MagicMock(return_value=[])
        storage = _make_storage_with_bucket(bucket)

        assert storage.list_folder("nonexistent") == []
        assert bucket.list.call_count == 1

    def test_files_only_filters_subfolder_placeholders(self):
        """Folder placeholders (id=None) excluded when files_only=True."""
        bucket = MagicMock()
        bucket.list = MagicMock(return_value=[
            _file_entry("email.eml"),
            _folder_entry("attachments"),
            _file_entry("metadata.json"),
        ])
        storage = _make_storage_with_bucket(bucket)

        result = storage.list_folder("doc-id", files_only=True)

        assert [f["name"] for f in result] == ["email.eml", "metadata.json"]

    def test_files_only_false_includes_folder_entries(self):
        """Root listing needs folder entries for orphan detection."""
        bucket = MagicMock()
        bucket.list = MagicMock(return_value=[
            _folder_entry("aaaa1111bbbb2222"),
            _folder_entry("cccc3333dddd4444"),
        ])
        storage = _make_storage_with_bucket(bucket)

        result = storage.list_folder("", files_only=False)

        assert len(result) == 2
        assert all(f["id"] is None for f in result)


class TestListFolderRetry:
    def test_retries_transient_jsondecodeerror_then_succeeds(self):
        """Simulates the observed Supabase gateway failure surface."""
        bucket = MagicMock()
        bucket.list = MagicMock(side_effect=[
            JSONDecodeError("Expecting value", "", 0),
            [_file_entry("a")],
        ])
        storage = _make_storage_with_bucket(bucket)

        with patch.object(archive_storage_module, "time") as mock_time:
            result = storage.list_folder("doc-id", backoff_base=1.0)

        assert len(result) == 1
        assert bucket.list.call_count == 2
        assert mock_time.sleep.call_count == 1
        assert mock_time.sleep.call_args_list[0].args[0] == 1.0  # 1.0 * 2**0

    def test_retries_with_exponential_backoff(self):
        bucket = MagicMock()
        bucket.list = MagicMock(side_effect=[
            JSONDecodeError("boom", "", 0),
            JSONDecodeError("boom", "", 0),
            [_file_entry("a")],
        ])
        storage = _make_storage_with_bucket(bucket)

        with patch.object(archive_storage_module, "time") as mock_time:
            storage.list_folder("doc-id", backoff_base=1.0)

        assert bucket.list.call_count == 3
        delays = [c.args[0] for c in mock_time.sleep.call_args_list]
        assert delays == [1.0, 2.0]  # 1*2^0, 1*2^1

    def test_raises_after_all_retries_exhausted(self):
        bucket = MagicMock()
        bucket.list = MagicMock(side_effect=JSONDecodeError("boom", "", 0))
        storage = _make_storage_with_bucket(bucket)

        with patch.object(archive_storage_module, "time"):
            with pytest.raises(ArchiveStorageError, match="after 3 attempts"):
                storage.list_folder("doc-id", backoff_base=0.01)

        assert bucket.list.call_count == 3

    def test_retry_recovers_mid_pagination(self):
        """Transient failure on page 2 of 3 must retry that page, not restart."""
        page1 = [_file_entry(f"a{i}") for i in range(100)]
        page2 = [_file_entry(f"b{i}") for i in range(100)]
        page3 = [_file_entry("c0")]
        bucket = MagicMock()
        bucket.list = MagicMock(side_effect=[
            page1,
            JSONDecodeError("transient", "", 0),
            page2,
            page3,
        ])
        storage = _make_storage_with_bucket(bucket)

        with patch.object(archive_storage_module, "time"):
            result = storage.list_folder("doc-id", backoff_base=0.01)

        assert len(result) == 201
        # second attempt at offset=100 after failure, then offset=200
        offsets = [c.args[1]["offset"] for c in bucket.list.call_args_list]
        assert offsets == [0, 100, 100, 200]
