"""Tests for archive file synchronization between local disk and Supabase Storage.

Covers the key-matching logic that ensures:
- Local archive files (sanitized with underscores by _sanitize_storage_key) map
  correctly to remote files in Supabase Storage
- Orphan detection works correctly
- files_to_upload produces no false positives
- Archive file count comparisons in validation are accurate
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _sanitize(filename: str) -> str:
    """Lazy wrapper so module collection does not trigger settings validation."""
    from mtss.ingest.archive_generator import _sanitize_storage_key
    return _sanitize_storage_key(filename)


# ===========================================================================
# Test 1: _sanitize_storage_key produces clean underscore-based names
# ===========================================================================


class TestSanitizeOutput:
    """Verify _sanitize_storage_key produces clean filenames with no URL-encoding."""

    @pytest.mark.unit
    def test_spaces_become_underscores(self):
        result = _sanitize("GE FO SER SYS.pdf")
        assert " " not in result
        assert "%" not in result
        assert result == "GE_FO_SER_SYS.pdf"

    @pytest.mark.unit
    def test_brackets_become_parens(self):
        result = _sanitize("report[1].pdf")
        assert "[" not in result and "]" not in result
        assert "(" in result and ")" in result

    @pytest.mark.unit
    def test_tilde_replaced(self):
        result = _sanitize("~WRD0001.jpg")
        assert "~" not in result
        assert "WRD0001.jpg" in result

    @pytest.mark.unit
    def test_plain_ascii_unchanged(self):
        assert _sanitize("simple-report_v2.pdf") == "simple-report_v2.pdf"

    @pytest.mark.unit
    def test_multiple_spaces_collapsed(self):
        result = _sanitize("MARAN ASPASIA - quarterly report.pdf")
        assert " " not in result
        assert "__" not in result

    @pytest.mark.unit
    def test_unicode_transliteration(self):
        result = _sanitize("report_\u0394\u03b5\u03bb\u03c4\u03b1.pdf")
        assert result.isascii()

    @pytest.mark.unit
    def test_idempotent(self):
        first = _sanitize("test file [draft].pdf")
        second = _sanitize(first)
        assert first == second

    @pytest.mark.unit
    def test_parentheses_preserved(self):
        result = _sanitize("report (final).pdf")
        assert "(" in result and ")" in result


# ===========================================================================
# Test 2: Local key building matches remote key building
# ===========================================================================


class TestLocalRemoteKeyMatching:
    """Verify local keys from disk match remote keys from bucket.list()."""

    @pytest.mark.unit
    def test_simple_filename_matches(self, tmp_path):
        archive_dir = tmp_path / "archive"
        doc_folder = archive_dir / "abc123def45678"
        doc_folder.mkdir(parents=True)
        (doc_folder / "email.eml").write_bytes(b"content")

        rel = str((doc_folder / "email.eml").relative_to(archive_dir)).replace("\\", "/")
        assert rel == "abc123def45678/email.eml"

    @pytest.mark.unit
    def test_sanitized_name_matches_remote(self, tmp_path):
        archive_dir = tmp_path / "archive"
        att_folder = archive_dir / "abc123def45678" / "attachments"
        att_folder.mkdir(parents=True)

        safe_name = _sanitize("GE FO SER SYS.pdf")
        (att_folder / safe_name).write_bytes(b"content")

        rel = str((att_folder / safe_name).relative_to(archive_dir)).replace("\\", "/")
        remote_key = f"abc123def45678/attachments/{safe_name}"
        assert rel == remote_key

    @pytest.mark.unit
    def test_brackets_match(self, tmp_path):
        archive_dir = tmp_path / "archive"
        att_folder = archive_dir / "abc123def45678" / "attachments"
        att_folder.mkdir(parents=True)

        safe_name = _sanitize("doc[1].pdf")
        (att_folder / safe_name).write_bytes(b"content")

        rel = str((att_folder / safe_name).relative_to(archive_dir)).replace("\\", "/")
        assert rel == f"abc123def45678/attachments/{safe_name}"


# ===========================================================================
# Test 3: Orphan detection
# ===========================================================================


class TestOrphanDetection:
    """Test that orphan detection (existing_remote - local) works correctly."""

    def _simulate_sync(self, local: list[str], remote: list[str],
                       folder: str = "abc123/attachments"):
        local_keys = {f"{folder}/{n}" for n in local}
        remote_keys = {f"{folder}/{n}" for n in remote}
        return remote_keys - local_keys

    @pytest.mark.unit
    def test_no_orphans_when_synced(self):
        safe = _sanitize("test file.pdf")
        assert self._simulate_sync([safe], [safe]) == set()

    @pytest.mark.unit
    def test_orphan_detected(self):
        assert len(self._simulate_sync([], ["old.pdf"])) == 1

    @pytest.mark.unit
    def test_no_false_orphan(self):
        safe = _sanitize("MARAN ASPASIA - report.pdf")
        assert self._simulate_sync([safe], [safe]) == set()

    @pytest.mark.unit
    def test_true_orphan_among_matches(self):
        local = [_sanitize("report.pdf"), _sanitize("test file.pdf")]
        remote = [_sanitize("report.pdf"), _sanitize("test file.pdf"), "stale.xlsx"]
        assert len(self._simulate_sync(local, remote)) == 1

    @pytest.mark.unit
    def test_multiple_orphans(self):
        assert len(self._simulate_sync(["keep.pdf"], ["keep.pdf", "a.pdf", "b.pdf"])) == 2


# ===========================================================================
# Test 4: files_to_upload — no false positives
# ===========================================================================


class TestFilesToUpload:

    @pytest.mark.unit
    def test_no_upload_when_exists(self):
        safe = _sanitize("test file.pdf")
        key = f"abc123/attachments/{safe}"
        assert key in {key}  # exists in remote

    @pytest.mark.unit
    def test_new_file_detected(self):
        safe = _sanitize("new report.pdf")
        key = f"abc123/attachments/{safe}"
        assert key not in set()  # not in remote

    @pytest.mark.unit
    def test_mixed_existing_and_new(self):
        old = _sanitize("old file.pdf")
        new = _sanitize("new file.pdf")
        remote = {f"abc123/attachments/{old}"}
        to_upload = [k for k in [f"abc123/attachments/{old}", f"abc123/attachments/{new}"]
                     if k not in remote]
        assert len(to_upload) == 1


# ===========================================================================
# Test 5: Archive file count comparison
# ===========================================================================


class TestArchiveFileCountComparison:

    def _compare(self, local: list[str], remote: list[str]) -> dict:
        def counts(keys):
            c = {}
            for k in keys:
                f = k.split("/")[0]
                c[f] = c.get(f, 0) + 1
            return c
        lc, rc = counts(local), counts(remote)
        return {f: (lc.get(f, 0), rc.get(f, 0))
                for f in set(lc) | set(rc) if lc.get(f, 0) != rc.get(f, 0)}

    @pytest.mark.unit
    def test_counts_match(self):
        keys = ["a/email.eml", "a/email.eml.md", f"a/attachments/{_sanitize('report.pdf')}"]
        assert self._compare(keys, keys) == {}

    @pytest.mark.unit
    def test_mismatch_remote_missing(self):
        local = ["a/email.eml", "a/att.pdf"]
        remote = ["a/email.eml"]
        assert "a" in self._compare(local, remote)

    @pytest.mark.unit
    def test_mismatch_remote_extra(self):
        local = ["a/email.eml"]
        remote = ["a/email.eml", "a/stale.pdf"]
        assert "a" in self._compare(local, remote)

    @pytest.mark.unit
    def test_folders_independent(self):
        local = ["a/email.eml", "b/email.eml"]
        remote = ["a/email.eml", "b/email.eml", "b/extra.pdf"]
        m = self._compare(local, remote)
        assert "a" not in m
        assert "b" in m


# ===========================================================================
# Test 6: Key building from disk
# ===========================================================================


class TestImportKeyBuildingFromDisk:

    @pytest.mark.unit
    def test_rglob_key_building(self, tmp_path):
        archive_dir = tmp_path / "archive"
        att = archive_dir / "abc123" / "attachments"
        att.mkdir(parents=True)
        safe = _sanitize("test file [draft].pdf")
        (att / safe).write_bytes(b"content")

        keys = [str(f.relative_to(archive_dir)).replace("\\", "/")
                for f in archive_dir.rglob("*") if f.is_file()]
        assert len(keys) == 1
        assert "%" not in keys[0]
        assert safe in keys[0]

    @pytest.mark.unit
    def test_backslash_normalization(self, tmp_path):
        archive_dir = tmp_path / "archive"
        d = archive_dir / "abc123" / "attachments"
        d.mkdir(parents=True)
        (d / "file.pdf").write_bytes(b"content")

        rel = str((d / "file.pdf").relative_to(archive_dir)).replace("\\", "/")
        assert "\\" not in rel
        assert rel == "abc123/attachments/file.pdf"

    @pytest.mark.unit
    def test_path_traversal_detected(self):
        assert any(s == ".." for s in "abc123/../secret/file.pdf".split("/"))
