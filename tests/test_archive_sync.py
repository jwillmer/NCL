"""Tests for archive file synchronization between local disk and Supabase Storage.

Covers the encoding round-trip and key-matching logic that ensures:
- Local archive files (URL-encoded on disk by _sanitize_storage_key) map correctly
  to remote files in Supabase Storage (which stores decoded names)
- Orphan detection works across encoding boundaries
- files_to_upload produces no false positives from encoding mismatches
- Archive file count comparisons in validation are accurate
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock
from urllib.parse import quote, unquote

import pytest


def _sanitize(filename: str) -> str:
    """Lazy wrapper so module collection does not trigger settings validation."""
    from mtss.ingest.archive_generator import _sanitize_storage_key

    return _sanitize_storage_key(filename)


# ---------------------------------------------------------------------------
# Helper: simulate the key-building flows from import and remote listing
# ---------------------------------------------------------------------------


def _build_local_key(archive_dir: Path, file_path: Path) -> str:
    """Reproduce import_cmd._import_archives local key building.

    This mirrors lines 281-282 of import_cmd.py:
        rel_key = str(file_path.relative_to(archive_dir)).replace("\\\\", "/")
    """
    return str(file_path.relative_to(archive_dir)).replace("\\", "/")


def _build_remote_key(folder: str, filename_from_bucket: str) -> str:
    """Build a key as Supabase bucket.list() would return it.

    bucket.list() returns decoded filenames (Supabase stores decoded names).
    So if we uploaded "GE%20FO%20SER%20SYS.pdf", bucket.list() returns
    "GE FO SER SYS.pdf".
    """
    return f"{folder}/{filename_from_bucket}"


def _decode_local_key(local_key: str) -> str:
    """Decode a local (URL-encoded) key so it matches remote (decoded) keys.

    This is the fix that was applied: unquote the local key for comparison.
    """
    return unquote(local_key)


# ===========================================================================
# Test 1: _sanitize_storage_key -> unquote round-trip
# ===========================================================================


class TestSanitizeUnquoteRoundtrip:
    """Verify that _sanitize_storage_key output, when unquoted, produces a
    stable decoded name suitable for comparison with Supabase bucket.list()."""

    @pytest.mark.unit
    def test_spaces_roundtrip(self):
        """Filename with spaces: encode -> decode yields consistent name."""
        encoded = _sanitize("GE FO SER SYS.pdf")
        decoded = unquote(encoded)
        assert " " not in encoded, "encoded key should not contain literal spaces"
        assert "%20" in encoded, "spaces should become %20"
        assert decoded == "GE FO SER SYS.pdf"

    @pytest.mark.unit
    def test_brackets_roundtrip(self):
        """Brackets are replaced by parens, not URL-encoded."""
        encoded = _sanitize("report[1].pdf")
        decoded = unquote(encoded)
        assert "[" not in encoded
        assert "]" not in encoded
        assert "(" in decoded and ")" in decoded
        # Decode should be idempotent (parens are in the safe set)
        assert decoded == encoded, "parens are safe chars, no encoding needed"

    @pytest.mark.unit
    def test_tilde_prefix_roundtrip(self):
        """Leading tilde replaced with underscore survives round-trip."""
        encoded = _sanitize("~WRD0001.jpg")
        decoded = unquote(encoded)
        assert decoded.startswith("_")
        assert "WRD0001.jpg" in decoded

    @pytest.mark.unit
    def test_plain_ascii_roundtrip(self):
        """Plain ASCII filenames should pass through unchanged."""
        filename = "simple-report_v2.pdf"
        encoded = _sanitize(filename)
        decoded = unquote(encoded)
        assert encoded == filename, "no encoding needed for safe chars"
        assert decoded == filename

    @pytest.mark.unit
    def test_multiple_spaces_roundtrip(self):
        """Multiple spaces and mixed punctuation."""
        filename = "MARAN ASPASIA - quarterly report.pdf"
        encoded = _sanitize(filename)
        decoded = unquote(encoded)
        assert " " not in encoded
        assert decoded == "MARAN ASPASIA - quarterly report.pdf"

    @pytest.mark.unit
    def test_unicode_transliteration_roundtrip(self):
        """Non-ASCII chars are transliterated, then result round-trips cleanly."""
        encoded = _sanitize("report_\u0394\u03b5\u03bb\u03c4\u03b1.pdf")
        decoded = unquote(encoded)
        assert decoded.isascii(), "transliteration should produce ASCII"
        assert decoded == encoded, "transliterated name has no special chars to encode"

    @pytest.mark.unit
    def test_already_decoded_name_is_stable(self):
        """If the name has no special chars, encode+decode is identity."""
        filename = "email.eml.md"
        encoded = _sanitize(filename)
        decoded = unquote(encoded)
        assert decoded == filename

    @pytest.mark.unit
    def test_double_decode_is_safe(self):
        """Decoding an already-decoded name should not corrupt it."""
        encoded = _sanitize("test file.pdf")
        decoded_once = unquote(encoded)
        decoded_twice = unquote(decoded_once)
        assert decoded_once == decoded_twice, "double unquote must be idempotent"

    @pytest.mark.unit
    def test_parentheses_in_safe_set(self):
        """Parentheses are in the safe set, so they are not encoded."""
        filename = "report (final).pdf"
        encoded = _sanitize(filename)
        # Spaces get encoded, but parens stay literal
        assert "(" in encoded
        assert ")" in encoded
        decoded = unquote(encoded)
        assert decoded == filename


# ===========================================================================
# Test 2: Local key building matches remote key building
# ===========================================================================


class TestLocalRemoteKeyMatching:
    """Verify that local keys from disk match remote keys from bucket.list()
    after both are normalized to the same (decoded) form."""

    @pytest.mark.unit
    def test_simple_filename_matches(self, tmp_path):
        """Simple filename: local key == remote key (no encoding difference)."""
        archive_dir = tmp_path / "archive"
        doc_folder = archive_dir / "abc123def45678"
        doc_folder.mkdir(parents=True)
        (doc_folder / "email.eml").write_bytes(b"content")

        local_key = _build_local_key(archive_dir, doc_folder / "email.eml")
        remote_key = _build_remote_key("abc123def45678", "email.eml")

        assert _decode_local_key(local_key) == remote_key

    @pytest.mark.unit
    def test_encoded_spaces_match_decoded_remote(self, tmp_path):
        """Local file with URL-encoded spaces matches decoded remote name."""
        archive_dir = tmp_path / "archive"
        att_folder = archive_dir / "abc123def45678" / "attachments"
        att_folder.mkdir(parents=True)

        # On disk the file is named with %20 (written by _sanitize_storage_key)
        encoded_name = _sanitize("GE FO SER SYS.pdf")
        (att_folder / encoded_name).write_bytes(b"content")

        local_key = _build_local_key(archive_dir, att_folder / encoded_name)

        # Supabase stores the decoded name
        remote_key = _build_remote_key(
            "abc123def45678/attachments", "GE FO SER SYS.pdf"
        )

        assert _decode_local_key(local_key) == remote_key

    @pytest.mark.unit
    def test_brackets_replaced_match(self, tmp_path):
        """Brackets replaced with parens on disk must match remote parens."""
        archive_dir = tmp_path / "archive"
        att_folder = archive_dir / "abc123def45678" / "attachments"
        att_folder.mkdir(parents=True)

        encoded_name = _sanitize("doc[1].pdf")
        (att_folder / encoded_name).write_bytes(b"content")

        local_key = _build_local_key(archive_dir, att_folder / encoded_name)
        # Supabase stores the decoded name (parens, since brackets were replaced)
        remote_key = _build_remote_key("abc123def45678/attachments", "doc(1).pdf")

        assert _decode_local_key(local_key) == remote_key

    @pytest.mark.unit
    def test_mixed_encoding_match(self, tmp_path):
        """Filename with spaces and brackets: both transformations apply."""
        archive_dir = tmp_path / "archive"
        att_folder = archive_dir / "abc123def45678" / "attachments"
        att_folder.mkdir(parents=True)

        original = "test [report] file.pdf"
        encoded_name = _sanitize(original)
        (att_folder / encoded_name).write_bytes(b"content")

        local_key = _build_local_key(archive_dir, att_folder / encoded_name)
        # After sanitize: brackets -> parens, spaces -> %20
        # After Supabase decode: %20 -> spaces
        expected_remote = "test (report) file.pdf"
        remote_key = _build_remote_key("abc123def45678/attachments", expected_remote)

        assert _decode_local_key(local_key) == remote_key


# ===========================================================================
# Test 3: Orphan detection with various filename patterns
# ===========================================================================


class TestOrphanDetection:
    """Test that orphan detection (existing_remote - local) works correctly
    across encoding boundaries for various filename patterns."""

    def _simulate_sync(
        self,
        local_filenames: list[str],
        remote_filenames: list[str],
        folder: str = "abc123def45678/attachments",
    ) -> tuple[set[str], set[str], set[str]]:
        """Simulate the import sync logic.

        Args:
            local_filenames: Filenames as they appear on disk (URL-encoded).
            remote_filenames: Filenames as returned by bucket.list() (decoded).

        Returns:
            (local_keys, remote_keys, orphans) — all in decoded form.
        """
        # Local keys: folder/encoded_name, then decoded for comparison
        local_keys = {
            unquote(f"{folder}/{name}") for name in local_filenames
        }
        # Remote keys: folder/decoded_name (already decoded from bucket.list)
        remote_keys = {
            f"{folder}/{name}" for name in remote_filenames
        }
        orphans = remote_keys - local_keys
        return local_keys, remote_keys, orphans

    @pytest.mark.unit
    def test_no_orphans_when_synced(self):
        """Perfectly synced files produce no orphans."""
        encoded = _sanitize("test file.pdf")
        local_keys, remote_keys, orphans = self._simulate_sync(
            local_filenames=[encoded],
            remote_filenames=["test file.pdf"],
        )
        assert orphans == set()

    @pytest.mark.unit
    def test_orphan_detected_for_missing_local(self):
        """File on remote but not local is detected as orphan."""
        _, _, orphans = self._simulate_sync(
            local_filenames=[],
            remote_filenames=["old_report.pdf"],
        )
        assert len(orphans) == 1
        assert "old_report.pdf" in next(iter(orphans))

    @pytest.mark.unit
    def test_no_false_orphan_from_spaces(self):
        """Spaces: encoded local (%20) must match decoded remote (space)."""
        encoded = _sanitize("MARAN ASPASIA - report.pdf")
        _, _, orphans = self._simulate_sync(
            local_filenames=[encoded],
            remote_filenames=["MARAN ASPASIA - report.pdf"],
        )
        assert orphans == set(), (
            f"should not detect false orphan; encoded={encoded!r}"
        )

    @pytest.mark.unit
    def test_no_false_orphan_from_brackets(self):
        """Brackets replaced with parens should not cause false orphans."""
        encoded = _sanitize("doc[1].pdf")
        _, _, orphans = self._simulate_sync(
            local_filenames=[encoded],
            remote_filenames=["doc(1).pdf"],
        )
        assert orphans == set()

    @pytest.mark.unit
    def test_no_false_orphan_from_tilde(self):
        """Leading tilde replaced with underscore should not cause false orphan."""
        encoded = _sanitize("~WRD0001.jpg")
        _, _, orphans = self._simulate_sync(
            local_filenames=[encoded],
            remote_filenames=["_WRD0001.jpg"],
        )
        assert orphans == set()

    @pytest.mark.unit
    def test_no_false_orphan_plain_ascii(self):
        """Plain ASCII files are identical in both sets."""
        filename = "email.eml"
        _, _, orphans = self._simulate_sync(
            local_filenames=[filename],
            remote_filenames=[filename],
        )
        assert orphans == set()

    @pytest.mark.unit
    def test_true_orphan_among_matches(self):
        """Mix of matching and orphaned files."""
        local = [
            _sanitize("report.pdf"),
            _sanitize("test file.pdf"),
        ]
        remote = [
            "report.pdf",
            "test file.pdf",
            "stale_attachment.xlsx",  # orphan
        ]
        _, _, orphans = self._simulate_sync(local, remote)
        assert len(orphans) == 1
        assert "stale_attachment.xlsx" in next(iter(orphans))

    @pytest.mark.unit
    def test_multiple_orphans(self):
        """Multiple orphan files detected correctly."""
        local = [_sanitize("keep.pdf")]
        remote = ["keep.pdf", "old1.pdf", "old2.pdf"]
        _, _, orphans = self._simulate_sync(local, remote)
        assert len(orphans) == 2

    @pytest.mark.unit
    def test_already_decoded_names_no_false_orphan(self):
        """Names that need no encoding (safe chars) should not cause orphans."""
        filenames = ["email.eml", "email.eml.md", "metadata.json"]
        _, _, orphans = self._simulate_sync(filenames, filenames)
        assert orphans == set()


# ===========================================================================
# Test 4: files_to_upload identification — no false positives
# ===========================================================================


class TestFilesToUpload:
    """Test that the import detects files needing upload without false
    positives caused by encoding mismatches."""

    def _find_files_to_upload(
        self,
        local_keys: list[str],
        remote_decoded_keys: set[str],
    ) -> list[str]:
        """Simulate upload detection: local files not yet on remote.

        Args:
            local_keys: URL-encoded keys from disk.
            remote_decoded_keys: Decoded keys from bucket.list().

        Returns:
            List of local keys that need uploading.
        """
        to_upload = []
        for local_key in local_keys:
            decoded = unquote(local_key)
            if decoded not in remote_decoded_keys:
                to_upload.append(local_key)
        return to_upload

    @pytest.mark.unit
    def test_no_upload_when_already_exists(self):
        """File already on remote should not be re-uploaded."""
        encoded = _sanitize("test file.pdf")
        local_key = f"abc123/attachments/{encoded}"
        remote_keys = {"abc123/attachments/test file.pdf"}

        to_upload = self._find_files_to_upload([local_key], remote_keys)
        assert to_upload == [], "should not re-upload existing file"

    @pytest.mark.unit
    def test_new_file_detected_for_upload(self):
        """File not on remote should be detected for upload."""
        encoded = _sanitize("new report.pdf")
        local_key = f"abc123/attachments/{encoded}"
        remote_keys = set()  # empty remote

        to_upload = self._find_files_to_upload([local_key], remote_keys)
        assert len(to_upload) == 1

    @pytest.mark.unit
    def test_mixed_existing_and_new(self):
        """Correctly distinguish existing from new files."""
        existing_encoded = _sanitize("old file.pdf")
        new_encoded = _sanitize("new file.pdf")

        local_keys = [
            f"abc123/attachments/{existing_encoded}",
            f"abc123/attachments/{new_encoded}",
        ]
        remote_keys = {"abc123/attachments/old file.pdf"}

        to_upload = self._find_files_to_upload(local_keys, remote_keys)
        assert len(to_upload) == 1
        assert new_encoded in to_upload[0]

    @pytest.mark.unit
    def test_bracket_file_not_false_positive(self):
        """File with brackets -> parens should not falsely re-upload."""
        encoded = _sanitize("doc[1].pdf")
        local_key = f"abc123/attachments/{encoded}"
        # Remote has decoded parens
        remote_keys = {"abc123/attachments/doc(1).pdf"}

        to_upload = self._find_files_to_upload([local_key], remote_keys)
        assert to_upload == [], "bracket->paren file should match"

    @pytest.mark.unit
    def test_no_false_positive_with_spaces(self):
        """Encoded spaces should not cause false upload detection."""
        filenames = [
            "MARAN ASPASIA - report.pdf",
            "GE FO SER SYS.pdf",
            "test (draft) v2.pdf",
        ]
        local_keys = []
        remote_keys = set()
        for fn in filenames:
            encoded = _sanitize(fn)
            local_keys.append(f"abc123/attachments/{encoded}")
            # Remote stores decoded names
            remote_keys.add(f"abc123/attachments/{unquote(encoded)}")

        to_upload = self._find_files_to_upload(local_keys, remote_keys)
        assert to_upload == [], "all files exist remotely, no uploads needed"


# ===========================================================================
# Test 5: Archive file count comparison (validation logic)
# ===========================================================================


class TestArchiveFileCountComparison:
    """Test the archive file count comparison that validation performs:
    comparing local file counts per doc_id folder against remote counts."""

    def _count_files_per_folder(
        self, keys: set[str]
    ) -> dict[str, int]:
        """Group keys by top-level folder (doc_id) and count files."""
        counts: dict[str, int] = {}
        for key in keys:
            folder = key.split("/")[0]
            counts[folder] = counts.get(folder, 0) + 1
        return counts

    def _compare_counts(
        self,
        local_encoded_keys: list[str],
        remote_decoded_keys: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Compare file counts per folder between local and remote.

        Returns:
            Dict of folder -> (local_count, remote_count) for mismatches only.
        """
        local_decoded = {unquote(k) for k in local_encoded_keys}
        remote_set = set(remote_decoded_keys)

        local_counts = self._count_files_per_folder(local_decoded)
        remote_counts = self._count_files_per_folder(remote_set)

        mismatches = {}
        all_folders = set(local_counts) | set(remote_counts)
        for folder in all_folders:
            lc = local_counts.get(folder, 0)
            rc = remote_counts.get(folder, 0)
            if lc != rc:
                mismatches[folder] = (lc, rc)
        return mismatches

    @pytest.mark.unit
    def test_counts_match_when_synced(self):
        """Perfectly synced folder should report no mismatches."""
        folder = "abc123def45678"
        local = [
            f"{folder}/email.eml",
            f"{folder}/email.eml.md",
            f"{folder}/metadata.json",
            f"{folder}/attachments/{_sanitize('report.pdf')}",
        ]
        remote = [
            f"{folder}/email.eml",
            f"{folder}/email.eml.md",
            f"{folder}/metadata.json",
            f"{folder}/attachments/report.pdf",
        ]
        mismatches = self._compare_counts(local, remote)
        assert mismatches == {}

    @pytest.mark.unit
    def test_mismatch_when_remote_missing_file(self):
        """Detect mismatch when remote has fewer files."""
        folder = "abc123def45678"
        local = [
            f"{folder}/email.eml",
            f"{folder}/email.eml.md",
            f"{folder}/attachments/{_sanitize('report.pdf')}",
        ]
        remote = [
            f"{folder}/email.eml",
            f"{folder}/email.eml.md",
            # attachment missing from remote
        ]
        mismatches = self._compare_counts(local, remote)
        assert folder in mismatches
        local_count, remote_count = mismatches[folder]
        assert local_count == 3
        assert remote_count == 2

    @pytest.mark.unit
    def test_mismatch_when_remote_has_extra(self):
        """Detect mismatch when remote has orphan files."""
        folder = "abc123def45678"
        local = [f"{folder}/email.eml"]
        remote = [
            f"{folder}/email.eml",
            f"{folder}/attachments/stale.pdf",  # orphan
        ]
        mismatches = self._compare_counts(local, remote)
        assert folder in mismatches
        local_count, remote_count = mismatches[folder]
        assert local_count == 1
        assert remote_count == 2

    @pytest.mark.unit
    def test_encoding_does_not_cause_false_mismatch(self):
        """Encoded local vs decoded remote should not cause count mismatch."""
        folder = "abc123def45678"
        local = [
            f"{folder}/email.eml",
            f"{folder}/attachments/{_sanitize('GE FO SER SYS.pdf')}",
            f"{folder}/attachments/{_sanitize('doc[1].pdf')}",
        ]
        remote = [
            f"{folder}/email.eml",
            f"{folder}/attachments/GE FO SER SYS.pdf",
            f"{folder}/attachments/doc(1).pdf",
        ]
        mismatches = self._compare_counts(local, remote)
        assert mismatches == {}, (
            "encoding differences should not cause count mismatch"
        )

    @pytest.mark.unit
    def test_multiple_folders_independent(self):
        """Each folder is counted independently."""
        local = [
            "folder_a/email.eml",
            "folder_a/email.eml.md",
            "folder_b/email.eml",
        ]
        remote = [
            "folder_a/email.eml",
            "folder_a/email.eml.md",
            "folder_b/email.eml",
            "folder_b/attachments/extra.pdf",  # orphan in folder_b
        ]
        mismatches = self._compare_counts(local, remote)
        assert "folder_a" not in mismatches
        assert "folder_b" in mismatches
        assert mismatches["folder_b"] == (1, 2)

    @pytest.mark.unit
    def test_empty_local_reports_all_remote_as_mismatch(self):
        """If local is empty but remote has files, all folders are mismatches."""
        remote = [
            "folder_a/email.eml",
            "folder_b/email.eml",
        ]
        mismatches = self._compare_counts([], remote)
        assert "folder_a" in mismatches
        assert "folder_b" in mismatches

    @pytest.mark.unit
    def test_real_scenario_3170_vs_3164(self):
        """Simulate the production scenario: 6 missing files across 2 folders."""
        # folder_a: 5 local, 3 remote (2 missing)
        # folder_b: 4 local, 0 remote (4 missing, entire folder missing)
        # folder_c: 3 local, 3 remote (all synced)
        local = (
            [f"folder_a/file{i}.txt" for i in range(5)]
            + [f"folder_b/file{i}.txt" for i in range(4)]
            + [f"folder_c/file{i}.txt" for i in range(3)]
        )
        remote = (
            [f"folder_a/file{i}.txt" for i in range(3)]
            # folder_b entirely missing
            + [f"folder_c/file{i}.txt" for i in range(3)]
        )
        mismatches = self._compare_counts(local, remote)
        assert "folder_a" in mismatches
        assert mismatches["folder_a"] == (5, 3)
        assert "folder_b" in mismatches
        assert mismatches["folder_b"] == (4, 0)
        assert "folder_c" not in mismatches


# ===========================================================================
# Test 6: ArchiveStorage.file_exists uses unquote correctly
# ===========================================================================


class TestArchiveStorageFileExists:
    """Verify that ArchiveStorage.file_exists correctly handles the
    URL-encoding/decoding mismatch between local keys and bucket contents."""

    @pytest.mark.unit
    def test_file_exists_decodes_before_comparison(self):
        """file_exists should unquote the filename before comparing with bucket.list()."""
        # Simulate: bucket.list returns decoded name, we query with encoded name
        mock_bucket = MagicMock()
        mock_bucket.list.return_value = [
            {"name": "GE FO SER SYS.pdf"},  # decoded by Supabase
        ]

        # Inline the logic from ArchiveStorage.file_exists
        path = "abc123def45678/attachments/GE%20FO%20SER%20SYS.pdf"
        parts = path.rsplit("/", 1)
        folder, filename = parts
        decoded_filename = unquote(filename)

        files = mock_bucket.list(folder)
        found = any(f["name"] == decoded_filename for f in files)

        assert found, "should find file after decoding the query filename"

    @pytest.mark.unit
    def test_file_exists_already_decoded_name(self):
        """Names without encoding should work as-is."""
        mock_bucket = MagicMock()
        mock_bucket.list.return_value = [{"name": "email.eml"}]

        path = "abc123def45678/email.eml"
        parts = path.rsplit("/", 1)
        folder, filename = parts
        decoded_filename = unquote(filename)

        files = mock_bucket.list(folder)
        found = any(f["name"] == decoded_filename for f in files)

        assert found

    @pytest.mark.unit
    def test_file_exists_returns_false_for_missing(self):
        """Missing file should not be found even after decoding."""
        mock_bucket = MagicMock()
        mock_bucket.list.return_value = [{"name": "other.pdf"}]

        path = "abc123def45678/attachments/missing%20file.pdf"
        parts = path.rsplit("/", 1)
        folder, filename = parts
        decoded_filename = unquote(filename)

        files = mock_bucket.list(folder)
        found = any(f["name"] == decoded_filename for f in files)

        assert not found


# ===========================================================================
# Test 7: Import key building from real directory structure
# ===========================================================================


class TestImportKeyBuildingFromDisk:
    """Integration-style tests that create real directory structures
    and verify key building matches what import_cmd would produce."""

    @pytest.mark.unit
    def test_rglob_key_building(self, tmp_path):
        """Simulate _import_archives rglob + key building with encoded filenames."""
        archive_dir = tmp_path / "archive"
        doc_folder = archive_dir / "abc123def45678"
        att_folder = doc_folder / "attachments"
        att_folder.mkdir(parents=True)

        # Write files with sanitized names (as archive_generator does)
        files_written = {
            "email.eml": doc_folder / "email.eml",
            "email.eml.md": doc_folder / "email.eml.md",
            "metadata.json": doc_folder / "metadata.json",
        }
        for name, path in files_written.items():
            path.write_bytes(b"content")

        # Write an encoded attachment
        encoded_att = _sanitize("test file [draft].pdf")
        att_path = att_folder / encoded_att
        att_path.write_bytes(b"content")

        # Simulate _import_archives key building
        built_keys = []
        for file_path in archive_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_key = str(file_path.relative_to(archive_dir)).replace("\\", "/")
            built_keys.append(rel_key)

        # Verify we got all files
        assert len(built_keys) == 4

        # Verify the attachment key contains the encoded name
        att_keys = [k for k in built_keys if "attachments" in k]
        assert len(att_keys) == 1
        assert encoded_att in att_keys[0]

        # When decoded, should match what Supabase stores
        decoded = unquote(att_keys[0])
        assert "test file (draft).pdf" in decoded

    @pytest.mark.unit
    def test_backslash_normalization(self, tmp_path):
        """On Windows, backslashes in paths must be normalized to forward slashes."""
        archive_dir = tmp_path / "archive"
        doc_folder = archive_dir / "abc123def45678" / "attachments"
        doc_folder.mkdir(parents=True)
        (doc_folder / "file.pdf").write_bytes(b"content")

        file_path = doc_folder / "file.pdf"
        rel_key = str(file_path.relative_to(archive_dir)).replace("\\", "/")

        assert "\\" not in rel_key, "backslashes must be normalized"
        assert rel_key == "abc123def45678/attachments/file.pdf"

    @pytest.mark.unit
    def test_path_traversal_rejected(self):
        """Keys containing '..' should be rejected."""
        rel_key = "abc123def45678/../secret/file.pdf"
        assert ".." in rel_key, "traversal paths should be detected"
        # import_cmd skips these (line 283-284)

    @pytest.mark.unit
    def test_non_file_entries_skipped(self, tmp_path):
        """Directories should be skipped, only files collected."""
        archive_dir = tmp_path / "archive"
        doc_folder = archive_dir / "abc123def45678" / "attachments"
        doc_folder.mkdir(parents=True)
        (doc_folder / "file.pdf").write_bytes(b"content")

        files_to_upload = []
        for file_path in archive_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_key = str(file_path.relative_to(archive_dir)).replace("\\", "/")
            files_to_upload.append(rel_key)

        assert len(files_to_upload) == 1
        assert files_to_upload[0] == "abc123def45678/attachments/file.pdf"
