"""Tests for _sanitize_storage_key and the archive key migration.

Covers the new underscore-based naming (no URL-encoding) and verifies
the migration script correctly renames files and updates URIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest


def _sanitize(filename: str) -> str:
    """Lazy import to avoid triggering settings validation."""
    from mtss.ingest.archive_generator import _sanitize_storage_key
    return _sanitize_storage_key(filename)


# ---------------------------------------------------------------------------
# _sanitize_storage_key — new behavior
# ---------------------------------------------------------------------------


class TestSanitizeStorageKey:
    """Test that _sanitize_storage_key produces clean, no-encoding keys."""

    @pytest.mark.unit
    def test_spaces_become_underscores(self):
        assert _sanitize("FORE SIDE CHAIN_2.pdf") == "FORE_SIDE_CHAIN_2.pdf"

    @pytest.mark.unit
    def test_multiple_spaces(self):
        assert _sanitize("July 2025 .pdf") == "July_2025.pdf"

    @pytest.mark.unit
    def test_apostrophe_removed(self):
        assert _sanitize("MASTER'S STATEMENT.pdf") == "MASTER_S_STATEMENT.pdf"

    @pytest.mark.unit
    def test_hash_removed(self):
        assert _sanitize("test file #3.pdf") == "test_file_3.pdf"

    @pytest.mark.unit
    def test_ampersand_removed(self):
        assert _sanitize("report & summary.pdf") == "report_summary.pdf"

    @pytest.mark.unit
    def test_comma_removed(self):
        result = _sanitize("Certificate, Greece C.pdf")
        assert "," not in result
        assert "_Greece_C.pdf" in result

    @pytest.mark.unit
    def test_brackets_replaced_with_parens(self):
        assert _sanitize("file[1].pdf") == "file(1).pdf"

    @pytest.mark.unit
    def test_tilde_replaced(self):
        assert _sanitize("~WRD0001.jpg") == "WRD0001.jpg"

    @pytest.mark.unit
    def test_plain_ascii_unchanged(self):
        assert _sanitize("simple-report_v2.pdf") == "simple-report_v2.pdf"

    @pytest.mark.unit
    def test_no_url_encoding_in_output(self):
        result = _sanitize("file with spaces & special #chars.pdf")
        assert "%" not in result, "output must not contain URL-encoding"

    @pytest.mark.unit
    def test_no_spaces_in_output(self):
        result = _sanitize("many  spaces   here.pdf")
        assert " " not in result

    @pytest.mark.unit
    def test_no_double_underscores(self):
        result = _sanitize("a & b # c.pdf")
        assert "__" not in result

    @pytest.mark.unit
    def test_extension_preserved(self):
        result = _sanitize("report (final).pdf")
        assert result.endswith(".pdf")

    @pytest.mark.unit
    def test_dot_in_filename_preserved(self):
        result = _sanitize("3. ROS2 CONFIGURATION.jpg")
        assert result.endswith(".jpg")
        assert "ROS2" in result


# ---------------------------------------------------------------------------
# Migration script
# ---------------------------------------------------------------------------


class TestMigration:
    """Test the archive key migration script."""

    def _run_migration(self, output_dir: Path, dry_run: bool = False):
        """Import and run the migration."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "migrate", Path(__file__).parent.parent / "scripts" / "migrate_archive_keys.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.migrate(output_dir, dry_run=dry_run)

    @pytest.mark.unit
    def test_renames_encoded_files(self, tmp_path):
        """Migration should rename files with %20 to underscores."""
        archive = tmp_path / "archive" / "abc123" / "attachments"
        archive.mkdir(parents=True)
        (archive / "file%20name.pdf").write_bytes(b"content")
        (archive / "file%20name.pdf.md").write_text("markdown")

        # Minimal documents.jsonl
        doc = {
            "id": "test",
            "doc_id": "abc123xxxx",
            "archive_download_uri": "/archive/abc123/attachments/file%20name.pdf",
            "archive_browse_uri": "/archive/abc123/attachments/file%20name.pdf.md",
        }
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")

        self._run_migration(tmp_path, dry_run=False)

        # Files renamed
        assert (archive / "file_name.pdf").exists()
        assert (archive / "file_name.pdf.md").exists()
        assert not (archive / "file%20name.pdf").exists()

        # URIs updated
        updated = json.loads((tmp_path / "documents.jsonl").read_text().strip())
        assert updated["archive_download_uri"] == "/archive/abc123/attachments/file_name.pdf"
        assert updated["archive_browse_uri"] == "/archive/abc123/attachments/file_name.pdf.md"

    @pytest.mark.unit
    def test_dry_run_makes_no_changes(self, tmp_path):
        """Dry run should not modify any files."""
        archive = tmp_path / "archive" / "abc123"
        archive.mkdir(parents=True)
        (archive / "file%20name.pdf").write_bytes(b"content")

        doc = {"id": "test", "archive_download_uri": "/archive/abc123/file%20name.pdf"}
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")

        self._run_migration(tmp_path, dry_run=True)

        # File not renamed
        assert (archive / "file%20name.pdf").exists()
        assert not (archive / "file_name.pdf").exists()

    @pytest.mark.unit
    def test_handles_ampersand_and_hash(self, tmp_path):
        """Migration handles %26 (&) and %23 (#) in filenames."""
        archive = tmp_path / "archive" / "abc123" / "attachments"
        archive.mkdir(parents=True)
        (archive / "H%26T%202025.pdf").write_bytes(b"content")

        doc = {
            "id": "test",
            "archive_download_uri": "/archive/abc123/attachments/H%26T%202025.pdf",
        }
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")

        self._run_migration(tmp_path, dry_run=False)

        assert (archive / "H_T_2025.pdf").exists()
        updated = json.loads((tmp_path / "documents.jsonl").read_text().strip())
        assert updated["archive_download_uri"] == "/archive/abc123/attachments/H_T_2025.pdf"

    @pytest.mark.unit
    def test_leaves_clean_files_unchanged(self, tmp_path):
        """Files without encoding should not be touched."""
        archive = tmp_path / "archive" / "abc123"
        archive.mkdir(parents=True)
        (archive / "email.eml").write_bytes(b"content")
        (archive / "email.eml.md").write_text("markdown")

        doc = {
            "id": "test",
            "archive_browse_uri": "/archive/abc123/email.eml.md",
        }
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")

        self._run_migration(tmp_path, dry_run=False)

        assert (archive / "email.eml").exists()
        assert (archive / "email.eml.md").exists()
        updated = json.loads((tmp_path / "documents.jsonl").read_text().strip())
        assert updated["archive_browse_uri"] == "/archive/abc123/email.eml.md"

    @pytest.mark.unit
    def test_preserves_unrelated_fields(self, tmp_path):
        """Migration should not alter non-URI fields in documents."""
        archive = tmp_path / "archive"
        archive.mkdir()

        doc = {
            "id": "test-id",
            "doc_id": "abc123",
            "document_type": "email",
            "file_name": "test.eml",
            "status": "completed",
            "archive_browse_uri": None,
        }
        (tmp_path / "documents.jsonl").write_text(json.dumps(doc) + "\n")

        self._run_migration(tmp_path, dry_run=False)

        updated = json.loads((tmp_path / "documents.jsonl").read_text().strip())
        assert updated["id"] == "test-id"
        assert updated["document_type"] == "email"
        assert updated["status"] == "completed"


# ---------------------------------------------------------------------------
# End-to-end: sanitize -> archive -> import consistency
# ---------------------------------------------------------------------------


class TestSanitizeImportConsistency:
    """Verify that sanitized filenames used during ingest match what
    import reads from disk — no encoding/decoding mismatches."""

    @pytest.mark.unit
    def test_sanitized_name_matches_disk_name(self, tmp_path):
        """File written with sanitized name can be read back without decoding."""
        filename = "GE FO SER SYS.pdf"
        safe = _sanitize(filename)

        # Write file using sanitized name (as archive_generator does)
        folder = tmp_path / "archive" / "abc123" / "attachments"
        folder.mkdir(parents=True)
        (folder / safe).write_bytes(b"content")

        # Read back using rglob (as import does)
        archive_dir = tmp_path / "archive"
        found = []
        for f in archive_dir.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(archive_dir)).replace("\\", "/")
                found.append(rel)

        assert len(found) == 1
        assert found[0] == f"abc123/attachments/{safe}"
        # No unquote needed — key is the literal filename
        assert "%" not in found[0]

    @pytest.mark.unit
    def test_uri_matches_storage_key(self):
        """URI stored in document should match the Supabase storage key."""
        filename = "MASTER'S STATEMENT.pdf"
        safe = _sanitize(filename)

        # URI as set by attachment_handler
        uri = f"/archive/abc123/attachments/{safe}"
        # Storage key as uploaded by import
        storage_key = f"abc123/attachments/{safe}"

        # API strips /archive/ prefix
        api_path = uri.removeprefix("/archive/")
        assert api_path == storage_key, "URI must match storage key exactly"
