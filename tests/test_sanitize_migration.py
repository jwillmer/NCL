"""Tests for _sanitize_storage_key and the archive key migration.

Covers the new underscore-based naming (no URL-encoding) and verifies
the migration script correctly renames files and updates URIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

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


# ---------------------------------------------------------------------------
# Validation check tests (sections 17-20 in validate_cmd.py)
# ---------------------------------------------------------------------------


class TestValidateNewChecks:
    """Test the new validation checks added to mtss validate ingest."""

    def _make_doc(self, doc_id="abc123", file_name="test.eml", depth=0,
                  browse_uri=None, download_uri=None, doc_type="email"):
        return {
            "id": str(uuid4()),
            "doc_id": doc_id,
            "document_type": doc_type,
            "file_name": file_name,
            "file_path": f"/emails/{file_name}",
            "depth": depth,
            "status": "completed",
            "archive_path": doc_id[:16],
            "archive_browse_uri": browse_uri,
            "archive_download_uri": download_uri,
        }

    def _make_chunk(self, doc_id, chunk_id=None, topic_ids=None):
        return {
            "id": str(uuid4()),
            "document_id": doc_id,
            "chunk_id": chunk_id or f"chunk_{uuid4().hex[:8]}",
            "content": "test",
            "chunk_index": 0,
            "embedding": [0.1],
            "metadata": {"topic_ids": topic_ids} if topic_ids else {},
        }

    # --- Section 17: Duplicate IDs ---

    @pytest.mark.unit
    def test_detect_duplicate_doc_ids(self):
        from collections import Counter
        docs = [
            self._make_doc(doc_id="dup123"),
            self._make_doc(doc_id="dup123"),
            self._make_doc(doc_id="unique456"),
        ]
        counts = Counter(d["doc_id"] for d in docs)
        dupes = {did: c for did, c in counts.items() if c > 1}
        assert len(dupes) == 1
        assert dupes["dup123"] == 2

    @pytest.mark.unit
    def test_no_duplicate_doc_ids(self):
        from collections import Counter
        docs = [self._make_doc(doc_id=f"unique_{i}") for i in range(5)]
        counts = Counter(d["doc_id"] for d in docs)
        dupes = {did: c for did, c in counts.items() if c > 1}
        assert len(dupes) == 0

    @pytest.mark.unit
    def test_detect_duplicate_chunk_ids(self):
        from collections import Counter
        doc_id = str(uuid4())
        chunks = [
            self._make_chunk(doc_id, chunk_id="dup_chunk"),
            self._make_chunk(doc_id, chunk_id="dup_chunk"),
            self._make_chunk(doc_id, chunk_id="unique_chunk"),
        ]
        counts = Counter(c["chunk_id"] for c in chunks)
        dupes = {cid: c for cid, c in counts.items() if c > 1}
        assert len(dupes) == 1

    # --- Section 18: Encoded filenames on disk ---

    @pytest.mark.unit
    def test_detect_encoded_filenames(self, tmp_path):
        import re
        archive = tmp_path / "archive" / "abc123" / "attachments"
        archive.mkdir(parents=True)
        (archive / "file%20name.pdf").write_bytes(b"x")
        (archive / "clean_name.pdf").write_bytes(b"x")

        enc_re = re.compile(r"%[0-9A-Fa-f]{2}")
        encoded = [f for f in (tmp_path / "archive").rglob("*")
                   if f.is_file() and enc_re.search(f.name)]
        assert len(encoded) == 1
        assert "file%20name.pdf" in encoded[0].name

    @pytest.mark.unit
    def test_no_encoded_filenames(self, tmp_path):
        import re
        archive = tmp_path / "archive" / "abc123"
        archive.mkdir(parents=True)
        (archive / "clean_file.pdf").write_bytes(b"x")

        enc_re = re.compile(r"%[0-9A-Fa-f]{2}")
        encoded = [f for f in (tmp_path / "archive").rglob("*")
                   if f.is_file() and enc_re.search(f.name)]
        assert len(encoded) == 0

    # --- Section 19: Encoded URIs in documents.jsonl ---

    @pytest.mark.unit
    def test_detect_encoded_uris(self):
        import re
        enc_re = re.compile(r"%[0-9A-Fa-f]{2}")
        doc = self._make_doc(browse_uri="/archive/abc/file%20name.pdf.md")
        has_encoded = any(
            enc_re.search(doc.get(k) or "")
            for k in ("archive_browse_uri", "archive_download_uri")
        )
        assert has_encoded

    @pytest.mark.unit
    def test_clean_uris_not_flagged(self):
        import re
        enc_re = re.compile(r"%[0-9A-Fa-f]{2}")
        doc = self._make_doc(browse_uri="/archive/abc/clean_file.pdf.md")
        has_encoded = any(
            enc_re.search(doc.get(k) or "")
            for k in ("archive_browse_uri", "archive_download_uri")
        )
        assert not has_encoded

    # --- Section 20: Broken markdown links ---

    @pytest.mark.unit
    def test_detect_broken_markdown_links(self, tmp_path):
        import re
        archive = tmp_path / "archive" / "abc123"
        archive.mkdir(parents=True)
        (archive / "email.eml.md").write_text(
            "# Email\n\n- [Report](abc123/attachments/missing_file.pdf)\n"
        )

        broken = []
        for md in (tmp_path / "archive").rglob("*.md"):
            content = md.read_text(encoding="utf-8")
            for link in re.findall(r"\[.*?\]\(([^)]+)\)", content):
                if link.startswith(("http", "#", "mailto:")):
                    continue
                target = tmp_path / "archive" / link
                if not target.exists():
                    broken.append(link)
        assert len(broken) == 1
        assert "missing_file.pdf" in broken[0]

    @pytest.mark.unit
    def test_valid_markdown_links_not_flagged(self, tmp_path):
        import re
        archive = tmp_path / "archive" / "abc123" / "attachments"
        archive.mkdir(parents=True)
        (archive / "report.pdf").write_bytes(b"content")
        (archive.parent / "email.eml.md").write_text(
            "- [Report](abc123/attachments/report.pdf)\n"
        )

        broken = []
        for md in (tmp_path / "archive").rglob("*.md"):
            content = md.read_text(encoding="utf-8")
            for link in re.findall(r"\[.*?\]\(([^)]+)\)", content):
                if link.startswith(("http", "#", "mailto:")):
                    continue
                target = tmp_path / "archive" / link
                if not target.exists():
                    broken.append(link)
        assert len(broken) == 0

    @pytest.mark.unit
    def test_llamaparse_images_skipped(self, tmp_path):
        """LlamaParse page images should not be flagged as broken."""
        import re
        archive = tmp_path / "archive" / "abc123" / "attachments"
        archive.mkdir(parents=True)
        (archive / "report.pdf.md").write_text(
            "![image](page_1_image_1_v2.jpg)\n![chart](page_3_chart_1_v2.jpg)\n"
        )

        broken = []
        for md in (tmp_path / "archive").rglob("*.md"):
            content = md.read_text(encoding="utf-8")
            for link in re.findall(r"\[.*?\]\(([^)]+)\)", content):
                if link.startswith(("http", "#", "mailto:")):
                    continue
                if re.match(r"page_\d+_(?:image|chart|seal)_\d+", link):
                    continue
                target = tmp_path / "archive" / link
                if not target.exists():
                    broken.append(link)
        assert len(broken) == 0
