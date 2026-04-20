"""Tests for scripts/repair_zip_archives.py retroactive migration.

Builds a miniature `data/output/` layout matching the live shape and
asserts the migration reconstructs ZIP-member archive originals and
`.md` previews without touching chunks or re-parsing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_repair_module():
    """Load scripts/repair_zip_archives.py as a module (not on sys.path by default)."""
    spec = importlib.util.spec_from_file_location(
        "repair_zip_archives",
        REPO_ROOT / "scripts" / "repair_zip_archives.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["repair_zip_archives"] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Seed SQLite ingest.db with the given rows.

    Route by filename stem: ``documents.jsonl`` → documents table,
    ``chunks.jsonl`` → chunks table. Keeps the legacy helper name so the
    existing tests don't churn — the JSONL files themselves are gone.
    """
    from mtss.storage.sqlite_client import SqliteStorageClient

    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    client = SqliteStorageClient(output_dir=output_dir)
    try:
        conn = client._conn
        now = "2026-04-20T00:00:00"
        if path.stem == "documents":
            with conn:
                conn.execute("BEGIN")
                # Re-seed idempotently so tests that rewrite the same path work.
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM documents")
                for r in rows:
                    conn.execute(
                        "INSERT INTO documents("
                        "id, doc_id, source_id, document_type, status, "
                        "file_hash, file_name, file_path, parent_id, root_id, "
                        "depth, content_version, ingest_version, archive_path, "
                        "title, source_title, mime_type, content_type, size_bytes, "
                        "embedding_mode, archive_browse_uri, archive_download_uri, "
                        "metadata_json, processed_at, created_at, updated_at"
                        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            r["id"],
                            r.get("doc_id") or r["id"][:16],
                            r.get("source_id") or r.get("file_name") or r["id"][:16],
                            r.get("document_type") or "attachment_other",
                            r.get("status") or "pending",
                            r.get("file_hash"), r.get("file_name"), r.get("file_path"),
                            r.get("parent_id"), r.get("root_id") or r["id"],
                            r.get("depth", 0), r.get("content_version", 1),
                            r.get("ingest_version", 1), r.get("archive_path"),
                            r.get("title"), r.get("source_title"),
                            r.get("mime_type"), r.get("content_type"),
                            r.get("size_bytes"), r.get("embedding_mode"),
                            r.get("archive_browse_uri"), r.get("archive_download_uri"),
                            None, r.get("processed_at"),
                            r.get("created_at") or now, r.get("updated_at") or now,
                        ),
                    )
        elif path.stem == "chunks":
            with conn:
                conn.execute("BEGIN")
                for r in rows:
                    conn.execute(
                        "INSERT INTO chunks("
                        "id, chunk_id, document_id, content, chunk_index, created_at"
                        ") VALUES (?,?,?,?,?,?)",
                        (
                            r["id"], r.get("chunk_id") or r["id"],
                            r["document_id"], r.get("content") or "",
                            r.get("chunk_index", 0), now,
                        ),
                    )
    finally:
        conn.close()


def _docs_from_db(output_dir: Path) -> list[dict]:
    """Read all documents from ingest.db as plain dicts (matches old JSONL shape
    the repair tests inspect — archive_browse_uri / archive_download_uri etc.)."""
    from mtss.storage.sqlite_client import SqliteStorageClient

    client = SqliteStorageClient(output_dir=output_dir)
    try:
        return list(client.iter_documents())
    finally:
        try:
            client._conn.close()
        except Exception:
            pass


@pytest.fixture
def sample_output_dir(tmp_path):
    """Build a fake output dir with one ZIP-member doc needing repair."""
    output_dir = tmp_path / "output"
    archive_dir = output_dir / "archive"

    # IDs
    email_uuid = str(uuid4())
    zip_uuid = str(uuid4())
    member_uuid = str(uuid4())

    from mtss.utils import compute_folder_id

    email_doc_id = "a" * 32
    zip_doc_id = "b" * 32
    member_doc_id = "c" * 32

    folder_id = compute_folder_id(email_doc_id)

    # On-disk ZIP containing one docx-like file
    folder = archive_dir / folder_id / "attachments"
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "container.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Member nested one level deep — the script should match by basename
        zf.writestr("nested/report.docx", b"DOCX BINARY BYTES HERE")

    docs = [
        {
            "id": email_uuid,
            "doc_id": email_doc_id,
            "document_type": "email",
            "file_name": "src.eml",
            "source_id": "src.eml",
            "depth": 0,
            "root_id": email_uuid,
            "parent_id": None,
            "status": "completed",
            "archive_browse_uri": f"/archive/{folder_id}/email.eml.md",
            "archive_download_uri": f"/archive/{folder_id}/email.eml",
        },
        {
            "id": zip_uuid,
            "doc_id": zip_doc_id,
            "document_type": "attachment_other",
            "file_name": "container.zip",
            "source_id": "src.eml/container.zip",
            "depth": 1,
            "root_id": email_uuid,
            "parent_id": email_uuid,
            "status": "completed",
            "attachment_metadata": {"content_type": "application/zip"},
            "archive_browse_uri": None,  # ZIPs themselves get no .md
            "archive_download_uri": f"/archive/{folder_id}/attachments/container.zip",
        },
        {
            # The ZIP-extracted member — the doc this script fixes.
            "id": member_uuid,
            "doc_id": member_doc_id,
            "document_type": "attachment_docx",
            "file_name": "report.docx",
            "source_id": "src.eml/container.zip/report.docx",
            "depth": 2,
            "root_id": email_uuid,
            "parent_id": zip_uuid,
            "status": "completed",
            "attachment_metadata": {
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            },
            "archive_browse_uri": None,
            "archive_download_uri": None,
        },
    ]
    _write_jsonl(output_dir / "documents.jsonl", docs)

    chunks = [
        {
            "id": str(uuid4()),
            "document_id": member_uuid,
            "chunk_id": "c1",
            "content": "Heading one\n\nFirst section of parsed content.",
            "chunk_index": 0,
        },
        {
            "id": str(uuid4()),
            "document_id": member_uuid,
            "chunk_id": "c2",
            "content": "Second section with more detail.",
            "chunk_index": 1,
        },
    ]
    _write_jsonl(output_dir / "chunks.jsonl", chunks)

    return {
        "output_dir": output_dir,
        "folder_id": folder_id,
        "member_uuid": member_uuid,
    }


class TestRepairZipArchivesDetection:
    def test_candidate_scan_finds_zip_member_missing_uris(self, sample_output_dir):
        mod = _load_repair_module()
        candidates, docs = mod.build_candidates(
            sample_output_dir["output_dir"], verbose=False
        )
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.member_basename == "report.docx"
        assert cand.safe_member_name == "report.docx"
        assert cand.folder_id == sample_output_dir["folder_id"]
        assert "First section" in cand.parsed_content
        assert "Second section" in cand.parsed_content

    def test_candidate_scan_skips_doc_with_uris_already(self, sample_output_dir, tmp_path):
        """A doc that already has archive URIs must not be re-flagged."""
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        from mtss.storage.sqlite_client import SqliteStorageClient

        client = SqliteStorageClient(output_dir=out)
        try:
            client._conn.execute(
                "UPDATE documents SET archive_browse_uri = ?, archive_download_uri = ? "
                "WHERE id = ?",
                ("/archive/x/y.md", "/archive/x/y", sample_output_dir["member_uuid"]),
            )
        finally:
            client._conn.close()

        candidates, _ = mod.build_candidates(out, verbose=False)
        assert candidates == []

    def test_candidate_scan_skips_image_attachments(self, sample_output_dir):
        """Image attachments never had .md previews — not this migration's job."""
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        from mtss.storage.sqlite_client import SqliteStorageClient

        client = SqliteStorageClient(output_dir=out)
        try:
            client._conn.execute(
                "UPDATE documents SET document_type = 'attachment_image' WHERE id = ?",
                (sample_output_dir["member_uuid"],),
            )
        finally:
            client._conn.close()

        candidates, _ = mod.build_candidates(out, verbose=False)
        assert candidates == []


class TestRepairZipArchivesExecution:
    def test_dry_run_makes_no_changes_to_disk(self, sample_output_dir):
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        folder_id = sample_output_dir["folder_id"]
        member_path = out / "archive" / folder_id / "attachments" / "report.docx"
        md_path = out / "archive" / folder_id / "attachments" / "report.docx.md"

        assert not member_path.exists()
        assert not md_path.exists()

        rc = mod.repair(out, dry_run=True, limit=0, verbose=False)
        assert rc == 0
        assert not member_path.exists()
        assert not md_path.exists()

        # documents row must not have been touched
        docs = _docs_from_db(out)
        member = next(d for d in docs if d["id"] == sample_output_dir["member_uuid"])
        assert member["archive_browse_uri"] is None
        assert member["archive_download_uri"] is None

    def test_live_run_writes_original_md_and_updates_doc(self, sample_output_dir):
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        folder_id = sample_output_dir["folder_id"]
        member_path = out / "archive" / folder_id / "attachments" / "report.docx"
        md_path = out / "archive" / folder_id / "attachments" / "report.docx.md"

        rc = mod.repair(out, dry_run=False, limit=0, verbose=False)
        assert rc == 0

        # 1. Original bytes extracted from the ZIP and written
        assert member_path.exists()
        assert member_path.read_bytes() == b"DOCX BINARY BYTES HERE"

        # 2. .md preview written with reconstructed content
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert "# report.docx" in md_text
        assert "First section of parsed content." in md_text
        assert "Second section with more detail." in md_text
        assert f"[Download Original]({folder_id}/attachments/report.docx)" in md_text

        # 3. documents row updated with the new URIs
        docs = _docs_from_db(out)
        member = next(d for d in docs if d["id"] == sample_output_dir["member_uuid"])
        assert member["archive_browse_uri"] == f"/archive/{folder_id}/attachments/report.docx.md"
        assert member["archive_download_uri"] == f"/archive/{folder_id}/attachments/report.docx"

        # 4. Other docs were not mangled (email + zip rows preserved)
        assert any(d["document_type"] == "email" and d["archive_browse_uri"] for d in docs)
        zip_row = next(d for d in docs if d["file_name"] == "container.zip")
        assert zip_row["archive_download_uri"] == f"/archive/{folder_id}/attachments/container.zip"

    def test_live_run_is_idempotent(self, sample_output_dir):
        """Re-running after success must be a no-op (candidates list becomes empty)."""
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]

        mod.repair(out, dry_run=False, limit=0, verbose=False)
        # After first run, doc has URIs set, so candidate scan returns [].
        candidates, _ = mod.build_candidates(out, verbose=False)
        assert candidates == []

    def test_live_run_skips_when_member_not_in_zip(self, sample_output_dir):
        """If the ZIP doesn't contain the expected member, skip without error
        and do not update the doc's URIs."""
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        folder_id = sample_output_dir["folder_id"]
        zip_path = out / "archive" / folder_id / "attachments" / "container.zip"
        # Rewrite zip without the expected member
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("other.txt", b"nope")

        rc = mod.repair(out, dry_run=False, limit=0, verbose=False)
        assert rc == 0

        docs = _docs_from_db(out)
        member = next(d for d in docs if d["id"] == sample_output_dir["member_uuid"])
        assert member["archive_browse_uri"] is None
        assert member["archive_download_uri"] is None


class TestRepairDetectsRealPipelineParentage:
    """Regression: the live pipeline's process_zip_attachment builds
    extracted-member docs with parent_id = email (not the ZIP doc).
    Detection must match docs via sibling ZIPs under the same root_id,
    not via parent_id equality.
    """

    def test_member_with_parent_id_eq_email_still_detected(self, tmp_path):
        mod = _load_repair_module()
        output_dir = tmp_path / "output"
        archive_dir = output_dir / "archive"

        email_uuid = str(uuid4())
        zip_uuid = str(uuid4())
        member_uuid = str(uuid4())

        from mtss.utils import compute_folder_id
        email_doc_id = "e" * 32
        folder_id = compute_folder_id(email_doc_id)

        folder = archive_dir / folder_id / "attachments"
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / "bundle.zip", "w") as zf:
            zf.writestr("report.pdf", b"PDF BYTES")

        docs = [
            {
                "id": email_uuid,
                "doc_id": email_doc_id,
                "document_type": "email",
                "file_name": "src.eml",
                "source_id": "src.eml",
                "root_id": email_uuid,
                "parent_id": None,
                "status": "completed",
                "archive_browse_uri": f"/archive/{folder_id}/email.eml.md",
                "archive_download_uri": f"/archive/{folder_id}/email.eml",
            },
            {
                "id": zip_uuid,
                "doc_id": "z" * 32,
                "document_type": "attachment_other",
                "file_name": "bundle.zip",
                "source_id": "src.eml/bundle.zip",
                "root_id": email_uuid,
                # NB: ZIP-member and ZIP both have parent_id = email.
                "parent_id": email_uuid,
                "status": "completed",
                "attachment_metadata": {"content_type": "application/zip"},
                "archive_browse_uri": None,
                "archive_download_uri": f"/archive/{folder_id}/attachments/bundle.zip",
            },
            {
                "id": member_uuid,
                "doc_id": "m" * 32,
                "document_type": "attachment_pdf",
                "file_name": "report.pdf",
                "source_id": "src.eml/bundle.zip/report.pdf",
                "root_id": email_uuid,
                # THE CRITICAL BIT: parent_id is the EMAIL, not the ZIP doc.
                "parent_id": email_uuid,
                "status": "completed",
                "attachment_metadata": {"content_type": "application/pdf"},
                "archive_browse_uri": None,
                "archive_download_uri": None,
            },
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)

        chunks = [
            {
                "id": str(uuid4()),
                "document_id": member_uuid,
                "chunk_id": "c1",
                "content": "Reconstructed content.",
                "chunk_index": 0,
            },
        ]
        _write_jsonl(output_dir / "chunks.jsonl", chunks)

        candidates, _ = mod.build_candidates(output_dir, verbose=False)
        assert len(candidates) == 1
        assert candidates[0].member_basename == "report.pdf"
        assert candidates[0].source_zip_path.name == "bundle.zip"
        assert candidates[0].folder_id == folder_id

    def test_member_detected_even_when_zip_has_no_doc_row(self, tmp_path):
        """Live pipeline reality: process_zip_attachment never creates a
        doc row for the ZIP itself. Detection must find the source ZIP
        by enumerating the archive folder on disk, not by looking up
        sibling doc rows.
        """
        mod = _load_repair_module()
        output_dir = tmp_path / "output"
        archive_dir = output_dir / "archive"

        email_uuid = str(uuid4())
        member_uuid = str(uuid4())
        from mtss.utils import compute_folder_id
        email_doc_id = "f" * 32
        folder_id = compute_folder_id(email_doc_id)

        folder = archive_dir / folder_id / "attachments"
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / "only-on-disk.zip", "w") as zf:
            zf.writestr("INCINERATOR FLUE GAS FAN.docx", b"DOCX")

        docs = [
            {
                "id": email_uuid,
                "doc_id": email_doc_id,
                "document_type": "email",
                "file_name": "src.eml",
                "source_id": "src.eml",
                "root_id": email_uuid,
                "parent_id": None,
                "status": "completed",
                "archive_browse_uri": f"/archive/{folder_id}/email.eml.md",
                "archive_download_uri": f"/archive/{folder_id}/email.eml",
            },
            # No ZIP doc row — matches live pipeline.
            {
                "id": member_uuid,
                "doc_id": "m" * 32,
                "document_type": "attachment_docx",
                "file_name": "INCINERATOR FLUE GAS FAN.docx",
                "source_id": "src.eml/only-on-disk.zip/incinerator flue gas fan.docx",
                "root_id": email_uuid,
                "parent_id": email_uuid,
                "status": "completed",
                "attachment_metadata": {
                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
                "archive_browse_uri": None,
                "archive_download_uri": None,
            },
        ]
        _write_jsonl(output_dir / "documents.jsonl", docs)
        _write_jsonl(
            output_dir / "chunks.jsonl",
            [{
                "id": str(uuid4()),
                "document_id": member_uuid,
                "chunk_id": "c1",
                "content": "Body content for the extracted docx.",
                "chunk_index": 0,
            }],
        )

        candidates, _ = mod.build_candidates(output_dir, verbose=False)
        assert len(candidates) == 1
        assert candidates[0].source_zip_path.name == "only-on-disk.zip"
