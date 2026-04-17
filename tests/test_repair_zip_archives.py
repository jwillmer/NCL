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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture
def sample_output_dir(tmp_path):
    """Build a fake output dir with one ZIP-member doc needing repair."""
    output_dir = tmp_path / "output"
    archive_dir = output_dir / "archive"

    # IDs
    email_uuid = str(uuid4())
    zip_uuid = str(uuid4())
    member_uuid = str(uuid4())

    email_doc_id = "a" * 32  # folder_id[:16] = "aaaaaaaaaaaaaaaa"
    zip_doc_id = "b" * 32
    member_doc_id = "c" * 32

    folder_id = email_doc_id[:16]

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
        docs_path = out / "documents.jsonl"
        docs = [json.loads(l) for l in docs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for d in docs:
            if d["id"] == sample_output_dir["member_uuid"]:
                d["archive_browse_uri"] = "/archive/x/y.md"
                d["archive_download_uri"] = "/archive/x/y"
        _write_jsonl(docs_path, docs)

        candidates, _ = mod.build_candidates(out, verbose=False)
        assert candidates == []

    def test_candidate_scan_skips_image_attachments(self, sample_output_dir):
        """Image attachments never had .md previews — not this migration's job."""
        mod = _load_repair_module()
        out = sample_output_dir["output_dir"]
        docs_path = out / "documents.jsonl"
        docs = [json.loads(l) for l in docs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for d in docs:
            if d["id"] == sample_output_dir["member_uuid"]:
                d["document_type"] = "attachment_image"
        _write_jsonl(docs_path, docs)

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

        # documents.jsonl must not have been touched
        docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
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

        # 3. documents.jsonl row updated with the new URIs
        docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        member = next(d for d in docs if d["id"] == sample_output_dir["member_uuid"])
        assert member["archive_browse_uri"] == f"/archive/{folder_id}/attachments/report.docx.md"
        assert member["archive_download_uri"] == f"/archive/{folder_id}/attachments/report.docx"

        # 4. Other docs in the file were not mangled (email + zip rows preserved)
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

        docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        member = next(d for d in docs if d["id"] == sample_output_dir["member_uuid"])
        assert member["archive_browse_uri"] is None
        assert member["archive_download_uri"] is None
