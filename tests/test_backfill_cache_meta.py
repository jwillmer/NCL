"""Tests for scripts/backfill_cache_meta.py — dry-run + apply paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.utils import compute_folder_id  # noqa: E402

_EMAIL_DOC_ID = "stable1234567890abc"
_FOLDER_ID = compute_folder_id(_EMAIL_DOC_ID)


@pytest.fixture
def seeded_archive(tmp_path):
    """An output dir with one email folder, one cached .md without sidecar,
    and a documents.jsonl + metadata.json trail entry pointing to local_text.
    """
    output = tmp_path / "output"
    archive = output / "archive"
    folder = archive / _FOLDER_ID
    attachments = folder / "attachments"
    attachments.mkdir(parents=True)

    # The cached .md with no sidecar yet.
    (attachments / "report.pdf.md").write_text(
        "# header\n\n## Content\nparsed content",
        encoding="utf-8",
    )

    # metadata.json in the folder — the processing trail.
    metadata = {
        "processing_trail": {
            "attachments": {
                "report.pdf": {
                    "steps": {
                        "parse": {"parser": "local_pdf", "model": None}
                    }
                }
            }
        }
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    # Seed ingest.db with one email doc + one attachment doc.
    from mtss.storage.sqlite_client import SqliteStorageClient

    client = SqliteStorageClient(output_dir=output)
    try:
        conn = client._conn
        now = "2026-04-20T00:00:00"
        docs = [
            ("11111111-1111-1111-1111-111111111111", _EMAIL_DOC_ID, "email", "source.eml", None),
            ("22222222-2222-2222-2222-222222222222", "attdoc1234567890", "attachment_pdf",
             "report.pdf", _EMAIL_DOC_ID),
        ]
        for uid, doc_id, dtype, fname, parent_id in docs:
            conn.execute(
                "INSERT INTO documents("
                "id, doc_id, source_id, document_type, status, file_name, "
                "parent_id, root_id, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, doc_id, fname, dtype, "completed", fname,
                 ("11111111-1111-1111-1111-111111111111" if parent_id else None),
                 "11111111-1111-1111-1111-111111111111",
                 now, now),
            )
    finally:
        conn.close()

    return output, attachments / "report.pdf.md", attachments / "report.pdf.meta.json"


def test_dry_run_does_not_write_sidecar(seeded_archive):
    """Default (dry-run) must never write files under data/output/."""
    output, md_path, meta_path = seeded_archive
    result = subprocess.run(
        [sys.executable, "scripts/backfill_cache_meta.py", "--output-dir", str(output)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert not meta_path.exists(), "dry-run must not create sidecar"
    assert "Would write 1 sidecars" in result.stdout
    assert "Dry-run complete" in result.stdout


def test_apply_writes_sidecar_from_trail(seeded_archive):
    """--apply writes the sidecar with parser info inferred from the trail."""
    output, md_path, meta_path = seeded_archive
    result = subprocess.run(
        [sys.executable, "scripts/backfill_cache_meta.py", "--output-dir", str(output), "--apply"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["parser"] == "local_pdf"
    assert data["backfilled"] is True


def test_existing_sidecar_preserved(seeded_archive):
    """Backfill must not overwrite a sidecar that already exists."""
    output, md_path, meta_path = seeded_archive
    meta_path.write_text(
        json.dumps({"parser": "gemini_pdf", "model": "x", "parsed_at": "pre-existing"}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/backfill_cache_meta.py", "--output-dir", str(output), "--apply"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["parser"] == "gemini_pdf"
    assert data["parsed_at"] == "pre-existing"
