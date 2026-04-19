"""Tests for scripts/backfill_cache_meta.py — dry-run + apply paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def seeded_archive(tmp_path):
    """An output dir with one email folder, one cached .md without sidecar,
    and a documents.jsonl + metadata.json trail entry pointing to local_text.
    """
    output = tmp_path / "output"
    archive = output / "archive"
    folder = archive / "stable1234567890"
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

    # documents.jsonl: one email doc + one attachment doc.
    docs = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "doc_id": "stable1234567890abc",
            "document_type": "email",
            "file_name": "source.eml",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "doc_id": "attdoc1234567890",
            "parent_id": "stable1234567890abc",
            "document_type": "attachment_pdf",
            "file_name": "report.pdf",
        },
    ]
    (output / "documents.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n",
        encoding="utf-8",
    )

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
