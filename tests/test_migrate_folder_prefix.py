"""Tests for scripts/migrate_folder_prefix.py — dry-run + apply paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the script's helpers importable for direct-call unit tests.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from migrate_folder_prefix import (  # noqa: E402
    FolderPlan,
    _rewrite_uri,
    apply_migration,
    plan_migration,
)
from mtss.utils import compute_folder_id  # noqa: E402


@pytest.fixture
def seeded_archive(tmp_path):
    """An output dir with one email folder named with the old 16-char prefix."""
    output = tmp_path / "output"
    archive = output / "archive"

    email_doc_id = "abcdef0123456789" + "0" * 16  # 32-char doc_id stub
    old_folder = email_doc_id[:16]
    new_folder = compute_folder_id(email_doc_id)

    folder = archive / old_folder
    (folder / "attachments").mkdir(parents=True)
    (folder / "email.md").write_text("# email\n\ncontent", encoding="utf-8")
    (folder / "attachments" / "report.pdf.md").write_text(
        "# report", encoding="utf-8"
    )
    (folder / "metadata.json").write_text(
        json.dumps({"folder_id": old_folder, "email_doc_id": email_doc_id}),
        encoding="utf-8",
    )

    docs = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "doc_id": email_doc_id,
            "document_type": "email",
            "archive_browse_uri": f"/archive/{old_folder}/email.md",
            "archive_download_uri": f"/archive/{old_folder}/email",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "doc_id": "attdoc1234567890abcdef0000000000",
            "parent_id": email_doc_id,
            "document_type": "attachment_pdf",
            "archive_browse_uri": f"/archive/{old_folder}/attachments/report.pdf.md",
            "archive_download_uri": f"/archive/{old_folder}/attachments/report.pdf",
        },
    ]
    (output / "documents.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8"
    )

    chunks = [
        {
            "chunk_id": "c0000000001",
            "doc_id": email_doc_id,
            "archive_browse_uri": f"/archive/{old_folder}/email.md",
            "archive_download_uri": f"/archive/{old_folder}/email",
        },
        {
            "chunk_id": "c0000000002",
            "doc_id": "attdoc1234567890abcdef0000000000",
            "archive_browse_uri": f"/archive/{old_folder}/attachments/report.pdf.md",
            "archive_download_uri": f"/archive/{old_folder}/attachments/report.pdf",
        },
    ]
    (output / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )

    return output, old_folder, new_folder, email_doc_id


def test_rewrite_uri_exact_prefix():
    assert _rewrite_uri("/archive/aaaaaaaaaaaaaaaa/x.md", "aaaaaaaaaaaaaaaa", "bb" * 16) == (
        "/archive/" + "bb" * 16 + "/x.md"
    )


def test_rewrite_uri_no_match_unchanged():
    assert _rewrite_uri("/archive/other/x.md", "aaaaaaaaaaaaaaaa", "bb" * 16) == (
        "/archive/other/x.md"
    )


def test_rewrite_uri_bare_folder():
    assert _rewrite_uri("/archive/aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "bb" * 16) == (
        "/archive/" + "bb" * 16
    )


def test_rewrite_uri_non_string_unchanged():
    assert _rewrite_uri(None, "a", "b") is None


def test_plan_migration_identifies_folder(seeded_archive):
    output, old_folder, new_folder, email_doc_id = seeded_archive
    result = plan_migration(output)
    assert len(result.plans) == 1
    plan = result.plans[0]
    assert plan.old_folder == old_folder
    assert plan.new_folder == new_folder
    assert plan.email_doc_id == email_doc_id
    # Two URI fields (browse + download) across two docs / two chunks.
    assert plan.doc_rewrites == 4
    assert plan.chunk_rewrites == 4


def test_dry_run_does_not_touch_data(seeded_archive):
    """Default (dry-run) must never create/rename anything."""
    output, old_folder, new_folder, _ = seeded_archive
    result = subprocess.run(
        [sys.executable, "scripts/migrate_folder_prefix.py", "--output-dir", str(output)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "archive" / old_folder).exists(), "dry-run must not rename"
    assert not (output / "archive" / new_folder).exists(), "dry-run must not create target"


def test_apply_renames_folder_and_rewrites_uris(seeded_archive):
    output, old_folder, new_folder, _ = seeded_archive
    result = plan_migration(output)
    apply_migration(output, result)

    # Folder renamed.
    assert not (output / "archive" / old_folder).exists()
    assert (output / "archive" / new_folder).exists()
    # Nested files preserved.
    assert (output / "archive" / new_folder / "email.md").exists()
    assert (output / "archive" / new_folder / "attachments" / "report.pdf.md").exists()

    # metadata.json folder_id updated.
    meta = json.loads(
        (output / "archive" / new_folder / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["folder_id"] == new_folder

    # URIs rewritten everywhere.
    for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        assert old_folder not in (doc.get("archive_browse_uri") or "")
        assert old_folder not in (doc.get("archive_download_uri") or "")
        assert new_folder in doc["archive_browse_uri"]
    for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        assert old_folder not in (chunk.get("archive_browse_uri") or "")
        assert new_folder in chunk["archive_browse_uri"]


def test_plan_skips_already_migrated_folder(tmp_path):
    """Folders already 32 chars are left alone (no double-migration)."""
    output = tmp_path / "output"
    archive = output / "archive"
    already = archive / ("c" * 32)
    already.mkdir(parents=True)
    (output / "documents.jsonl").write_text("", encoding="utf-8")
    result = plan_migration(output)
    assert result.plans == []


def test_plan_reports_unmatched_folders(tmp_path):
    """An archive folder with no matching email doc is flagged, not migrated."""
    output = tmp_path / "output"
    archive = output / "archive"
    (archive / "ffffffffffffffff").mkdir(parents=True)
    (output / "documents.jsonl").write_text("", encoding="utf-8")
    result = plan_migration(output)
    assert result.plans == []
    assert "ffffffffffffffff" in result.skipped_no_match


def test_plan_skips_when_target_already_exists(tmp_path):
    """If new_folder already exists (partial prior migration), skip."""
    output = tmp_path / "output"
    archive = output / "archive"

    email_doc_id = "deadbeef01234567" + "0" * 16
    old_folder = email_doc_id[:16]
    new_folder = compute_folder_id(email_doc_id)

    (archive / old_folder).mkdir(parents=True)
    (archive / new_folder).mkdir(parents=True)
    (output / "documents.jsonl").write_text(
        json.dumps({"doc_id": email_doc_id, "document_type": "email"}) + "\n",
        encoding="utf-8",
    )
    result = plan_migration(output)
    assert result.plans == []
    assert old_folder in result.skipped_target_exists


def test_compute_folder_id_is_deterministic():
    doc_id = "abcdef0123456789" * 2
    assert compute_folder_id(doc_id) == compute_folder_id(doc_id)
    assert len(compute_folder_id(doc_id)) == 32


def test_compute_folder_id_differs_from_doc_id_prefix():
    """Folder id must NOT be the doc_id[:16] nor doc_id[:32]."""
    doc_id = "abcdef0123456789" * 2
    folder_id = compute_folder_id(doc_id)
    assert folder_id != doc_id[:16]
    assert folder_id != doc_id[:32]
