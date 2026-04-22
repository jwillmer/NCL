"""Tests for mtss backup CLI command."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest
import typer


def _make_populated_db(db_path: Path) -> None:
    """Create a toy ingest.db so VACUUM INTO has real data to copy."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT);
            INSERT INTO documents (doc_id) VALUES ('a'), ('b'), ('c');
            """
        )
        conn.commit()
    finally:
        conn.close()


def _make_archive_tree(archive_dir: Path) -> list[Path]:
    """Populate archive/ with a couple of folders + files. Returns the file paths."""
    folder_a = archive_dir / "folder_a"
    folder_b = archive_dir / "folder_b" / "sub"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)
    files = [
        folder_a / "doc.md",
        folder_a / "attachment.pdf",
        folder_b / "nested.md",
    ]
    for f in files:
        f.write_bytes(f"content-of-{f.name}".encode())
    return files


@pytest.fixture
def staged_output(tmp_path: Path) -> Path:
    """Prepare <tmp>/output/ with ingest.db + archive/ populated."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _make_populated_db(output_dir / "ingest.db")
    _make_archive_tree(output_dir / "archive")
    return output_dir


def test_backup_creates_timestamped_snapshot_with_db_and_zip(
    staged_output: Path, tmp_path: Path
):
    from mtss.cli.backup_cmd import _run_backup

    dest = tmp_path / "backup"

    _run_backup(
        output_dir=staged_output,
        dest=dest,
        compression_level=1,
        split_size_mb=0,
        dry_run=False,
    )

    snapshots = list(dest.iterdir())
    assert len(snapshots) == 1, "exactly one timestamped snapshot dir expected"
    snapshot = snapshots[0]
    # Name shape: YYYYMMDD-HHMMSS (8 digits, dash, 6 digits).
    assert len(snapshot.name) == 15 and snapshot.name[8] == "-"

    db_copy = snapshot / "ingest.db"
    zip_copy = snapshot / "archive.zip"
    assert db_copy.exists(), "backup ingest.db missing"
    assert zip_copy.exists(), "backup archive.zip missing"

    # DB content preserved by VACUUM INTO.
    conn = sqlite3.connect(db_copy)
    try:
        rows = conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["a", "b", "c"]

    # Zip entries use forward slashes + contain the original file contents.
    with zipfile.ZipFile(zip_copy) as zf:
        names = set(zf.namelist())
        assert names == {
            "folder_a/doc.md",
            "folder_a/attachment.pdf",
            "folder_b/sub/nested.md",
        }
        assert zf.read("folder_a/doc.md") == b"content-of-doc.md"


def test_backup_dry_run_writes_nothing(staged_output: Path, tmp_path: Path):
    from mtss.cli.backup_cmd import _run_backup

    dest = tmp_path / "backup"

    _run_backup(
        output_dir=staged_output,
        dest=dest,
        compression_level=1,
        split_size_mb=0,
        dry_run=True,
    )

    assert not dest.exists(), "dry-run must not create backup root"


def test_backup_missing_db_exits_nonzero(tmp_path: Path):
    from mtss.cli.backup_cmd import _run_backup

    empty_output = tmp_path / "output"
    empty_output.mkdir()

    with pytest.raises(typer.Exit) as exc:
        _run_backup(
            output_dir=empty_output,
            dest=tmp_path / "backup",
            compression_level=1,
            split_size_mb=0,
            dry_run=False,
        )
    assert exc.value.exit_code == 1


def test_backup_skips_zip_when_archive_dir_missing(tmp_path: Path):
    from mtss.cli.backup_cmd import _run_backup

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _make_populated_db(output_dir / "ingest.db")
    # No archive/ directory on purpose.

    dest = tmp_path / "backup"
    _run_backup(
        output_dir=output_dir,
        dest=dest,
        compression_level=1,
        split_size_mb=0,
        dry_run=False,
    )

    snapshot = next(dest.iterdir())
    assert (snapshot / "ingest.db").exists()
    assert not (snapshot / "archive.zip").exists()


def test_resolve_paths_prefers_nextcloud_when_exists(tmp_path: Path, monkeypatch):
    from mtss.cli import backup_cmd

    fake_preferred = tmp_path / "nextcloud_backups"
    fake_preferred.mkdir()
    monkeypatch.setattr(backup_cmd, "PREFERRED_DEST", fake_preferred)

    source, dest = backup_cmd._resolve_paths(output_dir=tmp_path / "src", dest=None)
    assert dest == fake_preferred.resolve()


def test_resolve_paths_prompts_when_preferred_missing(tmp_path: Path, monkeypatch):
    from mtss.cli import backup_cmd

    missing_preferred = tmp_path / "does_not_exist"
    monkeypatch.setattr(backup_cmd, "PREFERRED_DEST", missing_preferred)

    chosen = tmp_path / "chosen"
    monkeypatch.setattr(backup_cmd.typer, "prompt", lambda *a, **kw: str(chosen))

    source, dest = backup_cmd._resolve_paths(output_dir=tmp_path / "src", dest=None)
    assert dest == chosen.resolve()


def test_explicit_dest_bypasses_preferred_path(tmp_path: Path, monkeypatch):
    from mtss.cli import backup_cmd

    # Even though preferred path exists, --dest wins.
    fake_preferred = tmp_path / "nextcloud_backups"
    fake_preferred.mkdir()
    monkeypatch.setattr(backup_cmd, "PREFERRED_DEST", fake_preferred)

    explicit = tmp_path / "explicit"
    _, dest = backup_cmd._resolve_paths(output_dir=tmp_path / "src", dest=explicit)
    assert dest == explicit.resolve()


def test_backup_store_mode_uses_no_compression(staged_output: Path, tmp_path: Path):
    from mtss.cli.backup_cmd import _run_backup

    dest = tmp_path / "backup"
    _run_backup(
        output_dir=staged_output,
        dest=dest,
        compression_level=0,
        split_size_mb=0,
        dry_run=False,
    )

    snapshot = next(dest.iterdir())
    with zipfile.ZipFile(snapshot / "archive.zip") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED


def test_backup_splits_artifacts_and_concat_restores_originals(
    staged_output: Path, tmp_path: Path
):
    """Split both artifacts into small chunks; confirm concat restores bytes."""
    from mtss.cli.backup_cmd import _run_backup, REASSEMBLE_README

    # Grow the archive so the zip is big enough to actually span multiple chunks.
    big = staged_output / "archive" / "folder_a" / "big.bin"
    big.write_bytes(b"X" * (300 * 1024))  # 300 KB raw; with compression_level=0 zip is ~300 KB

    dest = tmp_path / "backup"
    _run_backup(
        output_dir=staged_output,
        dest=dest,
        compression_level=0,  # uncompressed so zip size stays predictable
        split_size_mb=1,  # 1 MB per chunk; produces >=1 part for both artifacts
        dry_run=False,
    )

    snapshot = next(dest.iterdir())

    # Originals must be gone; at least one .part001 must exist for each artifact.
    assert not (snapshot / "ingest.db").exists()
    assert not (snapshot / "archive.zip").exists()
    db_parts = sorted(snapshot.glob("ingest.db.part*"))
    zip_parts = sorted(snapshot.glob("archive.zip.part*"))
    assert db_parts, "expected at least one DB part file"
    assert zip_parts, "expected at least one archive part file"
    assert (snapshot / REASSEMBLE_README).exists(), "REASSEMBLE readme missing"

    # Concatenating the parts back must produce a byte-identical DB and zip.
    reassembled_zip = tmp_path / "reassembled.zip"
    reassembled_zip.write_bytes(b"".join(p.read_bytes() for p in zip_parts))
    with zipfile.ZipFile(reassembled_zip) as zf:
        names = set(zf.namelist())
        assert "folder_a/big.bin" in names
        assert zf.read("folder_a/doc.md") == b"content-of-doc.md"

    reassembled_db = tmp_path / "reassembled.db"
    reassembled_db.write_bytes(b"".join(p.read_bytes() for p in db_parts))
    conn = sqlite3.connect(reassembled_db)
    try:
        rows = conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["a", "b", "c"]


def test_split_file_small_input_produces_single_part(tmp_path: Path):
    """Files smaller than one chunk still get a .part001 so glob patterns work."""
    from mtss.cli.backup_cmd import _split_file

    src = tmp_path / "tiny.bin"
    payload = b"hello world"
    src.write_bytes(payload)

    parts = _split_file(src, chunk_mb=1)

    assert len(parts) == 1
    assert parts[0].name == "tiny.bin.part001"
    assert parts[0].read_bytes() == payload
    assert not src.exists()


def test_split_size_zero_disables_splitting(staged_output: Path, tmp_path: Path):
    from mtss.cli.backup_cmd import _run_backup, REASSEMBLE_README

    dest = tmp_path / "backup"
    _run_backup(
        output_dir=staged_output,
        dest=dest,
        compression_level=1,
        split_size_mb=0,
        dry_run=False,
    )

    snapshot = next(dest.iterdir())
    assert (snapshot / "ingest.db").exists()
    assert (snapshot / "archive.zip").exists()
    assert not list(snapshot.glob("*.part*"))
    assert not (snapshot / REASSEMBLE_README).exists()
