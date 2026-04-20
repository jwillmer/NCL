"""Validate a single backup snapshot.

Checks, in order:
  1. SQLite ``PRAGMA integrity_check`` on the backup DB.
  2. Row-count parity between backup and live DB (documents, chunks, topics,
     chunk_topics, ingest_events, processing_log).
  3. ``ZipFile.testzip()`` — CRC-checks every entry in archive.zip.
  4. File-count + cumulative-byte parity between the zip and live archive/.

Read-only. Opens the live DB with ``mode=ro`` so a running ingest can't be
affected. Run this when no ingest is active to avoid the live DB shifting
mid-check (the comparison assumes a stable source).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import zipfile
from pathlib import Path


DB_NAME = "ingest.db"
ZIP_NAME = "archive.zip"
TABLES = ["documents", "chunks", "topics", "chunk_topics", "ingest_events", "processing_log"]


def _open_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in TABLES:
        try:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[t] = -1  # table absent
    return counts


def _scan_archive_dir(archive_dir: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for p in archive_dir.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return files, total


def validate(snapshot_dir: Path, live_output_dir: Path) -> int:
    db_backup = snapshot_dir / DB_NAME
    zip_backup = snapshot_dir / ZIP_NAME
    db_live = live_output_dir / DB_NAME
    archive_live = live_output_dir / "archive"

    failures: list[str] = []
    warnings: list[str] = []

    print(f"Snapshot: {snapshot_dir}")
    print(f"Live:     {live_output_dir}")
    print()

    # --- 1. Backup DB integrity ---
    print("[1/4] Backup DB integrity_check...")
    if not db_backup.exists():
        failures.append(f"backup db missing: {db_backup}")
    else:
        conn = _open_ro(db_backup)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            conn.close()
        if result == [("ok",)]:
            print("      ok")
        else:
            failures.append(f"integrity_check failed: {result}")

    # --- 2. Row-count parity ---
    print("[2/4] Row-count parity...")
    if db_backup.exists() and db_live.exists():
        b_conn = _open_ro(db_backup)
        l_conn = _open_ro(db_live)
        try:
            backup_counts = _row_counts(b_conn)
            live_counts = _row_counts(l_conn)
        finally:
            b_conn.close()
            l_conn.close()
        for t in TABLES:
            b, l = backup_counts[t], live_counts[t]
            marker = "==" if b == l else "!="
            print(f"      {t:<16} backup={b:<10} {marker} live={l}")
            if b == -1 and l == -1:
                continue
            if b != l:
                # Live DB may have changed since backup — downgrade to warning unless backup>live.
                if b > l:
                    failures.append(f"{t}: backup has more rows than live ({b} > {l})")
                else:
                    warnings.append(
                        f"{t}: live grew since backup ({l - b} new rows) — expected if ingest ran after"
                    )
    else:
        failures.append("can't compare DB row counts (missing file)")

    # --- 3. Zip CRC check ---
    print("[3/4] Zip testzip (CRC)...")
    if not zip_backup.exists():
        failures.append(f"archive.zip missing: {zip_backup}")
        zip_file_count = 0
        zip_uncompressed = 0
    else:
        with zipfile.ZipFile(zip_backup) as zf:
            bad = zf.testzip()
            if bad is not None:
                failures.append(f"zip CRC failure on entry: {bad}")
            else:
                print("      ok")
            infos = zf.infolist()
            zip_file_count = len(infos)
            zip_uncompressed = sum(i.file_size for i in infos)

    # --- 4. Zip vs live archive parity ---
    print("[4/4] Zip vs live archive/ parity...")
    if archive_live.is_dir() and zip_backup.exists():
        live_files, live_bytes = _scan_archive_dir(archive_live)
        print(f"      files:  zip={zip_file_count:<8} live={live_files}")
        print(f"      bytes:  zip={zip_uncompressed:<14} live={live_bytes}")
        if zip_file_count > live_files:
            failures.append(
                f"zip contains more files than live archive ({zip_file_count} > {live_files})"
            )
        elif zip_file_count < live_files:
            warnings.append(
                f"live archive has {live_files - zip_file_count} more file(s) than zip — added after backup?"
            )
        if zip_uncompressed != live_bytes:
            # Byte mismatch is interesting but not necessarily fatal — matches above file-count verdict.
            delta = live_bytes - zip_uncompressed
            warnings.append(f"byte total differs (live - zip = {delta})")
    else:
        warnings.append("can't compare zip vs live archive (missing file/dir)")

    print()
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — backup validated.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Validate an mtss backup snapshot.")
    ap.add_argument("snapshot", type=Path, help="Path to <dest>/<YYYYMMDD-HHMMSS>/")
    ap.add_argument(
        "--live",
        type=Path,
        default=Path("data/output"),
        help="Live output dir to compare against (default: data/output)",
    )
    args = ap.parse_args()
    sys.exit(validate(args.snapshot.resolve(), args.live.resolve()))


if __name__ == "__main__":
    main()
