"""Backfill ``embedding_mode`` for image attachment documents.

Background
----------
Image-attachment processing in ``attachment_handler.py`` historically did not
stamp ``Document.embedding_mode`` (the path bypasses the embedding decider —
images produce a single vision-derived chunk instead of being chunked from
text). The column landed in SQLite as ``NULL`` for every image attachment.

Downstream filters (``mtss re-embed``, ``mtss validate``, future RAG queries)
treat ``embedding_mode IS NULL`` as "not yet decided" and either skip the row
or surface it as a warning. This script repairs existing rows in-place by
setting ``embedding_mode = 'metadata_only'`` (the same value the live
ingest now stamps for images) for any document whose ``document_type`` is
``attachment_image`` and whose ``embedding_mode`` is NULL.

No re-parse, no LLM calls, no API cost.

Usage
-----
    # Dry-run (default). Reports how many rows would change, no writes.
    uv run python scripts/repair_image_embedding_modes.py --output-dir data/output

    # Real run. Only run after inspecting the dry-run output.
    uv run python scripts/repair_image_embedding_modes.py --output-dir data/output --apply

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger("repair_image_embedding_modes")


def _candidate_rows(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    rows = conn.execute(
        """
        SELECT id, file_name
        FROM documents
        WHERE document_type = 'attachment_image'
          AND embedding_mode IS NULL
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _apply(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        UPDATE documents
           SET embedding_mode = 'metadata_only',
               updated_at     = datetime('now')
         WHERE document_type = 'attachment_image'
           AND embedding_mode IS NULL
        """
    )
    return cursor.rowcount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing ingest.db (default: data/output).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the UPDATE. Without this flag the script only reports.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = args.output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found at %s", db_path)
        return 1

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        candidates = _candidate_rows(conn)
        logger.info("Found %d image documents with embedding_mode = NULL", len(candidates))
        if not candidates:
            logger.info("Nothing to do.")
            return 0

        sample = candidates[:10]
        for row_id, name in sample:
            logger.info("  %s  %s", row_id, name or "(no file_name)")
        if len(candidates) > len(sample):
            logger.info("  ... and %d more", len(candidates) - len(sample))

        if not args.apply:
            logger.info("Dry-run only. Re-run with --apply to UPDATE.")
            return 0

        updated = _apply(conn)
        logger.info("Updated %d rows.", updated)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
