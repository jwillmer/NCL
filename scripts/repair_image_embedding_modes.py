"""Backfill ``embedding_mode`` for attachment documents that bypass the decider.

Background
----------
Two attachment-handler paths historically completed without stamping
``Document.embedding_mode``:

1. **Image branch** (any ``attachment_image`` row). Image processing
   produces a single vision-derived chunk and skips the embedding decider
   entirely; ``embedding_mode`` was never assigned.
2. **Empty-parse branch** (any ``status='completed'`` doc with 0 chunks).
   When the parser opened the file but extracted no text (image-only
   PDFs, ``.url`` shortcut files), the handler logged ``no_body_chunks``
   and marked the doc COMPLETED, but skipped ``_decide_and_build_chunks``
   because it was gated on ``if parsed_content:``. ``embedding_mode``
   was never assigned.

Both branches now stamp ``METADATA_ONLY`` at ingest time. This script
repairs the existing rows that landed before the fix:

- Image rows: ``document_type = 'attachment_image' AND embedding_mode IS NULL``
- Empty-parse rows: ``status = 'completed' AND embedding_mode IS NULL``
  (the image clause is a subset; UNION dedupes)

Downstream filters (``mtss re-embed``, ``mtss validate``, future RAG
queries) treat ``embedding_mode IS NULL`` as "not yet decided" and either
skip the row or surface it as a warning, so leaving the rows NULL is a
silent observability gap.

FAILED rows are intentionally untouched — the parser blew up before the
decider could run, so NULL is the correct state for those.

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


def _candidate_rows(conn: sqlite3.Connection) -> list[tuple[str, str, str | None]]:
    """Image OR completed-with-no-mode rows. FAILED rows excluded — NULL is
    the right state for those (the parser blew up before the decider)."""
    rows = conn.execute(
        """
        SELECT id, document_type, file_name
        FROM documents
        WHERE embedding_mode IS NULL
          AND (
              document_type = 'attachment_image'
              OR status = 'completed'
          )
        ORDER BY document_type, file_name
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _apply(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        UPDATE documents
           SET embedding_mode = 'metadata_only',
               updated_at     = datetime('now')
         WHERE embedding_mode IS NULL
           AND (
               document_type = 'attachment_image'
               OR status = 'completed'
           )
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
        logger.info(
            "Found %d documents needing embedding_mode backfill "
            "(image OR completed-with-NULL)",
            len(candidates),
        )
        if not candidates:
            logger.info("Nothing to do.")
            return 0

        # Per-type breakdown to make the impact obvious before --apply.
        by_type: dict[str, int] = {}
        for _, doc_type, _ in candidates:
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
        for doc_type, n in sorted(by_type.items()):
            logger.info("  %-22s %d", doc_type, n)

        sample = candidates[:10]
        for row_id, doc_type, name in sample:
            logger.info("  %s  [%s]  %s", row_id, doc_type, name or "(no file_name)")
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
