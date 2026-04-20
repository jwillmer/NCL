"""Repair ``processing_log`` schema drift in an existing ``ingest.db``.

Background
----------
Until ``PROCESSING_LOG_SCHEMA_SQL`` was extracted as a shared constant,
``SqliteProgressTracker.__init__`` issued its own ``CREATE TABLE IF NOT
EXISTS processing_log (...)`` that omitted the ``ingest_version`` column
present in ``SqliteStorageClient._SCHEMA_SQL``.

Whichever class first opened a fresh DB won the schema. If the tracker won
(e.g. ``mtss reset-stale`` ran before ``mtss ingest`` on a clean install),
the table was created without ``ingest_version`` and any code path that
writes that column silently no-op'd it (or raised inside SQLite without
surfacing).

This script detects the drift and adds the column with a single
``ALTER TABLE``. SQLite's ``ALTER TABLE … ADD COLUMN`` is a metadata-only
operation — no row rewrite, no downtime risk.

Usage
-----
    # Dry-run (default). Reports the schema, does not modify.
    uv run python scripts/repair_processing_log_schema.py --output-dir data/output

    # Real run.
    uv run python scripts/repair_processing_log_schema.py --output-dir data/output --apply

Exit codes:
    0 = success (including dry-run, including no-op when schema is already current)
    1 = unrecoverable error (db missing, db locked, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger("repair_processing_log_schema")

EXPECTED_COLUMNS = {
    "file_path",
    "file_hash",
    "status",
    "started_at",
    "completed_at",
    "duration_seconds",
    "attempts",
    "error",
    "ingest_version",
}


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(processing_log)")}


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
        help="Run the ALTER TABLE. Without this flag the script only reports.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = args.output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found at %s", db_path)
        return 1

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        present = _existing_columns(conn)
        missing = EXPECTED_COLUMNS - present
        extra = present - EXPECTED_COLUMNS
        logger.info("processing_log columns present: %s", sorted(present))
        if extra:
            logger.warning("Unexpected extra columns (left as-is): %s", sorted(extra))
        if not missing:
            logger.info("Schema already current. Nothing to do.")
            return 0

        logger.info("Missing columns: %s", sorted(missing))
        if not args.apply:
            logger.info("Dry-run only. Re-run with --apply to ALTER.")
            return 0

        for col in sorted(missing):
            if col == "ingest_version":
                conn.execute(
                    "ALTER TABLE processing_log ADD COLUMN ingest_version INTEGER"
                )
                logger.info("Added column: ingest_version INTEGER")
            else:
                logger.warning(
                    "No ALTER recipe for column %r — add to the script before re-running",
                    col,
                )
                return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
