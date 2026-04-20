"""SQLite-backed progress tracker.

Drop-in replacement for ``LocalProgressTracker`` — same public methods,
but writes land in the shared ``processing_log`` table inside
``ingest.db`` instead of a standalone ``processing_log.jsonl`` file. All
writes are single-row UPSERTs inside short transactions, so partial
flush / torn JSONL is structurally impossible.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SqliteProgressTracker:
    """Track ingest progress using the ``processing_log`` table.

    Shares the same ``ingest.db`` used by :class:`SqliteStorageClient`.
    Uses a dedicated connection so progress writes do not contend with
    ingest transactions, and so the tracker can outlive a single
    ``SqliteStorageClient`` instance (e.g. CLI wrappers that open/close
    the client multiple times).
    """

    def __init__(self, output_dir: Path, db_filename: str = "ingest.db"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.output_dir / db_filename
        self._conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        # Ensure table exists even if tracker is instantiated before the
        # ingest client (e.g. ``mtss reset-stale`` on a fresh install).
        # Schema is owned by ``sqlite_client.PROCESSING_LOG_SCHEMA_SQL`` so
        # both classes produce identical columns + indexes — drift previously
        # caused the tracker version to omit ``ingest_version``, silently
        # discarding any writes routed through it.
        from .sqlite_client import PROCESSING_LOG_SCHEMA_SQL
        self._conn.executescript(PROCESSING_LOG_SCHEMA_SQL)

    def close(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as e:
            logger.debug("wal_checkpoint failed on close: %s", e)
        self._conn.close()

    def compact(self) -> None:
        """No-op: SQLite stores one row per file by PK. Kept for API parity."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error as e:
            logger.debug("wal_checkpoint during compact failed: %s", e)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_entry(self, file_path: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM processing_log WHERE file_path = ?", (file_path,)
        ).fetchone()

    async def mark_started(self, file_path: Path, file_hash: str) -> None:
        fp = str(file_path)
        prior = self._get_entry(fp)
        attempts = (prior["attempts"] if prior else 0) + 1
        self._conn.execute(
            """
            INSERT INTO processing_log (file_path, file_hash, status, started_at, attempts)
            VALUES (?, ?, 'PROCESSING', ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash = excluded.file_hash,
                status = 'PROCESSING',
                started_at = excluded.started_at,
                completed_at = NULL,
                duration_seconds = NULL,
                error = NULL,
                attempts = excluded.attempts
            """,
            (fp, file_hash, datetime.now(timezone.utc).isoformat(), attempts),
        )

    async def mark_completed(self, file_path: Path) -> None:
        fp = str(file_path)
        prior = self._get_entry(fp)
        now = datetime.now(timezone.utc)
        duration = None
        if prior and prior["started_at"]:
            try:
                started = datetime.fromisoformat(prior["started_at"])
                duration = round((now - started).total_seconds(), 1)
            except (ValueError, TypeError):
                pass
        self._conn.execute(
            """
            UPDATE processing_log SET
                status = 'COMPLETED',
                completed_at = ?,
                duration_seconds = ?,
                error = NULL
            WHERE file_path = ?
            """,
            (now.isoformat(), duration, fp),
        )

    async def mark_failed(self, file_path: Path, error: str) -> None:
        fp = str(file_path)
        self._conn.execute(
            """
            INSERT INTO processing_log (file_path, file_hash, status, completed_at, error)
            VALUES (?, '', 'FAILED', ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                status = 'FAILED',
                completed_at = excluded.completed_at,
                error = excluded.error
            """,
            (fp, datetime.now(timezone.utc).isoformat(), (error or "")[:1000]),
        )

    async def get_pending_files(self, source_dir: Path) -> List[Path]:
        completed_hashes = {
            row["file_hash"]
            for row in self._conn.execute(
                "SELECT file_hash FROM processing_log WHERE status = 'COMPLETED'"
            )
        }
        pending: list[Path] = []
        for eml in sorted(source_dir.rglob("*.eml")):
            file_hash = self.compute_file_hash(eml)
            if file_hash not in completed_hashes:
                pending.append(eml)
        return pending

    async def get_failed_files(self) -> List[Path]:
        return [
            Path(row["file_path"])
            for row in self._conn.execute(
                "SELECT file_path FROM processing_log WHERE status = 'FAILED'"
            )
        ]

    async def get_processing_stats(self) -> Dict[str, int]:
        stats = {"total": 0, "pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM processing_log GROUP BY status"
        ):
            stats["total"] += row["n"]
            key = (row["status"] or "").lower()
            if key in stats:
                stats[key] += row["n"]
        return stats

    async def reset_stale_processing(self, max_age_minutes: int = 60) -> int:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        ).isoformat()
        cur = self._conn.execute(
            """
            UPDATE processing_log
            SET status = 'FAILED',
                error = 'Stale processing (>' || ? || 'min)',
                completed_at = ?
            WHERE status = 'PROCESSING' AND started_at IS NOT NULL AND started_at < ?
            """,
            (max_age_minutes, datetime.now(timezone.utc).isoformat(), cutoff),
        )
        return cur.rowcount or 0

    async def get_outdated_files(self, source_dir: Path, target_version: int) -> List[Path]:
        return []

    # ── raw row access for CLIs that render tables ──────────────────
    def iter_entries(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM processing_log")]
