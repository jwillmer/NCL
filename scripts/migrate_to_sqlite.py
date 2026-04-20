"""Migrate legacy JSONL ingest output into ``data/output/ingest.db`` (SQLite).

One-shot migration script. Reads:

- ``documents.jsonl``
- ``chunks.jsonl``
- ``topics.jsonl``
- ``ingest_events.jsonl``
- ``processing_log.jsonl`` (if present — usually a different location, see
  ``LocalProgressTracker``)
- ``run_history.jsonl``
- ``manifest.json``

Writes ``ingest.db`` alongside the JSONLs. The JSONL files are *not*
touched. A three-way mode flag controls the output:

- ``--dry-run`` (default) — parse inputs + report counts. Writes nothing.
- ``--apply``   — write ``ingest.db.tmp`` then atomically rename to
  ``ingest.db`` on clean exit. If the new DB already exists, refuse
  unless ``--overwrite`` is passed.
- ``--verify``  — after a previous ``--apply`` run, re-scan both sides
  and report per-table row-count diffs. Read-only.

The script is idempotent only under ``--overwrite``: re-running will
produce a fresh DB from the JSONLs. Safe to run against live data:
writes go through a tmp path, errors delete the tmp.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Make ``src`` importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mtss.storage.sqlite_client import (  # noqa: E402
    _SCHEMA_SQL,
    _dumps,
    _encode_embedding,
    SCHEMA_VERSION,
)

logger = logging.getLogger("migrate_to_sqlite")


# ── JSONL streaming ─────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON in %s:%d: %s", path, line_no, e)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


# ── row mapping ─────────────────────────────────────────────────────

def _doc_row_from_jsonl(d: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        k: d.get(k)
        for k in (
            "path",
            "email_subject",
            "email_participants",
            "email_initiator",
            "email_date_start",
            "email_date_end",
            "email_message_count",
            "attachment_content_type",
            "attachment_size_bytes",
        )
        if d.get(k) is not None
    }
    return {
        "id":                d["id"],
        "doc_id":            d.get("doc_id"),
        "source_id":         d.get("source_id") or "",
        "document_type":     d.get("document_type") or "attachment_other",
        "status":            d.get("status") or "completed",
        "error_message":     d.get("error_message"),
        "file_hash":         d.get("file_hash"),
        "file_name":         d.get("file_name"),
        "file_path":         d.get("file_path"),
        "parent_id":         d.get("parent_id"),
        "root_id":           d.get("root_id") or d["id"],
        "depth":             d.get("depth", 0),
        "content_version":   d.get("content_version", 1),
        "ingest_version":    d.get("ingest_version", 1),
        "archive_path":      d.get("archive_path"),
        "title":             d.get("email_subject") or d.get("source_title"),
        "source_title":      d.get("source_title"),
        "mime_type":         d.get("attachment_content_type"),
        "content_type":      d.get("attachment_content_type"),
        "size_bytes":        d.get("attachment_size_bytes"),
        "embedding_mode":    d.get("embedding_mode"),
        "archive_browse_uri":   d.get("archive_browse_uri"),
        "archive_download_uri": d.get("archive_download_uri"),
        "metadata_json":     _dumps(meta) if meta else None,
        "processed_at":      d.get("processed_at"),
        "created_at":        d.get("created_at") or "",
        "updated_at":        d.get("updated_at") or d.get("created_at") or "",
    }


def _chunk_row_and_topics_from_jsonl(
    c: Dict[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    blob, dim = _encode_embedding(c.get("embedding"))
    meta = c.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    topic_ids = [str(t) for t in (meta.get("topic_ids") or [])]
    meta_stripped = {k: v for k, v in meta.items() if k != "topic_ids"}
    if not meta_stripped:
        meta_stripped = None
    row = {
        "id":                c["id"],
        "chunk_id":          c.get("chunk_id"),
        "document_id":       c["document_id"],
        "source_id":         c.get("source_id"),
        "content":           c.get("content") or "",
        "chunk_index":       c.get("chunk_index", 0),
        "char_start":        c.get("char_start"),
        "char_end":          c.get("char_end"),
        "line_from":         c.get("line_from"),
        "line_to":           c.get("line_to"),
        "page_number":       c.get("page_number"),
        "section_title":     c.get("section_title"),
        "section_path_json": _dumps(c.get("section_path") or []),
        "context_summary":   c.get("context_summary"),
        "embedding_text":    c.get("embedding_text"),
        "embedding":         blob,
        "embedding_dim":     dim,
        "embedding_mode":    c.get("embedding_mode"),
        "source_title":      c.get("source_title"),
        "archive_browse_uri":   c.get("archive_browse_uri"),
        "archive_download_uri": c.get("archive_download_uri"),
        "metadata_json":     _dumps(meta_stripped) if meta_stripped else None,
        "created_at":        "",
    }
    return row, topic_ids


def _topic_row_from_jsonl(t: Dict[str, Any]) -> Dict[str, Any]:
    blob, dim = _encode_embedding(t.get("embedding"))
    return {
        "id":              t["id"],
        "name":            t.get("name") or "",
        "display_name":    t.get("display_name") or t.get("name") or "",
        "description":     t.get("description"),
        "keywords_json":   _dumps(t.get("keywords") if isinstance(t.get("keywords"), list) else None),
        "embedding":       blob,
        "embedding_dim":   dim,
        "chunk_count":     t.get("chunk_count", 0),
        "document_count":  t.get("document_count", 0),
        "created_at":      t.get("created_at") or "",
        "updated_at":      t.get("updated_at") or t.get("created_at") or "",
    }


def _event_row_from_jsonl(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type":         e.get("event_type") or "unknown",
        "severity":           e.get("severity") or "warning",
        "reason":             e.get("reason"),
        "message":            e.get("message"),
        "file_path":          e.get("file_path"),
        "file_name":          e.get("file_name"),
        "file_size":          e.get("file_size"),
        "mime_type":          e.get("mime_type"),
        "source_eml_path":    e.get("source_eml_path"),
        "source_zip_path":    e.get("source_zip_path"),
        "parent_document_id": e.get("parent_document_id"),
        "document_id":        e.get("document_id"),
        "timestamp":          e.get("timestamp") or "",
    }


def _processing_log_row(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fp = e.get("file_path")
    if not fp:
        return None
    fh = e.get("file_hash") or e.get("hash") or ""
    return {
        "file_path":        fp,
        "file_hash":        fh,
        "status":           (e.get("status") or "pending").upper(),
        "started_at":       e.get("started_at"),
        "completed_at":     e.get("completed_at") or e.get("timestamp"),
        "duration_seconds": e.get("duration_seconds"),
        "attempts":         e.get("attempts") or 0,
        "error":            e.get("error") or e.get("error_message"),
        "ingest_version":   e.get("ingest_version"),
    }


def _run_history_row(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp":        e.get("timestamp") or "",
        "elapsed_seconds":  e.get("elapsed_seconds"),
        "files_attempted":  e.get("files_attempted"),
        "files_processed":  e.get("files_processed"),
        "files_failed":     e.get("files_failed"),
        "cumulative_json":  _dumps(e.get("cumulative")),
        "services_json":    _dumps(e.get("services")),
        "errors_json":      _dumps(e.get("errors")),
    }


# ── migration ────────────────────────────────────────────────────────

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA_SQL)
    return conn


def _insert_many(conn: sqlite3.Connection, table: str, rows: List[Dict[str, Any]], *, replace: bool = False) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    prefix = "INSERT OR REPLACE" if replace else "INSERT"
    sql = f"{prefix} INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])


def _manifest_from_json(manifest_path: Path) -> Dict[str, str]:
    """Read ``manifest.json`` if it exists and flatten nested values."""
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Unreadable manifest.json: %s", e)
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = v if isinstance(v, str) else json.dumps(v, default=str)
    return out


def migrate(output_dir: Path, db_path: Path) -> Dict[str, int]:
    """Stream JSONLs → DB. Returns counts per table."""
    conn = _open_db(db_path)
    counts = {
        "documents": 0,
        "chunks": 0,
        "chunk_topics": 0,
        "topics": 0,
        "ingest_events": 0,
        "processing_log": 0,
        "run_history": 0,
        "manifest": 0,
    }
    try:
        # Documents — one transaction total.
        conn.execute("BEGIN")
        seen_ids: set[str] = set()
        batch: List[Dict[str, Any]] = []
        for doc in _iter_jsonl(output_dir / "documents.jsonl"):
            did = doc.get("id")
            if not did or did in seen_ids:
                continue
            seen_ids.add(did)
            batch.append(_doc_row_from_jsonl(doc))
            if len(batch) >= 500:
                _insert_many(conn, "documents", batch)
                counts["documents"] += len(batch)
                batch = []
        if batch:
            _insert_many(conn, "documents", batch)
            counts["documents"] += len(batch)
        conn.execute("COMMIT")

        # Topics — must precede chunks so chunk_topics FK resolves.
        conn.execute("BEGIN")
        seen_topic_ids: set[str] = set()
        topic_batch: List[Dict[str, Any]] = []
        for t in _iter_jsonl(output_dir / "topics.jsonl"):
            row = _topic_row_from_jsonl(t)
            tid = row.get("id")
            if tid:
                seen_topic_ids.add(tid)
            topic_batch.append(row)
            if len(topic_batch) >= 500:
                _insert_many(conn, "topics", topic_batch, replace=True)
                counts["topics"] += len(topic_batch)
                topic_batch = []
        if topic_batch:
            _insert_many(conn, "topics", topic_batch, replace=True)
            counts["topics"] += len(topic_batch)
        conn.execute("COMMIT")

        # Chunks. Stream + batch. chunk_topics built alongside.
        # Skip chunks whose document_id is not in documents.jsonl — these are
        # orphan debris from pre-SQLite partial-flush bugs; the new schema's
        # FK CASCADE is exactly what prevents this class from reappearing.
        # Skip chunk_topics rows whose topic_id is not in topics.jsonl.
        conn.execute("BEGIN")
        seen_chunk_pks: set[str] = set()
        chunk_batch: List[Dict[str, Any]] = []
        ct_batch: List[tuple[str, str]] = []
        skipped_orphans = 0
        skipped_dupes = 0
        skipped_missing_topics = 0
        for c in _iter_jsonl(output_dir / "chunks.jsonl"):
            cid = c.get("id")
            if not cid:
                continue
            if cid in seen_chunk_pks:
                skipped_dupes += 1
                continue
            row, topic_ids = _chunk_row_and_topics_from_jsonl(c)
            if row.get("document_id") not in seen_ids:
                skipped_orphans += 1
                continue
            seen_chunk_pks.add(cid)
            chunk_batch.append(row)
            for tid in topic_ids:
                if tid in seen_topic_ids:
                    ct_batch.append((cid, tid))
                else:
                    skipped_missing_topics += 1
            if len(chunk_batch) >= 500:
                _insert_many(conn, "chunks", chunk_batch, replace=True)
                counts["chunks"] += len(chunk_batch)
                chunk_batch = []
            if len(ct_batch) >= 2000:
                conn.executemany(
                    "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) VALUES (?,?)",
                    ct_batch,
                )
                counts["chunk_topics"] += len(ct_batch)
                ct_batch = []
        if chunk_batch:
            _insert_many(conn, "chunks", chunk_batch, replace=True)
            counts["chunks"] += len(chunk_batch)
        if ct_batch:
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) VALUES (?,?)",
                ct_batch,
            )
            counts["chunk_topics"] += len(ct_batch)
        conn.execute("COMMIT")
        if skipped_orphans:
            logger.warning(
                "Skipped %d orphan chunks (document_id not found in documents.jsonl)",
                skipped_orphans,
            )
        if skipped_dupes:
            logger.warning("Skipped %d duplicate chunk rows by id", skipped_dupes)
        if skipped_missing_topics:
            logger.warning(
                "Skipped %d chunk_topics rows (topic_id missing from topics.jsonl)",
                skipped_missing_topics,
            )

        # Events.
        conn.execute("BEGIN")
        event_batch: List[Dict[str, Any]] = []
        for e in _iter_jsonl(output_dir / "ingest_events.jsonl"):
            event_batch.append(_event_row_from_jsonl(e))
            if len(event_batch) >= 2000:
                _insert_many(conn, "ingest_events", event_batch)
                counts["ingest_events"] += len(event_batch)
                event_batch = []
        if event_batch:
            _insert_many(conn, "ingest_events", event_batch)
            counts["ingest_events"] += len(event_batch)
        conn.execute("COMMIT")

        # processing_log. Dedup by file_path (the new PK), keeping the last
        # JSONL entry per file — matches ``LocalProgressTracker._load`` which
        # built an in-memory dict keyed by file_path and overwrote on each line.
        conn.execute("BEGIN")
        by_path: Dict[str, Dict[str, Any]] = {}
        for e in _iter_jsonl(output_dir / "processing_log.jsonl"):
            row = _processing_log_row(e)
            if row is None:
                continue
            by_path[row["file_path"]] = row
        pl_batch = list(by_path.values())
        if pl_batch:
            _insert_many(conn, "processing_log", pl_batch, replace=True)
            counts["processing_log"] = len(pl_batch)
        conn.execute("COMMIT")

        # run_history.
        conn.execute("BEGIN")
        rh_batch: List[Dict[str, Any]] = []
        for e in _iter_jsonl(output_dir / "run_history.jsonl"):
            rh_batch.append(_run_history_row(e))
            if len(rh_batch) >= 500:
                _insert_many(conn, "run_history", rh_batch)
                counts["run_history"] += len(rh_batch)
                rh_batch = []
        if rh_batch:
            _insert_many(conn, "run_history", rh_batch)
            counts["run_history"] += len(rh_batch)
        conn.execute("COMMIT")

        # manifest.
        manifest_data = _manifest_from_json(output_dir / "manifest.json")
        manifest_data.setdefault("schema_version", str(SCHEMA_VERSION))
        manifest_data["migrated_at"] = _now_iso()
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT OR REPLACE INTO manifest(key, value) VALUES (?, ?)",
            list(manifest_data.items()),
        )
        counts["manifest"] = len(manifest_data)
        conn.execute("COMMIT")

        # Recompute topic counts so they match junction-table reality.
        conn.execute("BEGIN")
        conn.execute(
            "UPDATE topics SET "
            "  chunk_count = (SELECT COUNT(*) FROM chunk_topics ct WHERE ct.topic_id = topics.id), "
            "  document_count = (SELECT COUNT(DISTINCT c.document_id) "
            "                    FROM chunk_topics ct JOIN chunks c ON c.id = ct.chunk_id "
            "                    WHERE ct.topic_id = topics.id)"
        )
        conn.execute("COMMIT")

        # Wipe WAL so the output is a single tidy .db file.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return counts


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── verify ───────────────────────────────────────────────────────────

def verify(output_dir: Path, db_path: Path) -> Dict[str, Dict[str, int]]:
    """Per-table count diffs between JSONL and SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(f"No DB to verify: {db_path}")
    conn = _open_db(db_path)
    try:
        jsonl_counts = {
            "documents":      _count_jsonl(output_dir / "documents.jsonl"),
            "chunks":         _count_jsonl(output_dir / "chunks.jsonl"),
            "topics":         _count_jsonl(output_dir / "topics.jsonl"),
            "ingest_events":  _count_jsonl(output_dir / "ingest_events.jsonl"),
            "run_history":    _count_jsonl(output_dir / "run_history.jsonl"),
            "processing_log": _count_jsonl(output_dir / "processing_log.jsonl"),
        }
        db_counts = {}
        for table in jsonl_counts:
            db_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()
    return {
        "jsonl": jsonl_counts,
        "sqlite": db_counts,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Migrate JSONL ingest output to SQLite. Default is --dry-run."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "output",
        help="Directory containing the existing *.jsonl files (default: data/output).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Output DB path (default: <output-dir>/ingest.db).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write the DB.")
    mode.add_argument("--verify", action="store_true", help="Compare counts only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --apply to replace an existing ingest.db.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    db_path: Path = args.db or (output_dir / "ingest.db")

    if not output_dir.is_dir():
        logger.error("output-dir not found: %s", output_dir)
        return 2

    # Verify mode — read-only, early exit.
    if args.verify:
        diffs = verify(output_dir, db_path)
        print(f"{'table':<18} {'jsonl':>10} {'sqlite':>10} {'diff':>10}")
        print("-" * 52)
        ok = True
        for table, jsonl_n in diffs["jsonl"].items():
            db_n = diffs["sqlite"].get(table, 0)
            diff = db_n - jsonl_n
            marker = "" if diff == 0 else " *"
            print(f"{table:<18} {jsonl_n:>10} {db_n:>10} {diff:>+10d}{marker}")
            if diff != 0 and table not in ("processing_log",):
                # processing_log routinely differs (dedup by file_hash in DB)
                ok = False
        return 0 if ok else 1

    # Count JSONL rows up front for the plan.
    planned = {
        "documents":      _count_jsonl(output_dir / "documents.jsonl"),
        "chunks":         _count_jsonl(output_dir / "chunks.jsonl"),
        "topics":         _count_jsonl(output_dir / "topics.jsonl"),
        "ingest_events":  _count_jsonl(output_dir / "ingest_events.jsonl"),
        "run_history":    _count_jsonl(output_dir / "run_history.jsonl"),
        "processing_log": _count_jsonl(output_dir / "processing_log.jsonl"),
    }
    logger.info("Input: %s", output_dir)
    logger.info("Target DB: %s", db_path)
    for k, v in planned.items():
        logger.info("  %-18s %8d rows", k, v)

    if not args.apply:
        logger.info("[dry-run] Not writing. Use --apply to commit.")
        return 0

    # Apply mode. Write to tmp, rename on success.
    if db_path.exists() and not args.overwrite:
        logger.error(
            "Refusing to overwrite existing %s (pass --overwrite if intended).",
            db_path,
        )
        return 2

    tmp_path = db_path.with_name(db_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    t0 = time.perf_counter()
    try:
        counts = migrate(output_dir, tmp_path)
        # Atomic swap.
        if db_path.exists():
            db_path.unlink()
        tmp_path.replace(db_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    elapsed = time.perf_counter() - t0

    logger.info("Wrote %s in %.1fs", db_path, elapsed)
    for k, v in counts.items():
        logger.info("  %-18s %8d rows", k, v)
    size_mb = db_path.stat().st_size / (1024 * 1024)
    logger.info("  DB size: %.1f MB", size_mb)
    logger.info("Run `python scripts/migrate_to_sqlite.py --verify` to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
