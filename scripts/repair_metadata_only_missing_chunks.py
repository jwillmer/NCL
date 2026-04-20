"""Repair ``metadata_only`` / ``summary`` docs with chunk_count != 1.

Background
----------
``mtss validate ingest`` check #27 (``single_chunk_modes``) enforces that
every document stamped ``embedding_mode IN ('summary','metadata_only')``
holds exactly one chunk. Two production-scale bugs violated that invariant:

**Bug A — missing filename stub (v=6 cutover).**
``attachment_handler._process_non_zip_attachment`` (and the ZIP-member
equivalent) stamps ``METADATA_ONLY`` when the parser opens a file but
extracts no text, but the chunker dispatcher was only called when
``parsed_content`` was truthy. Result: 30 docs completed with
``embedding_mode = metadata_only`` and **0 chunks**. All PDFs, JPGs, or
``.txt`` files that parsed to an empty string.

**Bug B — image chunk_id instability.**
``helpers.enrich_chunks_with_document_metadata`` fell back to
``crc32(chunk.content) % 1000`` as pseudo char-offsets when a chunk had
no markdown range. Image chunks always hit that branch (vision output
has no offsets). Re-ingesting the same image produced a slightly
different vision description → different chunk_id → ``UNIQUE(chunk_id)``
didn't dedup → **9 image docs with 2-3 duplicate chunks**.

Both bugs are patched in the ingest pipeline; this script repairs the
rows already in ``ingest.db``. Complementary, not mutually exclusive —
run both passes in a single invocation.

Strategy
--------
* **0-chunk repair**: build the filename-stub chunk (filename + type
  + source_title) and ``INSERT OR IGNORE`` it with ``chunk_id`` derived
  from ``(doc_id, METADATA_CHUNK_POS)`` — identical to the
  ingest-side ``build_chunks_metadata_only``. No LLM calls, no
  embedding generated (metadata_only chunks are not embedded anyway).
* **2+ chunk dedup**: keep the most-recent row per doc (``created_at DESC``,
  fallback ``chunk_id DESC`` for determinism), ``DELETE`` the rest.
  Foreign-key ``chunk_topics`` rows CASCADE automatically.

No re-parse, no LLM calls, no vision calls, no API cost.

Usage
-----
    # Dry-run (default).
    uv run python scripts/repair_metadata_only_missing_chunks.py --output-dir data/output

    # Real run.
    uv run python scripts/repair_metadata_only_missing_chunks.py --output-dir data/output --apply

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("repair_metadata_only_missing_chunks")

# Keep in sync with mtss.utils — compute_chunk_id + METADATA_CHUNK_POS.
_CHUNK_ID_LENGTH = 12
_METADATA_CHUNK_POS: tuple[int, int] = (-2, 0)


def _compute_chunk_id(doc_id: str, char_start: int, char_end: int) -> str:
    """Mirror ``mtss.utils.compute_chunk_id`` without importing the package.
    Keeping the script standalone avoids pulling in settings / config side-
    effects that would complicate running against a backup DB.
    """
    combined = f"{doc_id}:{char_start}:{char_end}"
    return hashlib.sha256(combined.encode()).hexdigest()[:_CHUNK_ID_LENGTH]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_stub_content(
    *,
    file_name: str | None,
    document_type: str | None,
    source_title: str | None,
    email_subject: str | None,
) -> str:
    """Mirror ``chunker.build_chunks_metadata_only``'s content assembly."""
    parts: list[str] = []
    if file_name:
        parts.append(f"File: {file_name}")
    if document_type:
        parts.append(f"Type: {document_type}")
    if source_title and source_title != file_name:
        parts.append(f"Title: {source_title}")
    if email_subject:
        parts.append(f"Subject: {email_subject}")
    return "\n".join(parts)


# ── pass 1: 0-chunk repair ───────────────────────────────────────────


def _zero_chunk_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT d.id, d.doc_id, d.file_name, d.document_type, d.source_id,
               d.source_title, d.archive_browse_uri, d.archive_download_uri,
               d.content_type, d.mime_type, d.root_id, d.parent_id,
               d.embedding_mode
        FROM documents d
        WHERE d.embedding_mode IN ('summary', 'metadata_only')
          AND (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) = 0
        ORDER BY d.embedding_mode, d.file_name
        """
    ).fetchall()


def _lookup_email_subject(
    conn: sqlite3.Connection, doc_row: sqlite3.Row
) -> str | None:
    """Best-effort: return the owning email's subject from metadata_json.
    Falls back through root_id → parent_id. Returns None if unavailable."""
    for key in ("root_id", "parent_id"):
        parent_uuid = doc_row[key]
        if not parent_uuid or parent_uuid == doc_row["id"]:
            continue
        row = conn.execute(
            "SELECT metadata_json FROM documents WHERE id = ?",
            (parent_uuid,),
        ).fetchone()
        if not row or not row[0]:
            continue
        try:
            meta = json.loads(row[0])
        except (TypeError, ValueError):
            continue
        if isinstance(meta, dict):
            email = meta.get("email_metadata") or {}
            subject = email.get("subject") if isinstance(email, dict) else None
            if subject:
                return subject
    return None


def _insert_stub_chunk(
    conn: sqlite3.Connection, doc_row: sqlite3.Row, dry_run: bool
) -> bool:
    """Returns True when a stub would be / was inserted."""
    doc_uuid = doc_row["id"]
    doc_id = doc_row["doc_id"]
    if not doc_id:
        logger.warning(
            "doc %s has no doc_id — cannot compute stable chunk_id; skipping",
            doc_uuid,
        )
        return False

    subject = _lookup_email_subject(conn, doc_row)
    content = _build_stub_content(
        file_name=doc_row["file_name"],
        document_type=doc_row["document_type"],
        source_title=doc_row["source_title"],
        email_subject=subject,
    )
    if not content:
        logger.warning(
            "doc %s has no file_name/type/title/subject — stub would be empty; skipping",
            doc_uuid,
        )
        return False

    chunk_id = _compute_chunk_id(doc_id, *_METADATA_CHUNK_POS)
    chunk_uuid = str(uuid.uuid4())
    metadata = {
        "type": "metadata_stub",
        "source_file": doc_row["file_name"] or "",
        "repaired_by": "repair_metadata_only_missing_chunks",
    }
    now = _utcnow_iso()

    if dry_run:
        return True

    conn.execute(
        """
        INSERT OR IGNORE INTO chunks (
            id, chunk_id, document_id, source_id, content, chunk_index,
            section_path_json, embedding_text, embedding_mode, source_title,
            archive_browse_uri, archive_download_uri, metadata_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, 0, '[]', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_uuid,
            chunk_id,
            doc_uuid,
            doc_row["source_id"],
            content,
            content,
            doc_row["embedding_mode"],
            doc_row["source_title"],
            doc_row["archive_browse_uri"],
            doc_row["archive_download_uri"],
            json.dumps(metadata),
            now,
        ),
    )
    return True


# ── pass 2: 2+ chunk dedup ───────────────────────────────────────────


def _duplicate_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT d.id, d.doc_id, d.file_name, d.embedding_mode, d.content_type,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
        FROM documents d
        WHERE d.embedding_mode IN ('summary', 'metadata_only')
          AND (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) > 1
        ORDER BY chunk_count DESC, d.file_name
        """
    ).fetchall()


def _dedup_chunks(
    conn: sqlite3.Connection, doc_row: sqlite3.Row, dry_run: bool
) -> int:
    """Delete all but the most-recent chunk. Returns the delete count."""
    rows = conn.execute(
        """
        SELECT id, chunk_id, created_at
        FROM chunks
        WHERE document_id = ?
        ORDER BY created_at DESC NULLS LAST, chunk_id DESC
        """,
        (doc_row["id"],),
    ).fetchall()
    if len(rows) <= 1:
        return 0
    keep = rows[0]
    drop_ids = [r["id"] for r in rows[1:]]
    if dry_run:
        return len(drop_ids)
    conn.executemany(
        "DELETE FROM chunks WHERE id = ?",
        [(d,) for d in drop_ids],
    )
    logger.debug(
        "doc %s (%s): kept chunk_id=%s, dropped %d",
        doc_row["id"],
        doc_row["file_name"],
        keep["chunk_id"],
        len(drop_ids),
    )
    return len(drop_ids)


# ── driver ───────────────────────────────────────────────────────────


def _report(rows: Sequence[sqlite3.Row], label: str, fields: Sequence[str]) -> None:
    logger.info("  %s: %d", label, len(rows))
    for r in rows[:10]:
        bits = "  ".join(f"{f}={r[f]}" for f in fields if f in r.keys())
        logger.info("    %s", bits)
    if len(rows) > 10:
        logger.info("    ... and %d more", len(rows) - 10)


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
        help="Perform writes. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--skip-stub",
        action="store_true",
        help="Skip pass 1 (0-chunk stub insertion).",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip pass 2 (duplicate-chunk delete).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = args.output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found at %s", db_path)
        return 1

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("Mode: %s", mode)
    logger.info("DB:   %s", db_path)

    inserted = 0
    deleted = 0
    try:
        if not args.skip_stub:
            logger.info("")
            logger.info("Pass 1 — missing filename stubs (single_chunk_modes w/ 0 chunks):")
            zero = _zero_chunk_candidates(conn)
            _report(
                zero,
                "candidates",
                ["embedding_mode", "content_type", "file_name"],
            )
            if zero and args.apply:
                conn.execute("BEGIN")
                try:
                    for r in zero:
                        if _insert_stub_chunk(conn, r, dry_run=False):
                            inserted += 1
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                logger.info("  inserted %d stub chunks", inserted)
            elif zero:
                for r in zero:
                    if _insert_stub_chunk(conn, r, dry_run=True):
                        inserted += 1
                logger.info("  would insert %d stub chunks (dry-run)", inserted)

        if not args.skip_dedup:
            logger.info("")
            logger.info("Pass 2 — duplicate chunks under single_chunk_modes docs:")
            dups = _duplicate_candidates(conn)
            _report(
                dups,
                "candidates",
                ["embedding_mode", "chunk_count", "content_type", "file_name"],
            )
            if dups and args.apply:
                conn.execute("BEGIN")
                try:
                    for r in dups:
                        deleted += _dedup_chunks(conn, r, dry_run=False)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                logger.info("  deleted %d duplicate chunks", deleted)
            elif dups:
                for r in dups:
                    deleted += _dedup_chunks(conn, r, dry_run=True)
                logger.info("  would delete %d duplicate chunks (dry-run)", deleted)

        if not args.apply:
            logger.info("")
            logger.info("Dry-run complete. Re-run with --apply to perform writes.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
