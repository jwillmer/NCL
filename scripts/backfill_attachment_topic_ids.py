"""Backfill attachment chunk topic_ids from the parent email's topics.

Background
----------
Before 2026-04-22, ``process_attachment`` / ``process_zip_attachment`` did not
thread ``topic_ids`` from the enclosing email down into attachment chunk
metadata. The email body chunks got stamped correctly (via
``_create_body_chunks``), but every attachment chunk — PDFs, Office files,
images, ZIP members — was stamped with vessel metadata only, no topics.

Consequence on the local SQLite store: ``chunks.metadata`` never had
``topic_ids`` for attachment rows AND the ``chunk_topics`` junction table
received no rows for attachments (the M:N write in ``insert_chunks`` only
fires when ``metadata.topic_ids`` is present at insert time).

Consequence on Supabase: the pgvector ``match_chunks`` topic filter is
``c.metadata @> jsonb_build_object('topic_ids', …)``, so attachment chunks
were invisible to topic-filtered retrieval even though the parent email
surfaced correctly.

What this script does
---------------------
1. Reads ``ingest.db``.
2. For every email whose body chunk has topic links in ``chunk_topics``,
   gathers the set of ``topic_id`` values.
3. For every attachment chunk under that email (``document.root_id == email.id``),
   inserts the same set of rows into ``chunk_topics`` (``INSERT OR IGNORE``).
4. Also rewrites ``chunks.metadata_json`` so downstream exports / Supabase
   sync see ``topic_ids`` in the blob too.

No LLM calls, no re-parse, no re-embed. The email-level topics were already
extracted at ingest time; we're only propagating them.

Why the script and not ``mtss import``
--------------------------------------
``mtss import`` is insert-only: its ``_import_documents`` path skips docs whose
``doc_id`` already exists on Supabase. It will never update ``chunks.metadata``
on rows that are already there. A separate remote-sync script
(``sync_attachment_topic_ids_to_supabase.py``) covers the Supabase side.

Usage
-----
    # Dry-run (default) — reports counts + sample chunks, no writes.
    python scripts/backfill_attachment_topic_ids.py

    # Apply for real. Run only after inspecting the dry-run output.
    python scripts/backfill_attachment_topic_ids.py --apply

    # Scope to a specific ingest_version (the original bug only affects v=6
    # attachments whose email bodies actually got topics; v=5 chunks are a
    # separate backfill because their emails don't have topics either).
    python scripts/backfill_attachment_topic_ids.py --min-ingest-version 6

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing ingest.db (default: data/output)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--min-ingest-version",
        type=int,
        default=6,
        help="Only backfill attachments whose root email has ingest_version >= N (default: 6)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="How many sample rows to print in the dry-run report (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = args.output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found at %s", db_path)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("[%s] Backfill attachment topic_ids from %s", mode, db_path)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Gather candidate rows:
    #   - attachment chunk rows (chunks.document_id -> documents where depth>0)
    #   - whose ROOT email has topics in chunk_topics (via email's body chunks)
    # and collect the topic_id set per email.
    sql = """
    WITH email_topics AS (
        SELECT d.id AS email_id, ct.topic_id
        FROM documents d
        JOIN chunks ec ON ec.document_id = d.id
        JOIN chunk_topics ct ON ct.chunk_id = ec.id
        WHERE d.document_type = 'email'
          AND d.depth = 0
          AND d.ingest_version >= ?
    )
    SELECT
        c.id AS chunk_id,
        c.metadata_json,
        et.email_id,
        et.topic_id
    FROM chunks c
    JOIN documents att ON att.id = c.document_id
    JOIN email_topics et ON et.email_id = att.root_id
    WHERE att.depth > 0
      AND NOT EXISTS (
        SELECT 1 FROM chunk_topics ct WHERE ct.chunk_id = c.id
      )
    """
    cur = con.execute(sql, (args.min_ingest_version,))

    # chunk_id -> (metadata_json, set[topic_id])
    pending: dict[str, tuple[str | None, set[str]]] = {}
    for row in cur:
        cid = row["chunk_id"]
        topics_for_chunk = pending.setdefault(cid, (row["metadata_json"], set()))[1]
        topics_for_chunk.add(row["topic_id"])
        # metadata_json captured on first seen; ignored on later rows (same chunk)
        pending[cid] = (pending[cid][0], topics_for_chunk)

    total_chunks = len(pending)
    total_links = sum(len(tids) for _, tids in pending.values())
    logger.info(
        "Found %d attachment chunks under %d topic-linked ingest_version>=%d emails",
        total_chunks,
        len({row["email_id"] for row in con.execute(sql, (args.min_ingest_version,))}),
        args.min_ingest_version,
    )
    logger.info("Total chunk_topics rows to insert: %d", total_links)

    if args.sample and total_chunks:
        logger.info("Sample (first %d):", args.sample)
        for i, (cid, (_, tids)) in enumerate(pending.items()):
            if i >= args.sample:
                break
            logger.info("  chunk=%s  topics=%s", cid, sorted(tids))

    if not args.apply:
        logger.info("[DRY-RUN] No writes performed. Re-run with --apply to commit.")
        con.close()
        return 0

    # Apply: one transaction for the whole backfill.
    link_inserts = 0
    meta_updates = 0
    with con:
        con.execute("BEGIN")
        for cid, (meta_json, tids) in pending.items():
            # Junction rows — INSERT OR IGNORE covers any race / re-run.
            con.executemany(
                "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) VALUES (?, ?)",
                [(cid, tid) for tid in sorted(tids)],
            )
            link_inserts += len(tids)

            # Keep the metadata blob in sync so JSONL export / Supabase sync
            # round-trip the field. The SQLite insert path strips topic_ids
            # from ``metadata_json`` on fresh inserts (line 1345 of
            # sqlite_client.py) and re-merges from chunk_topics on read
            # (line 1445), so an empty blob is fine functionally — but any
            # consumer that reads ``metadata_json`` directly (e.g.
            # ad-hoc SQL dumps) still benefits from a populated value.
            current: dict = {}
            if meta_json:
                try:
                    loaded = json.loads(meta_json)
                    if isinstance(loaded, dict):
                        current = loaded
                except (TypeError, ValueError):
                    current = {}
            current["topic_ids"] = sorted(tids)
            con.execute(
                "UPDATE chunks SET metadata_json = ? WHERE id = ?",
                (json.dumps(current), cid),
            )
            meta_updates += 1

    logger.info(
        "[APPLY] Inserted %d chunk_topics rows, updated metadata on %d chunks",
        link_inserts,
        meta_updates,
    )

    # Refresh topic chunk_count / document_count so the new membership shows
    # in validate output. Same SQL as ``_recompute_topic_counts`` in the
    # SQLite client (keeps the junction table authoritative).
    with con:
        con.execute("BEGIN")
        con.execute(
            "UPDATE topics SET "
            "  chunk_count = (SELECT COUNT(*) FROM chunk_topics ct WHERE ct.topic_id = topics.id), "
            "  document_count = (SELECT COUNT(DISTINCT c.document_id) "
            "                    FROM chunk_topics ct JOIN chunks c ON c.id = ct.chunk_id "
            "                    WHERE ct.topic_id = topics.id)"
        )
    logger.info("[APPLY] Recomputed topic chunk_count / document_count")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
