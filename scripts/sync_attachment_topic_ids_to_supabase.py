"""Sync attachment chunk ``topic_ids`` from local SQLite to Supabase.

Context
-------
Companion to ``backfill_attachment_topic_ids.py``.

The local backfill fixes SQLite (``chunk_topics`` + ``metadata_json``).
``mtss import`` is insert-only for existing ``doc_id`` rows (see
``_import_documents`` in ``src/mtss/cli/import_cmd.py``), so Supabase rows
imported before 2026-04-22 still have ``metadata`` blobs without
``topic_ids`` — the pgvector ``match_chunks`` topic filter never sees them.

This script updates Supabase rows in place, mirroring the local source of
truth. Runs as a single server-side UPDATE per parent email so network
round-trips stay bounded. No LLM, no re-parse, no re-embed.

Safety
------
- Dry-run by default; reports the blast radius before any write.
- Emits a rollback JSON (``<plan>.backup.json``) capturing the current
  ``metadata->'topic_ids'`` value of every affected row before writing.
- Wraps all updates in a single transaction; a failure partway through
  rolls back cleanly.
- Only affects chunks whose attachment document has ``root_id`` matching an
  email with topics locally — leaves unrelated chunks untouched.

Usage
-----
    # Dry-run (default). Reports counts + sample; no writes to Supabase.
    uv run python scripts/sync_attachment_topic_ids_to_supabase.py

    # Real run after inspecting the dry-run output.
    uv run python scripts/sync_attachment_topic_ids_to_supabase.py --apply

Exit codes: 0 success (incl. dry-run), 1 unrecoverable error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_local_assignments(db_path: Path, min_ingest_version: int) -> dict[str, list[str]]:
    """Return ``doc_id -> sorted(topic_ids)`` for every email that has topics.

    ``doc_id`` is the stable 16-char identifier shared between local SQLite
    and Supabase — NOT the UUID ``id`` column. Supabase joins on ``doc_id``
    for the same reason ``mtss import`` does.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            """
            SELECT d.doc_id AS email_doc_id, ct.topic_id
            FROM documents d
            JOIN chunks ec ON ec.document_id = d.id
            JOIN chunk_topics ct ON ct.chunk_id = ec.id
            WHERE d.document_type = 'email'
              AND d.depth = 0
              AND d.ingest_version >= ?
              AND d.doc_id IS NOT NULL
            """,
            (min_ingest_version,),
        )
        by_doc: dict[str, set[str]] = {}
        for row in cur:
            by_doc.setdefault(row["email_doc_id"], set()).add(row["topic_id"])
    finally:
        con.close()
    return {k: sorted(v) for k, v in by_doc.items()}


async def _async_main(args) -> int:
    # Lazy import so --help works without the full settings bootstrap.
    from mtss.storage.supabase_client import SupabaseClient

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("[%s] Sync attachment topic_ids → Supabase", mode)

    assignments = _load_local_assignments(
        args.output_dir / "ingest.db", args.min_ingest_version
    )
    logger.info(
        "Local emails (ingest_version>=%d) with topics: %d",
        args.min_ingest_version,
        len(assignments),
    )
    if not assignments:
        logger.info("Nothing to sync.")
        return 0

    db = SupabaseClient()
    try:
        pool = await db.get_pool()

        # Resolve local email doc_ids to Supabase document UUIDs (used as
        # attachments.root_id on the server). Do this in chunks of 1000 so
        # the array param stays comfortable on large corpora.
        email_doc_ids = list(assignments.keys())
        remote_email_rows: list[dict] = []
        async with pool.acquire() as conn:
            for i in range(0, len(email_doc_ids), 1000):
                batch = email_doc_ids[i : i + 1000]
                rows = await conn.fetch(
                    "SELECT doc_id, id FROM documents "
                    "WHERE doc_id = ANY($1::text[]) AND depth = 0",
                    batch,
                )
                remote_email_rows.extend(dict(r) for r in rows)

        email_uuid_by_doc = {r["doc_id"]: str(r["id"]) for r in remote_email_rows}
        missing_locally_mapped = [d for d in email_doc_ids if d not in email_uuid_by_doc]
        logger.info(
            "Matched %d of %d local emails to Supabase (missing: %d)",
            len(email_uuid_by_doc),
            len(email_doc_ids),
            len(missing_locally_mapped),
        )

        # Build update payload: list of (attachment_root_uuid, topic_ids[]).
        update_payload: list[tuple[str, list[str]]] = []
        for doc_id, topics in assignments.items():
            uid = email_uuid_by_doc.get(doc_id)
            if uid is None:
                continue
            update_payload.append((uid, topics))

        # Preview: count chunks that WOULD be updated.
        async with pool.acquire() as conn:
            preview_total = await conn.fetchval(
                """
                SELECT COUNT(*)::bigint FROM chunks c
                JOIN documents att ON att.id = c.document_id
                WHERE att.root_id = ANY($1::uuid[])
                  -- Cover email body chunks (depth=0) AND attachments (depth>0).
                  -- v=5 emails had no topic extraction at ingest, so BOTH are
                  -- missing topic_ids on Supabase. v=6 body chunks already
                  -- had topic_ids stamped so the NOT-populated filter skips
                  -- them; no risk of double-stamping.
                  AND att.depth >= 0
                  AND (
                    NOT (c.metadata ? 'topic_ids')
                    OR c.metadata->'topic_ids' = '[]'::jsonb
                  )
                """,
                [u for u, _ in update_payload],
            )
        logger.info("Attachment chunks missing topic_ids on Supabase: %s", preview_total)

        if args.sample and preview_total:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT c.id, att.doc_id AS att_doc_id, att.root_id
                    FROM chunks c JOIN documents att ON att.id = c.document_id
                    WHERE att.root_id = ANY($1::uuid[])
                      -- Cover email body chunks (depth=0) AND attachments (depth>0).
                  -- v=5 emails had no topic extraction at ingest, so BOTH are
                  -- missing topic_ids on Supabase. v=6 body chunks already
                  -- had topic_ids stamped so the NOT-populated filter skips
                  -- them; no risk of double-stamping.
                  AND att.depth >= 0
                      AND (NOT (c.metadata ? 'topic_ids') OR c.metadata->'topic_ids' = '[]'::jsonb)
                    LIMIT $2
                    """,
                    [u for u, _ in update_payload],
                    args.sample,
                )
            logger.info("Sample chunk ids to update:")
            for r in rows:
                logger.info("  chunk=%s  att_doc=%s", r["id"], r["att_doc_id"])

        if not args.apply:
            logger.info("[DRY-RUN] No writes performed. Re-run with --apply to commit.")
            return 0

        if preview_total == 0:
            logger.info("Nothing to update on Supabase.")
            return 0

        # Rollback snapshot of affected rows (pre-update metadata->topic_ids).
        snapshot_path = args.output_dir / "sync_attachment_topic_ids.backup.json"
        async with pool.acquire() as conn:
            backup_rows = await conn.fetch(
                """
                SELECT c.id::text AS id, c.metadata->'topic_ids' AS topic_ids
                FROM chunks c JOIN documents att ON att.id = c.document_id
                WHERE att.root_id = ANY($1::uuid[]) AND att.depth > 0
                  AND (NOT (c.metadata ? 'topic_ids') OR c.metadata->'topic_ids' = '[]'::jsonb)
                """,
                [u for u, _ in update_payload],
            )
        snapshot_path.write_text(
            json.dumps(
                [{"id": r["id"], "topic_ids": r["topic_ids"]} for r in backup_rows],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("Rollback snapshot: %s (%d rows)", snapshot_path, len(backup_rows))

        # Apply. One UPDATE per email filtered by document.id (not root_id),
        # so the planner uses the chunks.document_id index directly and avoids
        # a seq scan over ``documents`` that triggered Supabase's statement
        # timeout on the first run. Pre-resolve each email's attachment doc
        # UUIDs once, then batch the chunks UPDATE on ``c.document_id = ANY($1)``.
        # Runs inside a single transaction; statement timeout bumped so large
        # emails (50+ attachments) don't trip the pooler's default.
        updated_total = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL statement_timeout = 0")
                for i, (email_uuid, topics) in enumerate(update_payload, 1):
                    att_rows = await conn.fetch(
                        "SELECT id FROM documents WHERE root_id = $1::uuid",
                        email_uuid,
                    )
                    att_uuids = [str(r["id"]) for r in att_rows]
                    if not att_uuids:
                        continue
                    updated = await conn.execute(
                        """
                        UPDATE chunks
                        SET metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{topic_ids}',
                            $2::jsonb,
                            true
                        )
                        WHERE document_id = ANY($1::uuid[])
                          AND (
                            NOT (metadata ? 'topic_ids')
                            OR metadata->'topic_ids' = '[]'::jsonb
                          )
                        """,
                        att_uuids,
                        json.dumps(topics),
                    )
                    try:
                        updated_total += int(updated.rsplit(" ", 1)[-1])
                    except ValueError:
                        pass
                    if i % 200 == 0:
                        logger.info(
                            "  progress: %d/%d emails, %d chunks updated so far",
                            i, len(update_payload), updated_total,
                        )

        logger.info("[APPLY] Updated %d chunk row(s) on Supabase", updated_total)
        return 0
    finally:
        await db.close()


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
        help="Actually update Supabase. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--min-ingest-version",
        type=int,
        default=6,
        help="Only sync topics for local emails with ingest_version >= N (default: 6)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="How many sample chunk IDs to print in the dry-run report (default: 5)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file for Supabase credentials (default: .env). "
             "Use .env.test to target the test project.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from dotenv import load_dotenv
    if not args.env_file.exists():
        logger.error("env file not found: %s", args.env_file)
        return 1
    load_dotenv(args.env_file, override=True)
    logger.info("Loaded credentials from %s", args.env_file)

    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
