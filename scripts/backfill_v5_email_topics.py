"""Backfill chunk topic_ids for ingest_version=5 emails (body + attachments).

Background
----------
v=5 emails (~3,396 of them, ~9,324 docs, ~16,926 chunks in production) were
ingested before topic extraction was wired into the pipeline. As a result,
``chunk_topics`` has zero rows for any chunk under a v=5 email (body chunks
*and* attachment chunks), and ``chunks.metadata_json`` has no ``topic_ids``
key.

Everything from v=6 onward is already correct:

* v=6 email-body chunks were topic-stamped at ingest time.
* v=6 attachment chunks were backfilled by
  ``scripts/backfill_attachment_topic_ids.py`` (propagates the email-level
  topics onto its attachment rows — no LLM call).

v=5 needs a *fresh* extraction because the email itself never had topics —
we can't propagate what doesn't exist. This script runs ``TopicExtractor``
(gpt-4o-mini via the configured ``context_llm_model``) against the archived
markdown for each v=5 email, resolves the extracted topics through
``TopicMatcher.get_or_create_topics_batch`` against the existing local
ontology, and writes the resulting topic IDs onto every chunk under the
email (body + every attachment) via ``chunk_topics`` + a patched
``metadata_json``. The Supabase side is handled separately by
``scripts/sync_attachment_topic_ids_to_supabase.py``.

Flow (mirrors ``src/mtss/ingest/pipeline.py::_extract_topics``):

1. Find candidate emails: ``document_type='email' AND depth=0 AND
   ingest_version=5``.
2. Load ``archive/<folder_id>/email.md`` where ``folder_id =
   compute_folder_id(doc_id)``.
3. Build the topic input: ``Subject: ...`` + ``Content:\n<first 3000 chars
   after ## Message / ## Content / ---``>. Add the email body chunk's
   ``context_summary`` as ``Summary: ...`` if present.
4. ``extract_topics(input)`` -> LLM call (counts toward the run budget).
5. ``get_or_create_topics_batch([(name, desc), ...])`` -> UUIDs.
6. For every chunk under ``root_id = email.id`` -- ``INSERT OR IGNORE`` into
   ``chunk_topics`` + patch ``chunks.metadata_json`` to include the
   ``topic_ids`` array.
7. After the loop, recompute ``topics.chunk_count`` /
   ``topics.document_count``.

Cost estimate
-------------
3,396 emails x (~1,500 input tokens x $0.15/1M + ~300 output x $0.60/1M) ~=
**$1 - $2** for gpt-4o-mini. The dry-run itself runs extraction on
``--sample`` emails only (default 5) so proof-of-path is near-free.

Usage
-----
    # Dry-run (default): proves the pipeline on --sample emails and
    # reports counts + cost.
    uv run python scripts/backfill_v5_email_topics.py

    # Apply for real, bounded concurrency so we don't hammer OpenRouter.
    uv run python scripts/backfill_v5_email_topics.py --apply \\
        --concurrency 4

    # Scope to N emails (smoke-test + abort).
    uv run python scripts/backfill_v5_email_topics.py --apply --limit 50

Safety
------
* Local SQLite only. Does **not** touch Supabase. Run
  ``scripts/sync_attachment_topic_ids_to_supabase.py --apply`` separately
  after this script completes on ``./data``.
* Does not re-parse emails. Does not re-embed chunks.
* Per-email writes are wrapped in their own transaction so a mid-run
  failure leaves the DB consistent at the email boundary.

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.utils import compute_folder_id  # noqa: E402

logger = logging.getLogger(__name__)


# ── Small helpers ──────────────────────────────────────────────────────


def _loads(blob: str | None) -> dict[str, Any]:
    if not blob:
        return {}
    try:
        value = json.loads(blob)
        if isinstance(value, dict):
            return value
    except (TypeError, ValueError):
        return {}
    return {}


def _build_topic_input(
    subject: str | None,
    archived_md: str | None,
    context_summary: str | None,
) -> str:
    """Mirror of ``pipeline._extract_topics`` topic-input construction.

    Source of truth: ``src/mtss/ingest/pipeline.py`` lines 164-247.
    Kept in sync: if the pipeline logic changes, update both. The dupe is
    intentional — this script is a one-shot backfill and we don't want a
    cross-module import of a private helper making the pipeline harder to
    refactor later.
    """
    parts: list[str] = []

    if subject:
        parts.append(f"Subject: {subject}")

    topic_content = archived_md or ""
    if topic_content:
        is_markdown = topic_content.strip().startswith("#")
        if is_markdown:
            msg_start = topic_content.find("## Message")
            if msg_start == -1:
                msg_start = topic_content.find("## Content")
            if msg_start == -1:
                msg_start = topic_content.find("---")
                if msg_start != -1:
                    msg_start = topic_content.find("\n", msg_start + 3)
            message_content = (
                topic_content[msg_start:] if msg_start > 0 else topic_content
            )
            parts.append(f"Content:\n{message_content[:3000]}")
        else:
            # Archive markdown should always be markdown-shaped; take the
            # raw first 3000 chars as a defensive fallback.
            parts.append(f"Content:\n{topic_content[:3000]}")

    if context_summary:
        parts.append(f"Summary: {context_summary}")

    return "\n\n".join(parts).strip()


# ── list_all_topics_lightweight shim ───────────────────────────────────


def _attach_list_topics_shim(client) -> None:
    """Give ``SqliteStorageClient`` the method ``TopicCache`` expects.

    ``entity_cache.TopicCache.ensure_loaded`` calls
    ``db.list_all_topics_lightweight()``; the Supabase client has it but the
    SQLite client does not (the method is only exercised by Supabase-backed
    paths in production). We monkey-patch an equivalent reader that pulls
    the same columns out of the local ``topics`` table — no embeddings, so
    the cache stays lean.
    """

    async def list_all_topics_lightweight() -> list[SimpleNamespace]:
        rows = client._conn.execute(
            """
            SELECT id, name, display_name, description,
                   chunk_count, document_count,
                   created_at, updated_at
            FROM topics
            """
        ).fetchall()
        out: list[SimpleNamespace] = []
        for row in rows:
            out.append(
                SimpleNamespace(
                    id=UUID(row["id"]),
                    name=row["name"],
                    display_name=row["display_name"] or row["name"],
                    description=row["description"],
                    embedding=None,
                    chunk_count=row["chunk_count"],
                    document_count=row["document_count"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return out

    client.list_all_topics_lightweight = list_all_topics_lightweight  # type: ignore[attr-defined]


# ── Candidate loading ──────────────────────────────────────────────────


def _load_candidates(
    con: sqlite3.Connection, limit: int | None
) -> list[dict[str, Any]]:
    """Return v=5 email rows with subject, folder path, body context_summary."""
    sql = """
    SELECT d.id          AS email_row_id,
           d.doc_id       AS doc_id,
           d.source_title AS source_title,
           d.metadata_json AS metadata_json,
           (
             SELECT c.context_summary
             FROM chunks c
             WHERE c.document_id = d.id
             ORDER BY c.chunk_index
             LIMIT 1
           ) AS body_context_summary
      FROM documents d
     WHERE d.document_type = 'email'
       AND d.depth = 0
       AND d.ingest_version = 5
     ORDER BY d.id
    """
    if limit is not None:
        sql += f"\n LIMIT {int(limit)}"
    cur = con.execute(sql)
    out: list[dict[str, Any]] = []
    for row in cur:
        meta = _loads(row["metadata_json"])
        out.append(
            {
                "email_row_id": row["email_row_id"],
                "doc_id": row["doc_id"],
                "subject": meta.get("email_subject") or row["source_title"] or "",
                "context_summary": row["body_context_summary"],
            }
        )
    return out


def _read_archived_email_md(output_dir: Path, doc_id: str) -> str | None:
    folder = output_dir / "archive" / compute_folder_id(doc_id)
    # Historical / current layouts both exist: older archives wrote
    # ``email.md``, current ingest writes ``email.eml.md`` (keeps the
    # extension visible for consistency with attachments). Try both.
    for candidate in (folder / "email.eml.md", folder / "email.md"):
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def _chunks_under_email(
    con: sqlite3.Connection, email_row_id: str
) -> list[dict[str, Any]]:
    """All chunks whose document's ``root_id`` is the email row id."""
    sql = """
    SELECT c.id AS chunk_id, c.metadata_json AS metadata_json
      FROM chunks c
      JOIN documents d ON d.id = c.document_id
     WHERE d.root_id = ?
    """
    return [
        {"chunk_id": row["chunk_id"], "metadata_json": row["metadata_json"]}
        for row in con.execute(sql, (email_row_id,))
    ]


def _write_topics_for_email(
    con: sqlite3.Connection,
    email_row_id: str,
    topic_ids: Iterable[str],
) -> tuple[int, int]:
    """Apply (topic_ids) to every chunk under ``email_row_id``.

    Wrapped in a single transaction. Returns (inserted_chunk_topic_rows,
    updated_metadata_rows).
    """
    topic_id_list = sorted({str(t) for t in topic_ids})
    chunks = _chunks_under_email(con, email_row_id)
    if not chunks or not topic_id_list:
        return 0, 0

    inserts = 0
    meta_updates = 0
    try:
        con.execute("BEGIN")
        for entry in chunks:
            cid = entry["chunk_id"]
            # Junction rows (INSERT OR IGNORE — safe on re-run).
            con.executemany(
                "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) VALUES (?, ?)",
                [(cid, tid) for tid in topic_id_list],
            )
            inserts += len(topic_id_list)

            # Merge topic_ids into chunks.metadata_json so ad-hoc SQL dumps
            # + downstream tools see the field. Mirrors the pattern in
            # ``scripts/backfill_attachment_topic_ids.py``.
            current = _loads(entry["metadata_json"])
            current["topic_ids"] = topic_id_list
            con.execute(
                "UPDATE chunks SET metadata_json = ? WHERE id = ?",
                (json.dumps(current), cid),
            )
            meta_updates += 1
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    return inserts, meta_updates


def _recompute_topic_counts(con: sqlite3.Connection) -> None:
    """Same SQL as the attachment backfill — keeps the junction authoritative."""
    with con:
        con.execute("BEGIN")
        con.execute(
            "UPDATE topics SET "
            "  chunk_count = (SELECT COUNT(*) FROM chunk_topics ct "
            "                 WHERE ct.topic_id = topics.id), "
            "  document_count = (SELECT COUNT(DISTINCT c.document_id) "
            "                    FROM chunk_topics ct "
            "                    JOIN chunks c ON c.id = ct.chunk_id "
            "                    WHERE ct.topic_id = topics.id)"
        )


# ── Topic extraction (async per email, sem-bounded) ────────────────────


async def _extract_topic_ids_for_email(
    topic_extractor,
    topic_matcher,
    topic_input: str,
    doc_id: str,
) -> list[str]:
    """Mirror of ``pipeline._extract_topics`` LLM + resolve path.

    Returns ``[]`` on LLM failure / unparseable response — the caller logs
    and continues (same pattern as pipeline.py lines 245-247).
    """
    try:
        extracted = await topic_extractor.extract_topics(topic_input)
    except Exception as e:  # noqa: BLE001 — mirror pipeline's broad except
        logger.warning("[%s] topic extraction failed: %s", doc_id, e)
        return []

    if not extracted:
        return []

    try:
        batch_ids = await topic_matcher.get_or_create_topics_batch(
            [(t.name, t.description) for t in extracted]
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] topic matching failed: %s", doc_id, e)
        return []

    return [str(tid) for tid in batch_ids]


async def _run(args: argparse.Namespace) -> int:
    db_path: Path = args.output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found at %s", db_path)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("[%s] v=5 email topic backfill from %s", mode, db_path)

    # Imports deferred so `--help` stays fast and doesn't eat the import
    # cost of the embeddings stack when the user just wants docs.
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.processing.topics import TopicExtractor, TopicMatcher
    from mtss.storage.sqlite_client import SqliteStorageClient

    client = SqliteStorageClient(output_dir=args.output_dir)
    _attach_list_topics_shim(client)
    embeddings = EmbeddingGenerator()
    topic_extractor = TopicExtractor()
    topic_matcher = TopicMatcher(client, embeddings)

    con = client._conn  # reuse the same connection so reads see our writes

    candidates = _load_candidates(con, args.limit)
    logger.info("Candidate v=5 emails: %d", len(candidates))

    # Estimate chunk-link count by summing chunks under each candidate — one
    # SELECT per email is fine at 3,396 candidates (it's a cheap indexed scan).
    total_chunks = 0
    missing_archive = 0
    empty_input = 0
    for entry in candidates:
        chunks = _chunks_under_email(con, entry["email_row_id"])
        total_chunks += len(chunks)
        md = _read_archived_email_md(args.output_dir, entry["doc_id"])
        if md is None:
            missing_archive += 1
            entry["_archived_md"] = None
        else:
            entry["_archived_md"] = md
        topic_input = _build_topic_input(
            entry["subject"], entry["_archived_md"], entry["context_summary"]
        )
        entry["_topic_input"] = topic_input
        if not topic_input:
            empty_input += 1

    logger.info(
        "Reachable chunks to link: %d (across %d emails); missing archive: %d; "
        "empty topic input: %d",
        total_chunks,
        len(candidates),
        missing_archive,
        empty_input,
    )

    # Cost estimate (rough, input+output for gpt-4o-mini):
    #   in ~1,500 tok * $0.15/1M = $0.000225 per email
    #   out ~300 tok * $0.60/1M  = $0.00018 per email
    # -> ~$0.000405 per email, ~$1.40 at 3,396 emails
    per_email_usd = (1500 * 0.15 + 300 * 0.60) / 1_000_000
    est_cost = per_email_usd * len(candidates)
    logger.info(
        "Estimated LLM cost (gpt-4o-mini @ ~1.5k in / ~300 out tok): "
        "~$%.2f for %d emails",
        est_cost,
        len(candidates),
    )

    if not candidates:
        logger.info("Nothing to do.")
        return 0

    # ── Dry-run path: extract topics on the first --sample emails only ──
    if not args.apply:
        sample_n = min(args.sample, len(candidates))
        logger.info(
            "[DRY-RUN] Running topic extraction on %d sample email(s) to "
            "prove the path (no DB writes)",
            sample_n,
        )
        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def _sample_one(entry: dict[str, Any]) -> None:
            async with sem:
                if not entry["_topic_input"]:
                    logger.info(
                        "  [skip] %s — empty topic input",
                        entry["doc_id"],
                    )
                    return
                tids = await _extract_topic_ids_for_email(
                    topic_extractor,
                    topic_matcher,
                    entry["_topic_input"],
                    entry["doc_id"],
                )
                logger.info(
                    "  %s subject=%r -> %d topics: %s",
                    entry["doc_id"],
                    (entry["subject"] or "")[:60],
                    len(tids),
                    tids,
                )

        await asyncio.gather(*(_sample_one(c) for c in candidates[:sample_n]))
        logger.info("[DRY-RUN] No writes performed. Re-run with --apply to commit.")
        return 0

    # ── Apply path ──────────────────────────────────────────────────────
    sem = asyncio.Semaphore(max(1, args.concurrency))
    emails_processed = 0
    emails_skipped_empty = 0
    emails_topics_zero = 0
    total_links_inserted = 0
    total_meta_updated = 0
    llm_calls = 0
    write_lock = asyncio.Lock()  # serialise sqlite writes (single conn)

    async def _process(entry: dict[str, Any]) -> None:
        nonlocal emails_processed, emails_skipped_empty, emails_topics_zero
        nonlocal total_links_inserted, total_meta_updated, llm_calls

        async with sem:
            doc_id = entry["doc_id"]
            topic_input = entry["_topic_input"]
            if not topic_input:
                emails_skipped_empty += 1
                logger.info("[skip] %s — empty topic input", doc_id)
                return

            llm_calls += 1
            topic_ids = await _extract_topic_ids_for_email(
                topic_extractor, topic_matcher, topic_input, doc_id
            )
            if not topic_ids:
                emails_topics_zero += 1
                return

            # Writes must be serialised — sqlite3.Connection is not thread
            # or task safe for concurrent writes. Async coroutines are
            # single-threaded but we still want only one BEGIN...COMMIT
            # in flight at a time.
            async with write_lock:
                inserts, meta = _write_topics_for_email(
                    con, entry["email_row_id"], topic_ids
                )
            total_links_inserted += inserts
            total_meta_updated += meta
            emails_processed += 1

            if emails_processed and emails_processed % 100 == 0:
                logger.info(
                    "[progress] %d/%d emails done, %d chunk_topics inserted, "
                    "%d LLM calls",
                    emails_processed,
                    len(candidates),
                    total_links_inserted,
                    llm_calls,
                )

    await asyncio.gather(*(_process(c) for c in candidates))

    logger.info(
        "[APPLY] Done. emails_processed=%d emails_with_zero_topics=%d "
        "emails_empty_input=%d llm_calls=%d chunk_topics_inserted=%d "
        "metadata_updated=%d",
        emails_processed,
        emails_topics_zero,
        emails_skipped_empty,
        llm_calls,
        total_links_inserted,
        total_meta_updated,
    )

    _recompute_topic_counts(con)
    logger.info("[APPLY] Recomputed topic chunk_count / document_count")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing ingest.db + archive/ (default: data/output)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N candidate v=5 emails (default: all).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max parallel LLM extractions (default: 4).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="How many emails to run through extraction in dry-run "
        "(default: 5).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
