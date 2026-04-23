"""Re-stamp chunks.metadata vessel_ids/types/classes using DB content only.

Why this exists (and not `mtss vessels retag`): the built-in command downloads
archive markdown keyed off `documents.doc_id`, but the actual storage folder
is `documents.archive_path` (a separate hash). That means the built-in finds
zero files on the current corpus. This script sidesteps storage entirely —
all the text we need to match vessel names (title, file_name, email_subject)
is already in the DB.

Scope per root document:
  - Collect text = title + file_name + metadata.email_subject, union'd across
    the root email and every attachment (root_id=self or id=self).
  - Run VesselMatcher against the combined text.
  - Compare against the current `vessel_ids` on the root document's chunks.
  - If changed, update via `update_chunks_vessel_metadata` (UPDATE touches
    the root doc's chunks AND every child doc's chunks, because those
    references are all the same logical email thread).

Defaults to --dry-run. Pass --apply to write.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/retag_vessel_ids_from_db.py --dry-run
    uv run python scripts/retag_vessel_ids_from_db.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set
from uuid import UUID

from mtss.processing.vessel_matcher import VesselMatcher
from mtss.storage.supabase_client import SupabaseClient


async def _collect_doc_text(pool, root_id: UUID) -> str:
    """Return concatenated text from the root doc + all descendants.

    Remote schema flattens email fields to columns (source_title, file_name,
    email_subject) rather than nesting them inside a metadata JSON blob —
    which is what the local SQLite store does. We read the remote shape here
    because that's where the retag runs.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_title, file_name, email_subject
            FROM documents
            WHERE id = $1 OR root_id = $1
            """,
            root_id,
        )
    parts: List[str] = []
    for row in rows:
        if row["source_title"]:
            parts.append(row["source_title"])
        if row["file_name"]:
            parts.append(row["file_name"])
        if row["email_subject"]:
            parts.append(row["email_subject"])
    return "\n".join(parts)


async def _get_current_vessel_ids(pool, root_id: UUID) -> Set[str]:
    """Union of vessel_ids across all chunks belonging to the root doc tree."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT jsonb_array_elements_text(metadata->'vessel_ids') AS vid
            FROM chunks c
            WHERE c.document_id IN (
                SELECT id FROM documents WHERE id = $1 OR root_id = $1
            )
            """,
            root_id,
        )
    return {r["vid"] for r in rows if r["vid"]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (flips off --dry-run)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Parallel root-doc workers")
    args = ap.parse_args()
    dry_run = not args.apply

    db = SupabaseClient()
    try:
        # Load vessel registry from the target Supabase project — NOT from
        # the CSV. load_vessels_from_csv mints fresh uuid4 per row every
        # call, so CSV-derived UUIDs never match the vessels table and any
        # chunks tagged with those UUIDs become instantly orphaned. Source
        # of truth for the retag is always the live DB.
        vessels = await db.get_all_vessels()
        if not vessels:
            print("No vessels in target DB — abort", file=sys.stderr)
            return 2
        matcher = VesselMatcher(vessels)
        print(f"Loaded {matcher.vessel_count} vessels from DB ({matcher.name_count} names)")

        pool = await db._domain.get_pool()

        # Pull root email documents (depth=0). These drive retag scope.
        async with pool.acquire() as conn:
            roots = await conn.fetch(
                """
                SELECT id, doc_id, file_name, source_title
                FROM documents
                WHERE depth = 0 AND status = 'completed'
                ORDER BY created_at DESC
                """
                + (f"\nLIMIT {int(args.limit)}" if args.limit else "")
            )
        print(f"Scanning {len(roots)} root documents (dry_run={dry_run}, "
              f"concurrency={args.concurrency})")

        sem = asyncio.Semaphore(args.concurrency)
        counters = Counter()
        per_vessel_deltas: Counter = Counter()

        async def process_one(row) -> None:
            async with sem:
                root_id: UUID = row["id"]
                try:
                    text = await _collect_doc_text(pool, root_id)
                    matched_uuid_set = matcher.find_vessels(text)
                    new_ids = sorted(str(v) for v in matched_uuid_set)
                    new_types = matcher.get_types_for_ids(matched_uuid_set)
                    new_classes = matcher.get_classes_for_ids(matched_uuid_set)

                    current_ids = await _get_current_vessel_ids(pool, root_id)
                    added = set(new_ids) - current_ids
                    removed = current_ids - set(new_ids)

                    if added or removed:
                        counters["docs_changed"] += 1
                        counters["vessel_tags_added"] += len(added)
                        counters["vessel_tags_removed"] += len(removed)
                        for v in added:
                            per_vessel_deltas[f"+{v}"] += 1
                        for v in removed:
                            per_vessel_deltas[f"-{v}"] += 1
                        if not dry_run:
                            updated = await db._domain.update_chunks_vessel_metadata(
                                root_id, new_ids, new_types, new_classes
                            )
                            counters["chunks_updated"] += updated
                    else:
                        counters["docs_unchanged"] += 1
                    counters["docs_scanned"] += 1
                except Exception as e:
                    counters["docs_errored"] += 1
                    print(f"! {row['file_name']} ({row['doc_id']}): {e}",
                          file=sys.stderr)

        # Batch progress: print every 500 docs so long runs stay legible.
        batch_size = 500
        for start in range(0, len(roots), batch_size):
            batch = roots[start : start + batch_size]
            await asyncio.gather(*(process_one(r) for r in batch))
            print(f"  {start + len(batch)}/{len(roots)}  "
                  f"changed={counters['docs_changed']} "
                  f"+tags={counters['vessel_tags_added']} "
                  f"-tags={counters['vessel_tags_removed']} "
                  f"chunks_updated={counters['chunks_updated']}",
                  flush=True)

        print()
        print("=== SUMMARY ===")
        for k, v in counters.most_common():
            print(f"  {k:22s} {v}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
