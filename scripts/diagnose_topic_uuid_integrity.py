"""Diagnose whether chunk.metadata.topic_ids still reference valid topics.

Paired with diagnose_vessel_uuid_integrity.py — same idea, different table.
Reports orphan ratio so we can confirm topics don't have the same
UUID-clobbering issue that vessels did.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/diagnose_topic_uuid_integrity.py
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter

from mtss.storage.supabase_client import SupabaseClient


async def main() -> int:
    db = SupabaseClient()
    try:
        pool = await db._domain.get_pool()
        async with pool.acquire() as conn:
            topic_ids_live = {
                str(r["id"])
                for r in await conn.fetch("SELECT id FROM topics")
            }
        print(f"Loaded {len(topic_ids_live)} topics from topics table")

        async with pool.acquire() as conn:
            # Pull all topic_id refs from chunks.metadata.topic_ids as one
            # server-side unnest — faster than streaming metadata blobs.
            rows = await conn.fetch(
                """
                SELECT tid AS tid, count(*) AS n
                FROM chunks,
                     LATERAL jsonb_array_elements_text(metadata->'topic_ids') AS tid
                WHERE metadata ? 'topic_ids'
                GROUP BY tid
                """
            )

        hit = 0
        orphan = 0
        orphan_top: Counter = Counter()
        for r in rows:
            if r["tid"] in topic_ids_live:
                hit += int(r["n"])
            else:
                orphan += int(r["n"])
                orphan_top[r["tid"]] += int(r["n"])

        # Also count chunks that have any topic_ids populated (for scale)
        async with pool.acquire() as conn:
            chunks_with_topics = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE metadata ? 'topic_ids' "
                "AND jsonb_array_length(metadata->'topic_ids') > 0"
            )

        total_refs = hit + orphan
        print(f"\nChunks with topic_ids: {chunks_with_topics}")
        print(f"Total topic_id references: {total_refs}")
        print(f"  Valid (match topics table): {hit}")
        print(f"  Orphan (no matching topic):  {orphan}")
        if total_refs:
            pct = 100.0 * orphan / total_refs
            print(f"  Orphan ratio: {pct:.2f}%")

        if orphan_top:
            print("\nTop 10 orphan topic UUIDs:")
            for uuid, n in orphan_top.most_common(10):
                print(f"  {n:6d}  {uuid}")

        # Cross-check: chunk_topics junction table consistency (if used)
        async with pool.acquire() as conn:
            junction_orphans = await conn.fetchval(
                """
                SELECT count(*) FROM chunk_topics ct
                LEFT JOIN topics t ON t.id = ct.topic_id
                WHERE t.id IS NULL
                """
            )
        print(f"\nchunk_topics junction rows with missing topic: {junction_orphans}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
