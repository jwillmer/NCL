"""Diagnose whether chunk.metadata.vessel_ids still reference valid vessels.

Problem: `upsert_vessel` (src/mtss/storage/repositories/domain.py:41) sends a
freshly-generated UUID on every import. Postgres upsert-on-name-conflict does
UPDATE including the id column -> existing vessels silently get new UUIDs,
orphaning every chunk.metadata.vessel_ids reference.

This script is read-only. It loads:
  - current vessels table (name -> current UUID)
  - chunks.metadata.vessel_ids across a sample of chunks
  - counts which vessel_ids referenced by chunks are still present in vessels

Prints a summary: total chunks with vessel_ids, hits, orphans, and per-name
stats (how many chunks reference each current vessel UUID). If most chunks
are orphan, UUIDs have been corrupted and need restoration.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/diagnose_vessel_uuid_integrity.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter

from mtss.storage.supabase_client import SupabaseClient


async def main() -> int:
    db = SupabaseClient()
    try:
        # Load current vessels table
        vessels = await db.get_all_vessels()
        current_uuids = {str(v.id) for v in vessels}
        name_by_uuid = {str(v.id): v.name for v in vessels}
        print(f"Loaded {len(vessels)} vessels from vessels table")

        # Sample chunks with vessel_ids in metadata. PostgREST has a default
        # row cap; fetch in pages.
        client = db._domain.client  # low-level for this diagnostic
        total_chunks_with_vessel = 0
        total_vessel_id_refs = 0
        orphan_refs = 0
        hit_refs = 0
        uuid_hits: Counter = Counter()
        orphan_uuids: Counter = Counter()

        page = 0
        page_size = 1000
        while True:
            res = (
                client.table("chunks")
                .select("metadata")
                .not_.is_("metadata", "null")
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            rows = res.data or []
            if not rows:
                break
            for row in rows:
                meta = row.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        continue
                vids = meta.get("vessel_ids") or []
                if not vids:
                    continue
                total_chunks_with_vessel += 1
                for vid in vids:
                    total_vessel_id_refs += 1
                    if vid in current_uuids:
                        hit_refs += 1
                        uuid_hits[vid] += 1
                    else:
                        orphan_refs += 1
                        orphan_uuids[vid] += 1
            if len(rows) < page_size:
                break
            page += 1

        print(f"\nChunks scanned with vessel_ids metadata: {total_chunks_with_vessel}")
        print(f"Total vessel_id references: {total_vessel_id_refs}")
        print(f"  Valid (match vessels table): {hit_refs}")
        print(f"  Orphan (no matching vessel): {orphan_refs}")
        if total_vessel_id_refs:
            pct_orphan = 100.0 * orphan_refs / total_vessel_id_refs
            print(f"  Orphan ratio: {pct_orphan:.1f}%")

        if hit_refs:
            print("\nTop 10 vessels actually referenced by chunks:")
            for uuid, n in uuid_hits.most_common(10):
                print(f"  {n:6d}  {uuid}  {name_by_uuid.get(uuid, '?')}")

        if orphan_refs:
            print("\nTop 10 orphan vessel UUIDs (in chunks but not in vessels table):")
            for uuid, n in orphan_uuids.most_common(10):
                print(f"  {n:6d}  {uuid}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
