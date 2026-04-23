"""Re-stamp chunks.metadata vessel_ids/types/classes using DB content only.

Delegates to `mtss.storage.vessel_retag.retag_vessels` so the standalone
manual pass and the post-import auto-heal share one implementation.

Why this exists (and not `mtss vessels retag`): the built-in command
downloads archive markdown keyed off `documents.doc_id`, but the actual
storage folder is `documents.archive_path` (a separate hash). That means
the built-in finds zero files on the current corpus. This script sidesteps
storage entirely — all the text we need to match vessel names (source_title,
file_name, email_subject) is already in the DB.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/retag_vessel_ids_from_db.py --dry-run
    uv run python scripts/retag_vessel_ids_from_db.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from mtss.storage.supabase_client import SupabaseClient
from mtss.storage.vessel_retag import retag_vessels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (flips off --dry-run)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap root docs processed (debug only)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Parallel root-doc workers")
    args = ap.parse_args()
    dry_run = not args.apply

    db = SupabaseClient()
    try:
        if args.limit:
            # Preserve --limit behaviour by pre-picking the scope.
            pool = await db._domain.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id FROM documents
                    WHERE depth = 0 AND status = 'completed'
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    args.limit,
                )
            scope = [r["id"] for r in rows]
        else:
            scope = None

        stats = await retag_vessels(
            db,
            root_ids=scope,
            concurrency=args.concurrency,
            dry_run=dry_run,
        )
        print()
        print("=== SUMMARY ===")
        for k, v in stats.as_dict().items():
            print(f"  {k:22s} {v}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
