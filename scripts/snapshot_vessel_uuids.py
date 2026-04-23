"""Dump current vessel (name -> uuid) map from the Supabase project identified
by env. Used as a before/after snapshot when validating the upsert_vessel fix.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/snapshot_vessel_uuids.py > /tmp/vessels_before.json
    uv run mtss vessels import
    uv run python scripts/snapshot_vessel_uuids.py > /tmp/vessels_after.json
    diff /tmp/vessels_before.json /tmp/vessels_after.json
"""
from __future__ import annotations

import asyncio
import json
import sys

from mtss.storage.supabase_client import SupabaseClient


async def main() -> int:
    db = SupabaseClient()
    try:
        vessels = await db.get_all_vessels()
        payload = {v.name: str(v.id) for v in sorted(vessels, key=lambda x: x.name)}
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
