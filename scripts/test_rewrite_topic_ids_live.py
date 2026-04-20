"""End-to-end live test for the rewrite_chunk_topic_ids Postgres RPC.

Blast-radius context
--------------------
The RPC mutates `chunks.metadata.topic_ids` in place. A botched run
leaves live data pointing at stale topic UUIDs and forces a full
reingest (hours of downtime), so this script is the trusted
pre-flight: apply the migration, seed synthetic rows with known
topic_ids, call the RPC, and assert every case — single absorbed
ref, dedup against existing keeper, unrelated chunk, idempotence —
behaves exactly as the migration promises.

Runs against whatever SUPABASE_DB_URL points at. Cleans up after
itself regardless of pass/fail so a half-failed assertion does not
leave test rows behind. Safe to run repeatedly.

Usage::

    uv run python scripts/test_rewrite_topic_ids_live.py

Exits 0 on all-green, non-zero with a descriptive error otherwise.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg

MIGRATION_FILE = Path(__file__).resolve().parent.parent / "migrations" / "001_topic_rewrite_rpc.sql"


async def _ensure_migration(conn: asyncpg.Connection) -> None:
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    await conn.execute(sql)
    exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'rewrite_chunk_topic_ids')"
    )
    if not exists:
        raise RuntimeError("migration applied but rewrite_chunk_topic_ids not found in pg_proc")


async def _cleanup(conn: asyncpg.Connection, tag: str) -> None:
    """Remove every test row we inserted under ``tag`` regardless of prior state."""
    # Chunks cascade from documents via FK; delete documents first (the tag
    # lives in source_id so it is easy to target without touching real data).
    await conn.execute(
        "DELETE FROM documents WHERE source_id = $1", f"rpc-test-{tag}.eml"
    )
    await conn.execute(
        "DELETE FROM topics WHERE description = $1", f"rpc-live-test::{tag}"
    )


async def _seed(conn: asyncpg.Connection, tag: str) -> dict:
    """Insert synthetic topics + document + chunks for the test cases."""
    keeper_id = await conn.fetchval(
        "INSERT INTO topics(name, display_name, description, chunk_count, document_count) "
        "VALUES ($1, $2, $3, 0, 0) RETURNING id",
        f"rpc-test-keeper-{tag}", f"RPC Test Keeper {tag}", f"rpc-live-test::{tag}",
    )
    absorbed_id = await conn.fetchval(
        "INSERT INTO topics(name, display_name, description, chunk_count, document_count) "
        "VALUES ($1, $2, $3, 0, 0) RETURNING id",
        f"rpc-test-absorbed-{tag}", f"RPC Test Absorbed {tag}", f"rpc-live-test::{tag}",
    )
    unrelated_id = await conn.fetchval(
        "INSERT INTO topics(name, display_name, description, chunk_count, document_count) "
        "VALUES ($1, $2, $3, 0, 0) RETURNING id",
        f"rpc-test-unrelated-{tag}", f"RPC Test Unrelated {tag}", f"rpc-live-test::{tag}",
    )

    doc_id_local = str(uuid4())
    doc_pk = await conn.fetchval(
        """
        INSERT INTO documents(
            doc_id, file_name, file_path, file_hash, source_id,
            content_version, ingest_version, document_type, status, depth
        )
        VALUES ($1, $2, $3, $4, $5, 1, 1, 'email', 'completed', 0)
        RETURNING id
        """,
        doc_id_local,
        f"rpc-test-{tag}.eml",
        f"/tmp/rpc-test-{tag}.eml",
        f"hash-{tag}",
        f"rpc-test-{tag}.eml",
    )

    # Three chunks:
    #   c1 — references ONLY the absorbed topic. Post-rewrite must carry only the keeper.
    #   c2 — references BOTH absorbed + keeper. Post-rewrite must dedupe to keeper-only.
    #   c3 — references the unrelated topic. Post-rewrite must be unchanged.
    c1_pk, c2_pk, c3_pk = uuid4(), uuid4(), uuid4()
    await conn.executemany(
        """
        INSERT INTO chunks(
            id, document_id, chunk_id, content, chunk_index,
            char_start, char_end, line_from, line_to, section_path,
            metadata
        )
        VALUES ($1, $2, $3, $4, $5, 0, 0, 1, 1, $6, $7::jsonb)
        """,
        [
            (
                c1_pk, doc_pk, f"{tag}-chunk-1", "content-1", 0,
                [],
                json.dumps({"topic_ids": [str(absorbed_id)]}),
            ),
            (
                c2_pk, doc_pk, f"{tag}-chunk-2", "content-2", 1,
                [],
                json.dumps({"topic_ids": [str(absorbed_id), str(keeper_id)]}),
            ),
            (
                c3_pk, doc_pk, f"{tag}-chunk-3", "content-3", 2,
                [],
                json.dumps({"topic_ids": [str(unrelated_id)]}),
            ),
        ],
    )
    return {
        "keeper_id": str(keeper_id),
        "absorbed_id": str(absorbed_id),
        "unrelated_id": str(unrelated_id),
        "doc_pk": str(doc_pk),
        "c1_pk": str(c1_pk),
        "c2_pk": str(c2_pk),
        "c3_pk": str(c3_pk),
    }


async def _topic_ids_for(conn: asyncpg.Connection, chunk_pk: str) -> list[str]:
    row = await conn.fetchrow(
        "SELECT metadata->'topic_ids' AS topic_ids FROM chunks WHERE id = $1", chunk_pk
    )
    if row is None or row["topic_ids"] is None:
        return []
    parsed = json.loads(row["topic_ids"])
    return list(parsed)


async def main() -> int:
    import os
    # Prefer SUPABASE_DB_URL; fall back to DATABASE_URL.
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL (or DATABASE_URL) not set")
        return 2

    conn = await asyncpg.connect(db_url)
    tag = uuid4().hex[:12]
    failures: list[str] = []

    try:
        print(f"[1/5] applying migration from {MIGRATION_FILE.name}")
        await _ensure_migration(conn)

        print(f"[2/5] seeding test data (tag={tag})")
        ids = await _seed(conn, tag)

        mapping = {ids["absorbed_id"]: ids["keeper_id"]}
        print(f"[3/5] calling rewrite_chunk_topic_ids with mapping {mapping}")
        updated = await conn.fetchval(
            "SELECT rewrite_chunk_topic_ids($1::jsonb)", json.dumps(mapping)
        )
        print(f"      RPC reported {updated} row(s) updated")
        if updated != 2:
            failures.append(
                f"expected RPC to report 2 affected rows, got {updated}"
            )

        print("[4/5] verifying post-rewrite state")
        c1 = await _topic_ids_for(conn, ids["c1_pk"])
        c2 = await _topic_ids_for(conn, ids["c2_pk"])
        c3 = await _topic_ids_for(conn, ids["c3_pk"])

        if c1 != [ids["keeper_id"]]:
            failures.append(f"c1 (only-absorbed) must become [keeper]; got {c1}")
        if sorted(c2) != [ids["keeper_id"]]:
            failures.append(
                f"c2 (absorbed+keeper) must dedupe to [keeper]; got {c2}"
            )
        if c3 != [ids["unrelated_id"]]:
            failures.append(f"c3 (unrelated) must stay unchanged; got {c3}")

        # Idempotence: running the RPC a second time with the same mapping
        # should affect zero rows because no absorbed UUIDs remain.
        updated_again = await conn.fetchval(
            "SELECT rewrite_chunk_topic_ids($1::jsonb)", json.dumps(mapping)
        )
        if updated_again != 0:
            failures.append(
                f"second call should be a no-op; got {updated_again} row(s) updated"
            )

        # Input guard: the RPC refuses non-object input.
        try:
            await conn.fetchval(
                "SELECT rewrite_chunk_topic_ids($1::jsonb)", json.dumps([1, 2, 3])
            )
            failures.append("RPC accepted non-object input — input validation broken")
        except asyncpg.exceptions.RaiseError:
            pass  # expected

    finally:
        print(f"[5/5] cleaning up test rows (tag={tag})")
        try:
            await _cleanup(conn, tag)
        except Exception as exc:
            print(f"WARNING: cleanup raised {exc!r} — inspect DB for leftover rows")
        await conn.close()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nALL GREEN — RPC behaves as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
