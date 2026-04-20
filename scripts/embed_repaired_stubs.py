"""Embed the filename-stub chunks inserted by
``repair_metadata_only_missing_chunks.py``.

Background
----------
The repair script inserts metadata stubs but intentionally leaves the
embedding blob empty — it has no event loop and doesn't want to own
LiteLLM / OpenRouter configuration. Validate check #5
``embedding_completeness`` flags any chunk without an embedding, so
this follow-up pass embeds only the stubs tagged by the repair script
(metadata_json carries a ``repaired_by`` sentinel).

Identity is restricted to stubs *missing* embeddings; re-running is
safe — chunks that already have a blob are skipped.

Usage
-----
    uv run python scripts/embed_repaired_stubs.py --output-dir data/output
    uv run python scripts/embed_repaired_stubs.py --output-dir data/output --apply

Exit codes: 0 = success (including dry-run), 1 = error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("embed_repaired_stubs")

_SENTINEL = "repair_metadata_only_missing_chunks"


def _candidate_chunks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, chunk_id, content, embedding_text
        FROM chunks
        WHERE embedding IS NULL
          AND metadata_json LIKE ?
        ORDER BY chunk_id
        """,
        (f"%{_SENTINEL}%",),
    ).fetchall()


async def _embed_all(
    rows: list[sqlite3.Row],
) -> list[tuple[str, bytes, int]]:
    """Return (chunk_uuid, blob, dim) triples — ready for UPDATE."""
    from mtss.processing.embeddings import EmbeddingGenerator

    gen = EmbeddingGenerator()
    texts = [r["embedding_text"] or r["content"] for r in rows]
    vectors = await gen.generate_embeddings_batch(texts)
    out: list[tuple[str, bytes, int]] = []
    for r, vec in zip(rows, vectors):
        arr = np.asarray(vec, dtype=np.float32)
        out.append((r["id"], arr.tobytes(), int(arr.size)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the embedding calls + UPDATE. Without this flag the "
        "script only reports candidate count.",
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

    try:
        rows = _candidate_chunks(conn)
        logger.info("Candidates (repair-tagged stubs without embedding): %d", len(rows))
        if not rows:
            logger.info("Nothing to embed.")
            return 0

        for r in rows[:5]:
            logger.info("  chunk_id=%s  content=%r", r["chunk_id"], (r["content"] or "")[:60])
        if len(rows) > 5:
            logger.info("  ... and %d more", len(rows) - 5)

        if not args.apply:
            logger.info("Dry-run only. Re-run with --apply to embed + UPDATE.")
            return 0

        logger.info("Calling embedding API for %d chunks ...", len(rows))
        triples = asyncio.run(_embed_all(rows))

        conn.execute("BEGIN")
        try:
            for chunk_uuid, blob, dim in triples:
                conn.execute(
                    "UPDATE chunks SET embedding = ?, embedding_dim = ? WHERE id = ?",
                    (blob, dim, chunk_uuid),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.info("Embedded %d chunks.", len(triples))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
