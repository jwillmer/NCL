"""Unit test for ``scripts/backfill_v5_email_topics.py``.

Verifies that the --apply path, given a mocked ``TopicExtractor`` that
returns a fixed topic list, writes ``chunk_topics`` rows for every chunk
under the email's ``root_id`` — body + attachments — and patches
``chunks.metadata_json`` with the same topic_ids.

The test uses a real SQLite file under ``tmp_path`` so the script's
``SqliteStorageClient`` constructor (which runs schema migrations)
behaves identically to production.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.utils import compute_folder_id  # noqa: E402


# ── Fixtures ───────────────────────────────────────────────────────────


def _load_script_module():
    """Import the backfill script as a module.

    The script lives under ``scripts/`` which isn't on sys.path by default;
    load it via importlib so the test can drive ``_run`` directly.
    """
    path = REPO_ROOT / "scripts" / "backfill_v5_email_topics.py"
    spec = importlib.util.spec_from_file_location("backfill_v5_email_topics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_v5_email_topics"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seeded_db(tmp_path):
    """Build an ingest.db with one v=5 email + two attachments + 4 chunks.

    Layout:
        email doc (root)
          └─ body chunk           (under email doc)
          └─ attachment doc #1
                └─ att1 chunk
          └─ attachment doc #2
                └─ att2 chunk a
                └─ att2 chunk b

    All chunks are expected to receive the mocked topic_ids.
    """
    from mtss.storage.sqlite_client import SqliteStorageClient

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    client = SqliteStorageClient(output_dir=output_dir)
    conn = client._conn
    now = datetime.now(timezone.utc).isoformat()

    email_row = str(uuid4())
    email_doc_id = "emaildoc12345678"
    att1_row = str(uuid4())
    att2_row = str(uuid4())

    # 3 documents: email (depth=0, v=5), two attachments (depth=1)
    docs = [
        (
            email_row,
            email_doc_id,
            "email",
            "inbox/msg.eml",
            None,
            email_row,
            0,
            5,
            json.dumps({"email_subject": "Engine oil leak on MV Test"}),
            "Engine oil leak on MV Test",
        ),
        (
            att1_row,
            "att1doc0000000001",
            "attachment_pdf",
            "inbox/msg.eml::report.pdf",
            email_row,
            email_row,
            1,
            5,
            None,
            "report.pdf",
        ),
        (
            att2_row,
            "att2doc0000000002",
            "attachment_image",
            "inbox/msg.eml::photo.jpg",
            email_row,
            email_row,
            1,
            5,
            None,
            "photo.jpg",
        ),
    ]
    for (
        row_id,
        doc_id,
        dtype,
        source_id,
        parent,
        root,
        depth,
        iv,
        metadata_json,
        source_title,
    ) in docs:
        conn.execute(
            "INSERT INTO documents("
            "  id, doc_id, source_id, document_type, status, parent_id, "
            "  root_id, depth, ingest_version, metadata_json, source_title, "
            "  created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                doc_id,
                source_id,
                dtype,
                "completed",
                parent,
                root,
                depth,
                iv,
                metadata_json,
                source_title,
                now,
                now,
            ),
        )

    # 4 chunks total
    chunks = [
        (str(uuid4()), "bodychunk001", email_row, 0, "body text"),
        (str(uuid4()), "att1chunk001", att1_row, 0, "pdf text"),
        (str(uuid4()), "att2chunk001", att2_row, 0, "image ocr"),
        (str(uuid4()), "att2chunk002", att2_row, 1, "more image ocr"),
    ]
    for cid, chunk_id, doc_id_fk, chunk_index, content in chunks:
        conn.execute(
            "INSERT INTO chunks("
            "  id, chunk_id, document_id, content, chunk_index, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (cid, chunk_id, doc_id_fk, content, chunk_index, now),
        )

    # Seed the archive markdown the script will read.
    folder = output_dir / "archive" / compute_folder_id(email_doc_id)
    folder.mkdir(parents=True)
    (folder / "email.md").write_text(
        "# Email\n\n## Content\nEngine oil pressure drop on main engine.\n",
        encoding="utf-8",
    )

    conn.commit()
    # Close so the script opens its own handle — avoids WAL weirdness.
    client._conn.close()

    return {
        "output_dir": output_dir,
        "email_row_id": email_row,
        "email_doc_id": email_doc_id,
        "all_chunk_ids": [c[0] for c in chunks],
    }


# ── Test ───────────────────────────────────────────────────────────────


def test_apply_stamps_all_chunks_under_email(seeded_db, monkeypatch):
    """--apply must insert chunk_topics rows for every chunk (body + attachments)
    under the email's root_id, and patch each chunk's metadata_json."""
    import sqlite3

    module = _load_script_module()

    # Stub out heavy network-facing components:
    # * TopicExtractor.extract_topics → fixed list
    # * TopicMatcher.get_or_create_topics_batch → fixed UUIDs
    fixed_topic_ids = [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ]

    # chunk_topics has a FK to topics — seed the two target topic rows
    # directly (we mock ``get_or_create_topics_batch`` so the real insert
    # path is bypassed).
    import sqlite3 as _sqlite3
    seed_con = _sqlite3.connect(seeded_db["output_dir"] / "ingest.db")
    now_iso = datetime.now(timezone.utc).isoformat()
    for tid, tname in zip(fixed_topic_ids, ["engine maintenance", "oil leak"]):
        seed_con.execute(
            "INSERT INTO topics(id, name, display_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tid, tname, tname, now_iso, now_iso),
        )
    seed_con.commit()
    seed_con.close()

    async def fake_extract(self, content, max_topics=5):
        # Return two ExtractedTopic-like objects.
        return [
            SimpleNamespace(name="Engine Maintenance", description=None),
            SimpleNamespace(name="Oil Leak", description=None),
        ]

    async def fake_batch(self, topics):
        import uuid
        return [uuid.UUID(tid) for tid in fixed_topic_ids]

    # Patch the classes used inside ``_run`` after the import it does.
    from mtss.processing.topics import TopicExtractor, TopicMatcher

    monkeypatch.setattr(TopicExtractor, "extract_topics", fake_extract)
    monkeypatch.setattr(TopicMatcher, "get_or_create_topics_batch", fake_batch)

    # Prevent any accidental embedding calls from the matcher's name-cache
    # preload (the shim returns an empty list, so TopicCache._load has
    # nothing to do, but be defensive).
    from mtss.processing import embeddings as emb_mod

    async def _no_embed(self, text):
        return [0.0] * 16

    async def _no_embed_batch(self, texts):
        return [[0.0] * 16 for _ in texts]

    monkeypatch.setattr(
        emb_mod.EmbeddingGenerator, "generate_embedding", _no_embed, raising=False
    )
    monkeypatch.setattr(
        emb_mod.EmbeddingGenerator,
        "generate_embeddings_batch",
        _no_embed_batch,
        raising=False,
    )

    # Run --apply.
    import asyncio

    args = module._build_parser().parse_args(
        ["--output-dir", str(seeded_db["output_dir"]), "--apply", "--concurrency", "1"]
    )
    rc = asyncio.run(module._run(args))
    assert rc == 0

    # Verify every chunk got linked to both topic_ids.
    db_path = seeded_db["output_dir"] / "ingest.db"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT chunk_id, topic_id FROM chunk_topics ORDER BY chunk_id, topic_id"
        ).fetchall()
        by_chunk: dict[str, set[str]] = {}
        for r in rows:
            by_chunk.setdefault(r["chunk_id"], set()).add(r["topic_id"])

        assert set(by_chunk.keys()) == set(seeded_db["all_chunk_ids"]), (
            "every chunk under the email's root_id must be linked"
        )
        for cid, tids in by_chunk.items():
            assert tids == set(fixed_topic_ids), (
                f"chunk {cid} has topics {tids}, expected {fixed_topic_ids}"
            )

        # metadata_json should carry topic_ids too.
        meta_rows = con.execute(
            "SELECT id, metadata_json FROM chunks"
        ).fetchall()
        for row in meta_rows:
            meta = json.loads(row["metadata_json"])
            assert sorted(meta["topic_ids"]) == sorted(fixed_topic_ids), (
                f"chunk {row['id']} metadata_json missing topic_ids"
            )
    finally:
        con.close()
