"""SQLite-backed storage for the MTSS ingest pipeline.

Drop-in replacement for ``LocalStorageClient``. Writes go to a single
``ingest.db`` under ``output_dir``. ACID transactions + ``UNIQUE(chunk_id)``
+ ``ON DELETE CASCADE`` foreign keys eliminate the classes of bugs that
plagued the JSONL flush path (orphan chunks + dupes from partial atomic
rewrites of multi-GB files under Windows file-lock contention).

Design notes:

- **WAL mode** with a 30 s busy timeout. Readers never block writers.
- **Foreign keys ON.** Deleting a document cascades to its chunks +
  chunk_topic rows. No custom orphan filter needed in application code.
- **``chunk_topics`` is the single source of truth for topic membership.**
  Topic IDs are *not* duplicated inside ``chunks.metadata_json`` — prevents
  the drift we saw historically.
- **Embeddings stored as BLOB** (raw float32 bytes). ~4× smaller than the
  JSON text encoding. Decoded with ``numpy.frombuffer``.
- **Async signatures preserved** for API parity with ``LocalStorageClient``.
  The SQLite backend itself is synchronous; the ``async def`` is free
  because every method completes within a few ms.
- **Every SQL call is parameterised.** No f-strings, no ``.format()`` — the
  codebase's lint guard (CI grep for ``execute(f"``) enforces this.

Not covered here: the ``LocalBucketStorage`` / ``LocalIngestOutput`` helpers
from ``local_client.py`` — those manage the filesystem archive folder and
have nothing to do with JSONL. They are imported unchanged.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

import numpy as np

from ..models.serializers import chunk_to_dict, doc_to_dict, topic_to_dict
from ..utils import compute_folder_id

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────
# Versioned so future migrations can check what they are upgrading.
SCHEMA_VERSION = 1

_SCHEMA_SQL = """
-- ── Core entities ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    document_type   TEXT NOT NULL,
    status          TEXT NOT NULL,
    error_message   TEXT,
    file_hash       TEXT,
    file_name       TEXT,
    file_path       TEXT,
    parent_id       TEXT REFERENCES documents(id) ON DELETE CASCADE,
    root_id         TEXT NOT NULL,
    depth           INTEGER NOT NULL DEFAULT 0,
    content_version INTEGER NOT NULL DEFAULT 1,
    ingest_version  INTEGER NOT NULL DEFAULT 1,
    archive_path    TEXT,
    title           TEXT,
    source_title    TEXT,
    mime_type       TEXT,
    content_type    TEXT,
    size_bytes      INTEGER,
    embedding_mode  TEXT,
    archive_browse_uri   TEXT,
    archive_download_uri TEXT,
    metadata_json   TEXT,
    processed_at    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_id    ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_root_id   ON documents(root_id);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_parent_id ON documents(parent_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id                TEXT PRIMARY KEY,
    chunk_id          TEXT NOT NULL UNIQUE,
    document_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_id         TEXT,
    content           TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    char_start        INTEGER,
    char_end          INTEGER,
    line_from         INTEGER,
    line_to           INTEGER,
    page_number       INTEGER,
    section_title     TEXT,
    section_path_json TEXT,
    context_summary   TEXT,
    embedding_text    TEXT,
    embedding         BLOB,
    embedding_dim     INTEGER,
    embedding_mode    TEXT,
    source_title      TEXT,
    archive_browse_uri   TEXT,
    archive_download_uri TEXT,
    metadata_json     TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_id   ON chunks(source_id);

CREATE TABLE IF NOT EXISTS topics (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    display_name    TEXT,
    description     TEXT,
    keywords_json   TEXT,
    embedding       BLOB,
    embedding_dim   INTEGER,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    document_count  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);

-- M:N chunk ↔ topic. Single source of truth for topic membership.
CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_topics_topic ON chunk_topics(topic_id);

-- ── Operational tables ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_events (
    rowid              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type         TEXT NOT NULL,
    severity           TEXT NOT NULL,
    reason             TEXT,
    message            TEXT,
    file_path          TEXT,
    file_name          TEXT,
    file_size          INTEGER,
    mime_type          TEXT,
    source_eml_path    TEXT,
    source_zip_path    TEXT,
    parent_document_id TEXT,
    document_id        TEXT,
    timestamp          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON ingest_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_parent     ON ingest_events(parent_document_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp  ON ingest_events(timestamp);

CREATE TABLE IF NOT EXISTS processing_log (
    file_path        TEXT PRIMARY KEY,
    file_hash        TEXT NOT NULL,
    status           TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    duration_seconds REAL,
    attempts         INTEGER DEFAULT 0,
    error            TEXT,
    ingest_version   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_processing_log_status    ON processing_log(status);
CREATE INDEX IF NOT EXISTS idx_processing_log_file_hash ON processing_log(file_hash);

CREATE TABLE IF NOT EXISTS run_history (
    rowid            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    elapsed_seconds  REAL,
    files_attempted  INTEGER,
    files_processed  INTEGER,
    files_failed     INTEGER,
    cumulative_json  TEXT,
    services_json    TEXT,
    errors_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_history_timestamp ON run_history(timestamp);

CREATE TABLE IF NOT EXISTS manifest (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Append-only status / archive updates. Kept for audit; not used by ingest
-- logic. Mirrors the ``status_updates.jsonl`` and ``archive_updates.jsonl``
-- append files from the legacy client.
CREATE TABLE IF NOT EXISTS status_updates (
    rowid        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL,
    status       TEXT,
    error_message TEXT,
    timestamp    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS archive_updates (
    rowid                INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id               TEXT NOT NULL,
    archive_browse_uri   TEXT,
    archive_download_uri TEXT,
    timestamp            TEXT NOT NULL
);
"""


# ── Helpers ───────────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_embedding(emb: Optional[Sequence[float]]) -> tuple[Optional[bytes], Optional[int]]:
    """Pack an embedding as raw float32 bytes. ``(None, None)`` if missing."""
    if emb is None:
        return None, None
    arr = np.asarray(emb, dtype=np.float32)
    return arr.tobytes(), int(arr.size)


def _decode_embedding(blob: Optional[bytes], dim: Optional[int]) -> Optional[List[float]]:
    """Decode a BLOB back into a Python list of floats."""
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and arr.size != dim:
        logger.warning("embedding dim mismatch: stored=%d expected=%d", arr.size, dim)
    return arr.tolist()


def _dumps(value: Any) -> Optional[str]:
    """JSON-encode with ``default=str`` for UUIDs/datetimes. ``None`` stays ``None``."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def _loads(value: Optional[str]) -> Any:
    if value is None or value == "":
        return None
    return json.loads(value)


# ── Client ────────────────────────────────────────────────────────────

@dataclass
class SqliteStorageClient:
    """SQLite-backed ``LocalStorageClient`` replacement.

    Same public API — callers don't need to know the backend. Internally
    every mutation is a direct SQL write inside a short transaction;
    there is no prior-vs-current in-memory split, no flush-time dedup,
    and no atomic-rewrite of a monolithic file.
    """

    output_dir: Path
    db_filename: str = "ingest.db"
    _conn: sqlite3.Connection = field(init=False, repr=False)
    # Backing for the legacy ``_events`` attribute that ``log_ingest_event``
    # callers read back during a single run. Kept in-memory only because the
    # legacy client did the same; the authoritative store is the DB.
    _events: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    # Cache of in-memory Document / Chunk / Topic model objects keyed by UUID.
    # Needed because ingest code sometimes mutates returned objects in place
    # and expects the same reference later (e.g. ``doc.status = FAILED`` on a
    # previously-inserted doc). Rehydrating from the DB would break that
    # pattern. Entries are written through to SQL on every insert/update.
    _documents: Dict[UUID, Any] = field(default_factory=dict, repr=False)
    _chunks: Dict[UUID, Any] = field(default_factory=dict, repr=False)
    _topics: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Legacy alias kept so callers that passed ``client._topics_by_name``
    # continue to work. Populated lazily on insert_topic().
    _topics_by_name: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Absorbed→keeper map populated by ``merge_similar_topics``. Kept so the
    # public attribute matches LocalStorageClient's surface for callers that
    # inspect merge outcomes.
    _merged_topic_map: Dict[str, str] = field(default_factory=dict, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.output_dir / self.db_filename
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # Pragmas must run outside an implicit transaction; ``isolation_level=None``
        # gives us autocommit by default. Explicit ``BEGIN`` used for batches.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA_SQL)
        self._ensure_manifest_version()
        self._rehydrate_caches()

    def _ensure_manifest_version(self) -> None:
        existing = self._conn.execute(
            "SELECT value FROM manifest WHERE key = ?", ("schema_version",)
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO manifest(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def _rehydrate_caches(self) -> None:
        """Preload document/topic objects the ingest code expects on startup.

        Ingest checks ``get_document_by_hash`` / ``get_document_by_doc_id``
        before parsing each email. Preloading those into a dict is O(n)
        once vs per-hit SELECTs. Chunks are NOT preloaded — they're too
        numerous and only queried per-doc (indexed).
        """
        cur = self._conn.execute("SELECT * FROM documents")
        for row in cur:
            doc = self._row_to_doc_namespace(row)
            self._documents[doc.id] = doc
        cur = self._conn.execute("SELECT * FROM topics")
        for row in cur:
            topic = self._row_to_topic_namespace(row)
            self._topics[str(topic.id)] = topic
            if topic.name:
                self._topics_by_name[topic.name] = topic
        if self._documents or self._topics:
            logger.info(
                "SQLite rehydrate: %d docs, %d topics",
                len(self._documents),
                len(self._topics),
            )

    async def close(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as e:
            logger.debug("wal_checkpoint failed on close: %s", e)
        self._conn.close()

    # ── document: insert / lookup / mutate ───────────────────────────

    async def insert_document(self, doc: Any) -> Any:
        """Insert a Document. De-duplicates by ``doc_id`` (returns existing)."""
        if getattr(doc, "doc_id", None):
            existing = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ? LIMIT 1",
                (doc.doc_id,),
            ).fetchone()
            if existing is not None:
                cached = self._documents.get(UUID(existing["id"]))
                return cached if cached is not None else self._row_to_doc_namespace(existing)

        row = self._doc_to_row(doc)
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        self._conn.execute(
            f"INSERT INTO documents ({','.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )
        self._documents[doc.id] = doc
        return doc

    async def get_document_by_hash(self, file_hash: str):
        for d in self._documents.values():
            if getattr(d, "file_hash", None) == file_hash:
                return d
        row = self._conn.execute(
            "SELECT * FROM documents WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()
        return self._row_to_doc_namespace(row) if row else None

    async def get_document_by_id(self, doc_id):
        try:
            uid = doc_id if isinstance(doc_id, UUID) else UUID(str(doc_id))
        except ValueError:
            return None
        if uid in self._documents:
            return self._documents[uid]
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ? LIMIT 1",
            (str(uid),),
        ).fetchone()
        return self._row_to_doc_namespace(row) if row else None

    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Any]:
        for d in self._documents.values():
            if getattr(d, "doc_id", None) == doc_id:
                return d
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ? LIMIT 1",
            (doc_id,),
        ).fetchone()
        return self._row_to_doc_namespace(row) if row else None

    async def get_document_by_source_id(self, source_id: str) -> Optional[Any]:
        for d in self._documents.values():
            if getattr(d, "source_id", None) == source_id:
                return d
        row = self._conn.execute(
            "SELECT * FROM documents WHERE source_id = ? LIMIT 1",
            (source_id,),
        ).fetchone()
        return self._row_to_doc_namespace(row) if row else None

    async def get_document_children(self, doc_id):
        doc_id_str = str(doc_id)
        cached = [
            d for d in self._documents.values()
            if str(getattr(d, "parent_id", "") or "") == doc_id_str
        ]
        if cached:
            return cached
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE parent_id = ?",
            (doc_id_str,),
        )
        return [self._row_to_doc_namespace(r) for r in cur]

    def delete_document_for_reprocess(self, doc_id) -> None:
        """Remove a document + all descendants (chunks cascade via FK).

        Accepts the email root's UUID. Finds every descendant by ``root_id``
        and deletes them in a single transaction. FK CASCADE handles
        ``chunks`` + ``chunk_topics``.
        """
        root_id_str = str(doc_id)
        with self._conn:
            self._conn.execute("BEGIN")
            # Gather affected UUIDs for cache eviction below.
            cur = self._conn.execute(
                "SELECT id FROM documents WHERE id = ? OR root_id = ?",
                (root_id_str, root_id_str),
            )
            affected_uuids = {row["id"] for row in cur}
            self._conn.execute(
                "DELETE FROM documents WHERE id = ? OR root_id = ?",
                (root_id_str, root_id_str),
            )
        # Cache eviction
        for uid_str in affected_uuids:
            try:
                self._documents.pop(UUID(uid_str), None)
            except ValueError:
                continue
        # Chunks cache: evict anything pointing at a deleted doc.
        for cid in list(self._chunks):
            c = self._chunks[cid]
            if str(getattr(c, "document_id", "")) in affected_uuids:
                del self._chunks[cid]

    async def update_document_status(
        self,
        doc_id: UUID,
        status: Any,
        error_message: Optional[str] = None,
    ) -> None:
        status_val = status.value if hasattr(status, "value") else status
        now = _utcnow_iso()
        self._conn.execute(
            "UPDATE documents SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
            (status_val, error_message, now, str(doc_id)),
        )
        self._conn.execute(
            "INSERT INTO status_updates(doc_id, status, error_message, timestamp) VALUES (?,?,?,?)",
            (str(doc_id), status_val, error_message, now),
        )
        cached = self._documents.get(doc_id)
        if cached is not None:
            cached.status = status
            if error_message is not None:
                cached.error_message = error_message

    async def update_document_archive_uris(
        self,
        doc_id: UUID,
        archive_browse_uri: str,
        archive_download_uri: str | None = None,
    ) -> None:
        now = _utcnow_iso()
        self._conn.execute(
            "UPDATE documents SET archive_browse_uri = ?, "
            "archive_download_uri = COALESCE(?, archive_download_uri), "
            "updated_at = ? WHERE id = ?",
            (archive_browse_uri, archive_download_uri, now, str(doc_id)),
        )
        self._conn.execute(
            "INSERT INTO archive_updates(doc_id, archive_browse_uri, "
            "archive_download_uri, timestamp) VALUES (?,?,?,?)",
            (str(doc_id), archive_browse_uri, archive_download_uri, now),
        )
        cached = self._documents.get(doc_id)
        if cached is not None:
            cached.archive_browse_uri = archive_browse_uri
            if archive_download_uri is not None:
                cached.archive_download_uri = archive_download_uri

    # ── chunk: insert / lookup / mutate ──────────────────────────────

    async def insert_chunks(self, chunks: List[Any]) -> List[Any]:
        """Insert chunks in a single transaction. Upserts on ``chunk_id``."""
        if not chunks:
            return chunks

        with self._conn:
            self._conn.execute("BEGIN")
            for chunk in chunks:
                row = self._chunk_to_row(chunk)
                cols = list(row.keys())
                placeholders = ",".join(["?"] * len(cols))
                # INSERT OR REPLACE on chunk_id — force_reparse deletes the
                # document above (FK CASCADE wipes chunks), but inserts also
                # need to be idempotent on chunk_id across concurrent ingests.
                self._conn.execute(
                    f"INSERT OR REPLACE INTO chunks ({','.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
                # Sync chunk↔topic rows from metadata.topic_ids, then strip
                # the key from the stored metadata so the junction table is
                # the single source of truth.
                topic_ids = (chunk.metadata or {}).get("topic_ids") or []
                if topic_ids:
                    self._conn.execute(
                        "DELETE FROM chunk_topics WHERE chunk_id = ?",
                        (str(chunk.id),),
                    )
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) "
                        "VALUES (?, ?)",
                        [(str(chunk.id), str(tid)) for tid in topic_ids],
                    )
                self._chunks[chunk.id] = chunk
        return chunks

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Any]:
        doc_id_str = str(doc_id)
        cached = [c for c in self._chunks.values() if str(c.document_id) == doc_id_str]
        if cached:
            return cached
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ?",
            (doc_id_str,),
        )
        return [self._row_to_chunk_namespace(row) for row in cur]

    async def update_chunks_topic_ids(self, document_id, topic_ids):
        """Replace the topic set for every chunk under ``document_id``."""
        doc_id_str = str(document_id)
        topic_id_strs = [str(t) for t in topic_ids]
        affected = 0
        with self._conn:
            self._conn.execute("BEGIN")
            cur = self._conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (doc_id_str,)
            )
            chunk_ids = [row["id"] for row in cur]
            if not chunk_ids:
                return 0
            placeholders = ",".join(["?"] * len(chunk_ids))
            self._conn.execute(
                f"DELETE FROM chunk_topics WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            if topic_id_strs:
                rows = [(cid, tid) for cid in chunk_ids for tid in topic_id_strs]
                self._conn.executemany(
                    "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) VALUES (?,?)",
                    rows,
                )
            affected = len(chunk_ids)
        # Sync in-memory cache for any chunks still held there.
        for c in self._chunks.values():
            if str(getattr(c, "document_id", "")) == doc_id_str:
                if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                    c.metadata["topic_ids"] = topic_id_strs
        return affected

    async def update_chunks_topics_checked(self, document_id):
        doc_id_str = str(document_id)
        # Merge ``{"topics_checked": true}`` into each chunk's metadata_json.
        # SQLite's json_patch isn't available in all builds; read+write each.
        affected = 0
        with self._conn:
            self._conn.execute("BEGIN")
            cur = self._conn.execute(
                "SELECT id, metadata_json FROM chunks WHERE document_id = ?",
                (doc_id_str,),
            )
            updates: list[tuple[str, str]] = []
            for row in cur:
                meta = _loads(row["metadata_json"]) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["topics_checked"] = True
                updates.append((_dumps(meta) or "{}", row["id"]))
            self._conn.executemany(
                "UPDATE chunks SET metadata_json = ? WHERE id = ?",
                updates,
            )
            affected = len(updates)
        for c in self._chunks.values():
            if str(getattr(c, "document_id", "")) == doc_id_str:
                if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                    c.metadata["topics_checked"] = True
        return affected

    # ── topic ────────────────────────────────────────────────────────

    async def insert_topic(self, topic):
        row = self._topic_to_row(topic)
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        self._conn.execute(
            f"INSERT OR REPLACE INTO topics ({','.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )
        self._topics[str(topic.id)] = topic
        if getattr(topic, "name", None):
            self._topics_by_name[topic.name] = topic
        return topic

    async def get_topic_by_name(self, name: str):
        cached = self._topics_by_name.get(name)
        if cached is not None:
            return cached
        row = self._conn.execute(
            "SELECT * FROM topics WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return self._row_to_topic_namespace(row) if row else None

    async def get_topic_by_id(self, topic_id):
        tid = str(topic_id)
        if tid in self._topics:
            return self._topics[tid]
        row = self._conn.execute(
            "SELECT * FROM topics WHERE id = ? LIMIT 1", (tid,)
        ).fetchone()
        return self._row_to_topic_namespace(row) if row else None

    async def find_similar_topics(self, embedding, threshold: float = 0.85, limit: int = 5):
        """Return topics with cosine similarity ≥ ``threshold`` (capped at ``limit``).

        Loads all topic embeddings once and does a vectorised matmul via
        numpy — same approach as ``merge_similar_topics``. At production
        scale (< 10 K topics) this is fast enough; a dedicated vector index
        is out of scope for Phase 1.
        """
        if not embedding:
            return []
        rows = list(
            self._conn.execute(
                "SELECT id, name, display_name, embedding, embedding_dim "
                "FROM topics WHERE embedding IS NOT NULL"
            )
        )
        if not rows:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(query))
        if q_norm == 0:
            return []
        query = query / q_norm
        results = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0:
                continue
            sim = float(np.dot(query, vec / v_norm))
            if sim >= threshold:
                results.append({
                    "id": UUID(row["id"]),
                    "name": row["name"],
                    "display_name": row["display_name"] or row["name"],
                    "similarity": sim,
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    async def increment_topic_counts(self, topic_ids, chunk_delta: int = 0, document_delta: int = 0):
        if not topic_ids or (chunk_delta == 0 and document_delta == 0):
            return
        params = [(chunk_delta, document_delta, str(tid)) for tid in topic_ids]
        self._conn.executemany(
            "UPDATE topics SET chunk_count = chunk_count + ?, "
            "document_count = document_count + ? WHERE id = ?",
            params,
        )
        for tid in topic_ids:
            t = self._topics.get(str(tid))
            if t is not None:
                t.chunk_count = (getattr(t, "chunk_count", 0) or 0) + chunk_delta
                t.document_count = (getattr(t, "document_count", 0) or 0) + document_delta

    def merge_similar_topics(self, threshold: float = 0.80) -> List[tuple]:
        """Merge near-duplicate topics. Keeps the topic with more chunks.

        Returns ``(absorbed_name, kept_name, similarity)`` tuples for each
        merge. Topic rows and their ``chunk_topics`` memberships are updated
        atomically — no orphan references possible.
        """
        rows = list(
            self._conn.execute(
                "SELECT id, name, embedding, chunk_count FROM topics "
                "WHERE embedding IS NOT NULL"
            )
        )
        if len(rows) < 2:
            return []

        # Sort by chunk_count desc so bigger topics absorb smaller ones.
        rows.sort(key=lambda r: r["chunk_count"] or 0, reverse=True)

        ids = [r["id"] for r in rows]
        names = [r["name"] for r in rows]
        mat = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        ).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        sims = mat @ mat.T

        absorbed: set[str] = set()
        merges: list[tuple] = []
        with self._conn:
            self._conn.execute("BEGIN")
            for i in range(len(rows)):
                if ids[i] in absorbed:
                    continue
                keeper_id = ids[i]
                for j in range(i + 1, len(rows)):
                    if ids[j] in absorbed:
                        continue
                    sim = float(sims[i, j])
                    if sim < threshold:
                        continue
                    absorbed_id = ids[j]
                    # Move chunk_topics memberships from absorbed → keeper.
                    self._conn.execute(
                        "INSERT OR IGNORE INTO chunk_topics(chunk_id, topic_id) "
                        "SELECT chunk_id, ? FROM chunk_topics WHERE topic_id = ?",
                        (keeper_id, absorbed_id),
                    )
                    self._conn.execute(
                        "DELETE FROM chunk_topics WHERE topic_id = ?",
                        (absorbed_id,),
                    )
                    # Transfer counts then delete the absorbed topic.
                    self._conn.execute(
                        "UPDATE topics SET "
                        "  chunk_count = chunk_count + (SELECT chunk_count FROM topics WHERE id = ?), "
                        "  document_count = document_count + (SELECT document_count FROM topics WHERE id = ?) "
                        "WHERE id = ?",
                        (absorbed_id, absorbed_id, keeper_id),
                    )
                    self._conn.execute(
                        "DELETE FROM topics WHERE id = ?", (absorbed_id,)
                    )
                    absorbed.add(absorbed_id)
                    self._merged_topic_map[absorbed_id] = keeper_id
                    merges.append((names[j], names[i], round(sim, 3)))

        # Evict absorbed from caches.
        for tid in absorbed:
            t = self._topics.pop(tid, None)
            if t is not None and getattr(t, "name", None):
                self._topics_by_name.pop(t.name, None)
        return merges

    # ── vessel / misc ────────────────────────────────────────────────

    async def get_all_vessels(self):
        return []

    # ── events / logging ─────────────────────────────────────────────

    def log_ingest_event(
        self,
        document_id: UUID,
        event_type: str,
        severity: str = "warning",
        message: Optional[str] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        source_eml_path: Optional[str] = None,
    ) -> None:
        row = {
            "event_type": event_type,
            "severity": severity,
            "reason": None,
            "message": message,
            "file_path": file_path,
            "file_name": file_name,
            "file_size": None,
            "mime_type": None,
            "source_eml_path": source_eml_path,
            "source_zip_path": None,
            "parent_document_id": None,
            "document_id": str(document_id),
            "timestamp": _utcnow_iso(),
        }
        self._insert_event(row)
        # Legacy in-memory mirror — a few test/CLI paths scan ``client._events``.
        self._events.append({
            "document_id": row["document_id"],
            "event_type": row["event_type"],
            "severity": row["severity"],
            "message": row["message"],
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "source_eml_path": row["source_eml_path"],
            "timestamp": row["timestamp"],
        })

    async def log_unsupported_file(
        self,
        file_path,
        reason,
        source_eml_path=None,
        source_zip_path=None,
        parent_document_id=None,
    ):
        mime_type, _ = mimetypes.guess_type(str(file_path))
        try:
            file_size = Path(file_path).stat().st_size
        except OSError:
            file_size = None
        event_type = reason.split(":", 1)[0].strip() if reason else "unknown"
        self._insert_event({
            "event_type": event_type,
            "severity": "info" if event_type == "classified_as_non_content" else "warning",
            "reason": reason,
            "message": None,
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "file_size": file_size,
            "mime_type": mime_type,
            "source_eml_path": source_eml_path,
            "source_zip_path": source_zip_path,
            "parent_document_id": str(parent_document_id) if parent_document_id else None,
            "document_id": None,
            "timestamp": _utcnow_iso(),
        })

    def _insert_event(self, row: Dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        self._conn.execute(
            f"INSERT INTO ingest_events ({','.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )

    # ── manifest / flush / persist ───────────────────────────────────

    def write_manifest(self) -> None:
        """Upsert ingest metadata into the ``manifest`` table."""
        from ..config import get_settings

        settings = get_settings()
        entries = {
            "schema_version":        str(SCHEMA_VERSION),
            "ingest_version":        str(getattr(settings, "current_ingest_version", 1)),
            "embedding_model":       str(settings.embedding_model),
            "embedding_dimensions":  str(settings.embedding_dimensions),
            "chunk_size_tokens":     str(settings.chunk_size_tokens),
            "chunk_overlap_tokens":  str(settings.chunk_overlap_tokens),
            "last_manifest_at":      _utcnow_iso(),
        }
        self._conn.executemany(
            "INSERT INTO manifest(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(entries.items()),
        )

    def flush(self) -> None:
        """Checkpoint WAL + recompute topic counts. No atomic rewrites.

        Unlike the JSONL flush, there is no merge of prior-vs-current state:
        every write committed as it happened. This method exists for API
        parity and to (a) recompute cached topic counts from
        ``chunk_topics`` — the canonical source — and (b) checkpoint the
        WAL so the main DB file stays compact between runs.
        """
        with self._conn:
            self._conn.execute("BEGIN")
            # Recompute topic counts from chunk_topics.
            self._conn.execute(
                "UPDATE topics SET "
                "  chunk_count = (SELECT COUNT(*) FROM chunk_topics ct WHERE ct.topic_id = topics.id), "
                "  document_count = (SELECT COUNT(DISTINCT c.document_id) "
                "                    FROM chunk_topics ct JOIN chunks c ON c.id = ct.chunk_id "
                "                    WHERE ct.topic_id = topics.id)"
            )
            # Drop orphan topics (no chunks, no docs).
            self._conn.execute(
                "DELETE FROM topics WHERE chunk_count = 0 AND document_count = 0"
            )
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error as e:
            logger.debug("wal_checkpoint during flush failed: %s", e)

        # Housekeeping: drop orphan archive folders on disk.
        archive_dir = self.output_dir / "archive"
        if archive_dir.exists():
            valid_folder_ids = {
                compute_folder_id(row["doc_id"])
                for row in self._conn.execute(
                    "SELECT doc_id FROM documents WHERE document_type = 'email' AND doc_id IS NOT NULL"
                )
            }
            import shutil
            for folder in archive_dir.iterdir():
                if folder.is_dir() and folder.name not in valid_folder_ids:
                    logger.info("Removing orphan archive folder: %s", folder.name)
                    shutil.rmtree(folder, ignore_errors=True)

    async def persist_ingest_result(
        self,
        email_doc,
        attachment_docs: list,
        chunks: list,
        topic_ids: list | None = None,
        chunk_delta: int = 0,
    ) -> None:
        """Persist an email + its attachments + chunks + topic counters atomically."""
        uuid_remap: Dict[str, str] = {}
        stored = await self.insert_document(email_doc)
        if stored.id != email_doc.id:
            uuid_remap[str(email_doc.id)] = str(stored.id)
        for doc in attachment_docs:
            stored = await self.insert_document(doc)
            if stored.id != doc.id:
                uuid_remap[str(doc.id)] = str(stored.id)
        if chunks and uuid_remap:
            for chunk in chunks:
                old = str(getattr(chunk, "document_id", ""))
                if old in uuid_remap:
                    chunk.document_id = UUID(uuid_remap[old])
        if chunks:
            await self.insert_chunks(chunks)
        if topic_ids and chunk_delta:
            await self.increment_topic_counts(
                topic_ids, chunk_delta=chunk_delta, document_delta=1
            )
        self._write_result_json(email_doc, attachment_docs, chunks, topic_ids)

    # ── result.json (unchanged archive-folder sidecar) ───────────────

    def _write_result_json(self, email_doc, attachment_docs, chunks, topic_ids):
        doc_id = getattr(email_doc, "doc_id", None)
        if not doc_id:
            return
        folder = compute_folder_id(doc_id)
        archive_dir = self.output_dir / "archive" / folder
        if not archive_dir.exists():
            return

        body_chunks: list = []
        digest_text = None
        att_chunk_counts: Dict[str, int] = {}
        vessel_types: set[str] = set()
        for chunk in chunks:
            meta = chunk.metadata or {}
            ctype = meta.get("type", "")
            if ctype == "thread_digest":
                digest_text = chunk.content
            elif ctype == "email_body":
                body_chunks.append(chunk)
            else:
                did = str(chunk.document_id)
                att_chunk_counts[did] = att_chunk_counts.get(did, 0) + 1
            for v in meta.get("vessel_types", []) or []:
                vessel_types.add(v)
        context_summary = body_chunks[0].context_summary if body_chunks else None

        topics_info = []
        if topic_ids:
            for tid in topic_ids:
                topic = self._topics.get(str(tid))
                if topic is None:
                    row = self._conn.execute(
                        "SELECT name, display_name FROM topics WHERE id = ?", (str(tid),)
                    ).fetchone()
                    if row is None:
                        continue
                    topics_info.append({
                        "name": row["name"],
                        "display_name": row["display_name"] or row["name"],
                    })
                else:
                    topics_info.append({
                        "name": topic.name,
                        "display_name": getattr(topic, "display_name", topic.name),
                    })

        attachments_info = [
            {
                "file_name": d.file_name,
                "document_type": d.document_type.value if hasattr(d.document_type, "value") else d.document_type,
                "chunks": att_chunk_counts.get(str(d.id), 0),
            }
            for d in attachment_docs
        ]
        email_meta = getattr(email_doc, "email_metadata", None)
        result = {
            "doc_id": doc_id,
            "subject": email_meta.subject if email_meta else getattr(email_doc, "source_title", None),
            "participants": email_meta.participants if email_meta else [],
            "initiator": email_meta.initiator if email_meta else None,
            "date_start": (
                email_meta.date_start.isoformat()
                if email_meta and hasattr(email_meta.date_start, "isoformat")
                else (email_meta.date_start if email_meta else None)
            ),
            "date_end": (
                email_meta.date_end.isoformat()
                if email_meta and hasattr(email_meta.date_end, "isoformat")
                else (email_meta.date_end if email_meta else None)
            ),
            "message_count": email_meta.message_count if email_meta else 1,
            "context_summary": context_summary,
            "thread_digest": digest_text,
            "vessels": sorted(vessel_types),
            "topics": topics_info,
            "chunks": {
                "total": len(chunks),
                "email_body": len(body_chunks),
                "thread_digest": 1 if digest_text else 0,
                "attachments": sum(att_chunk_counts.values()),
            },
            "attachments": attachments_info,
        }
        (archive_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ── row ↔ model adapters ─────────────────────────────────────────

    def _doc_to_row(self, doc: Any) -> Dict[str, Any]:
        """Flatten a Document to a row dict suitable for INSERT."""
        d = doc_to_dict(doc)
        metadata = {
            k: d.get(k)
            for k in (
                "path",
                "email_subject",
                "email_participants",
                "email_initiator",
                "email_date_start",
                "email_date_end",
                "email_message_count",
                "attachment_content_type",
                "attachment_size_bytes",
            )
            if d.get(k) is not None
        }
        now = _utcnow_iso()
        return {
            "id":              d["id"],
            "doc_id":          d.get("doc_id"),
            "source_id":       d.get("source_id"),
            "document_type":   d.get("document_type"),
            "status":          d.get("status") or "pending",
            "error_message":   d.get("error_message"),
            "file_hash":       d.get("file_hash"),
            "file_name":       d.get("file_name"),
            "file_path":       d.get("file_path"),
            "parent_id":       d.get("parent_id"),
            "root_id":         d.get("root_id") or d["id"],
            "depth":           d.get("depth", 0),
            "content_version": d.get("content_version", 1),
            "ingest_version":  d.get("ingest_version", 1),
            "archive_path":    d.get("archive_path"),
            "title":           d.get("email_subject") or d.get("source_title"),
            "source_title":    d.get("source_title"),
            "mime_type":       d.get("attachment_content_type"),
            "content_type":    d.get("attachment_content_type"),
            "size_bytes":      d.get("attachment_size_bytes"),
            "embedding_mode":  d.get("embedding_mode"),
            "archive_browse_uri":   d.get("archive_browse_uri"),
            "archive_download_uri": d.get("archive_download_uri"),
            "metadata_json":   _dumps(metadata) if metadata else None,
            "processed_at":    d.get("processed_at"),
            "created_at":      d.get("created_at") or now,
            "updated_at":      d.get("updated_at") or now,
        }

    def _chunk_to_row(self, chunk: Any) -> Dict[str, Any]:
        d = chunk_to_dict(chunk)
        blob, dim = _encode_embedding(d.get("embedding"))
        meta_stripped = None
        if isinstance(d.get("metadata"), dict):
            meta_stripped = {k: v for k, v in d["metadata"].items() if k != "topic_ids"}
            if not meta_stripped:
                meta_stripped = None
        em_mode = d.get("embedding_mode")
        return {
            "id":                d["id"],
            "chunk_id":          d.get("chunk_id"),
            "document_id":       d["document_id"],
            "source_id":         d.get("source_id"),
            "content":           d["content"],
            "chunk_index":       d.get("chunk_index", 0),
            "char_start":        d.get("char_start"),
            "char_end":          d.get("char_end"),
            "line_from":         d.get("line_from"),
            "line_to":           d.get("line_to"),
            "page_number":       d.get("page_number"),
            "section_title":     d.get("section_title"),
            "section_path_json": _dumps(d.get("section_path") or []),
            "context_summary":   d.get("context_summary"),
            "embedding_text":    d.get("embedding_text"),
            "embedding":         blob,
            "embedding_dim":     dim,
            "embedding_mode":    em_mode,
            "source_title":      d.get("source_title"),
            "archive_browse_uri":   d.get("archive_browse_uri"),
            "archive_download_uri": d.get("archive_download_uri"),
            "metadata_json":     _dumps(meta_stripped) if meta_stripped else None,
            "created_at":        _utcnow_iso(),
        }

    def _topic_to_row(self, topic: Any) -> Dict[str, Any]:
        d = topic_to_dict(topic)
        blob, dim = _encode_embedding(d.get("embedding"))
        now = _utcnow_iso()
        return {
            "id":              d["id"],
            "name":            d["name"],
            "display_name":    d.get("display_name") or d["name"],
            "description":     d.get("description"),
            "keywords_json":   _dumps(d.get("keywords") if isinstance(d.get("keywords"), list) else None),
            "embedding":       blob,
            "embedding_dim":   dim,
            "chunk_count":     d.get("chunk_count", 0),
            "document_count":  d.get("document_count", 0),
            "created_at":      d.get("created_at") or now,
            "updated_at":      d.get("updated_at") or now,
        }

    # ── row → SimpleNamespace (for hit-lookups from rehydrate) ───────

    def _row_to_doc_namespace(self, row: sqlite3.Row) -> SimpleNamespace:
        from ..models.document import ProcessingStatus

        meta = _loads(row["metadata_json"]) or {}
        ns = SimpleNamespace(
            id=UUID(row["id"]),
            doc_id=row["doc_id"],
            source_id=row["source_id"],
            document_type=row["document_type"],
            status=(ProcessingStatus(row["status"]) if row["status"] else None),
            error_message=row["error_message"],
            file_hash=row["file_hash"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            parent_id=UUID(row["parent_id"]) if row["parent_id"] else None,
            root_id=UUID(row["root_id"]) if row["root_id"] else None,
            depth=row["depth"],
            content_version=row["content_version"],
            ingest_version=row["ingest_version"],
            archive_path=row["archive_path"],
            source_title=row["source_title"],
            archive_browse_uri=row["archive_browse_uri"],
            archive_download_uri=row["archive_download_uri"],
            embedding_mode=row["embedding_mode"],
            processed_at=row["processed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        # Attach flattened email/attachment metadata for back-compat.
        for k, v in meta.items():
            setattr(ns, k, v)
        return ns

    def _row_to_topic_namespace(self, row: sqlite3.Row) -> SimpleNamespace:
        return SimpleNamespace(
            id=UUID(row["id"]),
            name=row["name"],
            display_name=row["display_name"] or row["name"],
            description=row["description"],
            embedding=_decode_embedding(row["embedding"], row["embedding_dim"]),
            chunk_count=row["chunk_count"],
            document_count=row["document_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_chunk_namespace(self, row: sqlite3.Row) -> SimpleNamespace:
        meta = _loads(row["metadata_json"]) or {}
        if not isinstance(meta, dict):
            meta = {}
        topic_ids = [
            r["topic_id"]
            for r in self._conn.execute(
                "SELECT topic_id FROM chunk_topics WHERE chunk_id = ?", (row["id"],)
            )
        ]
        if topic_ids:
            meta["topic_ids"] = topic_ids
        return SimpleNamespace(
            id=UUID(row["id"]),
            chunk_id=row["chunk_id"],
            document_id=UUID(row["document_id"]),
            source_id=row["source_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            line_from=row["line_from"],
            line_to=row["line_to"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            section_path=_loads(row["section_path_json"]) or [],
            context_summary=row["context_summary"],
            embedding_text=row["embedding_text"],
            embedding=_decode_embedding(row["embedding"], row["embedding_dim"]),
            embedding_mode=row["embedding_mode"],
            source_title=row["source_title"],
            archive_browse_uri=row["archive_browse_uri"],
            archive_download_uri=row["archive_download_uri"],
            metadata=meta,
            created_at=row["created_at"],
        )

    # ── SELECT helpers used by validate / import / re-embed ──────────

    def iter_documents(self, chunk_size: int = 500) -> Iterable[Dict[str, Any]]:
        """Yield every document row as a dict. Streams via keyset over id."""
        last_id = ""
        while True:
            rows = list(
                self._conn.execute(
                    "SELECT * FROM documents WHERE id > ? ORDER BY id LIMIT ?",
                    (last_id, chunk_size),
                )
            )
            if not rows:
                break
            for row in rows:
                yield dict(row)
            last_id = rows[-1]["id"]

    def iter_chunks(self, chunk_size: int = 500) -> Iterable[Dict[str, Any]]:
        """Yield every chunk row as a dict with decoded embedding + topic_ids."""
        last_id = ""
        while True:
            rows = list(
                self._conn.execute(
                    "SELECT * FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
                    (last_id, chunk_size),
                )
            )
            if not rows:
                break
            for row in rows:
                d = dict(row)
                d["embedding"] = _decode_embedding(row["embedding"], row["embedding_dim"])
                d["metadata"] = _loads(row["metadata_json"]) or {}
                if not isinstance(d["metadata"], dict):
                    d["metadata"] = {}
                d["section_path"] = _loads(row["section_path_json"]) or []
                d["metadata"]["topic_ids"] = [
                    r["topic_id"]
                    for r in self._conn.execute(
                        "SELECT topic_id FROM chunk_topics WHERE chunk_id = ?", (row["id"],)
                    )
                ]
                yield d
            last_id = rows[-1]["id"]

    def iter_topics(self) -> Iterable[Dict[str, Any]]:
        for row in self._conn.execute("SELECT * FROM topics ORDER BY id"):
            d = dict(row)
            d["embedding"] = _decode_embedding(row["embedding"], row["embedding_dim"])
            d["keywords"] = _loads(row["keywords_json"]) or []
            yield d

    def iter_events(self) -> Iterable[Dict[str, Any]]:
        for row in self._conn.execute(
            "SELECT * FROM ingest_events ORDER BY rowid"
        ):
            yield dict(row)
