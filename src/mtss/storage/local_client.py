"""Local storage backend for running ingest without Supabase.

Provides in-memory + JSONL implementations of SupabaseClient, ArchiveStorage,
and a coordinator class, enabling fully offline ingest pipelines.
"""

from __future__ import annotations

import json
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models.serializers import chunk_to_dict, doc_to_dict, topic_to_dict

logger = logging.getLogger(__name__)


@dataclass
class LocalStorageClient:
    """Mock SupabaseClient that writes to local JSONL files.

    Provides the same interface as SupabaseClient but stores data
    in local JSONL files for easy inspection.
    """

    output_dir: Path
    _documents: Dict[UUID, Any] = field(default_factory=dict)
    _chunks: Dict[UUID, Any] = field(default_factory=dict)
    _events: List[Dict[str, Any]] = field(default_factory=list)

    # In-memory indexes for fast lookup
    _documents_by_hash: Dict[str, Any] = field(default_factory=dict)
    _documents_by_doc_id: Dict[str, Any] = field(default_factory=dict)
    _documents_by_source_id: Dict[str, Any] = field(default_factory=dict)
    _topics: Dict[str, Any] = field(default_factory=dict)
    _topics_by_name: Dict[str, Any] = field(default_factory=dict)
    _chunks_by_document: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure output directory exists and load prior data."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prior_documents: list[Dict[str, Any]] = []
        self._prior_topics: Dict[str, Dict[str, Any]] = {}
        self._merged_topic_map: Dict[str, str] = {}  # absorbed_id -> keeper_id
        self._load_prior_data()

    def _load_prior_data(self):
        """Load existing documents and topics from previous runs.

        Documents are stored as raw dicts for flush merge.
        Also populates in-memory indexes (hash, doc_id, source_id) so
        deduplication works correctly across batched runs.
        Topics are similarly loaded into indexes for topic dedup.
        """
        from ..models.document import ProcessingStatus

        docs_path = self.output_dir / "documents.jsonl"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            doc_dict = json.loads(line)
                            self._prior_documents.append(doc_dict)
                            # Populate dedup indexes with SimpleNamespace wrappers
                            doc_obj = SimpleNamespace(**doc_dict)
                            doc_obj.id = UUID(doc_dict["id"]) if isinstance(doc_dict["id"], str) else doc_dict["id"]
                            if doc_dict.get("status"):
                                try:
                                    doc_obj.status = ProcessingStatus(doc_dict["status"])
                                except ValueError:
                                    pass
                            if doc_dict.get("file_hash"):
                                self._documents_by_hash[doc_dict["file_hash"]] = doc_obj
                            if doc_dict.get("doc_id"):
                                self._documents_by_doc_id[doc_dict["doc_id"]] = doc_obj
                            if doc_dict.get("source_id"):
                                self._documents_by_source_id[doc_dict["source_id"]] = doc_obj
                        except (json.JSONDecodeError, KeyError):
                            pass

        topics_path = self.output_dir / "topics.jsonl"
        if topics_path.exists():
            with open(topics_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            t = json.loads(line)
                            self._prior_topics[t["id"]] = t
                            # Load into in-memory indexes for topic dedup
                            topic_obj = SimpleNamespace(**t)
                            topic_obj.id = UUID(t["id"]) if isinstance(t["id"], str) else t["id"]
                            self._topics[t["id"]] = topic_obj
                            if t.get("name"):
                                self._topics_by_name[t["name"]] = topic_obj
                        except (json.JSONDecodeError, KeyError):
                            pass
        if self._prior_documents:
            logger.info(f"Loaded {len(self._prior_documents)} prior documents, {len(self._prior_topics)} prior topics")

    def _append_jsonl(self, filename: str, data: Dict[str, Any]) -> None:
        """Append a JSON object as a line to a JSONL file."""
        file_path = self.output_dir / filename
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _doc_to_dict(self, doc: Any) -> Dict[str, Any]:
        """Convert Document model to dictionary for JSONL serialization."""
        return doc_to_dict(doc)

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """Convert Chunk model to dictionary."""
        return chunk_to_dict(chunk)

    def _topic_to_dict(self, topic: Any) -> Dict[str, Any]:
        """Convert Topic model to dictionary."""
        return topic_to_dict(topic)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Document operations ────────────────────────────────────────

    async def insert_document(self, doc: Any) -> Any:
        """Insert a document record to local storage. Skips if doc_id already exists."""
        if hasattr(doc, "doc_id") and doc.doc_id:
            if doc.doc_id in self._documents_by_doc_id:
                return self._documents_by_doc_id[doc.doc_id]
            self._documents_by_doc_id[doc.doc_id] = doc
        self._documents[doc.id] = doc
        if hasattr(doc, "file_hash") and doc.file_hash:
            self._documents_by_hash[doc.file_hash] = doc
        if hasattr(doc, "source_id") and doc.source_id:
            self._documents_by_source_id[doc.source_id] = doc
        self._append_jsonl("documents.jsonl", self._doc_to_dict(doc))
        return doc

    async def get_document_by_hash(self, file_hash: str):
        """Get document by file hash."""
        return self._documents_by_hash.get(file_hash)

    async def get_document_by_id(self, doc_id):
        """Get document by UUID."""
        doc_id_str = str(doc_id)
        for doc in self._documents.values():
            if str(doc.id) == doc_id_str:
                return doc
        return None

    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Any]:
        """Get document by doc_id from local storage."""
        result = self._documents_by_doc_id.get(doc_id)
        if result:
            return result
        # Fallback: linear scan
        for doc in self._documents.values():
            if doc.doc_id == doc_id:
                return doc
        return None

    async def get_document_by_source_id(self, source_id: str) -> Optional[Any]:
        """Get document by source_id from local storage."""
        result = self._documents_by_source_id.get(source_id)
        if result:
            return result
        # Fallback: linear scan
        for doc in self._documents.values():
            if doc.source_id == source_id:
                return doc
        return None

    async def get_document_children(self, doc_id):
        """Get all direct children of a document."""
        doc_id_str = str(doc_id)
        return [
            d for d in self._documents.values()
            if str(getattr(d, "parent_document_id", getattr(d, "parent_id", ""))) == doc_id_str
        ]

    def delete_document_for_reprocess(self, doc_id):
        """Remove a document and all its children + chunks for reprocessing."""
        doc_id_str = str(doc_id)

        # Collect all document UUIDs to remove (root + children by root_id)
        doc_ids_to_remove = {doc_id_str}
        # Check in-memory documents
        for did, d in list(self._documents.items()):
            if str(getattr(d, "root_id", "")) == doc_id_str:
                doc_ids_to_remove.add(str(did))
        # Check prior documents (from previous runs)
        for pd in list(self._prior_documents):
            if pd.get("root_id") == doc_id_str:
                doc_ids_to_remove.add(pd.get("id", ""))

        # Remove from in-memory caches (try both UUID and string keys)
        for did in doc_ids_to_remove:
            doc = self._documents.pop(did, None)
            if doc is None:
                try:
                    doc = self._documents.pop(UUID(did), None)
                except ValueError:
                    pass
            if doc:
                self._documents_by_hash.pop(getattr(doc, "file_hash", ""), None)
                self._documents_by_doc_id.pop(getattr(doc, "doc_id", ""), None)
                self._documents_by_source_id.pop(getattr(doc, "source_id", ""), None)
            self._chunks_by_document.pop(did, None)

        # Remove chunks for all removed documents
        chunks_to_remove = [
            cid for cid, c in self._chunks.items()
            if str(getattr(c, "document_id", "")) in doc_ids_to_remove
        ]
        for cid in chunks_to_remove:
            del self._chunks[cid]

        # Remove from prior documents (so flush doesn't write them back)
        self._prior_documents = [
            pd for pd in self._prior_documents
            if pd.get("id") not in doc_ids_to_remove
        ]

    async def update_document_status(
        self,
        doc_id: UUID,
        status: Any,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status in local storage."""
        # Update in-memory document object
        doc = self._documents.get(doc_id)
        if doc:
            doc.status = status

        event = {
            "type": "status_update",
            "doc_id": str(doc_id),
            "status": status.value if hasattr(status, "value") else status,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_jsonl("status_updates.jsonl", event)

    async def update_document_archive_uris(
        self,
        doc_id: UUID,
        archive_browse_uri: str,
        archive_download_uri: str | None = None,
    ) -> None:
        """Update document archive URIs."""
        # Update in-memory document object
        doc = self._documents.get(doc_id)
        if doc:
            doc.archive_browse_uri = archive_browse_uri
            if archive_download_uri is not None:
                doc.archive_download_uri = archive_download_uri

        event = {
            "type": "archive_uri_update",
            "doc_id": str(doc_id),
            "archive_browse_uri": archive_browse_uri,
            "archive_download_uri": archive_download_uri,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_jsonl("archive_updates.jsonl", event)

    # ── Chunk operations ───────────────────────────────────────────

    async def insert_chunks(self, chunks: List[Any]) -> List[Any]:
        """Insert chunks to local storage."""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._append_jsonl("chunks.jsonl", self._chunk_to_dict(chunk))
            # Populate chunks-by-document index
            doc_id_str = str(chunk.document_id)
            self._chunks_by_document.setdefault(doc_id_str, []).append(chunk)
        return chunks

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Any]:
        """Get chunks for a document from local storage."""
        cached = self._chunks_by_document.get(str(doc_id))
        if cached is not None:
            return cached
        return [c for c in self._chunks.values() if c.document_id == doc_id]

    async def update_chunks_topic_ids(self, document_id, topic_ids):
        """Update topic_ids in chunk metadata for a document."""
        doc_id_str = str(document_id)
        count = 0
        for chunk in self._chunks.values():
            if str(getattr(chunk, "document_id", "")) == doc_id_str:
                if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                    chunk.metadata["topic_ids"] = [str(t) for t in topic_ids]
                count += 1
        return count

    async def update_chunks_topics_checked(self, document_id):
        """Mark chunks as having been checked for topics."""
        doc_id_str = str(document_id)
        count = 0
        for chunk in self._chunks.values():
            if str(getattr(chunk, "document_id", "")) == doc_id_str:
                if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                    chunk.metadata["topics_checked"] = True
                count += 1
        return count

    # ── Topic operations ───────────────────────────────────────────

    async def insert_topic(self, topic):
        """Insert a topic record."""
        topic_id = str(topic.id)
        self._topics[topic_id] = topic
        if hasattr(topic, "name") and topic.name:
            self._topics_by_name[topic.name] = topic
        self._append_jsonl("topics.jsonl", self._topic_to_dict(topic))
        return topic

    async def get_topic_by_name(self, name: str):
        """Get topic by canonical name."""
        return self._topics_by_name.get(name)

    async def get_topic_by_id(self, topic_id):
        """Get topic by UUID."""
        return self._topics.get(str(topic_id))

    async def find_similar_topics(self, embedding, threshold: float = 0.85, limit: int = 5):
        """Find topics with similar embeddings using cosine similarity."""
        results = []
        for topic in self._topics.values():
            topic_emb = getattr(topic, "embedding", None)
            if topic_emb is None:
                continue
            sim = self._cosine_similarity(embedding, topic_emb)
            if sim >= threshold:
                results.append({
                    "id": topic.id,
                    "name": topic.name,
                    "display_name": getattr(topic, "display_name", topic.name),
                    "similarity": sim,
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    async def increment_topic_counts(self, topic_ids, chunk_delta: int = 0, document_delta: int = 0):
        """Increment chunk/document counts on topics."""
        for tid in topic_ids:
            topic = self._topics.get(str(tid))
            if topic:
                if hasattr(topic, "chunk_count"):
                    topic.chunk_count = (topic.chunk_count or 0) + chunk_delta
                if hasattr(topic, "document_count"):
                    topic.document_count = (topic.document_count or 0) + document_delta

    def merge_similar_topics(self, threshold: float = 0.80) -> List[tuple]:
        """Merge near-duplicate topics that slipped past the creation threshold.

        Finds topic pairs with cosine similarity >= threshold and merges the
        smaller topic into the larger one. Updates all chunk metadata references.

        Args:
            threshold: Similarity threshold for merging (default 0.80).

        Returns:
            List of (absorbed_name, kept_name, similarity) tuples.
        """
        topics = list(self._topics.values())
        if len(topics) < 2:
            return []

        # Build similarity matrix for all topic pairs
        merges = []
        absorbed_ids: set = set()

        # Sort by chunk_count descending so larger topics absorb smaller ones
        topics.sort(key=lambda t: getattr(t, "chunk_count", 0) or 0, reverse=True)

        for i, topic_a in enumerate(topics):
            if str(topic_a.id) in absorbed_ids:
                continue
            emb_a = getattr(topic_a, "embedding", None)
            if emb_a is None:
                continue

            for j in range(i + 1, len(topics)):
                topic_b = topics[j]
                if str(topic_b.id) in absorbed_ids:
                    continue
                emb_b = getattr(topic_b, "embedding", None)
                if emb_b is None:
                    continue

                sim = self._cosine_similarity(emb_a, emb_b)
                if sim >= threshold:
                    # Merge topic_b into topic_a (topic_a is larger)
                    self._merge_topic_into(topic_a, topic_b)
                    absorbed_ids.add(str(topic_b.id))
                    merges.append((topic_b.name, topic_a.name, round(sim, 3)))

        # Remove absorbed topics from all indexes (including prior, to prevent resurrection)
        for tid in absorbed_ids:
            topic = self._topics.pop(tid, None)
            if topic and hasattr(topic, "name"):
                self._topics_by_name.pop(topic.name, None)
            self._prior_topics.pop(tid, None)

        return merges

    def _merge_topic_into(self, keeper, absorbed):
        """Merge absorbed topic into keeper: transfer counts and update chunk metadata."""
        # Transfer counts
        keeper.chunk_count = (getattr(keeper, "chunk_count", 0) or 0) + (getattr(absorbed, "chunk_count", 0) or 0)
        keeper.document_count = (getattr(keeper, "document_count", 0) or 0) + (getattr(absorbed, "document_count", 0) or 0)

        # Track replacement for prior chunks rewrite during flush
        absorbed_id = str(absorbed.id)
        keeper_id = str(keeper.id)
        self._merged_topic_map[absorbed_id] = keeper_id
        for chunk in self._chunks.values():
            meta = getattr(chunk, "metadata", None)
            if not isinstance(meta, dict):
                continue
            topic_ids = meta.get("topic_ids")
            if topic_ids and absorbed_id in topic_ids:
                topic_ids = [keeper_id if t == absorbed_id else t for t in topic_ids]
                # Deduplicate (if keeper was already in the list)
                meta["topic_ids"] = list(dict.fromkeys(topic_ids))

    # ── Vessel operations ──────────────────────────────────────────

    async def get_all_vessels(self):
        """Return all vessels (empty for local mode)."""
        return []

    # ── Event / logging operations ─────────────────────────────────

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
        """Log an ingest event to local storage."""
        event = {
            "document_id": str(document_id),
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "file_path": file_path,
            "file_name": file_name,
            "source_eml_path": source_eml_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._events.append(event)
        self._append_jsonl("ingest_events.jsonl", event)

    async def log_unsupported_file(
        self,
        file_path,
        reason,
        source_eml_path=None,
        source_zip_path=None,
        parent_document_id=None,
    ):
        """Log unsupported/skipped files to ingest_events.jsonl."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        try:
            file_size = Path(file_path).stat().st_size
        except OSError:
            file_size = None
        self._append_jsonl("ingest_events.jsonl", {
            "event_type": reason,
            "severity": "info" if reason == "classified_as_non_content" else "warning",
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "reason": reason,
            "mime_type": mime_type,
            "file_size": file_size,
            "source_eml_path": source_eml_path,
            "source_zip_path": source_zip_path,
            "parent_document_id": str(parent_document_id) if parent_document_id else None,
        })

    # ── Result JSON ────────────────────────────────────────────────

    def _write_result_json(self, email_doc, attachment_docs, chunks, topic_ids):
        """Write result.json with AI-enriched data to the archive folder."""
        doc_id = getattr(email_doc, "doc_id", None)
        if not doc_id:
            return

        archive_dir = self.output_dir / "archive" / doc_id
        if not archive_dir.exists():
            return

        # Categorize chunks
        body_chunks = []
        digest_text = None
        att_chunk_counts: Dict[str, int] = {}  # document_id -> count

        for chunk in chunks:
            chunk_type = (chunk.metadata or {}).get("type", "")
            if chunk_type == "thread_digest":
                digest_text = chunk.content
            elif chunk_type == "email_body":
                body_chunks.append(chunk)
            else:
                did = str(chunk.document_id)
                att_chunk_counts[did] = att_chunk_counts.get(did, 0) + 1

        # Extract context summary from first body chunk
        context_summary = None
        if body_chunks:
            context_summary = body_chunks[0].context_summary

        # Collect vessel info from chunk metadata
        vessel_ids = set()
        vessel_types = set()
        vessel_classes = set()
        for chunk in chunks:
            meta = chunk.metadata or {}
            for vid in meta.get("vessel_ids", []):
                vessel_ids.add(vid)
            for vt in meta.get("vessel_types", []):
                vessel_types.add(vt)
            for vc in meta.get("vessel_classes", []):
                vessel_classes.add(vc)

        # Resolve topic names
        topics_info = []
        if topic_ids:
            for tid in topic_ids:
                topic = self._topics.get(str(tid))
                if topic:
                    topics_info.append({
                        "name": topic.name,
                        "display_name": getattr(topic, "display_name", topic.name),
                    })

        # Build attachment summary
        attachments_info = []
        for doc in attachment_docs:
            attachments_info.append({
                "file_name": doc.file_name,
                "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else doc.document_type,
                "chunks": att_chunk_counts.get(str(doc.id), 0),
            })

        # Email metadata
        email_meta = getattr(email_doc, "email_metadata", None)

        result = {
            "doc_id": doc_id,
            "subject": email_meta.subject if email_meta else getattr(email_doc, "source_title", None),
            "participants": email_meta.participants if email_meta else [],
            "initiator": email_meta.initiator if email_meta else None,
            "date_start": email_meta.date_start.isoformat() if email_meta and hasattr(email_meta.date_start, "isoformat") else (email_meta.date_start if email_meta else None),
            "date_end": email_meta.date_end.isoformat() if email_meta and hasattr(email_meta.date_end, "isoformat") else (email_meta.date_end if email_meta else None),
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

        result_path = archive_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    # ── Manifest / flush / close ───────────────────────────────────

    def write_manifest(self):
        """Write manifest.json with ingest metadata."""
        from ..config import get_settings

        settings = get_settings()
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "topic_count": len(self._topics),
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote manifest to {manifest_path}")

    def flush(self):
        """Rewrite JSONL files merging prior data with current run."""
        # Merge prior documents with current run (current overrides by id, dedup by doc_id)
        current_doc_ids = {str(doc.id) for doc in self._documents.values()}
        seen_doc_ids = {doc.doc_id for doc in self._documents.values() if hasattr(doc, "doc_id") and doc.doc_id}
        docs_path = self.output_dir / "documents.jsonl"

        # First pass: collect candidate prior docs (dedup + override filter)
        candidate_prior: list[dict] = []
        for prior_doc in self._prior_documents:
            if prior_doc.get("id") in current_doc_ids:
                continue
            did = prior_doc.get("doc_id")
            if did and did in seen_doc_ids:
                continue
            if did:
                seen_doc_ids.add(did)
            candidate_prior.append(prior_doc)

        # Build full UUID set to detect orphan children (root_id → non-existent parent)
        all_valid_doc_uuids: set[str] = (
            {pd["id"] for pd in candidate_prior}
            | {str(doc.id) for doc in self._documents.values()}
        )

        # Write, skipping orphan children whose parent was replaced
        with open(docs_path, "w") as f:
            for prior_doc in candidate_prior:
                root_id = prior_doc.get("root_id", "")
                if root_id and root_id != prior_doc["id"] and root_id not in all_valid_doc_uuids:
                    all_valid_doc_uuids.discard(prior_doc["id"])
                    continue
                f.write(json.dumps(prior_doc, default=str) + "\n")
            for doc in self._documents.values():
                f.write(json.dumps(self._doc_to_dict(doc), default=str) + "\n")

        # Rewrite chunks: current in-memory chunks (with merged topic IDs) + prior chunks
        current_chunk_ids = {str(cid) for cid in self._chunks}
        seen_chunk_ids = {c.chunk_id for c in self._chunks.values() if hasattr(c, "chunk_id") and c.chunk_id}
        chunks_path = self.output_dir / "chunks.jsonl"
        # Read prior chunks from disk, apply topic ID replacements, skip orphan/duplicate chunks
        prior_lines = []
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            c = json.loads(line)
                            if c.get("id") in current_chunk_ids:
                                continue
                            # Drop orphan chunks (document was deleted for reprocess)
                            if c.get("document_id") not in all_valid_doc_uuids:
                                continue
                            cid = c.get("chunk_id")
                            if cid and cid in seen_chunk_ids:
                                continue
                            if cid:
                                seen_chunk_ids.add(cid)
                            # Apply absorbed topic ID replacements to prior chunks
                            meta = c.get("metadata")
                            if isinstance(meta, dict) and self._merged_topic_map:
                                tids = meta.get("topic_ids")
                                if tids:
                                    meta["topic_ids"] = list(dict.fromkeys(
                                        self._merged_topic_map.get(t, t) for t in tids
                                    ))
                            prior_lines.append(json.dumps(c, default=str))
                        except json.JSONDecodeError:
                            pass
        with open(chunks_path, "w", encoding="utf-8") as f:
            for line in prior_lines:
                f.write(line + "\n")
            for chunk in self._chunks.values():
                # Also filter current-run orphan chunks (doc_id dedup can leave unregistered UUIDs)
                if str(getattr(chunk, "document_id", "")) not in all_valid_doc_uuids:
                    continue
                f.write(json.dumps(self._chunk_to_dict(chunk), default=str) + "\n")

        # Merge prior topics with current run (current overrides by id)
        # Drop orphan topics: created during extraction but never linked to chunks
        topics_path = self.output_dir / "topics.jsonl"
        current_topic_ids = set(self._topics.keys())
        with open(topics_path, "w") as f:
            for tid, prior_topic in self._prior_topics.items():
                if tid not in current_topic_ids:
                    f.write(json.dumps(prior_topic, default=str) + "\n")
            for topic in self._topics.values():
                if (getattr(topic, "chunk_count", 0) or 0) == 0 and (getattr(topic, "document_count", 0) or 0) == 0:
                    logger.debug(f"Dropping orphan topic: {topic.name}")
                    continue
                f.write(json.dumps(self._topic_to_dict(topic), default=str) + "\n")

        # Remove orphan archive folders (on disk but not in documents)
        archive_dir = self.output_dir / "archive"
        if archive_dir.exists():
            valid_folder_ids = set()
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            if d.get("document_type") == "email" and d.get("doc_id"):
                                valid_folder_ids.add(d["doc_id"][:16])
                        except json.JSONDecodeError:
                            pass
            import shutil
            for folder in archive_dir.iterdir():
                if folder.is_dir() and folder.name not in valid_folder_ids:
                    logger.info(f"Removing orphan archive folder: {folder.name}")
                    shutil.rmtree(folder)

    async def persist_ingest_result(
        self,
        email_doc,
        attachment_docs: list,
        chunks: list,
        topic_ids: list | None = None,
        chunk_delta: int = 0,
    ) -> None:
        """Persist all documents + chunks atomically (buffer in memory, flush on close)."""
        # Track UUID remaps from doc_id dedup (insert_document may return existing doc)
        uuid_remap: Dict[str, str] = {}
        stored = await self.insert_document(email_doc)
        if stored.id != email_doc.id:
            uuid_remap[str(email_doc.id)] = str(stored.id)
        for doc in attachment_docs:
            stored = await self.insert_document(doc)
            if stored.id != doc.id:
                uuid_remap[str(doc.id)] = str(stored.id)
        # Remap chunk document_ids if any docs were deduped
        if chunks and uuid_remap:
            for chunk in chunks:
                old_did = str(getattr(chunk, "document_id", ""))
                if old_did in uuid_remap:
                    chunk.document_id = UUID(uuid_remap[old_did])
        if chunks:
            await self.insert_chunks(chunks)
        if topic_ids and chunk_delta:
            await self.increment_topic_counts(topic_ids, chunk_delta=chunk_delta, document_delta=1)

        # Write result.json to archive folder
        self._write_result_json(email_doc, attachment_docs, chunks, topic_ids)

    async def close(self) -> None:
        """No resources to release for local storage."""
        pass


@dataclass
class LocalBucketStorage:
    """Mock ArchiveStorage that writes to local folders.

    Provides the same interface as ArchiveStorage but stores files
    in a local directory structure.
    """

    bucket_dir: Path

    def __post_init__(self):
        """Ensure bucket directory exists."""
        self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(
        self,
        path: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Upload file to local bucket directory."""
        file_path = self.bucket_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return path

    def upload_text(
        self,
        path: str,
        text: str,
        content_type: str = "text/markdown; charset=utf-8",
    ) -> str:
        """Upload text file to local bucket directory."""
        return self.upload_file(path, text.encode("utf-8"), content_type)

    def download_file(self, path: str) -> bytes:
        """Download file from local bucket directory."""
        file_path = self.bucket_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_bytes()

    def download_text(self, path: str, encoding: str = "utf-8") -> str:
        """Download text file from local bucket directory."""
        return self.download_file(path).decode(encoding)

    def file_exists(self, path: str) -> bool:
        """Check if file exists in local bucket directory."""
        return (self.bucket_dir / path).exists()

    def delete_folder(self, doc_id: str, preserve_md: bool = False) -> None:
        """Delete folder from local bucket directory."""
        folder_path = self.bucket_dir / doc_id
        if folder_path.exists():
            import shutil
            if preserve_md:
                # Delete non-md files only
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and not file_path.suffix == ".md":
                        file_path.unlink()
            else:
                shutil.rmtree(folder_path)

    def list_files(self, doc_id: str) -> List[Dict[str, Any]]:
        """List files in a folder."""
        folder_path = self.bucket_dir / doc_id
        if not folder_path.exists():
            return []

        files = []
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.bucket_dir)),
                    "size": file_path.stat().st_size,
                })
        return files


@dataclass
class LocalIngestOutput:
    """Coordinates local storage for test capture.

    Provides a unified interface for capturing all ingest output
    to local files for inspection and debugging.
    """

    db: LocalStorageClient
    bucket: LocalBucketStorage

    @classmethod
    def create(cls, output_root: Path) -> "LocalIngestOutput":
        """Create a LocalIngestOutput with default directory structure.

        Args:
            output_root: Root directory for all output.

        Returns:
            Configured LocalIngestOutput instance.
        """
        return cls(
            db=LocalStorageClient(output_root / "database"),
            bucket=LocalBucketStorage(output_root / "bucket"),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of captured output."""
        return {
            "documents": len(self.db._documents),
            "chunks": len(self.db._chunks),
            "events": len(self.db._events),
            "bucket_files": len(list(self.bucket.bucket_dir.rglob("*"))),
        }

    def _read_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """Read all records from a JSONL file."""
        file_path = self.db.output_dir / filename
        if not file_path.exists():
            return []
        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def read_documents_jsonl(self) -> List[Dict[str, Any]]:
        return self._read_jsonl("documents.jsonl")

    def read_chunks_jsonl(self) -> List[Dict[str, Any]]:
        return self._read_jsonl("chunks.jsonl")

    def read_events_jsonl(self) -> List[Dict[str, Any]]:
        return self._read_jsonl("ingest_events.jsonl")
