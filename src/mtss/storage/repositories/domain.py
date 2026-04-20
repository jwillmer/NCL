"""Vessels, Topics CRUD, counts, and admin operations."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from ...models.topic import Topic, TopicSummary
from ...models.vessel import Vessel, VesselSummary
from .base import BaseRepository

logger = logging.getLogger(__name__)


class DomainRepository(BaseRepository):
    """Handles vessel, topic, and admin database operations."""

    # ==================== Vessel Operations ====================

    async def insert_vessel(self, vessel: Vessel) -> Vessel:
        """Insert a vessel record.

        Args:
            vessel: Vessel to insert.

        Returns:
            The inserted vessel.
        """
        data = {
            "id": str(vessel.id),
            "name": vessel.name,
            "vessel_type": vessel.vessel_type,
            "vessel_class": vessel.vessel_class,
            "aliases": vessel.aliases,
        }
        self.client.table("vessels").insert(data).execute()
        return vessel

    async def upsert_vessel(self, vessel: Vessel) -> Vessel:
        """Insert or update a vessel record by name.

        Args:
            vessel: Vessel to upsert.

        Returns:
            The upserted vessel.
        """
        data = {
            "id": str(vessel.id),
            "name": vessel.name,
            "vessel_type": vessel.vessel_type,
            "vessel_class": vessel.vessel_class,
            "aliases": vessel.aliases,
        }
        self.client.table("vessels").upsert(data, on_conflict="name").execute()
        return vessel

    async def get_all_vessels(self) -> List[Vessel]:
        """Get all vessels from the registry.

        Returns:
            List of all vessels ordered by name.
        """
        result = (
            self.client.table("vessels")
            .select("*")
            .order("name")
            .execute()
        )
        return [self._row_to_vessel(row) for row in result.data]

    async def get_vessel_summaries(self) -> List[VesselSummary]:
        """Get minimal vessel info for dropdown lists.

        Returns:
            List of vessel summaries ordered by name.
        """
        result = (
            self.client.table("vessels")
            .select("id, name, vessel_type, vessel_class")
            .order("name")
            .execute()
        )
        return [
            VesselSummary(
                id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
                name=row["name"],
                vessel_type=row.get("vessel_type") or "",
                vessel_class=row.get("vessel_class") or "",
            )
            for row in result.data
        ]

    async def get_vessel_by_id(self, vessel_id: UUID) -> Optional[Vessel]:
        """Get vessel by ID.

        Args:
            vessel_id: Vessel UUID.

        Returns:
            Vessel if found, None otherwise.
        """
        result = (
            self.client.table("vessels")
            .select("*")
            .eq("id", str(vessel_id))
            .limit(1)
            .execute()
        )
        if result.data:
            return self._row_to_vessel(result.data[0])
        return None

    async def delete_all_vessels(self) -> int:
        """Delete all vessels from the registry.

        Returns:
            Number of vessels deleted.
        """
        result = (
            self.client.table("vessels")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        return len(result.data) if result.data else 0

    async def get_unique_vessel_types(self) -> List[str]:
        """Get distinct vessel types from the vessels table.

        Returns:
            List of unique vessel type strings sorted alphabetically.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT vessel_type FROM vessels
                WHERE vessel_type IS NOT NULL AND vessel_type != ''
                ORDER BY vessel_type
                """
            )
        return [row["vessel_type"] for row in rows]

    async def get_unique_vessel_classes(self) -> List[str]:
        """Get distinct vessel classes from the vessels table.

        Returns:
            List of unique vessel class strings sorted alphabetically.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT vessel_class FROM vessels
                WHERE vessel_class IS NOT NULL AND vessel_class != ''
                ORDER BY vessel_class
                """
            )
        return [row["vessel_class"] for row in rows]

    async def update_chunks_vessel_ids(
        self, document_id: UUID, vessel_ids: List[str]
    ) -> int:
        """Update vessel_ids in chunk metadata for all chunks of a document.

        Uses JSONB set operation to update the vessel_ids field in metadata.
        Also updates chunks for child documents (attachments).

        Args:
            document_id: Root document UUID.
            vessel_ids: List of vessel UUID strings to set.

        Returns:
            Number of chunks updated.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Update chunks for the root document and all its children
            # Using a subquery to get all document IDs in the hierarchy
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = CASE
                    WHEN $2::text[] = '{}'::text[] THEN
                        metadata - 'vessel_ids'
                    ELSE
                        jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{vessel_ids}',
                            to_jsonb($2::text[])
                        )
                END
                WHERE document_id IN (
                    SELECT id FROM documents
                    WHERE id = $1 OR root_id = $1
                )
                """,
                document_id,
                vessel_ids,
            )
            # Extract count from "UPDATE N" result
            count_str = result.split()[-1] if result else "0"
            return int(count_str)

    async def update_chunks_vessel_metadata(
        self,
        document_id: UUID,
        vessel_ids: List[str],
        vessel_types: List[str],
        vessel_classes: List[str],
    ) -> int:
        """Update vessel_ids, vessel_types, and vessel_classes in chunk metadata.

        Uses JSONB set operations to update all three fields in metadata.
        Also updates chunks for child documents (attachments).

        Args:
            document_id: Root document UUID.
            vessel_ids: List of vessel UUID strings to set.
            vessel_types: List of unique vessel types for matched vessels.
            vessel_classes: List of unique vessel classes for matched vessels.

        Returns:
            Number of chunks updated.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Build the metadata update - remove empty arrays, set non-empty ones
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = (
                    SELECT
                        CASE WHEN $4::text[] = '{}'::text[] THEN m ELSE jsonb_set(m, '{vessel_classes}', to_jsonb($4::text[])) END
                    FROM (
                        SELECT
                            CASE WHEN $3::text[] = '{}'::text[] THEN m ELSE jsonb_set(m, '{vessel_types}', to_jsonb($3::text[])) END AS m
                        FROM (
                            SELECT
                                CASE WHEN $2::text[] = '{}'::text[]
                                    THEN COALESCE(metadata, '{}'::jsonb) - 'vessel_ids'
                                    ELSE jsonb_set(COALESCE(metadata, '{}'::jsonb), '{vessel_ids}', to_jsonb($2::text[]))
                                END AS m
                        ) sub1
                    ) sub2
                )
                WHERE document_id IN (
                    SELECT id FROM documents
                    WHERE id = $1 OR root_id = $1
                )
                """,
                document_id,
                vessel_ids,
                vessel_types,
                vessel_classes,
            )
            # Extract count from "UPDATE N" result
            count_str = result.split()[-1] if result else "0"
            return int(count_str)

    async def get_root_documents_for_retagging(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get root documents (emails) for vessel retagging.

        Returns documents with their archive paths for content retrieval.

        Args:
            limit: Optional limit on number of documents.

        Returns:
            List of document records with id, doc_id, archive_browse_uri.
        """
        query = (
            self.client.table("documents")
            .select("id, doc_id, archive_browse_uri, file_name")
            .eq("depth", 0)
            .eq("status", "completed")
            .order("created_at", desc=True)
        )
        if limit:
            query = query.limit(limit)

        result = query.execute()
        return result.data or []

    async def get_current_vessel_ids(self, document_id: UUID) -> List[str]:
        """Get current vessel_ids from a document's chunks.

        Args:
            document_id: Document UUID.

        Returns:
            List of vessel UUID strings currently tagged on the document's chunks.
        """
        result = (
            self.client.table("chunks")
            .select("metadata")
            .eq("document_id", str(document_id))
            .limit(1)
            .execute()
        )
        if result.data and result.data[0].get("metadata"):
            return result.data[0]["metadata"].get("vessel_ids", [])
        return []

    # ==================== Topic Operations ====================

    async def insert_topic(self, topic: Topic) -> Topic:
        """Insert a new topic.

        Args:
            topic: Topic to insert.

        Returns:
            The inserted topic with generated ID.
        """
        data = {
            "name": topic.name,
            "display_name": topic.display_name,
            "description": topic.description,
            "embedding": topic.embedding,
            "chunk_count": topic.chunk_count,
            "document_count": topic.document_count,
        }
        result = self.client.table("topics").insert(data).execute()
        if result.data:
            return self._row_to_topic(result.data[0])
        return topic

    async def get_topic_by_name(self, name: str) -> Optional[Topic]:
        """Get topic by canonical name (exact match).

        Args:
            name: Canonical topic name (lowercase).

        Returns:
            Topic if found, None otherwise.
        """
        result = (
            self.client.table("topics")
            .select("*")
            .eq("name", name.lower().strip())
            .limit(1)
            .execute()
        )
        return self._row_to_topic(result.data[0]) if result.data else None

    async def get_topic_by_id(self, topic_id: UUID) -> Optional[Topic]:
        """Get topic by ID.

        Args:
            topic_id: Topic UUID.

        Returns:
            Topic if found, None otherwise.
        """
        result = (
            self.client.table("topics")
            .select("*")
            .eq("id", str(topic_id))
            .limit(1)
            .execute()
        )
        return self._row_to_topic(result.data[0]) if result.data else None

    async def find_similar_topics(
        self,
        embedding: List[float],
        threshold: float = 0.85,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find topics with similar embeddings.

        Args:
            embedding: Query embedding vector.
            threshold: Minimum similarity threshold (0-1).
            limit: Maximum results to return.

        Returns:
            List of dicts with id, name, display_name, similarity.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM find_similar_topics($1, $2, $3)
                """,
                embedding,
                threshold,
                limit,
            )
        return [dict(row) for row in rows]

    async def get_all_topics(self) -> List[TopicSummary]:
        """Get all topics ordered by chunk_count.

        Returns:
            List of topic summaries.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM get_topics_with_counts()")
        return [
            TopicSummary(
                id=row["id"],
                name=row["name"],
                display_name=row["display_name"],
                chunk_count=row["chunk_count"],
            )
            for row in rows
        ]

    async def get_chunks_count_for_topic(
        self, topic_id: UUID, vessel_filter: Optional[Dict] = None
    ) -> int:
        """Count chunks tagged with a specific topic.

        Args:
            topic_id: Topic UUID.
            vessel_filter: Optional vessel filter dict.

        Returns:
            Count of matching chunks.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT count_chunks_by_topic($1, $2::jsonb)",
                topic_id,
                json.dumps(vessel_filter) if vessel_filter else None,
            )
        return result or 0

    async def get_chunks_count_for_topics(
        self, topic_ids: List[UUID], vessel_filter: Optional[Dict] = None
    ) -> int:
        """Count chunks tagged with ANY of the given topics (OR logic).

        Args:
            topic_ids: List of topic UUIDs.
            vessel_filter: Optional vessel filter dict.

        Returns:
            Count of matching chunks (deduplicated).
        """
        if not topic_ids:
            return 0
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT count_chunks_by_topics($1, $2::jsonb)",
                topic_ids,
                json.dumps(vessel_filter) if vessel_filter else None,
            )
        return result or 0

    async def increment_topic_counts(
        self,
        topic_ids: List[UUID],
        chunk_delta: int = 0,
        document_delta: int = 0,
    ) -> None:
        """Increment usage counts for topics.

        Args:
            topic_ids: List of topic UUIDs to update.
            chunk_delta: Amount to add to chunk_count.
            document_delta: Amount to add to document_count.
        """
        if not topic_ids:
            return
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            for topic_id in topic_ids:
                await conn.execute(
                    """
                    UPDATE topics
                    SET chunk_count = chunk_count + $2,
                        document_count = document_count + $3,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    topic_id,
                    chunk_delta,
                    document_delta,
                )

    async def update_chunks_topic_ids(
        self, document_id: UUID, topic_ids: List[str]
    ) -> int:
        """Update topic_ids in chunk metadata for all chunks of a document.

        Uses JSONB set operation to update the topic_ids field in metadata.
        Also updates chunks for child documents (attachments).

        Args:
            document_id: Root document UUID.
            topic_ids: List of topic UUID strings to set.

        Returns:
            Number of chunks updated.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = CASE
                    WHEN $2::text[] = '{}'::text[] THEN
                        metadata - 'topic_ids'
                    ELSE
                        jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{topic_ids}',
                            to_jsonb($2::text[])
                        )
                END
                WHERE document_id IN (
                    SELECT id FROM documents
                    WHERE id = $1 OR root_id = $1
                )
                """,
                document_id,
                topic_ids,
            )
            count_str = result.split()[-1] if result else "0"
            return int(count_str)

    async def update_chunks_topics_checked(self, document_id: UUID) -> int:
        """Mark chunks as having topics checked (even if none found).

        Sets topics_checked: true in metadata to prevent re-processing
        documents that legitimately have no extractable topics (e.g.,
        marketing emails with no technical problems).

        Args:
            document_id: Root document UUID.

        Returns:
            Number of chunks updated.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{topics_checked}',
                    'true'::jsonb
                )
                WHERE document_id IN (
                    SELECT id FROM documents
                    WHERE id = $1 OR root_id = $1
                )
                """,
                document_id,
            )
            count_str = result.split()[-1] if result else "0"
            return int(count_str)

    async def delete_all_topics(self) -> int:
        """Delete all topics from the database.

        Returns:
            Number of topics deleted.
        """
        result = (
            self.client.table("topics")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        return len(result.data) if result.data else 0

    # ==================== Admin Operations ====================

    async def reset_failed_documents(self, file_paths: List[str]) -> Dict[str, int]:
        """Reset failed documents by deleting their database entries.

        This allows them to be reprocessed on the next ingest run.
        Used by the `MTSS reset-failures` command to clear failed documents
        from a JSON failure report.

        Deletion order:
        1. processing_log entries (tracks file processing state)
        2. Query child document IDs (attachments)
        3. ingest_events for root + children (no CASCADE)
        4. documents (CASCADE deletes children and chunks)
        5. archive folders from Supabase Storage

        Args:
            file_paths: List of absolute file paths to reset (e.g., from
                failure report JSON). These match the file_path column in
                both processing_log and documents tables.

        Returns:
            Dictionary with counts of deleted records:
            - "documents": Number of document records deleted
            - "processing_log": Number of processing log entries deleted
            - "archives": Number of archive folders deleted from storage
        """
        if not file_paths:
            return {"documents": 0, "processing_log": 0, "archives": 0}

        counts: Dict[str, int] = {"documents": 0, "processing_log": 0, "archives": 0}

        # Batch delete from processing_log (Supabase supports .in_() for batch ops)
        result = (
            self.client.table("processing_log")
            .delete()
            .in_("file_path", file_paths)
            .execute()
        )
        counts["processing_log"] = len(result.data) if result.data else 0

        # Get all document IDs and doc_ids for these file paths
        doc_result = (
            self.client.table("documents")
            .select("id, doc_id")
            .in_("file_path", file_paths)
            .execute()
        )
        doc_ids = [doc["id"] for doc in doc_result.data]
        doc_id_strings = [doc["doc_id"] for doc in doc_result.data if doc.get("doc_id")]

        if doc_ids:
            # Get all child document IDs (attachments) - these will be CASCADE deleted
            # but we need their IDs to clean up ingest_events first
            child_result = (
                self.client.table("documents")
                .select("id")
                .in_("parent_id", doc_ids)
                .execute()
            )
            child_doc_ids = [doc["id"] for doc in child_result.data]
            all_doc_ids = doc_ids + child_doc_ids

            # Delete ingest_events for root docs AND their children (no CASCADE)
            self.client.table("ingest_events").delete().in_(
                "parent_document_id", all_doc_ids
            ).execute()

            # Delete root documents (CASCADE handles child documents and all chunks)
            result = (
                self.client.table("documents")
                .delete()
                .in_("id", doc_ids)
                .execute()
            )
            counts["documents"] = len(result.data) if result.data else 0

        # Delete archive folders from Supabase Storage
        if doc_id_strings:
            from ..archive_storage import ArchiveStorage, ArchiveStorageError

            try:
                storage = ArchiveStorage()
                for doc_id in doc_id_strings:
                    try:
                        storage.delete_folder(doc_id)
                        counts["archives"] += 1
                    except ArchiveStorageError:
                        pass  # Folder may not exist
            except ArchiveStorageError:
                pass  # Storage not configured or bucket doesn't exist

        return counts

    def _batched_delete(
        self,
        table: str,
        *,
        id_col: str = "id",
        batch_size: int = 100,
        status: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Delete every row from ``table`` in small ID-scoped batches.

        PostgREST enforces a per-statement timeout (~8s) and a single
        ``DELETE ... WHERE true`` against a large table blows past it
        (observed on ``topics`` with ~3.4k rows — error 57014). Select
        a batch of primary keys, delete by ``.in_()``, repeat until the
        table drains. Returns total rows deleted.

        ``status`` is an optional live-progress sink (CLI spinner); it
        receives a fresh message after each batch so the caller can show
        the user which table is draining without waiting for the full
        coroutine to return.
        """
        total = 0
        if status:
            status(f"Deleting {table}...")
        while True:
            select_result = (
                self.client.table(table)
                .select(id_col)
                .limit(batch_size)
                .execute()
            )
            if not select_result.data:
                break
            ids = [row[id_col] for row in select_result.data]
            delete_result = (
                self.client.table(table)
                .delete()
                .in_(id_col, ids)
                .execute()
            )
            total += len(delete_result.data) if delete_result.data else 0
            if status:
                status(f"Deleting {table}... {total} rows")
            if len(ids) < batch_size:
                break
        return total

    async def delete_all_data(
        self,
        status: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, int]:
        """Delete all data from all tables.

        Deletes in correct order to respect foreign key constraints.

        ``status`` is an optional live-progress sink: on a large corpus
        this coroutine can easily run for minutes, and without a
        per-table signal the CLI looks frozen to the user. The callback
        receives a short human-readable message for each table it
        starts, plus per-batch updates for the batched-delete paths.

        Returns:
            Dictionary with table names and deleted row counts.
        """
        counts: Dict[str, int] = {}

        def _notify(msg: str) -> None:
            if status:
                status(msg)

        # Delete in order respecting foreign keys
        # conversations has FK to vessels — delete first
        _notify("Deleting conversations...")
        result = (
            self.client.table("conversations")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["conversations"] = len(result.data) if result.data else 0

        # LangGraph checkpoint tables (no FK to our tables, but related to conversations)
        # These are created by AsyncPostgresSaver and store conversation message history
        for table in ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]:
            _notify(f"Deleting {table}...")
            try:
                # Delete all rows - use neq with impossible value as Supabase requires a filter
                result = (
                    self.client.table(table)
                    .delete()
                    .neq("thread_id", "00000000-0000-0000-0000-000000000000")
                    .execute()
                )
                counts[table] = len(result.data) if result.data else 0
            except Exception:
                # Table may not exist yet if no conversations have been created
                counts[table] = 0

        # ingest_events has FK to documents
        _notify("Deleting ingest_events...")
        result = (
            self.client.table("ingest_events")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["ingest_events"] = len(result.data) if result.data else 0

        # chunks has FK to documents (will also be deleted by CASCADE, but explicit is clearer)
        # Delete in batches to avoid statement timeout on large tables
        # Use small batch size (100) to avoid URL length limits with .in_() queries
        counts["chunks"] = self._batched_delete("chunks", status=status)

        # processing_log has no FK
        counts["processing_log"] = self._batched_delete("processing_log", status=status)

        # documents is the main table (CASCADE will handle any remaining children)
        counts["documents"] = self._batched_delete("documents", status=status)

        # vessels — no FK dependencies remain after conversations deleted
        _notify("Deleting vessels...")
        result = (
            self.client.table("vessels")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["vessels"] = len(result.data) if result.data else 0

        # topics — no FK dependencies
        counts["topics"] = self._batched_delete("topics", status=status)

        return counts
