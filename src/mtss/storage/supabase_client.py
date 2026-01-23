"""Supabase client wrapper for database operations and vector search."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client, create_client

logger = logging.getLogger(__name__)

from ..config import get_settings
from ..models.chunk import Chunk
from ..models.document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    EmailMetadata,
    ProcessingStatus,
)
from ..models.vessel import Vessel, VesselSummary


class SupabaseClient:
    """Wrapper for Supabase operations including vector search.

    Handles document CRUD, chunk storage with embeddings, and vector similarity search.
    """

    def __init__(self):
        """Initialize Supabase client."""
        settings = get_settings()
        self.client: Client = create_client(settings.supabase_url, settings.supabase_key)
        self.db_url = settings.supabase_db_url
        self._pool = None

    async def get_pool(self):
        """Get or create asyncpg connection pool for direct Postgres access.

        Used for bulk operations and vector search that benefit from direct access.
        """
        if self._pool is None:
            import asyncpg
            from pgvector.asyncpg import register_vector

            async def init_connection(conn):
                # Register vector type with the extensions schema where pgvector is installed
                await register_vector(conn, schema="extensions")

            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                init=init_connection,
            )
        return self._pool

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def delete_document_for_reprocess(self, doc_id: UUID):
        """Delete a document and its children to prepare for reprocessing.

        Child documents (attachments) and chunks are automatically deleted
        due to ON DELETE CASCADE constraints in the database schema.
        ingest_events must be deleted explicitly as it lacks CASCADE.

        Args:
            doc_id: UUID of the document to delete.
        """
        # Delete ingest_events first (no CASCADE constraint)
        self.client.table("ingest_events").delete().eq(
            "parent_document_id", str(doc_id)
        ).execute()
        # Now delete the document (cascades to child docs and chunks)
        self.client.table("documents").delete().eq("id", str(doc_id)).execute()

    # ==================== Document Operations ====================

    async def insert_document(self, doc: Document) -> Document:
        """Insert a document record.

        Args:
            doc: Document to insert (should have all fields populated).

        Returns:
            The inserted document.
        """
        settings = get_settings()

        data = {
            "id": str(doc.id),
            # Stable identification (from Document model)
            "source_id": doc.source_id or "",
            "doc_id": doc.doc_id or "",
            # Versioning
            "content_version": doc.content_version,
            "ingest_version": doc.ingest_version or settings.current_ingest_version,
            # Hierarchy
            "parent_id": str(doc.parent_id) if doc.parent_id else None,
            "root_id": str(doc.root_id) if doc.root_id else None,
            "depth": doc.depth,
            "path": doc.path,
            # Identification
            "document_type": doc.document_type.value,
            "file_path": doc.file_path,
            "file_name": doc.file_name,
            "file_hash": doc.file_hash,
            # Citation metadata (from Document model)
            "source_title": doc.source_title,
            # Archive links (from Document model)
            "archive_path": doc.archive_path,
            "archive_browse_uri": doc.archive_browse_uri,
            "archive_download_uri": doc.archive_download_uri,
            # Status
            "status": doc.status.value,
        }

        # Add email conversation metadata if present
        if doc.email_metadata:
            data.update(
                {
                    "email_subject": doc.email_metadata.subject,
                    "email_participants": doc.email_metadata.participants,
                    "email_initiator": doc.email_metadata.initiator,
                    "email_date_start": (
                        doc.email_metadata.date_start.isoformat()
                        if doc.email_metadata.date_start
                        else None
                    ),
                    "email_date_end": (
                        doc.email_metadata.date_end.isoformat()
                        if doc.email_metadata.date_end
                        else None
                    ),
                    "email_message_count": doc.email_metadata.message_count,
                }
            )

        # Add attachment metadata if present
        if doc.attachment_metadata:
            data.update(
                {
                    "attachment_content_type": doc.attachment_metadata.content_type,
                    "attachment_size_bytes": doc.attachment_metadata.size_bytes,
                }
            )

        self.client.table("documents").insert(data).execute()
        return doc

    async def update_document_status(
        self,
        doc_id: UUID,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ):
        """Update document processing status.

        Args:
            doc_id: Document UUID.
            status: New processing status.
            error_message: Optional error message for failed status.
        """
        data: Dict[str, Any] = {"status": status.value, "updated_at": "now()"}
        if error_message:
            data["error_message"] = error_message
        if status == ProcessingStatus.COMPLETED:
            data["processed_at"] = "now()"

        self.client.table("documents").update(data).eq("id", str(doc_id)).execute()

    async def update_document_archive_browse_uri(
        self,
        doc_id: UUID,
        archive_browse_uri: str,
    ):
        """Update document's archive_browse_uri after markdown file is created.

        Args:
            doc_id: Document UUID.
            archive_browse_uri: URI to the browsable markdown file.
        """
        self.client.table("documents").update(
            {"archive_browse_uri": archive_browse_uri, "updated_at": "now()"}
        ).eq("id", str(doc_id)).execute()

    async def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Check if document with hash already exists.

        Args:
            file_hash: SHA-256 hash of file content.

        Returns:
            Document if found, None otherwise.
        """
        result = (
            self.client.table("documents")
            .select("*")
            .eq("file_hash", file_hash)
            .limit(1)
            .execute()
        )

        if result.data:
            return self._row_to_document(result.data[0])
        return None

    async def get_document_by_id(self, doc_id: UUID) -> Optional[Document]:
        """Get document by ID.

        Args:
            doc_id: Document UUID.

        Returns:
            Document if found, None otherwise.
        """
        result = (
            self.client.table("documents").select("*").eq("id", str(doc_id)).limit(1).execute()
        )

        if result.data:
            return self._row_to_document(result.data[0])
        return None

    async def get_document_ancestry(self, doc_id: UUID) -> List[Document]:
        """Get ancestry using database function.

        Args:
            doc_id: Document UUID.

        Returns:
            List of documents from root to the specified document.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM get_document_ancestry($1)", doc_id)
        return [self._row_to_document(dict(row)) for row in rows]

    async def get_document_children(self, doc_id: UUID) -> List[Document]:
        """Get direct children of a document.

        Args:
            doc_id: Document UUID.

        Returns:
            List of child documents.
        """
        result = (
            self.client.table("documents").select("*").eq("parent_id", str(doc_id)).execute()
        )

        return [self._row_to_document(row) for row in result.data]

    # ==================== Chunk Operations ====================

    async def insert_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Insert multiple chunks with embeddings and citation metadata.

        Args:
            chunks: List of chunks to insert.

        Returns:
            The inserted chunks.
        """
        if not chunks:
            return chunks

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            records = [
                (
                    chunk.id,
                    chunk.chunk_id or "",  # Stable chunk ID for citations
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.context_summary,  # LLM-generated context
                    chunk.embedding_text,  # Full text used for embedding
                    chunk.section_path,  # Renamed from heading_path
                    chunk.section_title,
                    chunk.source_title,  # Denormalized citation metadata
                    chunk.source_id,
                    chunk.page_number,
                    chunk.line_from,
                    chunk.line_to,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.archive_browse_uri,
                    chunk.archive_download_uri,
                    chunk.embedding,
                    json.dumps(chunk.metadata),
                )
                for chunk in chunks
            ]

            await conn.executemany(
                """
                INSERT INTO chunks
                (id, chunk_id, document_id, content, chunk_index,
                 context_summary, embedding_text, section_path, section_title,
                 source_title, source_id,
                 page_number, line_from, line_to, char_start, char_end,
                 archive_browse_uri, archive_download_uri,
                 embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """,
                records,
            )

        return chunks

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Chunk]:
        """Get all chunks for a document.

        Args:
            doc_id: Document UUID.

        Returns:
            List of chunks ordered by index.
        """
        # Select all fields except embedding (REST API returns vectors as strings)
        fields = (
            "id, document_id, chunk_id, content, chunk_index, context_summary, "
            "embedding_text, section_path, section_title, source_title, source_id, "
            "page_number, line_from, line_to, char_start, char_end, "
            "archive_browse_uri, archive_download_uri, metadata"
        )
        result = (
            self.client.table("chunks")
            .select(fields)
            .eq("document_id", str(doc_id))
            .order("chunk_index")
            .execute()
        )

        return [self._row_to_chunk(row) for row in result.data]

    async def delete_chunks_by_document(self, doc_id: UUID) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document UUID.

        Returns:
            Number of chunks deleted.
        """
        # First count how many chunks exist
        count_result = (
            self.client.table("chunks")
            .select("id", count="exact")
            .eq("document_id", str(doc_id))
            .execute()
        )
        count = count_result.count or 0

        if count > 0:
            # Delete the chunks
            self.client.table("chunks").delete().eq("document_id", str(doc_id)).execute()

        return count

    async def replace_chunks_atomic(self, doc_id: UUID, new_chunks: List[Chunk]) -> int:
        """Atomically delete old chunks and insert new ones.

        Uses database transaction to ensure no data loss on failure.
        If insert fails, delete is rolled back.

        Args:
            doc_id: Document UUID whose chunks to replace.
            new_chunks: New chunks to insert.

        Returns:
            Number of chunks inserted.
        """
        if not new_chunks:
            # Just delete if no new chunks
            await self.delete_chunks_by_document(doc_id)
            return 0

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Delete within transaction
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = $1", doc_id
                )

                # Insert within same transaction
                records = [
                    (
                        chunk.id,
                        chunk.chunk_id or "",
                        chunk.document_id,
                        chunk.content,
                        chunk.chunk_index,
                        chunk.context_summary,
                        chunk.embedding_text,
                        chunk.section_path,
                        chunk.section_title,
                        chunk.source_title,
                        chunk.source_id,
                        chunk.page_number,
                        chunk.line_from,
                        chunk.line_to,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.archive_browse_uri,
                        chunk.archive_download_uri,
                        chunk.embedding,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in new_chunks
                ]

                await conn.executemany(
                    """
                    INSERT INTO chunks
                    (id, chunk_id, document_id, content, chunk_index,
                     context_summary, embedding_text, section_path, section_title,
                     source_title, source_id,
                     page_number, line_from, line_to, char_start, char_end,
                     archive_browse_uri, archive_download_uri,
                     embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    """,
                    records,
                )

        return len(new_chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its chunk_id (the short hex ID used in citations).

        Uses synchronous Supabase client (suitable for single-row lookups).
        Excludes embedding field to avoid REST API string serialization issues.

        Args:
            chunk_id: The chunk's hex ID (e.g., "8f3a2b1c").

        Returns:
            Chunk if found, None otherwise.
        """
        # Select all fields except embedding (REST API returns vectors as strings)
        fields = (
            "id, document_id, chunk_id, content, chunk_index, context_summary, "
            "embedding_text, section_path, section_title, source_title, source_id, "
            "page_number, line_from, line_to, char_start, char_end, "
            "archive_browse_uri, archive_download_uri, metadata"
        )
        result = (
            self.client.table("chunks")
            .select(fields)
            .eq("chunk_id", chunk_id)
            .limit(1)
            .execute()
        )

        if result.data:
            return self._row_to_chunk(result.data[0])
        return None

    async def update_chunk_context(self, chunk: Chunk) -> None:
        """Update a chunk's context_summary, embedding_text, and embedding.

        Used by ingest-update when adding context summaries to existing chunks.

        Args:
            chunk: Chunk with updated context_summary, embedding_text, and embedding.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chunks
                SET context_summary = $2,
                    embedding_text = $3,
                    embedding = $4
                WHERE id = $1
                """,
                chunk.id,
                chunk.context_summary,
                chunk.embedding_text,
                chunk.embedding,
            )

    # ==================== Vector Search ====================

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector (1536 dimensions).
            match_threshold: Minimum similarity score (0-1).
            match_count: Maximum number of results.
            metadata_filter: Optional JSONB filter (e.g., {"vessel_ids": ["uuid"]}).

        Returns:
            List of matching chunks with document context.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            rows = await conn.fetch(
                """
                SELECT * FROM match_chunks($1, $2, $3, $4::jsonb)
                """,
                query_embedding,
                match_threshold,
                match_count,
                filter_json,
            )

        return [dict(row) for row in rows]

    # ==================== Version-aware Queries ====================

    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Document]:
        """Get document by content-addressable doc_id.

        Args:
            doc_id: Content-addressable document ID.

        Returns:
            Document if found, None otherwise.
        """
        result = (
            self.client.table("documents")
            .select("*")
            .eq("doc_id", doc_id)
            .limit(1)
            .execute()
        )

        if result.data:
            return self._row_to_document(result.data[0])
        return None

    async def get_document_by_source_id(self, source_id: str) -> Optional[Document]:
        """Get document by source identifier.

        Args:
            source_id: Normalized source path.

        Returns:
            Document if found, None otherwise.
        """
        result = (
            self.client.table("documents")
            .select("*")
            .eq("source_id", source_id)
            .limit(1)
            .execute()
        )

        if result.data:
            return self._row_to_document(result.data[0])
        return None

    async def get_documents_below_version(
        self, target_version: int, limit: int = 100
    ) -> List[Document]:
        """Get documents with ingest_version below target.

        Used for bulk re-processing when ingest logic is upgraded.

        Args:
            target_version: Get documents with version below this.
            limit: Maximum number of documents to return.

        Returns:
            List of documents needing re-processing.
        """
        result = (
            self.client.table("documents")
            .select("*")
            .lt("ingest_version", target_version)
            .limit(limit)
            .execute()
        )

        return [self._row_to_document(row) for row in result.data]

    async def count_documents_below_version(self, target_version: int) -> int:
        """Count documents with ingest_version below target.

        Args:
            target_version: Count documents with version below this.

        Returns:
            Count of documents needing re-processing.
        """
        result = (
            self.client.table("documents")
            .select("id", count="exact")
            .lt("ingest_version", target_version)
            .execute()
        )

        return result.count or 0

    # ==================== Ingest Events ====================

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
        """Log an ingest event to the ingest_events table.

        Used to track processing issues, encoding fallbacks, parse failures, etc.
        for visibility and debugging.

        Note: This is a synchronous method as it uses the sync Supabase client.

        Args:
            document_id: Parent document UUID.
            event_type: Type of event (e.g., 'parse_failure', 'encoding_fallback').
            severity: Event severity ('error', 'warning', 'info').
            message: Description of the event (truncated to 200 chars for security).
            file_path: Path to the file that caused the event.
            file_name: Original filename.
            source_eml_path: Source EML file path.
        """
        data: Dict[str, Any] = {
            "parent_document_id": str(document_id),
            "event_type": event_type,
            "severity": severity,
            "reason": message[:200] if message else event_type,
        }

        if file_path:
            data["file_path"] = file_path
        if file_name:
            data["file_name"] = file_name
        if source_eml_path:
            data["source_eml_path"] = source_eml_path

        try:
            self.client.table("ingest_events").insert(data).execute()
        except Exception as e:
            # Log but don't fail the ingest if event logging fails
            logger.warning(f"Failed to log ingest event: {e}")

    # ==================== Cleanup Operations ====================

    async def get_all_root_source_ids(self) -> Dict[str, UUID]:
        """Get all source_ids for root documents (depth=0).

        Returns:
            Dict mapping source_id -> document UUID.
        """
        result = (
            self.client.table("documents")
            .select("id, source_id")
            .eq("depth", 0)
            .execute()
        )

        return {row["source_id"]: UUID(row["id"]) for row in result.data if row.get("source_id")}

    async def delete_orphaned_documents(self, doc_ids: List[UUID]) -> int:
        """Delete orphaned documents and their children.

        Children and chunks are deleted via CASCADE constraints.

        Args:
            doc_ids: List of document UUIDs to delete.

        Returns:
            Number of root documents deleted.
        """
        if not doc_ids:
            return 0

        deleted = 0
        for doc_id in doc_ids:
            try:
                # Delete ingest_events first (no CASCADE)
                self.client.table("ingest_events").delete().eq(
                    "parent_document_id", str(doc_id)
                ).execute()

                # Delete document (cascades to children and chunks)
                self.client.table("documents").delete().eq("id", str(doc_id)).execute()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete orphaned document {doc_id}: {e}")

        return deleted

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
            from .archive_storage import ArchiveStorage, ArchiveStorageError

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

    async def delete_all_data(self) -> Dict[str, int]:
        """Delete all data from all tables.

        Deletes in correct order to respect foreign key constraints.

        Returns:
            Dictionary with table names and deleted row counts.
        """
        counts: Dict[str, int] = {}

        # Delete in order respecting foreign keys
        # conversations has FK to vessels
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
        chunks_deleted = 0
        batch_size = 100
        while True:
            # First, select a batch of IDs
            select_result = (
                self.client.table("chunks")
                .select("id")
                .limit(batch_size)
                .execute()
            )
            if not select_result.data:
                break
            ids_to_delete = [row["id"] for row in select_result.data]
            # Delete the batch by IDs
            result = (
                self.client.table("chunks")
                .delete()
                .in_("id", ids_to_delete)
                .execute()
            )
            batch_count = len(result.data) if result.data else 0
            chunks_deleted += batch_count
            if len(ids_to_delete) < batch_size:
                break
        counts["chunks"] = chunks_deleted

        # processing_log has no FK
        result = (
            self.client.table("processing_log")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["processing_log"] = len(result.data) if result.data else 0

        # documents is the main table (CASCADE will handle any remaining children)
        result = (
            self.client.table("documents")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["documents"] = len(result.data) if result.data else 0

        return counts

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

    def _row_to_vessel(self, row: Dict[str, Any]) -> Vessel:
        """Convert database row to Vessel model.

        Args:
            row: Database row as dictionary.

        Returns:
            Vessel model instance.
        """
        return Vessel(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            name=row["name"],
            vessel_type=row.get("vessel_type") or "",
            vessel_class=row.get("vessel_class") or "",
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    # ==================== Helper Methods ====================

    def _row_to_document(self, row: Dict[str, Any]) -> Document:
        """Convert database row to Document model.

        Args:
            row: Database row as dictionary.

        Returns:
            Document model instance.
        """
        email_metadata = None
        if row.get("email_subject") or row.get("email_participants"):
            email_metadata = EmailMetadata(
                subject=row.get("email_subject"),
                participants=row.get("email_participants") or [],
                initiator=row.get("email_initiator"),
                date_start=row.get("email_date_start"),
                date_end=row.get("email_date_end"),
                message_count=row.get("email_message_count") or 1,
            )

        attachment_metadata = None
        if row.get("attachment_content_type"):
            attachment_metadata = AttachmentMetadata(
                content_type=row["attachment_content_type"],
                size_bytes=row.get("attachment_size_bytes") or 0,
                original_filename=row["file_name"],
            )

        # Handle document_type whether it's a string or enum
        doc_type = row["document_type"]
        if isinstance(doc_type, str):
            doc_type = DocumentType(doc_type)

        # Handle status whether it's a string or enum
        status = row.get("status", "pending")
        if isinstance(status, str):
            status = ProcessingStatus(status)

        return Document(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            parent_id=UUID(row["parent_id"]) if row.get("parent_id") else None,
            root_id=UUID(row["root_id"]) if row.get("root_id") else None,
            depth=row["depth"],
            path=row.get("path") or [],
            document_type=doc_type,
            file_path=row["file_path"],
            file_name=row["file_name"],
            file_hash=row.get("file_hash"),
            # Stable identification for citations and version tracking
            source_id=row.get("source_id"),
            doc_id=row.get("doc_id"),
            content_version=row.get("content_version", 1),
            ingest_version=row.get("ingest_version", 1),
            source_title=row.get("source_title"),
            # Archive links
            archive_path=row.get("archive_path"),
            archive_browse_uri=row.get("archive_browse_uri"),
            archive_download_uri=row.get("archive_download_uri"),
            email_metadata=email_metadata,
            attachment_metadata=attachment_metadata,
            status=status,
            error_message=row.get("error_message"),
            processed_at=row.get("processed_at"),
        )

    def _row_to_chunk(self, row: Dict[str, Any]) -> Chunk:
        """Convert database row to Chunk model.

        Args:
            row: Database row as dictionary.

        Returns:
            Chunk model instance.
        """
        return Chunk(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            document_id=(
                UUID(row["document_id"])
                if isinstance(row["document_id"], str)
                else row["document_id"]
            ),
            chunk_id=row.get("chunk_id"),
            content=row["content"],
            chunk_index=row["chunk_index"],
            context_summary=row.get("context_summary"),
            embedding_text=row.get("embedding_text"),
            section_path=row.get("section_path") or row.get("heading_path") or [],
            section_title=row.get("section_title"),
            source_title=row.get("source_title"),
            source_id=row.get("source_id"),
            page_number=row.get("page_number"),
            line_from=row.get("line_from"),
            line_to=row.get("line_to"),
            char_start=row.get("char_start") or row.get("start_char"),
            char_end=row.get("char_end") or row.get("end_char"),
            archive_browse_uri=row.get("archive_browse_uri"),
            archive_download_uri=row.get("archive_download_uri"),
            embedding=row.get("embedding"),
            metadata=row.get("metadata") or {},
        )
