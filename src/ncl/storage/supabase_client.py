"""Supabase client wrapper for database operations and vector search."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client, create_client

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

        Args:
            doc_id: UUID of the document to delete.
        """
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
        result = (
            self.client.table("chunks")
            .select("*")
            .eq("document_id", str(doc_id))
            .order("chunk_index")
            .execute()
        )

        return [self._row_to_chunk(row) for row in result.data]

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

    # ==================== Cleanup Operations ====================

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

        # unsupported_files has FK to documents
        result = (
            self.client.table("unsupported_files")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["unsupported_files"] = len(result.data) if result.data else 0

        # chunks has FK to documents (will also be deleted by CASCADE, but explicit is clearer)
        result = (
            self.client.table("chunks")
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )
        counts["chunks"] = len(result.data) if result.data else 0

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
            "imo": vessel.imo,
            "vessel_type": vessel.vessel_type,
            "dwt": vessel.dwt,
            "aliases": vessel.aliases,
        }
        self.client.table("vessels").insert(data).execute()
        return vessel

    async def upsert_vessel(self, vessel: Vessel) -> Vessel:
        """Insert or update a vessel record by IMO number.

        Args:
            vessel: Vessel to upsert.

        Returns:
            The upserted vessel.
        """
        data = {
            "id": str(vessel.id),
            "name": vessel.name,
            "imo": vessel.imo,
            "vessel_type": vessel.vessel_type,
            "dwt": vessel.dwt,
            "aliases": vessel.aliases,
        }
        self.client.table("vessels").upsert(data, on_conflict="imo").execute()
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
            .select("id, name, imo, vessel_type")
            .order("name")
            .execute()
        )
        return [
            VesselSummary(
                id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
                name=row["name"],
                imo=row.get("imo"),
                vessel_type=row.get("vessel_type"),
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
            imo=row.get("imo"),
            vessel_type=row.get("vessel_type"),
            dwt=row.get("dwt"),
            aliases=row.get("aliases") or [],
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
