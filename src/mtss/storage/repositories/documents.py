"""Document and chunk CRUD, ancestry, context updates, ingest events."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...config import get_settings
from ...models.chunk import Chunk
from ...models.document import Document, ProcessingStatus
from .base import BaseRepository

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Handles all document and chunk database operations."""

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

    async def update_document_archive_uris(
        self,
        doc_id: UUID,
        archive_browse_uri: str,
        archive_download_uri: str | None = None,
    ):
        """Update document's archive URIs and propagate to chunks.

        Args:
            doc_id: Document UUID.
            archive_browse_uri: URI to the browsable markdown file.
            archive_download_uri: Optional URI to the original file for download.
        """
        data: Dict[str, Any] = {"archive_browse_uri": archive_browse_uri, "updated_at": "now()"}
        if archive_download_uri:
            data["archive_download_uri"] = archive_download_uri
        self.client.table("documents").update(data).eq("id", str(doc_id)).execute()

        # Propagate to chunks (prevents denormalization drift)
        chunk_data: Dict[str, Any] = {"archive_browse_uri": archive_browse_uri}
        if archive_download_uri:
            chunk_data["archive_download_uri"] = archive_download_uri
        self.client.table("chunks").update(chunk_data).eq("document_id", str(doc_id)).execute()

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

    async def get_documents_below_version(
        self, target_version: int, limit: int = 100
    ) -> List[Document]:
        """Get documents with ingest_version below target.

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
