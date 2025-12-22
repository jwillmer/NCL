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

            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                init=register_vector,
            )
        return self._pool

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ==================== Document Operations ====================

    async def insert_document(self, doc: Document) -> Document:
        """Insert a document record.

        Args:
            doc: Document to insert.

        Returns:
            The inserted document.
        """
        data = {
            "id": str(doc.id),
            "parent_id": str(doc.parent_id) if doc.parent_id else None,
            "root_id": str(doc.root_id) if doc.root_id else None,
            "depth": doc.depth,
            "path": doc.path,
            "document_type": doc.document_type.value,
            "file_path": doc.file_path,
            "file_name": doc.file_name,
            "file_hash": doc.file_hash,
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
        """Insert multiple chunks with embeddings.

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
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.heading_path,
                    chunk.section_title,
                    chunk.page_number,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.embedding,
                    json.dumps(chunk.metadata),
                )
                for chunk in chunks
            ]

            await conn.executemany(
                """
                INSERT INTO chunks
                (id, document_id, content, chunk_index, heading_path,
                 section_title, page_number, start_char, end_char, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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

    # ==================== Vector Search ====================

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector (1536 dimensions).
            match_threshold: Minimum similarity score (0-1).
            match_count: Maximum number of results.

        Returns:
            List of matching chunks with document context.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM match_chunks($1, $2, $3)
                """,
                query_embedding,
                match_threshold,
                match_count,
            )

        return [dict(row) for row in rows]

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
            content=row["content"],
            chunk_index=row["chunk_index"],
            heading_path=row.get("heading_path") or [],
            section_title=row.get("section_title"),
            page_number=row.get("page_number"),
            start_char=row.get("start_char"),
            end_char=row.get("end_char"),
            embedding=row.get("embedding"),
            metadata=row.get("metadata") or {},
        )
