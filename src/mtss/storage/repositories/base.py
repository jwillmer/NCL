"""Base repository with shared pool management and row mappers."""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from ...models.chunk import Chunk
from ...models.document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    EmailMetadata,
    ProcessingStatus,
)
from ...models.topic import Topic
from ...models.vessel import Vessel

logger = logging.getLogger(__name__)


class BaseRepository:
    """Shared infrastructure for all repositories.

    Owns the asyncpg connection pool and row-to-model mappers so that
    every repository can convert database rows without duplicating logic.
    """

    def __init__(self, client, db_url: str):
        self.client = client
        self.db_url = db_url
        self._pool = None
        self._pool_source: BaseRepository | None = None  # Delegate pool to another repo

    async def get_pool(self):
        """Get or create asyncpg connection pool for direct Postgres access.

        If _pool_source is set, delegates to that repository to ensure
        all repositories share a single connection pool.
        """
        if self._pool_source is not None:
            return await self._pool_source.get_pool()
        if self._pool is None:
            import asyncpg
            from pgvector.asyncpg import register_vector

            async def init_connection(conn):
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

    # ==================== Row Mappers ====================

    def _row_to_document(self, row: Dict[str, Any]) -> Document:
        """Convert database row to Document model."""
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

        doc_type = row["document_type"]
        if isinstance(doc_type, str):
            doc_type = DocumentType(doc_type)

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
            source_id=row.get("source_id"),
            doc_id=row.get("doc_id"),
            content_version=row.get("content_version", 1),
            ingest_version=row.get("ingest_version", 1),
            source_title=row.get("source_title"),
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
        """Convert database row to Chunk model."""
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

    def _row_to_vessel(self, row: Dict[str, Any]) -> Vessel:
        """Convert database row to Vessel model."""
        return Vessel(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            name=row["name"],
            vessel_type=row.get("vessel_type") or "",
            vessel_class=row.get("vessel_class") or "",
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def _row_to_topic(self, row: Dict[str, Any]) -> Topic:
        """Convert database row to Topic model."""
        return Topic(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            name=row["name"],
            display_name=row["display_name"],
            description=row.get("description"),
            embedding=row.get("embedding"),
            chunk_count=row.get("chunk_count", 0),
            document_count=row.get("document_count", 0),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
