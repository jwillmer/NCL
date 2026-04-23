"""Document and chunk CRUD, ancestry, context updates, ingest events."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...config import get_settings
from ...models.chunk import Chunk
from ...models.document import Document, ProcessingStatus
from .base import BaseRepository


def _to_datetime(val: Any) -> datetime | None:
    """Ensure value is a datetime object (asyncpg requires this for timestamptz)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return val


def _strip_nul(val: Any) -> Any:
    """Recursively strip NUL bytes (\\x00) from strings.

    Postgres TEXT columns reject `\\x00` as "invalid byte sequence for encoding
    UTF8". SQLite (the local ingest store) tolerates them, so malformed bytes
    surviving a parse can reach import without warning. Applied at the asyncpg
    bind boundary so every text column, array, and jsonb leaf is scrubbed.
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val.replace("\x00", "") if "\x00" in val else val
    if isinstance(val, list):
        return [_strip_nul(v) for v in val]
    if isinstance(val, dict):
        return {k: _strip_nul(v) for k, v in val.items()}
    return val


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
                        _to_datetime(doc.email_metadata.date_start)
                    ),
                    "email_date_end": (
                        _to_datetime(doc.email_metadata.date_end)
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

    async def get_corpus_stats(self) -> dict[str, int]:
        """Return `{emails, documents, topics, vessels}` counts for the UI footer.

        Single aggregate round trip — the stats only shift on ingest and the
        API caches this for 5 min, so the query load is negligible.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    (SELECT count(*) FILTER (WHERE document_type = 'email')  FROM documents) AS emails,
                    (SELECT count(*) FILTER (WHERE document_type <> 'email') FROM documents) AS documents,
                    (SELECT count(*) FROM topics)                                            AS topics,
                    (SELECT count(*) FROM vessels)                                           AS vessels
                """
            )
        return {
            "emails": int(row["emails"] or 0),
            "documents": int(row["documents"] or 0),
            "topics": int(row["topics"] or 0),
            "vessels": int(row["vessels"] or 0),
        }

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
            await self._insert_chunks_pg(conn, chunks)

        return chunks

    async def persist_ingest_result(
        self,
        email_doc: Document,
        attachment_docs: List[Document],
        chunks: List[Chunk],
        topic_ids: List[UUID] | None = None,
        chunk_delta: int = 0,
    ) -> None:
        """Persist all documents + chunks in a single asyncpg transaction.

        Ensures atomic persistence: either everything is committed or nothing.
        This prevents partial data from crashes mid-processing.
        """
        settings = get_settings()
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Insert all documents
                for doc in [email_doc] + attachment_docs:
                    await self._insert_document_pg(conn, doc, settings)

                # Batch insert all chunks
                if chunks:
                    await self._insert_chunks_pg(conn, chunks)

                # Increment topic counts
                if topic_ids and chunk_delta:
                    for topic_id in topic_ids:
                        await conn.execute(
                            """
                            UPDATE topics
                            SET chunk_count = chunk_count + $2,
                                document_count = document_count + $3,
                                updated_at = NOW()
                            WHERE id = $1
                            """,
                            topic_id, chunk_delta, 1,
                        )

    @staticmethod
    async def _insert_chunks_pg(conn, chunks: List[Chunk]) -> None:
        """Insert chunks via asyncpg connection (shared by insert_chunks and persist_ingest_result)."""
        records = [
            (
                chunk.id, _strip_nul(chunk.chunk_id or ""), chunk.document_id,
                _strip_nul(chunk.content), chunk.chunk_index,
                _strip_nul(chunk.context_summary), _strip_nul(chunk.embedding_text),
                _strip_nul(chunk.section_path), _strip_nul(chunk.section_title),
                _strip_nul(chunk.source_title), _strip_nul(chunk.source_id),
                chunk.page_number, chunk.line_from, chunk.line_to,
                chunk.char_start, chunk.char_end,
                _strip_nul(chunk.archive_browse_uri), _strip_nul(chunk.archive_download_uri),
                chunk.embedding, json.dumps(_strip_nul(chunk.metadata)),
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
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
            """,
            records,
        )

    @staticmethod
    async def _insert_document_pg(conn, doc: Document, settings) -> None:
        """Insert a document via asyncpg connection (for use within transactions)."""
        await conn.execute(
            """
            INSERT INTO documents (
                id, source_id, doc_id, content_version, ingest_version,
                parent_id, root_id, depth, path,
                document_type, file_path, file_name, file_hash,
                source_title, archive_path, archive_browse_uri, archive_download_uri,
                status, error_message, processed_at,
                email_subject, email_participants, email_initiator,
                email_date_start, email_date_end, email_message_count,
                attachment_content_type, attachment_size_bytes
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                $21,$22,$23,$24,$25,$26,$27,$28
            )
            """,
            doc.id,
            _strip_nul(doc.source_id or ""),
            _strip_nul(doc.doc_id or ""),
            doc.content_version,
            doc.ingest_version or settings.current_ingest_version,
            doc.parent_id,
            doc.root_id,
            doc.depth,
            _strip_nul(doc.path),
            doc.document_type.value,
            _strip_nul(doc.file_path),
            _strip_nul(doc.file_name),
            _strip_nul(doc.file_hash),
            _strip_nul(doc.source_title),
            _strip_nul(doc.archive_path),
            _strip_nul(doc.archive_browse_uri),
            _strip_nul(doc.archive_download_uri),
            doc.status.value,
            _strip_nul(doc.error_message),
            datetime.now(timezone.utc) if doc.status == ProcessingStatus.COMPLETED else None,
            _strip_nul(doc.email_metadata.subject) if doc.email_metadata else None,
            _strip_nul(doc.email_metadata.participants) if doc.email_metadata else None,
            _strip_nul(doc.email_metadata.initiator) if doc.email_metadata else None,
            _to_datetime(doc.email_metadata.date_start) if doc.email_metadata else None,
            _to_datetime(doc.email_metadata.date_end) if doc.email_metadata else None,
            doc.email_metadata.message_count if doc.email_metadata else None,
            _strip_nul(doc.attachment_metadata.content_type) if doc.attachment_metadata else None,
            doc.attachment_metadata.size_bytes if doc.attachment_metadata else None,
        )

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
                        _strip_nul(chunk.chunk_id or ""),
                        chunk.document_id,
                        _strip_nul(chunk.content),
                        chunk.chunk_index,
                        _strip_nul(chunk.context_summary),
                        _strip_nul(chunk.embedding_text),
                        _strip_nul(chunk.section_path),
                        _strip_nul(chunk.section_title),
                        _strip_nul(chunk.source_title),
                        _strip_nul(chunk.source_id),
                        chunk.page_number,
                        chunk.line_from,
                        chunk.line_to,
                        chunk.char_start,
                        chunk.char_end,
                        _strip_nul(chunk.archive_browse_uri),
                        _strip_nul(chunk.archive_download_uri),
                        chunk.embedding,
                        json.dumps(_strip_nul(chunk.metadata)),
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

    def get_origin_email_for_document(self, document_id: UUID | str) -> Optional[Dict[str, Any]]:
        """Resolve a chunk's originating email (root email doc).

        For attachment chunks, returns the root email's subject + the
        chunk_id of its first email-body chunk so the UI can re-open the
        dialog on the email. Returns None when the chunk is already part
        of the email (no deeper origin to surface).
        """
        doc_row = (
            self.client.table("documents")
            .select("id, parent_id, root_id, document_type, source_title")
            .eq("id", str(document_id))
            .limit(1)
            .execute()
        )
        if not doc_row.data:
            return None

        row = doc_row.data[0]
        doc_type = row.get("document_type")
        if isinstance(doc_type, str):
            doc_type_str = doc_type
        else:
            doc_type_str = getattr(doc_type, "value", "") or ""

        if doc_type_str == "email" or not row.get("parent_id"):
            return None

        root_id = row.get("root_id") or row.get("parent_id")
        if not root_id:
            return None

        root_row = (
            self.client.table("documents")
            .select("id, source_title")
            .eq("id", str(root_id))
            .limit(1)
            .execute()
        )
        if not root_row.data:
            return None

        subject = root_row.data[0].get("source_title")

        chunk_row = (
            self.client.table("chunks")
            .select("chunk_id, chunk_index")
            .eq("document_id", str(root_id))
            .gte("chunk_index", 0)
            .order("chunk_index")
            .limit(1)
            .execute()
        )
        if not chunk_row.data:
            return None

        origin_chunk_id = chunk_row.data[0].get("chunk_id")
        if not origin_chunk_id:
            return None

        return {"subject": subject, "chunk_id": origin_chunk_id}

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
