"""Supabase client facade that delegates to focused repositories."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import create_client

from ..config import get_settings
from ..models.chunk import Chunk
from ..models.document import Document, ProcessingStatus
from ..models.topic import Topic, TopicSummary
from ..models.vessel import Vessel, VesselSummary
from .repositories import DocumentRepository, DomainRepository, SearchRepository

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Facade for Supabase operations including vector search.

    Delegates to focused repositories while keeping a single public interface.
    All existing method signatures are preserved for backward compatibility.
    """

    def __init__(self):
        """Initialize Supabase client and repositories.

        Raises:
            ValueError: If Supabase config fields are not set (use --local-only mode instead).
        """
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_key or not settings.supabase_db_url:
            raise ValueError(
                "SUPABASE_URL, SUPABASE_KEY, and SUPABASE_DB_URL must be set. "
                "Use --local-only mode if you don't have a Supabase instance."
            )
        self.client = create_client(settings.supabase_url, settings.supabase_key)
        self.db_url = settings.supabase_db_url

        self._docs = DocumentRepository(self.client, self.db_url)
        self._search = SearchRepository(self.client, self.db_url)
        self._domain = DomainRepository(self.client, self.db_url)

        # Share a single pool across all repositories to avoid multiple connections
        self._search._pool_source = self._docs
        self._domain._pool_source = self._docs

    # ==================== Pool Management ====================

    @property
    def _pool(self):
        """Access the shared connection pool (for backward compatibility with tests)."""
        return self._docs._pool

    @_pool.setter
    def _pool(self, value):
        """Set pool on all repositories (for backward compatibility with tests)."""
        self._docs._pool = value
        self._search._pool = value
        self._domain._pool = value

    async def get_pool(self):
        """Get or create asyncpg connection pool for direct Postgres access."""
        return await self._docs.get_pool()

    async def close(self):
        """Close connection pool."""
        await self._docs.close()

    # ==================== Document Operations ====================

    async def insert_document(self, doc: Document) -> Document:
        return await self._docs.insert_document(doc)

    async def update_document_status(
        self, doc_id: UUID, status: ProcessingStatus, error_message: Optional[str] = None
    ):
        return await self._docs.update_document_status(doc_id, status, error_message)

    async def update_document_archive_uris(
        self, doc_id: UUID, archive_browse_uri: str, archive_download_uri: str | None = None
    ):
        return await self._docs.update_document_archive_uris(
            doc_id, archive_browse_uri, archive_download_uri
        )

    async def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        return await self._docs.get_document_by_hash(file_hash)

    async def get_document_by_id(self, doc_id: UUID) -> Optional[Document]:
        return await self._docs.get_document_by_id(doc_id)

    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Document]:
        return await self._docs.get_document_by_doc_id(doc_id)

    async def get_document_by_source_id(self, source_id: str) -> Optional[Document]:
        return await self._docs.get_document_by_source_id(source_id)

    async def get_document_ancestry(self, doc_id: UUID) -> List[Document]:
        return await self._docs.get_document_ancestry(doc_id)

    async def get_document_children(self, doc_id: UUID) -> List[Document]:
        return await self._docs.get_document_children(doc_id)

    async def get_documents_below_version(
        self, target_version: int, limit: int = 100
    ) -> List[Document]:
        return await self._docs.get_documents_below_version(target_version, limit)

    async def count_documents_below_version(self, target_version: int) -> int:
        return await self._docs.count_documents_below_version(target_version)

    def delete_document_for_reprocess(self, doc_id: UUID):
        return self._docs.delete_document_for_reprocess(doc_id)

    async def get_all_root_source_ids(self) -> Dict[str, UUID]:
        return await self._docs.get_all_root_source_ids()

    async def delete_orphaned_documents(self, doc_ids: List[UUID]) -> int:
        return await self._docs.delete_orphaned_documents(doc_ids)

    # ==================== Chunk Operations ====================

    async def insert_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        return await self._docs.insert_chunks(chunks)

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Chunk]:
        return await self._docs.get_chunks_by_document(doc_id)

    async def delete_chunks_by_document(self, doc_id: UUID) -> int:
        return await self._docs.delete_chunks_by_document(doc_id)

    async def replace_chunks_atomic(self, doc_id: UUID, new_chunks: List[Chunk]) -> int:
        return await self._docs.replace_chunks_atomic(doc_id, new_chunks)

    async def persist_ingest_result(
        self,
        email_doc: Document,
        attachment_docs: List[Document],
        chunks: List[Chunk],
        topic_ids: List[UUID] | None = None,
        chunk_delta: int = 0,
    ) -> None:
        """Persist all documents + chunks in a single asyncpg transaction."""
        return await self._docs.persist_ingest_result(
            email_doc, attachment_docs, chunks, topic_ids, chunk_delta,
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self._docs.get_chunk_by_id(chunk_id)

    async def update_chunk_context(self, chunk: Chunk) -> None:
        return await self._docs.update_chunk_context(chunk)

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
        return self._docs.log_ingest_event(
            document_id, event_type, severity, message, file_path, file_name, source_eml_path
        )

    # ==================== Vector Search ====================

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._search.search_similar_chunks(
            query_embedding, match_threshold, match_count, metadata_filter, query_text
        )

    # ==================== Vessel Operations ====================

    async def insert_vessel(self, vessel: Vessel) -> Vessel:
        return await self._domain.insert_vessel(vessel)

    async def upsert_vessel(self, vessel: Vessel) -> Vessel:
        return await self._domain.upsert_vessel(vessel)

    async def get_all_vessels(self) -> List[Vessel]:
        return await self._domain.get_all_vessels()

    async def get_vessel_summaries(self) -> List[VesselSummary]:
        return await self._domain.get_vessel_summaries()

    async def get_vessel_by_id(self, vessel_id: UUID) -> Optional[Vessel]:
        return await self._domain.get_vessel_by_id(vessel_id)

    async def delete_all_vessels(self) -> int:
        return await self._domain.delete_all_vessels()

    async def get_unique_vessel_types(self) -> List[str]:
        return await self._domain.get_unique_vessel_types()

    async def get_unique_vessel_classes(self) -> List[str]:
        return await self._domain.get_unique_vessel_classes()

    async def update_chunks_vessel_ids(self, document_id: UUID, vessel_ids: List[str]) -> int:
        return await self._domain.update_chunks_vessel_ids(document_id, vessel_ids)

    async def update_chunks_vessel_metadata(
        self,
        document_id: UUID,
        vessel_ids: List[str],
        vessel_types: List[str],
        vessel_classes: List[str],
    ) -> int:
        return await self._domain.update_chunks_vessel_metadata(
            document_id, vessel_ids, vessel_types, vessel_classes
        )

    async def get_root_documents_for_retagging(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return await self._domain.get_root_documents_for_retagging(limit)

    async def get_current_vessel_ids(self, document_id: UUID) -> List[str]:
        return await self._domain.get_current_vessel_ids(document_id)

    # ==================== Topic Operations ====================

    async def insert_topic(self, topic: Topic) -> Topic:
        return await self._domain.insert_topic(topic)

    async def get_topic_by_name(self, name: str) -> Optional[Topic]:
        return await self._domain.get_topic_by_name(name)

    async def get_topic_by_id(self, topic_id: UUID) -> Optional[Topic]:
        return await self._domain.get_topic_by_id(topic_id)

    async def find_similar_topics(
        self, embedding: List[float], threshold: float = 0.85, limit: int = 5
    ) -> List[Dict[str, Any]]:
        return await self._domain.find_similar_topics(embedding, threshold, limit)

    async def get_all_topics(self) -> List[TopicSummary]:
        return await self._domain.get_all_topics()

    async def get_chunks_count_for_topic(
        self, topic_id: UUID, vessel_filter: Optional[Dict] = None
    ) -> int:
        return await self._domain.get_chunks_count_for_topic(topic_id, vessel_filter)

    async def get_chunks_count_for_topics(
        self, topic_ids: List[UUID], vessel_filter: Optional[Dict] = None
    ) -> int:
        return await self._domain.get_chunks_count_for_topics(topic_ids, vessel_filter)

    async def increment_topic_counts(
        self, topic_ids: List[UUID], chunk_delta: int = 0, document_delta: int = 0
    ) -> None:
        return await self._domain.increment_topic_counts(topic_ids, chunk_delta, document_delta)

    async def update_chunks_topic_ids(self, document_id: UUID, topic_ids: List[str]) -> int:
        return await self._domain.update_chunks_topic_ids(document_id, topic_ids)

    async def update_chunks_topics_checked(self, document_id: UUID) -> int:
        return await self._domain.update_chunks_topics_checked(document_id)

    async def delete_all_topics(self) -> int:
        return await self._domain.delete_all_topics()

    # ==================== Admin Operations ====================

    async def reset_failed_documents(self, file_paths: List[str]) -> Dict[str, int]:
        return await self._domain.reset_failed_documents(file_paths)

    async def delete_all_data(self) -> Dict[str, int]:
        return await self._domain.delete_all_data()

