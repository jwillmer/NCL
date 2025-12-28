"""Ingest version management for document re-processing.

Tracks schema/logic versions and determines what action to take
for each document during ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID

from ..config import get_settings
from ..storage.supabase_client import SupabaseClient
from ..utils import compute_doc_id, normalize_source_id

logger = logging.getLogger(__name__)


@dataclass
class IngestDecision:
    """Decision about what action to take for a document during ingestion."""

    action: Literal["insert", "update", "skip", "reprocess"]
    reason: str
    existing_doc_id: Optional[UUID] = None


class VersionManager:
    """Manage document versions and re-ingestion decisions.

    Determines whether a document should be:
    - insert: New document, never seen before
    - update: Same source, but content changed
    - skip: Already processed with current ingest version
    - reprocess: Processed with older ingest logic version
    """

    def __init__(self, db: Optional[SupabaseClient] = None):
        """Initialize version manager.

        Args:
            db: Supabase client instance (creates new one if not provided).
        """
        self.db = db or SupabaseClient()
        settings = get_settings()
        self.current_version = settings.current_ingest_version

    async def check_document(
        self,
        source_id: str,
        file_hash: str,
    ) -> IngestDecision:
        """Determine what action to take for a document.

        Args:
            source_id: Normalized source identifier (from normalize_source_id).
            file_hash: SHA-256 hash of file content.

        Returns:
            IngestDecision with action and reason.
        """
        # Compute content-addressable doc_id
        doc_id = compute_doc_id(source_id, file_hash)

        # Check if this exact version exists
        existing = await self.db.get_document_by_doc_id(doc_id)

        if existing:
            # Same content exists - check ingest version
            # Note: We're using getattr with a fallback since existing models
            # may not have ingest_version attribute yet
            existing_version = getattr(existing, "ingest_version", 1)

            if existing_version < self.current_version:
                return IngestDecision(
                    action="reprocess",
                    reason=f"Ingest logic upgraded from v{existing_version} to v{self.current_version}",
                    existing_doc_id=existing.id,
                )
            return IngestDecision(
                action="skip",
                reason="Already processed with current version",
                existing_doc_id=existing.id,
            )

        # Check if same source exists with different content
        old_version = await self.db.get_document_by_source_id(source_id)

        if old_version:
            return IngestDecision(
                action="update",
                reason="Content changed since last ingest",
                existing_doc_id=old_version.id,
            )

        # New document
        return IngestDecision(
            action="insert",
            reason="New document",
        )

    async def get_reprocess_candidates(
        self,
        target_version: Optional[int] = None,
        limit: int = 100,
    ) -> list:
        """Get documents that need re-processing.

        Args:
            target_version: Re-process documents below this version.
                           Defaults to current_ingest_version.
            limit: Maximum number of documents to return.

        Returns:
            List of Document objects needing re-processing.
        """
        version = target_version or self.current_version
        return await self.db.get_documents_below_version(version, limit)

    async def count_reprocess_candidates(
        self,
        target_version: Optional[int] = None,
    ) -> int:
        """Count documents that need re-processing.

        Args:
            target_version: Count documents below this version.
                           Defaults to current_ingest_version.

        Returns:
            Count of documents needing re-processing.
        """
        version = target_version or self.current_version
        return await self.db.count_documents_below_version(version)
