"""Shared components for ingest and ingest-update commands.

Provides a factory function and dataclass to ensure both commands
use identical component initialization, preventing behavioral divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..parsers.attachment_processor import AttachmentProcessor
    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..parsers.eml_parser import EMLParser
    from ..processing.archive_generator import ArchiveGenerator
    from ..processing.embeddings import EmbeddingGenerator
    from ..processing.hierarchy_manager import HierarchyManager
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.archive_storage import ArchiveStorage
    from ..storage.supabase_client import SupabaseClient


@dataclass
class IngestComponents:
    """Shared components for ingest and ingest-update commands.

    All processing operations should use these components to ensure
    identical behavior between regular ingest and repair operations.
    """

    db: SupabaseClient
    eml_parser: EMLParser
    attachment_processor: AttachmentProcessor
    hierarchy_manager: HierarchyManager
    embeddings: EmbeddingGenerator
    archive_generator: ArchiveGenerator
    context_generator: ContextGenerator
    chunker: DocumentChunker
    archive_storage: ArchiveStorage
    vessel_matcher: Optional[VesselMatcher] = None


def create_ingest_components(
    db: SupabaseClient,
    source_dir: Path,
    vessels: Optional[list] = None,
) -> IngestComponents:
    """Create all ingest components with identical initialization.

    This factory function is used by both `ingest` and `ingest-update`
    commands to ensure they use the same component configuration.

    Args:
        db: Initialized Supabase client.
        source_dir: Root directory for email ingestion.
        vessels: Optional list of vessels for VesselMatcher.

    Returns:
        IngestComponents dataclass with all initialized components.
    """
    # Import here to avoid circular imports
    from ..parsers.attachment_processor import AttachmentProcessor
    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..parsers.eml_parser import EMLParser
    from ..processing.archive_generator import ArchiveGenerator
    from ..processing.embeddings import EmbeddingGenerator
    from ..processing.hierarchy_manager import HierarchyManager
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.archive_storage import ArchiveStorage

    return IngestComponents(
        db=db,
        eml_parser=EMLParser(),
        attachment_processor=AttachmentProcessor(),
        hierarchy_manager=HierarchyManager(db, ingest_root=source_dir),
        embeddings=EmbeddingGenerator(),
        archive_generator=ArchiveGenerator(ingest_root=source_dir),
        context_generator=ContextGenerator(),
        chunker=DocumentChunker(),
        archive_storage=ArchiveStorage(),
        vessel_matcher=VesselMatcher(vessels) if vessels else None,
    )
