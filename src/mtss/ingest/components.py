"""Shared components for ingest and ingest-update commands.

Provides factory functions and a dataclass to ensure all ingest commands
use identical component initialization, preventing behavioral divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..parsers.attachment_processor import AttachmentProcessor
    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..parsers.eml_parser import EMLParser
    from ..processing.embeddings import EmbeddingGenerator
    from ..processing.topics import TopicExtractor, TopicMatcher
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.archive_storage import ArchiveStorage
    from .archive_generator import ArchiveGenerator
    from .hierarchy_manager import HierarchyManager


@dataclass
class IngestComponents:
    """Shared components for ingest and ingest-update commands.

    All processing operations should use these components to ensure
    identical behavior between regular ingest and repair operations.
    """

    db: Any
    eml_parser: EMLParser
    attachment_processor: AttachmentProcessor
    hierarchy_manager: HierarchyManager
    embeddings: EmbeddingGenerator
    archive_generator: ArchiveGenerator
    context_generator: ContextGenerator
    chunker: DocumentChunker
    archive_storage: ArchiveStorage
    vessel_matcher: Optional[VesselMatcher] = None
    topic_extractor: Optional[TopicExtractor] = None
    topic_matcher: Optional[TopicMatcher] = None


def _create_shared_components(
    db,
    source_dir: Path,
    vessels: list | None = None,
    enable_topics: bool = True,
    archive_storage=None,
    archive_generator_storage=None,
) -> IngestComponents:
    """Internal helper: create components shared between Supabase and local modes."""
    # Import here to avoid circular imports
    from ..parsers.attachment_processor import AttachmentProcessor
    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..parsers.eml_parser import EMLParser
    from ..processing.embeddings import EmbeddingGenerator
    from ..processing.topics import TopicExtractor, TopicMatcher
    from ..processing.vessel_matcher import VesselMatcher
    from .archive_generator import ArchiveGenerator
    from .hierarchy_manager import HierarchyManager

    if archive_storage is None:
        from ..storage.archive_storage import ArchiveStorage
        archive_storage = ArchiveStorage()

    embeddings = EmbeddingGenerator()

    # Create topic components if enabled
    topic_extractor = None
    topic_matcher = None
    if enable_topics:
        topic_extractor = TopicExtractor()
        topic_matcher = TopicMatcher(db, embeddings)

    return IngestComponents(
        db=db,
        eml_parser=EMLParser(),
        attachment_processor=AttachmentProcessor(),
        hierarchy_manager=HierarchyManager(db, ingest_root=source_dir),
        embeddings=embeddings,
        archive_generator=ArchiveGenerator(
            ingest_root=source_dir,
            storage=archive_generator_storage,
        ),
        context_generator=ContextGenerator(),
        chunker=DocumentChunker(),
        archive_storage=archive_storage,
        vessel_matcher=VesselMatcher(vessels) if vessels else None,
        topic_extractor=topic_extractor,
        topic_matcher=topic_matcher,
    )


def create_ingest_components(
    db,
    source_dir: Path,
    vessels: list | None = None,
    enable_topics: bool = True,
) -> IngestComponents:
    """Create all ingest components for Supabase mode.

    This factory function is used by both ``ingest`` and ``ingest-update``
    commands to ensure they use the same component configuration.

    Args:
        db: Initialized Supabase client.
        source_dir: Root directory for email ingestion.
        vessels: Optional list of vessels for VesselMatcher.
        enable_topics: Whether to enable topic extraction (default True).

    Returns:
        IngestComponents dataclass with all initialized components.
    """
    return _create_shared_components(db, source_dir, vessels, enable_topics)


def create_local_ingest_components(
    db,
    output_dir: Path,
    source_dir: Path,
    vessels: list | None = None,
    enable_topics: bool = True,
) -> IngestComponents:
    """Create all ingest components for local-only mode (SQLite output).

    Args:
        db: Initialized SqliteStorageClient instance.
        output_dir: Directory holding ``ingest.db`` and the ``archive/`` tree.
        source_dir: Root directory for email ingestion.
        vessels: Optional list of vessels for VesselMatcher.
        enable_topics: Whether to enable topic extraction (default True).

    Returns:
        IngestComponents dataclass with all initialized components.
    """
    from ..storage.local_bucket_storage import LocalBucketStorage

    archive_storage = LocalBucketStorage(output_dir / "archive")

    return _create_shared_components(
        db=db,
        source_dir=source_dir,
        vessels=vessels,
        enable_topics=enable_topics,
        archive_storage=archive_storage,
        archive_generator_storage=archive_storage,
    )
