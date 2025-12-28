"""Document hierarchy manager for maintaining parent-child relationships."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from ..config import get_settings
from ..models.document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    ParsedEmail,
)
from ..storage.supabase_client import SupabaseClient
from ..utils import compute_doc_id, normalize_source_id

if TYPE_CHECKING:
    from .archive_generator import ArchiveResult, ContentFileResult


class HierarchyManager:
    """Manages document hierarchy relationships.

    Handles the creation and tracking of parent-child relationships
    between emails and their attachments for proper context in RAG.
    Generates stable IDs for citations and archive paths.
    """

    def __init__(self, db_client: SupabaseClient, ingest_root: Optional[Path] = None):
        """Initialize the hierarchy manager.

        Args:
            db_client: Supabase client for database operations.
            ingest_root: Root directory for ingestion (for stable source IDs).
        """
        self.db = db_client
        settings = get_settings()
        self.ingest_root = ingest_root or settings.eml_source_dir
        self.archive_dir = settings.archive_dir
        self.archive_base_url = settings.archive_base_url
        self.current_ingest_version = settings.current_ingest_version

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal hash string.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def create_email_document(
        self,
        eml_path: Path,
        parsed_email: ParsedEmail,
        archive_result: Optional["ArchiveResult"] = None,
    ) -> Document:
        """Create a root email document in the hierarchy.

        Args:
            eml_path: Path to the EML file.
            parsed_email: Parsed email content.
            archive_result: Optional archive generation result with paths.

        Returns:
            Created Document representing the email.
        """
        file_hash = self.compute_file_hash(eml_path)

        # Generate stable IDs for citations
        source_id = normalize_source_id(str(eml_path), self.ingest_root)
        doc_id = compute_doc_id(source_id, file_hash)

        # Build archive URIs
        archive_path = None
        archive_browse_uri = None
        archive_download_uri = None

        if archive_result:
            archive_path = archive_result.archive_path
            if self.archive_base_url:
                archive_browse_uri = f"{self.archive_base_url}/{archive_result.markdown_path}"
                archive_download_uri = f"{self.archive_base_url}/{archive_result.original_path}"
            else:
                archive_browse_uri = f"/archive/{archive_result.markdown_path}"
                archive_download_uri = f"/archive/{archive_result.original_path}"

        # Determine source title (email subject)
        source_title = parsed_email.metadata.subject or eml_path.stem

        doc = Document(
            parent_id=None,
            root_id=None,  # Will be set to self after creation
            depth=0,
            path=[],
            document_type=DocumentType.EMAIL,
            file_path=str(eml_path),
            file_name=eml_path.name,
            file_hash=file_hash,
            source_id=source_id,
            doc_id=doc_id,
            content_version=1,
            ingest_version=self.current_ingest_version,
            source_title=source_title,
            archive_path=archive_path,
            archive_browse_uri=archive_browse_uri,
            archive_download_uri=archive_download_uri,
            email_metadata=parsed_email.metadata,
        )

        # Root document references itself
        doc.root_id = doc.id
        doc.path = [str(doc.id)]

        # Save to database
        await self.db.insert_document(doc)

        return doc

    async def create_attachment_document(
        self,
        parent_doc: Document,
        attachment_path: Path,
        content_type: str,
        size_bytes: int,
        original_filename: str,
        archive_file_result: Optional["ContentFileResult"] = None,
    ) -> Document:
        """Create an attachment document linked to parent email.

        Args:
            parent_doc: Parent document (email or another attachment).
            attachment_path: Path to the saved attachment file.
            content_type: MIME type of the attachment.
            size_bytes: Size of the attachment in bytes.
            original_filename: Original filename from the email.
            archive_file_result: Optional archive file result with paths.

        Returns:
            Created Document representing the attachment.
        """
        # Lazy import to avoid circular dependency
        from ..parsers.attachment_processor import AttachmentProcessor

        processor = AttachmentProcessor()

        file_hash = self.compute_file_hash(attachment_path)
        doc_type = processor.get_document_type(content_type)

        # Generate stable IDs for citations
        # For attachments, derive source_id from parent's source_id + attachment filename
        # This ensures stable IDs regardless of temporary processing paths
        source_id = f"{parent_doc.source_id}/{original_filename}".lower()
        doc_id = compute_doc_id(source_id, file_hash)

        # Build archive URIs
        archive_path = None
        archive_browse_uri = None
        archive_download_uri = None

        # Determine file_path: use archive location if available (processed is cleaned up)
        # Otherwise fall back to the attachment_path (which may not exist after cleanup)
        actual_file_path = str(attachment_path)

        if archive_file_result:
            archive_path = archive_file_result.archive_path
            # Use archive path as the actual file location
            actual_file_path = str(self.archive_dir / archive_file_result.download_uri)
            if archive_file_result.browse_uri:
                if self.archive_base_url:
                    archive_browse_uri = f"{self.archive_base_url}/{archive_file_result.browse_uri}"
                else:
                    archive_browse_uri = f"/archive/{archive_file_result.browse_uri}"
            if archive_file_result.download_uri:
                if self.archive_base_url:
                    archive_download_uri = f"{self.archive_base_url}/{archive_file_result.download_uri}"
                else:
                    archive_download_uri = f"/archive/{archive_file_result.download_uri}"

        doc = Document(
            parent_id=parent_doc.id,
            root_id=parent_doc.root_id or parent_doc.id,
            depth=parent_doc.depth + 1,
            path=parent_doc.path + [str(parent_doc.id)],
            document_type=doc_type,
            file_path=actual_file_path,
            file_name=original_filename,
            file_hash=file_hash,
            source_id=source_id,
            doc_id=doc_id,
            content_version=1,
            ingest_version=self.current_ingest_version,
            source_title=original_filename,
            archive_path=archive_path,
            archive_browse_uri=archive_browse_uri,
            archive_download_uri=archive_download_uri,
            attachment_metadata=AttachmentMetadata(
                content_type=content_type,
                size_bytes=size_bytes,
                original_filename=original_filename,
            ),
        )

        await self.db.insert_document(doc)

        return doc

    async def get_document_ancestry(self, document_id: UUID) -> List[Document]:
        """Get full ancestry path from root to document.

        Args:
            document_id: Document UUID.

        Returns:
            List of documents from root to the specified document.
        """
        return await self.db.get_document_ancestry(document_id)

    async def get_children(self, document_id: UUID) -> List[Document]:
        """Get all direct children of a document.

        Args:
            document_id: Document UUID.

        Returns:
            List of child documents.
        """
        return await self.db.get_document_children(document_id)

    async def build_context_string(self, document_id: UUID) -> str:
        """Build a context string showing document's place in hierarchy.

        Args:
            document_id: Document UUID.

        Returns:
            Human-readable hierarchy path string.
        """
        ancestry = await self.get_document_ancestry(document_id)

        parts = []
        for doc in ancestry:
            if doc.document_type == DocumentType.EMAIL:
                subject = doc.email_metadata.subject if doc.email_metadata else "No Subject"
                parts.append(f"Email: {subject}")
            else:
                parts.append(f"{doc.document_type.value}: {doc.file_name}")

        return " > ".join(parts)

    async def get_root_email(self, document_id: UUID) -> Document | None:
        """Get the root email for any document in the hierarchy.

        Args:
            document_id: Document UUID.

        Returns:
            Root email document or None.
        """
        doc = await self.db.get_document_by_id(document_id)
        if not doc:
            return None

        if doc.document_type == DocumentType.EMAIL:
            return doc

        if doc.root_id:
            return await self.db.get_document_by_id(doc.root_id)

        return None
