"""Document hierarchy manager for maintaining parent-child relationships."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List
from uuid import UUID

from ..models.document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    ParsedEmail,
)
from ..storage.supabase_client import SupabaseClient


class HierarchyManager:
    """Manages document hierarchy relationships.

    Handles the creation and tracking of parent-child relationships
    between emails and their attachments for proper context in RAG.
    """

    def __init__(self, db_client: SupabaseClient):
        """Initialize the hierarchy manager.

        Args:
            db_client: Supabase client for database operations.
        """
        self.db = db_client

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
    ) -> Document:
        """Create a root email document in the hierarchy.

        Args:
            eml_path: Path to the EML file.
            parsed_email: Parsed email content.

        Returns:
            Created Document representing the email.
        """
        file_hash = self.compute_file_hash(eml_path)

        doc = Document(
            parent_id=None,
            root_id=None,  # Will be set to self after creation
            depth=0,
            path=[],
            document_type=DocumentType.EMAIL,
            file_path=str(eml_path),
            file_name=eml_path.name,
            file_hash=file_hash,
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
    ) -> Document:
        """Create an attachment document linked to parent email.

        Args:
            parent_doc: Parent document (email or another attachment).
            attachment_path: Path to the saved attachment file.
            content_type: MIME type of the attachment.
            size_bytes: Size of the attachment in bytes.
            original_filename: Original filename from the email.

        Returns:
            Created Document representing the attachment.
        """
        # Lazy import to avoid circular dependency
        from ..parsers.attachment_processor import AttachmentProcessor

        processor = AttachmentProcessor()

        file_hash = self.compute_file_hash(attachment_path)
        doc_type = processor.get_document_type(content_type)

        doc = Document(
            parent_id=parent_doc.id,
            root_id=parent_doc.root_id or parent_doc.id,
            depth=parent_doc.depth + 1,
            path=parent_doc.path + [str(parent_doc.id)],
            document_type=doc_type,
            file_path=str(attachment_path),
            file_name=original_filename,
            file_hash=file_hash,
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
