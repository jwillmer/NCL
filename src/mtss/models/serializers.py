"""JSONL serialization and deserialization for Document, Chunk, and Topic models.

Single source of truth for the flattened JSONL format used by:
- LocalStorageClient (ingest output)
- import_cmd (Supabase import)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID


def doc_to_dict(doc: Any) -> Dict[str, Any]:
    """Convert Document model to dictionary for JSONL serialization."""
    email_meta = getattr(doc, "email_metadata", None)
    att_meta = getattr(doc, "attachment_metadata", None)
    return {
        "id": str(doc.id),
        "source_id": doc.source_id,
        "doc_id": doc.doc_id,
        "content_version": getattr(doc, "content_version", 1),
        "ingest_version": doc.ingest_version,
        "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else doc.document_type,
        "file_path": doc.file_path,
        "file_name": doc.file_name,
        "file_hash": doc.file_hash,
        "parent_id": str(doc.parent_id) if doc.parent_id else None,
        "root_id": str(doc.root_id) if doc.root_id else None,
        "depth": doc.depth,
        "path": getattr(doc, "path", []),
        "source_title": doc.source_title,
        "archive_path": getattr(doc, "archive_path", None),
        "archive_browse_uri": doc.archive_browse_uri,
        "archive_download_uri": doc.archive_download_uri,
        "status": doc.status.value if hasattr(doc.status, "value") else doc.status,
        "error_message": getattr(doc, "error_message", None),
        "processed_at": doc.processed_at.isoformat() if getattr(doc, "processed_at", None) else None,
        "created_at": doc.created_at.isoformat() if getattr(doc, "created_at", None) else None,
        "updated_at": doc.updated_at.isoformat() if getattr(doc, "updated_at", None) else None,
        # Email metadata (flattened)
        "email_subject": email_meta.subject if email_meta else None,
        "email_participants": email_meta.participants if email_meta else None,
        "email_initiator": email_meta.initiator if email_meta else None,
        "email_date_start": email_meta.date_start.isoformat() if email_meta and email_meta.date_start else None,
        "email_date_end": email_meta.date_end.isoformat() if email_meta and email_meta.date_end else None,
        "email_message_count": email_meta.message_count if email_meta else None,
        # Attachment metadata (flattened)
        "attachment_content_type": att_meta.content_type if att_meta else None,
        "attachment_size_bytes": att_meta.size_bytes if att_meta else None,
    }


def chunk_to_dict(chunk: Any) -> Dict[str, Any]:
    """Convert Chunk model to dictionary for JSONL serialization."""
    return {
        "id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "context_summary": chunk.context_summary,
        "embedding_text": chunk.embedding_text,
        "section_path": chunk.section_path,
        "section_title": chunk.section_title,
        "source_title": chunk.source_title,
        "source_id": chunk.source_id,
        "line_from": chunk.line_from,
        "line_to": chunk.line_to,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "archive_browse_uri": chunk.archive_browse_uri,
        "archive_download_uri": chunk.archive_download_uri,
        "metadata": chunk.metadata,
        "embedding": chunk.embedding,
        "page_number": getattr(chunk, "page_number", None),
    }


def topic_to_dict(topic: Any) -> Dict[str, Any]:
    """Convert Topic model to dictionary for JSONL serialization."""
    return {
        "id": str(topic.id),
        "name": topic.name,
        "display_name": getattr(topic, "display_name", topic.name),
        "description": getattr(topic, "description", None),
        "embedding": getattr(topic, "embedding", None),
        "chunk_count": getattr(topic, "chunk_count", 0),
        "document_count": getattr(topic, "document_count", 0),
        "created_at": topic.created_at.isoformat() if getattr(topic, "created_at", None) else None,
        "updated_at": topic.updated_at.isoformat() if getattr(topic, "updated_at", None) else None,
    }


def dict_to_document(d: Dict[str, Any]) -> "Document":
    """Reconstruct a Document model from a JSONL dict."""
    from .document import (
        AttachmentMetadata,
        Document,
        DocumentType,
        EmailMetadata,
        ProcessingStatus,
    )

    email_metadata = None
    if d.get("email_subject") or d.get("email_participants"):
        email_metadata = EmailMetadata(
            subject=d.get("email_subject"),
            participants=d.get("email_participants") or [],
            initiator=d.get("email_initiator"),
            date_start=datetime.fromisoformat(d["email_date_start"]) if d.get("email_date_start") else None,
            date_end=datetime.fromisoformat(d["email_date_end"]) if d.get("email_date_end") else None,
            message_count=d.get("email_message_count") or 1,
        )

    attachment_metadata = None
    if d.get("attachment_content_type"):
        attachment_metadata = AttachmentMetadata(
            content_type=d["attachment_content_type"],
            size_bytes=d.get("attachment_size_bytes") or 0,
            original_filename=d.get("file_name", ""),
        )

    return Document(
        id=UUID(d["id"]),
        source_id=d.get("source_id"),
        doc_id=d.get("doc_id"),
        content_version=d.get("content_version", 1),
        ingest_version=d.get("ingest_version", 1),
        document_type=DocumentType(d["document_type"]),
        file_path=d["file_path"],
        file_name=d["file_name"],
        file_hash=d.get("file_hash"),
        parent_id=UUID(d["parent_id"]) if d.get("parent_id") else None,
        root_id=UUID(d["root_id"]) if d.get("root_id") else None,
        depth=d.get("depth", 0),
        path=d.get("path") or [],
        source_title=d.get("source_title"),
        archive_path=d.get("archive_path"),
        archive_browse_uri=d.get("archive_browse_uri"),
        archive_download_uri=d.get("archive_download_uri"),
        email_metadata=email_metadata,
        attachment_metadata=attachment_metadata,
        status=ProcessingStatus(d.get("status", "completed")),
        error_message=d.get("error_message"),
        processed_at=datetime.fromisoformat(d["processed_at"]) if d.get("processed_at") else None,
        created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else datetime.now(timezone.utc),
    )


def dict_to_chunk(d: Dict[str, Any]) -> "Chunk":
    """Reconstruct a Chunk model from a JSONL dict."""
    from .chunk import Chunk

    return Chunk(
        id=UUID(d["id"]),
        document_id=UUID(d["document_id"]),
        chunk_id=d.get("chunk_id"),
        content=d["content"],
        chunk_index=d.get("chunk_index", 0),
        context_summary=d.get("context_summary"),
        embedding_text=d.get("embedding_text"),
        section_path=d.get("section_path") or [],
        section_title=d.get("section_title"),
        source_title=d.get("source_title"),
        source_id=d.get("source_id"),
        page_number=d.get("page_number"),
        line_from=d.get("line_from"),
        line_to=d.get("line_to"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
        archive_browse_uri=d.get("archive_browse_uri"),
        archive_download_uri=d.get("archive_download_uri"),
        embedding=d.get("embedding"),
        metadata=d.get("metadata") or {},
    )


def dict_to_topic(d: Dict[str, Any]) -> "Topic":
    """Reconstruct a Topic model from a JSONL dict."""
    from .topic import Topic

    return Topic(
        id=UUID(d["id"]),
        name=d["name"],
        display_name=d.get("display_name", d["name"]),
        description=d.get("description"),
        embedding=d.get("embedding"),
        chunk_count=d.get("chunk_count", 0),
        document_count=d.get("document_count", 0),
        created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else datetime.now(timezone.utc),
    )
