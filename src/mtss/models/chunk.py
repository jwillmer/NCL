"""Chunk and RAG response data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .document import Document, EmbeddingMode


class Chunk(BaseModel):
    """Represents a text chunk with embedding."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID

    # Stable identification for citations
    chunk_id: Optional[str] = None  # Deterministic ID: hash(doc_id + char_start + char_end)

    content: str
    chunk_index: int

    # Contextual chunking for improved retrieval
    context_summary: Optional[str] = None  # Document-level context (LLM-generated)
    embedding_text: Optional[str] = None  # Full text for embedding: context + content

    # Hierarchy context from document structure
    section_path: List[str] = Field(default_factory=list)  # Renamed from heading_path
    section_title: Optional[str] = None

    # Citation metadata (denormalized from document)
    source_title: Optional[str] = None
    source_id: Optional[str] = None

    # Source location in original document
    page_number: Optional[int] = None
    line_from: Optional[int] = None
    line_to: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    # Archive links (denormalized for fast retrieval)
    archive_browse_uri: Optional[str] = None
    archive_download_uri: Optional[str] = None

    # Embedding vector (populated after generation)
    embedding: Optional[List[float]] = None

    # Additional metadata for filtering/context
    metadata: Dict[str, Any] = Field(default_factory=dict)

    embedding_mode: Optional[EmbeddingMode] = None

    # Backwards compatibility
    @property
    def heading_path(self) -> List[str]:
        """Alias for section_path (backwards compatibility)."""
        return self.section_path

    @property
    def start_char(self) -> Optional[int]:
        """Alias for char_start (backwards compatibility)."""
        return self.char_start

    @property
    def end_char(self) -> Optional[int]:
        """Alias for char_end (backwards compatibility)."""
        return self.char_end


class ChunkWithContext(BaseModel):
    """Chunk with full ancestry context for RAG responses."""

    chunk: Chunk
    document_path: List[Document]  # From root email to chunk's document
    similarity_score: float

    # Convenience properties for source linking
    source_file_path: str
    root_email_subject: Optional[str] = None
    root_email_from: Optional[str] = None


@dataclass
class RetrievalResult:
    """Single retrieval result with full citation metadata."""

    text: str
    score: float
    chunk_id: str
    doc_id: str
    source_id: str
    source_title: Optional[str]
    section_path: List[str]
    page_number: Optional[int] = None
    line_from: Optional[int] = None
    line_to: Optional[int] = None
    archive_browse_uri: Optional[str] = None
    archive_download_uri: Optional[str] = None
    image_uri: Optional[str] = None  # For image chunks - path to display inline

    # Context summary (LLM-generated document summary)
    context_summary: Optional[str] = None

    # Embedding mode inherited from parent document (full / summary /
    # metadata_only). Optional so existing callers that don't yet hydrate
    # the field stay valid; consumers that need to branch on it should
    # check for None and treat that as "unknown -> behave like full".
    embedding_mode: Optional[str] = None

    # Additional context
    document_type: Optional[str] = None
    email_subject: Optional[str] = None
    email_initiator: Optional[str] = None
    email_participants: Optional[List[str]] = None
    email_date: Optional[str] = None
    root_file_path: Optional[str] = None
    file_path: Optional[str] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for state storage."""
        return {
            "text": self.text,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "line_from": self.line_from,
            "line_to": self.line_to,
            "archive_browse_uri": self.archive_browse_uri,
            "archive_download_uri": self.archive_download_uri,
            "image_uri": self.image_uri,
            "context_summary": self.context_summary,
            "embedding_mode": self.embedding_mode,
            "document_type": self.document_type,
            "email_subject": self.email_subject,
            "email_initiator": self.email_initiator,
            "email_participants": self.email_participants,
            "email_date": self.email_date,
            "root_file_path": self.root_file_path,
            "file_path": self.file_path,
            "rerank_score": self.rerank_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ValidatedCitation:
    """Citation after validation against retrieved sources."""

    index: int  # Citation number in response (1-indexed)
    chunk_id: str
    source_title: Optional[str]
    source_id: Optional[str] = None  # Stable per-document id; used by the UI to dedupe chunks of the same source
    page: Optional[int] = None
    lines: Optional[Tuple[int, int]] = None
    archive_browse_uri: Optional[str] = None
    archive_download_uri: Optional[str] = None
    archive_verified: bool = False


@dataclass
class CitationValidationResult:
    """Result of validating citations in an LLM response."""

    response: str  # Response text with invalid citations removed
    citations: List[ValidatedCitation] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)  # chunk_ids that didn't exist
    missing_archives: List[str] = field(default_factory=list)  # chunk_ids with missing archive files
    needs_retry: bool = False  # True if too many citations were invalid


@dataclass
class EnhancedRAGResponse:
    """Enhanced RAG response with validated citations and archive links."""

    answer: str
    citations: List[ValidatedCitation]
    query: str
    had_invalid_citations: bool = False
    retry_count: int = 0
