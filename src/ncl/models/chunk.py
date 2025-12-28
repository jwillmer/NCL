"""Chunk and RAG response data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .document import Document


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
class SourceReference:
    """Reference to source document for a RAG response."""

    file_path: str
    document_type: str
    email_subject: Optional[str]
    email_initiator: Optional[str]  # Conversation starter
    email_participants: Optional[List[str]]  # All participants
    email_date: Optional[str]
    chunk_content: str
    similarity_score: float
    heading_path: List[str]
    root_file_path: Optional[str] = None
    rerank_score: Optional[float] = None  # Cross-encoder relevance score


@dataclass
class RAGResponse:
    """Response from RAG query with sources."""

    answer: str
    sources: List[SourceReference]
    query: str


# ============================================
# Citation System Models
# ============================================


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

    # Additional context
    document_type: Optional[str] = None
    email_subject: Optional[str] = None
    email_initiator: Optional[str] = None
    email_participants: Optional[List[str]] = None
    email_date: Optional[str] = None
    root_file_path: Optional[str] = None
    file_path: Optional[str] = None
    rerank_score: Optional[float] = None


@dataclass
class ValidatedCitation:
    """Citation after validation against retrieved sources."""

    index: int  # Citation number in response (1-indexed)
    chunk_id: str
    source_title: Optional[str]
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
