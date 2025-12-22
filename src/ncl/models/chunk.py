"""Chunk and RAG response data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .document import Document


class Chunk(BaseModel):
    """Represents a text chunk with embedding."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID

    content: str
    chunk_index: int

    # Hierarchy context from document structure
    heading_path: List[str] = Field(default_factory=list)
    section_title: Optional[str] = None

    # Source location in original document
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Embedding vector (populated after generation)
    embedding: Optional[List[float]] = None

    # Additional metadata for filtering/context
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
