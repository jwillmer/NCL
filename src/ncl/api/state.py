"""RAG State model for CopilotKit frontend/backend synchronization.

This module defines the shared state contract between the Python backend
and React frontend using Pydantic models that map to TypeScript types.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A source document chunk returned from RAG search."""

    chunk_id: str = Field(description="Unique identifier for this chunk")
    file_path: str = Field(description="Path to the source file")
    document_type: str = Field(description="Type of document (email, pdf, image, etc.)")
    email_subject: Optional[str] = Field(default=None, description="Email subject if applicable")
    email_initiator: Optional[str] = Field(default=None, description="Original email sender")
    email_participants: Optional[List[str]] = Field(default=None, description="All email participants")
    email_date: Optional[str] = Field(default=None, description="Email date as ISO string")
    chunk_content: str = Field(description="The text content of this chunk")
    similarity_score: float = Field(description="Vector similarity score (0-1)")
    rerank_score: Optional[float] = Field(default=None, description="Reranker score if applied")
    heading_path: Optional[str] = Field(default=None, description="Section path within document")
    root_file_path: Optional[str] = Field(default=None, description="Parent email path for attachments")


class RAGState(BaseModel):
    """Shared state between frontend and backend for RAG operations.

    This state enables the frontend to:
    - Display retrieved sources with metadata
    - Show loading/error states

    State flow (Backend → Frontend only):
    - sources, current_query, answer, is_searching, error_message
    """

    # Search results
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Sources retrieved from the last search"
    )
    current_query: Optional[str] = Field(
        default=None,
        description="The current/last search query"
    )
    answer: Optional[str] = Field(
        default=None,
        description="The generated answer"
    )

    # UI State
    is_searching: bool = Field(default=False, description="Whether a search is in progress")
    error_message: Optional[str] = Field(default=None, description="Error message if any")


def create_initial_state() -> RAGState:
    """Create a fresh initial state."""
    return RAGState()
