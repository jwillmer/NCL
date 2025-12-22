"""NCL Email Agent - Centralized AI logic for CopilotKit integration.

This agent provides RAG-based email search with source display.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from copilotkit import Action

from ..config import get_settings
from ..rag.query_engine import RAGQueryEngine
from .state import RAGState, SourceReference, create_initial_state

logger = logging.getLogger(__name__)

# Load system prompt from markdown file
_PROMPT_FILE = Path(__file__).parent / "prompts" / "system.md"


def _load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("System prompt file not found: %s", _PROMPT_FILE)
        return "You are a helpful AI assistant for searching email archives."


SYSTEM_PROMPT = _load_system_prompt()


class NCLEmailAgent:
    """Encapsulates all AI agent logic for the NCL email RAG system.

    This class centralizes:
    - System prompts and instructions (loaded from markdown)
    - CopilotKit action definitions
    - RAG state management for frontend synchronization
    - Query handling and sanitization

    Note: This agent uses a singleton pattern with shared state. In a multi-user
    deployment, each user would see the same state. For production multi-user
    support, state should be moved to session scope or a per-user store.
    """

    def __init__(self):
        """Initialize the agent."""
        self._engine: RAGQueryEngine | None = None
        self._state: RAGState = create_initial_state()

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the AI."""
        return SYSTEM_PROMPT

    @property
    def state(self) -> RAGState:
        """Get the current RAG state."""
        return self._state

    def update_state(self, **kwargs) -> RAGState:
        """Update state fields and return the new state."""
        self._state = self._state.model_copy(update=kwargs)
        return self._state

    def reset_state(self) -> RAGState:
        """Reset state to initial values."""
        self._state = create_initial_state()
        return self._state

    async def _get_engine(self) -> RAGQueryEngine:
        """Get or create RAG engine instance."""
        if self._engine is None:
            self._engine = RAGQueryEngine()
        return self._engine

    async def close(self):
        """Close the RAG engine connection."""
        if self._engine is not None:
            await self._engine.close()
            self._engine = None

    def get_actions(self) -> List[Action]:
        """Get the list of CopilotKit actions for this agent.

        Returns:
            List of Action objects that CopilotKit can use.
        """
        return [
            Action(
                name="queryEmailDocuments",
                description=(
                    "Search and answer questions about email documents and attachments "
                    "in the NCL archive. Use this to find information from emails, PDFs, "
                    "images, and other attachments. Always use this action when the user "
                    "asks about their emails or documents."
                ),
                parameters=[
                    {
                        "name": "question",
                        "type": "string",
                        "description": "The question to ask about the email documents",
                        "required": True,
                    },
                ],
                handler=self._query_emails,
            ),
            Action(
                name="getRAGState",
                description=(
                    "Get the current RAG state including retrieved sources. "
                    "Use this to check what sources were found in the last search."
                ),
                parameters=[],
                handler=self._get_rag_state,
            ),
        ]

    async def _query_emails(self, question: str) -> Dict[str, Any]:
        """Query email documents using RAG.

        Args:
            question: The user's question.

        Returns:
            Dict with answer and sources, plus state update.
        """
        question = self._sanitize_query(question)
        if not question:
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
            }

        # Update state: searching
        self.update_state(
            is_searching=True,
            current_query=question,
            error_message=None,
        )

        try:
            engine = await self._get_engine()
            settings = get_settings()

            response = await engine.query(
                question=question,
                top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
                use_rerank=settings.rerank_enabled,
            )

            # Convert sources to SourceReference models with unique IDs
            sources = []
            for idx, s in enumerate(response.sources):
                chunk_id = self._generate_chunk_id(s.file_path, s.chunk_content, idx)
                sources.append(SourceReference(
                    chunk_id=chunk_id,
                    file_path=s.file_path,
                    document_type=s.document_type,
                    email_subject=s.email_subject,
                    email_initiator=s.email_initiator,
                    email_participants=s.email_participants,
                    email_date=s.email_date,
                    chunk_content=s.chunk_content,
                    similarity_score=s.similarity_score,
                    rerank_score=s.rerank_score,
                    heading_path=s.heading_path,
                    root_file_path=s.root_file_path,
                ))

            # Update state with results
            self.update_state(
                is_searching=False,
                sources=sources,
                answer=response.answer,
            )

            return {
                "answer": response.answer,
                "sources": [s.model_dump() for s in sources],
                "state": self._state.model_dump(),
            }

        except Exception as e:
            logger.error("RAG query failed: %s", str(e), exc_info=True)
            self.update_state(
                is_searching=False,
                error_message=str(e),
            )
            return {
                "answer": "I encountered an error while searching the email archive. Please try again.",
                "sources": [],
                "error": str(e),
            }

    async def _get_rag_state(self) -> Dict[str, Any]:
        """Return the current RAG state.

        Returns:
            Dict containing the full RAG state.
        """
        return {
            "state": self._state.model_dump(),
            "source_count": len(self._state.sources),
        }

    @staticmethod
    def _generate_chunk_id(file_path: str, content: str, index: int) -> str:
        """Generate a unique ID for a chunk."""
        data = f"{file_path}:{index}:{content[:100]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize user query input.

        Args:
            query: Raw user input.

        Returns:
            Sanitized query string.
        """
        if not query or not isinstance(query, str):
            return ""

        # Strip whitespace and limit length
        query = query.strip()[:2000]

        # Remove potentially dangerous characters but keep normal punctuation
        query = re.sub(r"[<>{}\\]", "", query)

        return query
