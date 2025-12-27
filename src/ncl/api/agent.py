"""Pydantic AI Agent for NCL Email RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent

from ..config import get_settings
from ..rag.query_engine import RAGQueryEngine

logger = logging.getLogger(__name__)


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


# Create the Pydantic AI agent
agent = Agent(
    "openai:gpt-4o",
    instructions=_load_system_prompt(),
)


@agent.tool_plain
async def query_email_documents(question: str) -> dict:
    """Search and answer questions about email documents and attachments.

    Use this tool to find information from emails, PDFs, images, and other
    attachments in the NCL archive. Always use this tool when the user asks
    about their emails or documents.

    Args:
        question: The question to ask about the email documents

    Returns:
        Dict with answer and sources
    """
    # Sanitize input
    question = question.strip()[:2000] if question else ""
    if not question:
        return {"answer": "Please provide a valid question.", "sources": []}

    engine: Optional[RAGQueryEngine] = None
    try:
        engine = RAGQueryEngine()
        settings = get_settings()

        response = await engine.query(
            question=question,
            top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
            use_rerank=settings.rerank_enabled,
        )

        # Convert sources to serializable format
        sources = []
        for idx, s in enumerate(response.sources):
            sources.append({
                "file_path": s.file_path,
                "document_type": s.document_type,
                "email_subject": s.email_subject,
                "email_initiator": s.email_initiator,
                "email_participants": s.email_participants,
                "email_date": s.email_date,
                "chunk_content": s.chunk_content,
                "similarity_score": s.similarity_score,
                "rerank_score": s.rerank_score,
                "heading_path": " > ".join(s.heading_path) if s.heading_path else None,
                "root_file_path": s.root_file_path,
            })

        return {"answer": response.answer, "sources": sources}

    except Exception as e:
        logger.error("RAG query failed: %s", str(e), exc_info=True)
        return {
            "answer": "I encountered an error while searching the email archive. Please try again.",
            "sources": [],
        }
    finally:
        if engine:
            await engine.close()


# Expose as AG-UI ASGI app
agent_app = agent.to_ag_ui()
