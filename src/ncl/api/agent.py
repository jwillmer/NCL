"""Pydantic AI Agent for NCL Email RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ag_ui import StateDeps

from ..config import get_settings
from ..rag.query_engine import RAGQueryEngine

logger = logging.getLogger(__name__)


class RAGState(BaseModel):
    """Shared state for progress tracking between agent and frontend."""

    is_searching: bool = False
    current_query: Optional[str] = None
    error_message: Optional[str] = None


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


# Create the Pydantic AI agent with state sharing support
agent = Agent(
    "openai:gpt-4o",
    instructions=_load_system_prompt(),
    deps_type=StateDeps[RAGState],
)


@agent.tool
async def query_email_documents(
    ctx: RunContext[StateDeps[RAGState]],
    question: str,
) -> ToolReturn:
    """Search and answer questions about email documents and attachments.

    Use this tool to find information from emails, PDFs, images, and other
    attachments in the NCL archive. Always use this tool when the user asks
    about their emails or documents.

    Args:
        ctx: Run context with shared state
        question: The question to ask about the email documents

    Returns:
        ToolReturn with answer, sources, and state snapshot
    """
    state = ctx.deps.state

    # Sanitize input
    question = question.strip()[:2000] if question else ""
    if not question:
        return ToolReturn(
            return_value={"answer": "Please provide a valid question.", "sources": []},
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=state.model_dump(),
                )
            ],
        )

    # Update state to show searching
    state.is_searching = True
    state.current_query = question
    state.error_message = None

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

        # Update state - done searching
        state.is_searching = False
        state.current_query = None

        return ToolReturn(
            return_value={"answer": response.answer, "sources": sources},
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=state.model_dump(),
                )
            ],
        )

    except Exception as e:
        logger.error("RAG query failed: %s", str(e), exc_info=True)

        # Update state with error
        state.is_searching = False
        state.error_message = str(e)

        return ToolReturn(
            return_value={
                "answer": "I encountered an error while searching the email archive. Please try again.",
                "sources": [],
            },
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=state.model_dump(),
                )
            ],
        )
    finally:
        if engine:
            await engine.close()


# Expose as AG-UI ASGI app with initial state
agent_app = agent.to_ag_ui(deps=StateDeps(RAGState()))
