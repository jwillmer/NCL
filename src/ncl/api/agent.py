"""LangGraph Agent for NCL Email RAG with granular UI updates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional, cast

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..config import get_settings
from ..rag.query_engine import RAGQueryEngine

logger = logging.getLogger(__name__)


class AgentState(CopilotKitState):
    """Shared state for progress tracking between agent and frontend.

    Extends CopilotKitState to enable bidirectional state sync with frontend.
    """

    is_searching: bool = False
    search_progress: str = ""
    current_query: Optional[str] = None
    error_message: Optional[str] = None


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


# Tool definition for the LLM
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "query_email_documents",
        "description": (
            "Search and answer questions about email documents and attachments. "
            "Use this tool to find information from emails, PDFs, images, and other "
            "attachments in the NCL archive. Always use this tool when the user asks "
            "about their emails or documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask about the email documents",
                }
            },
            "required": ["question"],
        },
    },
}


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["search_node", "__end__"]]:
    """Main chat node - routes to search tool or responds directly."""
    model = ChatOpenAI(model="gpt-4o")
    model_with_tools = model.bind_tools([SEARCH_TOOL], parallel_tool_calls=False)

    system_message = SystemMessage(content=_load_system_prompt())

    response = await model_with_tools.ainvoke(
        [system_message, *state["messages"]],
        config,
    )

    # Check if the model wants to use a tool
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="search_node", update={"messages": [response]})

    return Command(goto=END, update={"messages": [response]})


async def search_node(state: AgentState, config: RunnableConfig) -> Command[Literal["chat_node"]]:
    """Execute RAG search with granular progress updates."""
    ai_message = cast(AIMessage, state["messages"][-1])

    if not ai_message.tool_calls:
        # No tool calls, return to chat
        return Command(goto="chat_node", update={})

    tool_call = ai_message.tool_calls[0]
    question = tool_call["args"].get("question", "").strip()[:2000]

    if not question:
        # Invalid question - create tool response and return to chat
        from langchain_core.messages import ToolMessage
        tool_response = ToolMessage(
            content='{"answer": "Please provide a valid question.", "sources": []}',
            tool_call_id=tool_call["id"],
        )
        return Command(goto="chat_node", update={"messages": [tool_response]})

    # Update state - starting search
    state["is_searching"] = True
    state["current_query"] = question
    state["search_progress"] = "Initializing search..."
    state["error_message"] = None
    await copilotkit_emit_state(config, state)

    engine: Optional[RAGQueryEngine] = None
    try:
        engine = RAGQueryEngine()
        settings = get_settings()

        # Progress callback to emit state updates
        async def on_progress(message: str) -> None:
            state["search_progress"] = message
            await copilotkit_emit_state(config, state)

        response = await engine.query(
            question=question,
            top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
            use_rerank=settings.rerank_enabled,
            on_progress=on_progress,
        )

        # Convert sources to serializable format
        sources = []
        for s in response.sources:
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

        # Create tool response
        import json
        from langchain_core.messages import ToolMessage

        result = {"answer": response.answer, "sources": sources}
        tool_response = ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call["id"],
        )

        # Update state - done searching
        state["is_searching"] = False
        state["search_progress"] = ""
        state["current_query"] = None
        await copilotkit_emit_state(config, state)

        return Command(goto="chat_node", update={"messages": [tool_response]})

    except Exception as e:
        logger.error("RAG query failed: %s", str(e), exc_info=True)

        # Update state with error
        state["is_searching"] = False
        state["search_progress"] = ""
        state["error_message"] = str(e)
        await copilotkit_emit_state(config, state)

        import json
        from langchain_core.messages import ToolMessage

        result = {
            "answer": "I encountered an error while searching the email archive. Please try again.",
            "sources": [],
        }
        tool_response = ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call["id"],
        )

        return Command(goto="chat_node", update={"messages": [tool_response]})

    finally:
        if engine:
            await engine.close()


def create_graph() -> StateGraph:
    """Create the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("search_node", search_node)

    # Set entry point
    graph.set_entry_point("chat_node")

    # Compile with in-memory checkpointer (required for AG-UI protocol)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Create compiled graph instance
agent_graph = create_graph()
