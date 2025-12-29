"""LangGraph Agent for NCL Email RAG with granular UI updates.

Architecture:
    User Question
         |
         v
    [chat_node] -> Decides to search
         |
         v
    [search_node] -> Returns RAW search results (no LLM generation)
         |           - chunk_id, content, metadata, archive URIs
         v
    [chat_node] -> Agent generates answer with [C:chunk_id] citations
         |
         v
    [validate_response_node] -> Validates citations, replaces with links
         |
         v
    Response to client (with validated markdown links)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..config import get_settings
from ..models.chunk import RetrievalResult
from ..rag.citation_processor import CitationProcessor
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
    # Citation map for validation (chunk_id -> serialized RetrievalResult)
    citation_map: Optional[Dict[str, Any]] = None


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


def _serialize_retrieval_result(result: RetrievalResult) -> Dict[str, Any]:
    """Serialize RetrievalResult for state storage."""
    return {
        "text": result.text,
        "score": result.score,
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "source_id": result.source_id,
        "source_title": result.source_title,
        "section_path": result.section_path,
        "page_number": result.page_number,
        "line_from": result.line_from,
        "line_to": result.line_to,
        "archive_browse_uri": result.archive_browse_uri,
        "archive_download_uri": result.archive_download_uri,
    }


def _deserialize_retrieval_result(data: Dict[str, Any]) -> RetrievalResult:
    """Deserialize RetrievalResult from state storage."""
    return RetrievalResult(**data)


# Tool definition for the LLM - now returns raw context, not answers
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search for relevant email documents and attachments. "
            "Returns document context with chunk IDs for citations. "
            "Use this tool when the user asks about emails, documents, or technical issues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                }
            },
            "required": ["question"],
        },
    },
}


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["search_node", "validate_response_node"]]:
    """Main chat node - routes to search tool or validates response."""
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

    # No tool call - go to validation node to process any citations
    return Command(goto="validate_response_node", update={"messages": [response]})


async def search_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["chat_node"]]:
    """Execute RAG search and return raw results (no LLM generation).

    The search tool returns document context with citation metadata.
    The agent (chat_node) then generates answers with [C:chunk_id] citations.
    """
    ai_message = cast(AIMessage, state["messages"][-1])

    if not ai_message.tool_calls:
        return Command(goto="chat_node", update={})

    tool_call = ai_message.tool_calls[0]
    question = tool_call["args"].get("question", "").strip()[:2000]

    if not question:
        tool_response = ToolMessage(
            content=json.dumps({
                "context": "Please provide a valid search question.",
                "available_chunk_ids": [],
            }),
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
        citation_processor = CitationProcessor()

        # Progress callback to emit state updates
        async def on_progress(message: str) -> None:
            state["search_progress"] = message
            await copilotkit_emit_state(config, state)

        # Get raw search results (no LLM generation)
        retrieval_results = await engine.search_only(
            question=question,
            top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
            use_rerank=settings.rerank_enabled,
            on_progress=on_progress,
        )

        if not retrieval_results:
            # No results found
            state["is_searching"] = False
            state["search_progress"] = ""
            state["current_query"] = None
            await copilotkit_emit_state(config, state)

            tool_response = ToolMessage(
                content=json.dumps({
                    "context": "No relevant documents found for this query.",
                    "available_chunk_ids": [],
                }),
                tool_call_id=tool_call["id"],
            )
            return Command(
                goto="chat_node",
                update={"messages": [tool_response], "citation_map": None},
            )

        # Build context with citation headers
        context = citation_processor.build_context(retrieval_results)
        citation_map = citation_processor.get_citation_map(retrieval_results)

        # Serialize citation_map for state storage
        serialized_citation_map = {
            k: _serialize_retrieval_result(v) for k, v in citation_map.items()
        }

        # Return context to agent (agent will generate answer with citations)
        tool_response = ToolMessage(
            content=json.dumps({
                "context": context,
                "available_chunk_ids": list(citation_map.keys()),
            }),
            tool_call_id=tool_call["id"],
        )

        # Update state - done searching
        state["is_searching"] = False
        state["search_progress"] = ""
        state["current_query"] = None
        await copilotkit_emit_state(config, state)

        return Command(
            goto="chat_node",
            update={
                "messages": [tool_response],
                "citation_map": serialized_citation_map,
            },
        )

    except Exception as e:
        logger.error("RAG search failed: %s", str(e), exc_info=True)

        # Update state with error
        state["is_searching"] = False
        state["search_progress"] = ""
        state["error_message"] = str(e)
        await copilotkit_emit_state(config, state)

        tool_response = ToolMessage(
            content=json.dumps({
                "context": "An error occurred while searching. Please try again.",
                "available_chunk_ids": [],
            }),
            tool_call_id=tool_call["id"],
        )

        return Command(
            goto="chat_node",
            update={"messages": [tool_response], "citation_map": None},
        )

    finally:
        if engine:
            await engine.close()


async def validate_response_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Validate citations in the agent's response before sending to client.

    Replaces [C:chunk_id] markers with numbered references [1], [2], etc.
    and appends a sources section with archive links.
    """
    last_message = state["messages"][-1]

    # Skip validation if no AI message or no citation_map
    if not isinstance(last_message, AIMessage):
        return Command(goto=END, update={})

    citation_map_data = state.get("citation_map")
    if not citation_map_data:
        # No citations to validate, pass through unchanged
        return Command(goto=END, update={})

    try:
        # Deserialize citation_map
        citation_map = {
            k: _deserialize_retrieval_result(v) for k, v in citation_map_data.items()
        }

        citation_processor = CitationProcessor()

        # Validate citations in the response
        validation = citation_processor.process_response(
            last_message.content, citation_map
        )

        # Replace [C:chunk_id] with [1], [2] and add sources section
        formatted = citation_processor.replace_citation_markers(
            validation.response, validation.citations
        )
        sources_section = citation_processor.format_sources_section(validation.citations)

        if sources_section:
            formatted = formatted + sources_section

        # Create updated message with validated citations
        updated_message = AIMessage(content=formatted)

        # Clear citation_map after validation
        return Command(
            goto=END,
            update={"messages": [updated_message], "citation_map": None},
        )

    except Exception as e:
        logger.error("Citation validation failed: %s", str(e), exc_info=True)
        # On error, pass through original message unchanged
        return Command(goto=END, update={"citation_map": None})


def create_graph() -> StateGraph:
    """Create the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("search_node", search_node)
    graph.add_node("validate_response_node", validate_response_node)

    # Set entry point
    graph.set_entry_point("chat_node")

    # Compile with in-memory checkpointer (required for AG-UI protocol)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Create compiled graph instance
agent_graph = create_graph()
