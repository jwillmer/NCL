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
) -> Command[Literal["search_node", "__end__"]]:
    """Main chat node - routes to search tool or returns final response."""
    settings = get_settings()
    model = ChatOpenAI(model=settings.llm_model)
    model_with_tools = model.bind_tools([SEARCH_TOOL], parallel_tool_calls=False)

    system_message = SystemMessage(content=_load_system_prompt())

    # Check if we have search context (coming back from search_node)
    citation_map_data = state.get("citation_map")
    logger.debug("chat_node called, citation_map_data present: %s", citation_map_data is not None)

    if citation_map_data:
        # Update progress - generating answer
        logger.debug("Emitting 'Generating answer...' progress")
        state["search_progress"] = "Generating answer..."
        state["is_searching"] = True  # Keep the searching indicator active
        await copilotkit_emit_state(config, state)

    response = await model_with_tools.ainvoke(
        [system_message, *state["messages"]],
        config,
    )

    # Check if the model wants to use a tool
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="search_node", update={"messages": [response]})

    # No tool call - process citations inline before returning
    if citation_map_data:
        try:
            # Update progress - validating citations
            logger.debug("Emitting 'Validating citations...' progress")
            state["search_progress"] = "Validating citations..."
            await copilotkit_emit_state(config, state)

            # Deserialize and process citations
            citation_map = {
                k: _deserialize_retrieval_result(v) for k, v in citation_map_data.items()
            }
            citation_processor = CitationProcessor()
            validation = citation_processor.process_response(
                response.content, citation_map
            )
            formatted = citation_processor.replace_citation_markers(
                validation.response, validation.citations
            )
            # Create new message with processed citations
            response = AIMessage(content=formatted)
        except Exception as e:
            logger.error("Citation processing failed: %s", str(e), exc_info=True)
            # Continue with original response on error
        finally:
            # Clear progress
            logger.debug("Clearing progress indicators")
            state["search_progress"] = ""
            state["is_searching"] = False
            await copilotkit_emit_state(config, state)

    return Command(goto=END, update={"messages": [response], "citation_map": None})


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

        # Update state - search complete, moving to answer generation
        # Keep is_searching=True so the UI continues showing progress
        # chat_node will clear it after generating the answer
        state["search_progress"] = "Processing results..."
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


def create_graph() -> StateGraph:
    """Create the LangGraph agent graph.

    Flow:
        chat_node -> search_node -> chat_node -> END
                  -> END (if no tool call needed)

    Citation validation and processing happens inline in chat_node before
    returning the final response, ensuring citations are validated against
    the citation_map and formatted with <cite> tags.
    """
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
