"""LangGraph Agent for MTSS Email RAG with granular UI updates.

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
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import Command

from ..config import get_settings
from ..models.chunk import RetrievalResult
from ..models.vessel import Vessel
from ..rag.citation_processor import CitationProcessor
from ..rag.query_engine import RAGQueryEngine
from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


def _group_by_incident(
    results: List[RetrievalResult],
) -> Dict[str, List[RetrievalResult]]:
    """Group retrieval results by root email thread (incident).

    Each unique root_file_path represents a distinct incident/email thread.
    Results are grouped together so the LLM can identify related chunks
    from the same incident and present them coherently.

    Args:
        results: List of retrieval results from RAG search.

    Returns:
        Dict mapping root_file_path (or fallback key) to list of results.
    """
    groups: Dict[str, List[RetrievalResult]] = {}

    for result in results:
        # Use root_file_path to identify the incident thread
        # Fall back to file_path or source_id if root is not available
        key = result.root_file_path or result.file_path or result.source_id or "unknown"
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    return groups


def _build_incident_summary(
    incident_groups: Dict[str, List[RetrievalResult]],
) -> str:
    """Build a summary of incidents found for the LLM context.

    Args:
        incident_groups: Dict of incident groups from _group_by_incident.

    Returns:
        Formatted summary string to append to context.
    """
    total_chunks = sum(len(chunks) for chunks in incident_groups.values())

    lines = [
        "",
        "---",
        "",
        "## Incident Summary",
        f"Found {total_chunks} relevant chunks from {len(incident_groups)} unique incident(s).",
        "",
    ]

    for i, (root_path, chunks) in enumerate(incident_groups.items(), 1):
        first_chunk = chunks[0]
        subject = first_chunk.email_subject or "Unknown Subject"
        date = first_chunk.email_date or "Unknown Date"

        lines.append(f"**Incident {i}:** {subject}")
        lines.append(f"  - Date: {date}")
        lines.append(f"  - Relevant sections: {len(chunks)}")
        lines.append("")

    return "\n".join(lines)

# Cache for vessel lookups to avoid repeated DB queries
_vessel_cache: Dict[str, Optional[Vessel]] = {}


class AgentState(MessagesState):
    """Shared state for progress tracking between agent and frontend.

    Extends MessagesState for LangGraph message handling.
    State is synced to frontend via emit_state() which triggers STATE_SNAPSHOT events.
    """

    search_progress: str = ""
    error_message: Optional[str] = None
    # Filter fields (mutually exclusive - only one can be active)
    selected_vessel_id: Optional[str] = None
    selected_vessel_type: Optional[str] = None
    selected_vessel_class: Optional[str] = None
    # Citation map for response validation (internal use)
    citation_map: Optional[Dict[str, Any]] = None


async def emit_state(config: RunnableConfig, state: Dict[str, Any]) -> None:
    """Emit state to frontend via AG-UI protocol.

    Dispatches a custom event named 'manually_emit_state' which ag-ui-langgraph
    converts to a STATE_SNAPSHOT event for the frontend.

    Args:
        config: LangGraph runnable config with callbacks
        state: State dictionary to emit to frontend
    """
    await adispatch_custom_event(
        name="manually_emit_state",
        data=state,
        config=config,
    )


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


async def _get_vessel_info(vessel_id: Optional[str]) -> Optional[Vessel]:
    """Get vessel info from cache or database."""
    if not vessel_id:
        return None

    if vessel_id in _vessel_cache:
        return _vessel_cache[vessel_id]

    try:
        from uuid import UUID
        client = SupabaseClient()
        vessel = await client.get_vessel_by_id(UUID(vessel_id))
        _vessel_cache[vessel_id] = vessel
        return vessel
    except Exception as e:
        logger.warning("Failed to fetch vessel info for %s: %s", vessel_id, e)
        return None


def _build_system_prompt(vessel: Optional[Vessel] = None) -> str:
    """Build system prompt with optional vessel context."""
    base_prompt = _load_system_prompt()

    if vessel:
        vessel_context = f"""
## Current Vessel Filter

You are currently searching within documents related to a specific vessel:
- **Vessel Name:** {vessel.name}
- **IMO Number:** {vessel.imo}

All search results are filtered to this vessel. When responding:
- Acknowledge that information is specific to {vessel.name}
- If the user asks about other vessels, remind them that results are filtered to this vessel
- Use the vessel name when referencing findings (e.g., "On {vessel.name}, the issue was...")
"""
        return base_prompt + vessel_context

    return base_prompt


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
        "image_uri": result.image_uri,
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

    # Build system prompt with vessel context if a vessel filter is active
    vessel_id = state.get("selected_vessel_id")
    vessel = await _get_vessel_info(vessel_id)
    system_message = SystemMessage(content=_build_system_prompt(vessel))

    # Check if we have search context (coming back from search_node)
    citation_map_data = state.get("citation_map")
    logger.debug("chat_node called, citation_map_data present: %s", citation_map_data is not None)

    if citation_map_data:
        # Update progress - generating answer
        logger.debug("Emitting 'Generating answer...' progress")
        state["search_progress"] = "Generating answer"
        await emit_state(config, state)

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
            state["search_progress"] = "Validating citations"
            await emit_state(config, state)

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
            await emit_state(config, state)

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
    state["search_progress"] = "Initializing search"
    state["error_message"] = None
    await emit_state(config, state)

    engine: Optional[RAGQueryEngine] = None
    try:
        engine = RAGQueryEngine()
        settings = get_settings()
        citation_processor = CitationProcessor()

        # Read vessel filters from shared state (synced from frontend)
        # Only one filter can be active at a time (mutually exclusive)
        vessel_id = state.get("selected_vessel_id")
        vessel_type = state.get("selected_vessel_type")
        vessel_class = state.get("selected_vessel_class")

        # Progress callback to emit state updates
        async def on_progress(message: str) -> None:
            state["search_progress"] = message
            await emit_state(config, state)

        # Get raw search results (no LLM generation)
        retrieval_results = await engine.search_only(
            question=question,
            top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
            use_rerank=settings.rerank_enabled,
            vessel_id=vessel_id,
            vessel_type=vessel_type,
            vessel_class=vessel_class,
            on_progress=on_progress,
        )

        if not retrieval_results:
            # No results found
            state["search_progress"] = ""
            await emit_state(config, state)

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

        # Group results by incident (root email thread) and build summary
        incident_groups = _group_by_incident(retrieval_results)
        incident_summary = _build_incident_summary(incident_groups)

        # Append incident summary to context for LLM awareness
        enhanced_context = context + incident_summary

        # Serialize citation_map for state storage
        serialized_citation_map = {
            k: _serialize_retrieval_result(v) for k, v in citation_map.items()
        }

        # Return context to agent (agent will generate answer with citations)
        tool_response = ToolMessage(
            content=json.dumps({
                "context": enhanced_context,
                "available_chunk_ids": list(citation_map.keys()),
                "incident_count": len(retrieval_results),
                "unique_incidents": len(incident_groups),
            }),
            tool_call_id=tool_call["id"],
        )

        # Update state - search complete, moving to answer generation
        state["search_progress"] = "Processing results"
        await emit_state(config, state)

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
        state["search_progress"] = ""
        state["error_message"] = str(e)
        await emit_state(config, state)

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


def create_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Create the LangGraph agent graph with persistence.

    Args:
        checkpointer: LangGraph checkpointer for conversation persistence.
                      Use AsyncPostgresSaver for production, MemorySaver for testing.

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

    # Compile with provided checkpointer for conversation persistence
    # Note: Langfuse callback handler is added per-request in patches/__init__.py
    # so that session_id metadata can be set correctly for each conversation
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
