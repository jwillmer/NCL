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

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import Command

from ..config import get_settings
from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY
from ..models.chunk import RetrievalResult
from ..models.vessel import Vessel
from ..processing.topics import TopicExtractor, TopicMatcher
from ..rag.citation_processor import CitationProcessor
from ..rag.intent_classifier import (
    IntentClassifier,
    QueryIntent,
    is_fresh_user_turn,
    latest_user_text,
)
from ..rag.query_engine import RAGQueryEngine
from ..rag.topic_filter import TopicFilter, TopicFilterResult
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

def _sanitize_messages_for_llm(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Remove orphaned tool calls that don't have matching tool responses.

    When a conversation checkpoint is restored after a tool call was made but
    before the tool response was added, the message history will have an
    AIMessage with tool_calls but no corresponding ToolMessage. This causes
    OpenAI to reject the request with a 400 error.

    This function detects and removes such orphaned tool call messages.

    Args:
        messages: List of messages from conversation history.

    Returns:
        Sanitized list of messages with orphaned tool calls removed.
    """
    if not messages:
        return messages

    # Build a set of tool_call_ids that have responses
    responded_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            responded_tool_call_ids.add(msg.tool_call_id)

    # Filter out AIMessages with tool_calls that don't have responses
    sanitized: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            # Check if all tool calls have corresponding responses
            orphaned_calls = [
                tc for tc in msg.tool_calls
                if tc.get("id") not in responded_tool_call_ids
            ]
            if orphaned_calls:
                # Log and skip this orphaned tool call message
                logger.warning(
                    "Removing orphaned AIMessage with tool_calls: %s",
                    [tc.get("id") for tc in orphaned_calls],
                )
                continue
        sanitized.append(msg)

    return sanitized


# Vessel lookups use the process-wide VesselCache (in-memory mirror of the
# ``vessels`` table, ~50 rows with a 5-minute TTL). See processing/entity_cache.py.


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
    """Emit state update to frontend via LangGraph custom event.

    Dispatches a custom event named 'manually_emit_state' which streaming.py
    converts to a Vercel AI SDK data annotation for progress display.

    Args:
        config: LangGraph runnable config with callbacks
        state: State dictionary containing search_progress etc.
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
    """Resolve vessel info through the process-wide VesselCache.

    The cache is a full in-memory mirror of the vessels table (tiny — ~50 rows),
    so we lazy-load it once per process (with a 5-minute TTL) instead of doing a
    DB round-trip per request.
    """
    if not vessel_id:
        return None

    from uuid import UUID
    from ..processing.entity_cache import get_vessel_cache

    try:
        vid = UUID(vessel_id)
    except (ValueError, TypeError):
        return None

    vessel_cache = get_vessel_cache()
    try:
        await vessel_cache.ensure_loaded(SupabaseClient())
    except Exception as e:
        logger.warning("Failed to load vessel cache: %s", e)
        return None

    return vessel_cache.get_by_id(vid)


# Prompt addenda appended when the intent classifier routes a turn away
# from RAG. Kept short — they modulate tone without rewriting the whole
# system prompt, so vessel context and citation rules still apply if a
# follow-up turn ends up calling search after all.
_EXPLORATORY_ADDENDUM = """

## Current turn routing: exploratory

The user is scoping a problem rather than asking a specific lookup question.
Do NOT call search_documents on this turn. Instead:
- Ask 1-2 concrete clarifying questions (component, symptom, vessel, time frame).
- Once the user narrows it down, offer: "Shall I search the knowledge base for similar cases?"
- Keep the reply short (under 120 words). No headers, no citations.
"""

_OFF_TOPIC_ADDENDUM = """

## Current turn routing: off-topic

The user's message is outside the maritime technical-support scope of MTSS.
Do NOT call search_documents. Reply in 1-2 sentences:
- Politely note you're focused on vessel operations, maintenance, and incident history.
- Invite them to ask a maritime-related question.
No headers, no citations.
"""

_GREETING_ADDENDUM = """

## Current turn routing: greeting

The user sent a greeting or small-talk message.
Do NOT call search_documents. Reply in 1-2 sentences:
- Greet them back briefly.
- Offer an example of what they can ask (e.g., "You can ask me about past incidents, maintenance history, or specific vessels.").
No headers, no citations.
"""


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
    return result.to_dict()


def _deserialize_retrieval_result(data: Dict[str, Any]) -> RetrievalResult:
    """Deserialize RetrievalResult from state storage."""
    return RetrievalResult.from_dict(data)


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
                },
                "skip_topic_filter": {
                    "type": "boolean",
                    "description": "Set to true to search across all categories, ignoring topic filtering. Use when user confirms they want a broader search.",
                },
            },
            "required": ["question"],
        },
    },
}


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["search_node", "__end__"]]:
    """Main chat node - routes to search tool or returns final response.

    On fresh user turns we first classify intent (factual / exploratory /
    off_topic / greeting). Factual turns force the search tool call
    (``tool_choice="required"``) — we no longer trust the chat model's
    discretion, baseline-01 showed gpt-5-mini skipped search on 31/37
    factual questions. Non-factual turns skip tool binding entirely so we
    don't waste a round trip on chit-chat or off-topic requests.
    """
    settings = get_settings()
    model_name = settings.llm_model.removeprefix("openrouter/")
    model = ChatOpenAI(
        model=model_name,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        extra_body=OPENROUTER_PRIVACY_EXTRA_BODY,
    )

    # Build system prompt with vessel context if a vessel filter is active
    vessel_id = state.get("selected_vessel_id")
    vessel = await _get_vessel_info(vessel_id)
    base_system = _build_system_prompt(vessel)

    # Check if we have search context (coming back from search_node)
    citation_map_data = state.get("citation_map")
    logger.debug("chat_node called, citation_map_data present: %s", citation_map_data is not None)

    if citation_map_data:
        # Update progress - generating answer
        logger.debug("Emitting 'Generating answer...' progress")
        state["search_progress"] = "Generating answer"
        await emit_state(config, state)

    # Sanitize messages to remove orphaned tool calls (can happen if conversation
    # was restored from checkpoint after a tool call but before the response)
    sanitized_messages = _sanitize_messages_for_llm(state["messages"])

    # Classify intent on fresh user turns only. Returning from search_node
    # (citation_map_data present) or mid-tool-call states skip this.
    intent: Optional[QueryIntent] = None
    system_prompt = base_system
    force_search = False
    bind_search_tool = True

    if (
        settings.intent_classifier_enabled
        and not citation_map_data
        and is_fresh_user_turn(sanitized_messages)
    ):
        user_text = latest_user_text(sanitized_messages)
        if user_text:
            state["search_progress"] = "Understanding your question"
            await emit_state(config, state)
            classifier = IntentClassifier()
            result = await classifier.classify(user_text)
            intent = result.intent
            logger.info(
                "Intent classified: %s (confidence=%.2f, reasoning=%s)",
                intent.value,
                result.confidence,
                result.reasoning,
            )
            if intent is QueryIntent.FACTUAL_QUERY:
                force_search = True
            elif intent is QueryIntent.EXPLORATORY:
                bind_search_tool = False
                system_prompt = base_system + _EXPLORATORY_ADDENDUM
            elif intent is QueryIntent.OFF_TOPIC:
                bind_search_tool = False
                system_prompt = base_system + _OFF_TOPIC_ADDENDUM
            elif intent is QueryIntent.GREETING:
                bind_search_tool = False
                system_prompt = base_system + _GREETING_ADDENDUM
            state["search_progress"] = ""
            await emit_state(config, state)

    system_message = SystemMessage(content=system_prompt)

    if bind_search_tool:
        invoker = model.bind_tools(
            [SEARCH_TOOL],
            parallel_tool_calls=False,
            tool_choice="search_documents" if force_search else "auto",
        )
    else:
        invoker = model

    response = await invoker.ainvoke(
        [system_message, *sanitized_messages],
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

        # Build vessel filter dict for topic pre-filtering
        vessel_filter = None
        if vessel_id:
            vessel_filter = {"vessel_ids": [vessel_id]}
        elif vessel_type:
            vessel_filter = {"vessel_types": [vessel_type]}
        elif vessel_class:
            vessel_filter = {"vessel_classes": [vessel_class]}

        # Progress callback to emit state updates
        async def on_progress(message: str) -> None:
            state["search_progress"] = message
            await emit_state(config, state)

        # Check if user requested broad search (skip topic filtering)
        skip_filter = tool_call["args"].get("skip_topic_filter", False)

        if skip_filter:
            # User requested broad search - skip topic filtering entirely
            filter_result = TopicFilterResult()
            query_embedding = None
        else:
            # Run topic analysis and query embedding concurrently
            await on_progress("Analyzing query")
            topic_filter = TopicFilter(
                topic_extractor=TopicExtractor(),
                topic_matcher=TopicMatcher(engine.retriever.db, engine.retriever.embeddings),
                db=engine.retriever.db,
            )
            filter_task = topic_filter.analyze_query(
                question, vessel_filter, on_progress=on_progress
            )
            embed_task = engine.retriever.embed_query(question)
            filter_result, query_embedding = await asyncio.gather(
                filter_task, embed_task
            )

        # EARLY RETURN: Skip RAG if no results possible
        if filter_result.should_skip_rag:
            state["search_progress"] = ""
            await emit_state(config, state)

            tool_response = ToolMessage(
                content=json.dumps({
                    "context": filter_result.message,
                    "available_chunk_ids": [],
                    "topic_info": {
                        "detected": filter_result.detected_topics,
                        "matched": filter_result.matched_topic_names,
                        "unmatched": filter_result.unmatched_topics,
                        "chunk_count": filter_result.total_chunk_count,
                        "should_skip": True,
                    },
                }),
                tool_call_id=tool_call["id"],
            )
            return Command(
                goto="chat_node",
                update={"messages": [tool_response], "citation_map": None},
            )

        # Build combined metadata filter for RAG
        # Uses OR logic across topic_ids (match any topic)
        metadata_filter = None
        if filter_result.matched_topic_ids or vessel_filter:
            metadata_filter = {}
            if filter_result.matched_topic_ids:
                metadata_filter["topic_ids"] = [
                    str(tid) for tid in filter_result.matched_topic_ids
                ]
            if vessel_filter:
                metadata_filter.update(vessel_filter)

        # Get raw search results (pass pre-computed embedding when available)
        retrieval_results = await engine.search_only(
            question=question,
            top_k=settings.retrieval_top_k,
            use_rerank=settings.rerank_enabled,
            metadata_filter=metadata_filter,
            on_progress=on_progress,
            query_embedding=query_embedding,
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

        # Surface total candidate count for completeness transparency
        candidate_count = filter_result.total_chunk_count or len(retrieval_results)

        # Return context to agent (agent will generate answer with citations)
        tool_response = ToolMessage(
            content=json.dumps({
                "context": enhanced_context,
                "available_chunk_ids": list(citation_map.keys()),
                "incident_count": len(retrieval_results),
                "unique_incidents": len(incident_groups),
                "total_candidate_count": candidate_count,
                "note": (
                    f"Showing top {len(retrieval_results)} results out of "
                    f"{candidate_count} candidates. If the answer seems incomplete, "
                    f"the user can ask for a broader search."
                ) if candidate_count > len(retrieval_results) else None,
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
