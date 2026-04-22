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
from ..observability.step_timing import record_step
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
    # Set by set_filter_node so the next chat_node turn still forces search.
    # Without this, the filter would be applied but no retrieval would run.
    filter_pending_search: bool = False


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


async def emit_filter_update(
    config: RunnableConfig,
    vessel_id: Optional[str],
    vessel_type: Optional[str],
    vessel_class: Optional[str],
) -> None:
    """Emit a filter-change notification to the frontend.

    streaming.py converts this to a `data-filter` part on the UI message stream
    so the React client can repaint its dropdowns + persist to the conversation
    row. Only one field should be non-null — frontend enforces mutual exclusion.
    """
    await adispatch_custom_event(
        name="emit_filter_update",
        data={
            "vessel_id": vessel_id,
            "vessel_type": vessel_type,
            "vessel_class": vessel_class,
        },
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
- **Vessel Type:** {vessel.vessel_type}
- **Vessel Class:** {vessel.vessel_class}

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


SET_FILTER_TOOL = {
    "type": "function",
    "function": {
        "name": "set_filter",
        "description": (
            "Set or clear the active retrieval filter when the user's question "
            "names a specific vessel, vessel type (e.g. VLCC, SUEZMAX, AFRAMAX), "
            "or vessel class (e.g. 'Canopus Class'). Call this BEFORE "
            "search_documents — the new filter applies to the next search. "
            "Filters are mutually exclusive: setting one clears the others. "
            "Use kind='clear' to remove the current filter (e.g. user asks "
            "for a fleet-wide view). Do NOT call if no specific vessel/type/"
            "class is mentioned — leave the user's existing filter alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["vessel", "vessel_type", "vessel_class", "clear"],
                    "description": "Filter field to set, or 'clear' to remove any active filter.",
                },
                "value": {
                    "type": "string",
                    "description": "Vessel name / type / class as written by the user. Ignored when kind='clear'.",
                },
            },
            "required": ["kind"],
        },
    },
}


async def _resolve_filter_value(
    kind: str, value: Optional[str]
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    """Resolve a set_filter tool call into (vessel_id, vessel_type, vessel_class, message).

    Returns a tuple where exactly one of the first three is set (or all None for
    'clear' / failure). The string is a human-readable result the tool reports
    back to the LLM so it can acknowledge the change in the final answer.

    Vessel names go through VesselCache (fuzzy on normalized name + aliases).
    Type/class values validate against the distinct values in the cached vessel
    table — no DB round-trip.
    """
    from ..processing.entity_cache import get_vessel_cache

    kind_norm = (kind or "").strip().lower()
    if kind_norm == "clear":
        return None, None, None, "Filter cleared."

    v = (value or "").strip()
    if not v:
        return None, None, None, f"Cannot set {kind_norm} filter: value is empty."

    vessel_cache = get_vessel_cache()
    try:
        await vessel_cache.ensure_loaded(SupabaseClient())
    except Exception as e:
        logger.warning("Failed to load vessel cache for set_filter: %s", e)
        return None, None, None, "Could not load vessel registry — filter unchanged."

    if kind_norm == "vessel":
        vessel = vessel_cache.get_by_name(v)
        if not vessel:
            return None, None, None, (
                f"No vessel matches '{v}'. Filter unchanged. "
                f"Ask the user to confirm the vessel name."
            )
        return str(vessel.id), None, None, f"Filter set to vessel: {vessel.name}."

    if kind_norm in ("vessel_type", "vessel_class"):
        attr = "vessel_type" if kind_norm == "vessel_type" else "vessel_class"
        target = v.casefold()
        match: Optional[str] = None
        for vessel in vessel_cache.list_all():
            candidate = getattr(vessel, attr, None)
            if candidate and candidate.casefold() == target:
                match = candidate
                break
        if not match:
            label = "type" if kind_norm == "vessel_type" else "class"
            return None, None, None, (
                f"No vessel {label} matches '{v}'. Filter unchanged."
            )
        if kind_norm == "vessel_type":
            return None, match, None, f"Filter set to vessel type: {match}."
        return None, None, match, f"Filter set to vessel class: {match}."

    return None, None, None, f"Unknown filter kind: '{kind}'."


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["search_node", "set_filter_node", "__end__"]]:
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
    bind_filter_tool = True

    # If the filter-setting tool just ran, force the next turn to actually
    # retrieve — otherwise the LLM may reply in text without searching.
    if state.get("filter_pending_search"):
        force_search = True

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
            async with record_step("intent_ms"):
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
                bind_filter_tool = False
                system_prompt = base_system + _EXPLORATORY_ADDENDUM
            elif intent is QueryIntent.OFF_TOPIC:
                bind_search_tool = False
                bind_filter_tool = False
                system_prompt = base_system + _OFF_TOPIC_ADDENDUM
            elif intent is QueryIntent.GREETING:
                bind_search_tool = False
                bind_filter_tool = False
                system_prompt = base_system + _GREETING_ADDENDUM
            state["search_progress"] = ""
            await emit_state(config, state)

    system_message = SystemMessage(content=system_prompt)

    if bind_search_tool or bind_filter_tool:
        available_tools = []
        if bind_search_tool:
            available_tools.append(SEARCH_TOOL)
        if bind_filter_tool:
            available_tools.append(SET_FILTER_TOOL)
        # force_search pins the choice to search_documents specifically —
        # the filter tool must have already run (or not be relevant) on this
        # turn, so there's no reason to let the model pick it again.
        if force_search and bind_search_tool:
            tool_choice: Any = "search_documents"
        else:
            tool_choice = "auto"
        invoker = model.bind_tools(
            available_tools,
            parallel_tool_calls=False,
            tool_choice=tool_choice,
        )
    else:
        invoker = model

    llm_step = "chat_llm2_ms" if citation_map_data else "chat_llm1_ms"
    async with record_step(llm_step):
        response = await invoker.ainvoke(
            [system_message, *sanitized_messages],
            config,
        )

    # Check if the model wants to use a tool, and route by name.
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_name = response.tool_calls[0].get("name")
        goto_node = "set_filter_node" if tool_name == "set_filter" else "search_node"
        # Clear the pending-search flag; if we just dispatched another
        # set_filter call, set_filter_node will re-arm it.
        return Command(
            goto=goto_node,
            update={"messages": [response], "filter_pending_search": False},
        )

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
            async with record_step("validate_ms"):
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


async def set_filter_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["chat_node"]]:
    """Resolve a set_filter tool call, mutate filter fields, notify the UI.

    Writes the resolved filter to AgentState so downstream search_node picks it
    up, and emits a `data-filter` annotation so the React client can repaint
    its dropdowns + persist the change to the conversation row.
    """
    ai_message = cast(AIMessage, state["messages"][-1])
    if not ai_message.tool_calls:
        return Command(goto="chat_node", update={})

    tool_call = ai_message.tool_calls[0]
    args = tool_call.get("args") or {}
    kind = str(args.get("kind") or "").strip()
    value = args.get("value")

    state["search_progress"] = "Updating filter"
    await emit_state(config, state)

    vid, vtype, vclass, message = await _resolve_filter_value(kind, value)

    # Emit to frontend regardless of outcome — on failure all three are None,
    # which the client treats as a no-op.
    resolved_ok = kind.lower() == "clear" or (vid or vtype or vclass)
    if resolved_ok:
        await emit_filter_update(config, vid, vtype, vclass)

    state["search_progress"] = ""
    await emit_state(config, state)

    tool_response = ToolMessage(
        content=json.dumps({
            "ok": bool(resolved_ok),
            "vessel_id": vid,
            "vessel_type": vtype,
            "vessel_class": vclass,
            "message": message,
        }),
        tool_call_id=tool_call["id"],
    )

    update: Dict[str, Any] = {
        "messages": [tool_response],
        "selected_vessel_id": vid,
        "selected_vessel_type": vtype,
        "selected_vessel_class": vclass,
        # Only gate the next turn to force search when the resolution succeeded —
        # otherwise the LLM should explain the failure and ask the user to
        # clarify rather than searching with stale filter state.
        "filter_pending_search": bool(resolved_ok),
    }
    return Command(goto="chat_node", update=update)


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

        # Check if user requested broad search (skip topic filtering),
        # or if the topic filter is disabled globally via settings.
        skip_filter = (
            tool_call["args"].get("skip_topic_filter", False)
            or not settings.topic_filter_enabled
        )

        if skip_filter:
            # Skip topic filtering entirely — either the LLM opted out for
            # a broad query, or the feature flag is off for the corpus.
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
            async def _timed_filter():
                async with record_step("topic_filter_ms"):
                    return await topic_filter.analyze_query(
                        question, vessel_filter, on_progress=on_progress
                    )

            async def _timed_embed():
                async with record_step("embed_ms"):
                    return await engine.retriever.embed_query(question)

            filter_result, query_embedding = await asyncio.gather(
                _timed_filter(), _timed_embed()
            )

        # NOTE: the `should_skip_rag` early-return used to short-circuit
        # here when the topic filter couldn't match any stored topics.
        # That logic misfires on corpora where ingest populates
        # topic_ids on only a minority of chunks (test DB: ~8%), which
        # lets the topic filter declare "nothing to search" while the
        # answer is sitting under an empty-topic chunk a hybrid search
        # would have found. Off-topic queries are already blocked by
        # the intent_classifier earlier in chat_node, so skipping this
        # return just falls through to the normal search path (with
        # the no-result topic-filter fallback below as a safety net).

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
        async with record_step("search_rerank_ms"):
            retrieval_results = await engine.search_only(
                question=question,
                top_k=settings.retrieval_top_k,
                use_rerank=settings.rerank_enabled,
                metadata_filter=metadata_filter,
                on_progress=on_progress,
                query_embedding=query_embedding,
            )

            # Fallback: if the topic-filtered search returned nothing,
            # retry without the topic_ids filter. Many chunks in the
            # corpus have empty topic_ids arrays (populated by an LLM
            # pass during ingest that often no-ops), so the
            # semantic-matched topic_ids sometimes point to zero
            # actual chunks. Vessel filters stay applied — they are
            # structural, not LLM-derived.
            if not retrieval_results and filter_result.matched_topic_ids:
                fallback_filter = dict(vessel_filter) if vessel_filter else None
                logger.info(
                    "Topic-filtered search returned 0 results; "
                    "retrying without topic_ids filter"
                )
                retrieval_results = await engine.search_only(
                    question=question,
                    top_k=settings.retrieval_top_k,
                    use_rerank=settings.rerank_enabled,
                    metadata_filter=fallback_filter,
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
    graph.add_node("set_filter_node", set_filter_node)

    # Set entry point
    graph.set_entry_point("chat_node")

    # Compile with provided checkpointer for conversation persistence
    # Note: Langfuse callback handler is added per-request in patches/__init__.py
    # so that session_id metadata can be set correctly for each conversation
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
