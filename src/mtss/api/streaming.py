"""Vercel AI SDK streaming endpoint for LangGraph agent.

Replaces the AG-UI protocol with Vercel AI SDK UI Message Stream v1 format.
Streams LangGraph events as newline-delimited type-prefixed chunks:
  0:"text"   - text delta from LLM
  2:[...]    - data annotation (progress updates)
  d:{...}    - finish signal

The endpoint validates JWT auth before starting the stream, injects vessel
filters into LangGraph state, and sets up Langfuse tracing per conversation.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..observability import (
    flush_langfuse_traces,
    get_langfuse_handler,
    get_user_id,
    set_session_id,
)
from .middleware.auth import UserPayload, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class VesselFilters(BaseModel):
    """Optional vessel filters — at most one should be set."""

    vessel_id: Optional[str] = None
    vessel_type: Optional[str] = None
    vessel_class: Optional[str] = None


class MessagePart(BaseModel):
    """A single part of a Vercel AI SDK UIMessage."""

    type: str
    text: str


class Message(BaseModel):
    """A single chat message (Vercel AI SDK UIMessage format)."""

    role: str
    parts: List[MessagePart] = Field(..., min_length=1)

    @property
    def content(self) -> str:
        return "\n".join(p.text for p in self.parts if p.type == "text")


class AgentRequest(BaseModel):
    """POST body accepted by /api/agent."""

    messages: List[Message] = Field(..., min_length=1)
    thread_id: UUID
    filters: VesselFilters = Field(default_factory=VesselFilters)


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

async def _stream_agent(
    request: Request,
    body: AgentRequest,
    user: UserPayload,
) -> AsyncGenerator[str, None]:
    """Run the LangGraph agent and yield Vercel AI SDK v1 stream chunks."""

    graph = request.app.state.agent_graph
    # LangGraph/Langfuse expect a string thread_id in config/metadata;
    # Pydantic validated it as a UUID — convert to string for downstream.
    thread_id = str(body.thread_id)

    # -- Langfuse tracing (mirrors patches/__init__.py) ---------------------
    set_session_id(thread_id)

    config: dict = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [],
        "metadata": {},
    }

    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        config["callbacks"].append(langfuse_handler)
        config["metadata"]["langfuse_session_id"] = thread_id
        user_id = get_user_id()
        if user_id:
            config["metadata"]["langfuse_user_id"] = user_id

    # -- Build input state --------------------------------------------------
    # Convert frontend messages to LangChain messages
    langchain_messages: list = []
    for msg in body.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        # System messages from the frontend are passed through; assistant
        # messages are already stored in the checkpoint and do not need to be
        # re-sent.

    # Inject vessel filters (same approach as patches/__init__.py)
    filters = body.filters
    input_state: dict = {}

    if filters.vessel_id:
        input_state["selected_vessel_id"] = filters.vessel_id
    if filters.vessel_type:
        input_state["selected_vessel_type"] = filters.vessel_type
    if filters.vessel_class:
        input_state["selected_vessel_class"] = filters.vessel_class

    # Attach vessel_id to the latest HumanMessage for per-message tracking
    if filters.vessel_id and langchain_messages:
        for msg in reversed(langchain_messages):
            if isinstance(msg, HumanMessage):
                msg.additional_kwargs["vessel_id"] = filters.vessel_id
                break

    # -- Conversation continuation (carry-forward from patches/__init__.py) -
    # If the checkpoint already has messages, only append the newest user
    # message to avoid re-sending the entire history.
    checkpoint = await graph.aget_state(config)
    stored_messages = checkpoint.values.get("messages", []) if checkpoint.values else []

    non_system = [m for m in langchain_messages if not isinstance(m, SystemMessage)]

    if stored_messages and non_system:
        # Continuing an existing thread — send only the last user message
        input_state["messages"] = [non_system[-1]]
        logger.debug(
            "Thread continuation: stored=%d, incoming=%d — appending latest message",
            len(stored_messages),
            len(non_system),
        )
    else:
        # New thread — send all messages
        input_state["messages"] = langchain_messages

    # -- Stream LangGraph events as Vercel AI SDK v1 chunks -----------------
    try:
        async for event in graph.astream_events(
            input_state, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                # Text delta from LLM token streaming
                content = event["data"]["chunk"].content
                if content:
                    yield f"0:{json.dumps(content)}\n"

            elif kind == "on_custom_event" and event["name"] == "manually_emit_state":
                # Progress update emitted by emit_state() in agent nodes
                state_data = event["data"]
                progress = state_data.get("search_progress", "")
                if progress:
                    yield f"2:{json.dumps(['progress', progress])}\n"

        # Finish signal
        yield f'd:{json.dumps({"finishReason": "stop"})}\n'
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Error streaming agent response for thread %s", thread_id)
        yield f'd:{json.dumps({"finishReason": "error"})}\n'
        yield "data: [DONE]\n\n"

    finally:
        try:
            flush_langfuse_traces()
        except Exception:
            logger.debug("Failed to flush Langfuse traces after stream")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/agent")
@limiter.limit("30/minute")
async def agent_stream(
    request: Request,
    body: AgentRequest,
    user: UserPayload = Depends(get_current_user),
):
    """Streaming agent endpoint using Vercel AI SDK UI Message Stream v1.

    JWT authentication is validated via the ``get_current_user`` dependency
    **before** the streaming response begins, ensuring unauthorised callers
    never trigger a LangGraph run.
    """
    return StreamingResponse(
        _stream_agent(request, body, user),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-ui-message-stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
