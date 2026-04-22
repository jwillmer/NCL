"""Vercel AI SDK streaming endpoint for LangGraph agent.

Emits the AI SDK v6 UI Message Stream protocol (SSE with typed JSON events):
  data: {"type":"start","messageId":"..."}
  data: {"type":"text-start","id":"..."}
  data: {"type":"text-delta","id":"...","delta":"..."}
  data: {"type":"text-end","id":"..."}
  data: {"type":"data-<name>","data":...}   # custom parts (progress, filter)
  data: {"type":"finish"}
  data: [DONE]

The legacy v4 `0:/2:/d:` format is silently dropped by v6 clients, which is
why the assistant message never appeared until the page was reloaded.
"""

import json
import logging
from typing import AsyncGenerator, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, Request
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

    # -- Stream LangGraph events as AI SDK v6 UI Message Stream chunks ------
    message_id = f"msg_{uuid4().hex}"
    text_id = f"txt_{uuid4().hex}"
    text_started = False

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    yield _sse({"type": "start", "messageId": message_id})

    try:
        async for event in graph.astream_events(
            input_state, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    if not text_started:
                        yield _sse({"type": "text-start", "id": text_id})
                        text_started = True
                    yield _sse({"type": "text-delta", "id": text_id, "delta": content})

            elif kind == "on_custom_event" and event["name"] == "chat_token":
                # Per-token delta from the litellm streaming bridge in
                # agent.py `_call_chat_llm(stream=True)`.
                text = event["data"].get("text") or ""
                if text:
                    if not text_started:
                        yield _sse({"type": "text-start", "id": text_id})
                        text_started = True
                    yield _sse({"type": "text-delta", "id": text_id, "delta": text})

            elif kind == "on_custom_event" and event["name"] == "manually_emit_state":
                progress = event["data"].get("search_progress", "")
                if progress:
                    yield _sse({"type": "data-progress", "data": progress})

            elif kind == "on_custom_event" and event["name"] == "emit_filter_update":
                yield _sse({"type": "data-filter", "data": event["data"]})

        if text_started:
            yield _sse({"type": "text-end", "id": text_id})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Error streaming agent response for thread %s", thread_id)
        if text_started:
            yield _sse({"type": "text-end", "id": text_id})
        yield _sse({"type": "error", "errorText": "Stream error"})
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
    body: AgentRequest = Body(...),
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
