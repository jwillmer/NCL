"""Conversations API for managing chat history.

Provides CRUD operations for conversation metadata.
Actual message history is managed by LangGraph's checkpointer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import litellm
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

# Drop unsupported parameters for models that don't support them
litellm.drop_params = True

from ..config import get_settings
from ..storage.supabase_client import SupabaseClient
from .middleware.auth import UserPayload, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


# ============================================
# Pydantic Models
# ============================================


class ConversationCreate(BaseModel):
    """Create a new conversation."""

    thread_id: Optional[UUID] = Field(default=None, description="Thread ID (auto-generated if not provided)")
    title: Optional[str] = Field(default=None, max_length=200)
    # Filter fields (mutually exclusive - only one should be set)
    vessel_id: Optional[UUID] = Field(default=None, description="Vessel filter (UUID)")
    vessel_type: Optional[str] = Field(default=None, description="Vessel type filter e.g. VLCC")
    vessel_class: Optional[str] = Field(default=None, description="Vessel class filter e.g. Canopus Class")


class ConversationUpdate(BaseModel):
    """Update conversation metadata."""

    title: Optional[str] = Field(default=None, max_length=200)
    # Filter fields (mutually exclusive - only one should be set)
    vessel_id: Optional[UUID] = Field(default=None, description="Vessel filter (UUID)")
    vessel_type: Optional[str] = Field(default=None, description="Vessel type filter e.g. VLCC")
    vessel_class: Optional[str] = Field(default=None, description="Vessel class filter e.g. Canopus Class")
    is_archived: Optional[bool] = None


class ConversationResponse(BaseModel):
    """Conversation response model."""

    id: UUID
    thread_id: UUID
    user_id: UUID
    title: Optional[str]
    # Filter fields (mutually exclusive)
    vessel_id: Optional[UUID]  # Vessel filter (UUID)
    vessel_type: Optional[str] = None  # Vessel type filter e.g. VLCC
    vessel_class: Optional[str] = None  # Vessel class filter e.g. Canopus Class
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]


class ConversationListResponse(BaseModel):
    """Paginated list of conversations."""

    items: list[ConversationResponse]
    total: int
    has_more: bool


class GenerateTitleRequest(BaseModel):
    """Request to generate title from message content."""

    content: str = Field(..., max_length=500)
    force: bool = Field(default=False, description="Force regeneration even if title exists")


class MessageResponse(BaseModel):
    """Message response for history loading."""

    id: str
    role: str  # "user" | "assistant"
    content: str
    vessel_id: Optional[str] = None  # Vessel filter active when message was sent


class MessagesListResponse(BaseModel):
    """List of messages for a conversation."""

    messages: list[MessageResponse]


# ============================================
# Endpoints
# ============================================


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    q: Optional[str] = Query(default=None, max_length=100, description="Search query"),
    archived: bool = Query(default=False, description="Include archived conversations"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: UserPayload = Depends(get_current_user),
):
    """List user's conversations, ordered by most recent activity."""
    client = SupabaseClient()

    # Build query
    query = (
        client.client.table("conversations")
        .select("*", count="exact")
        .eq("user_id", user.sub)
        .eq("is_archived", archived)
    )

    # Add search filter if provided
    if q:
        # Use PostgreSQL full-text search
        query = query.text_search("title", q, config="english")

    # Order by most recent activity
    query = (
        query.order("last_message_at", desc=True, nullsfirst=False)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    result = query.execute()

    conversations = [
        ConversationResponse(
            id=row["id"],
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            title=row["title"],
            vessel_id=row.get("vessel_id"),
            vessel_type=row.get("vessel_type"),
            vessel_class=row.get("vessel_class"),
            is_archived=row["is_archived"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_message_at=row.get("last_message_at"),
        )
        for row in result.data
    ]

    total = result.count or 0
    has_more = offset + len(conversations) < total

    return ConversationListResponse(items=conversations, total=total, has_more=has_more)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate = Body(default=ConversationCreate()),
    user: UserPayload = Depends(get_current_user),
):
    """Create a new conversation."""
    client = SupabaseClient()

    thread_id = data.thread_id or uuid4()

    result = (
        client.client.table("conversations")
        .insert(
            {
                "thread_id": str(thread_id),
                "user_id": user.sub,
                "title": data.title,
                "vessel_id": str(data.vessel_id) if data.vessel_id else None,
                "vessel_type": data.vessel_type,
                "vessel_class": data.vessel_class,
            }
        )
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation",
        )

    row = result.data[0]
    logger.info("Created conversation %s for user %s", row["id"], user.email or user.sub)

    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        vessel_id=row.get("vessel_id"),
        vessel_type=row.get("vessel_type"),
        vessel_class=row.get("vessel_class"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


async def _get_first_user_message(request: Request, thread_id: str) -> Optional[str]:
    """Get the first user message from LangGraph checkpoints.

    Uses the LangGraph checkpointer from app.state to properly deserialize
    messages from checkpoint_blobs (messages are stored as binary msgpack).
    """
    try:
        checkpointer = getattr(request.app.state, "checkpointer", None)
        if not checkpointer:
            logger.warning("Checkpointer not available in app.state")
            return None

        # Use LangGraph's API to get the checkpoint with deserialized messages
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)

        if not checkpoint_tuple:
            return None

        # Get messages from the properly deserialized checkpoint
        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])

        # Find the first human/user message
        for msg in messages:
            # LangGraph deserializes to LangChain message objects
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            # Fallback for dict representation
            elif isinstance(msg, dict):
                msg_type = msg.get("type", "")
                if msg_type == "human":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return None
    except Exception as e:
        logger.warning("Failed to get first user message: %s", e)
        return None


@router.get("/{thread_id}", response_model=ConversationResponse)
async def get_conversation(
    request: Request,
    thread_id: UUID,
    user: UserPayload = Depends(get_current_user),
):
    """Get a specific conversation by thread_id.

    Automatically regenerates fallback titles (truncated messages) using LLM.
    """
    client = SupabaseClient()

    result = (
        client.client.table("conversations")
        .select("*")
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .maybe_single()
        .execute()
    )

    if not result or not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    row = result.data
    title = (row.get("title") or "").strip()
    has_messages = row.get("last_message_at") is not None

    # Regenerate title if conversation has messages and:
    # 1. No title (empty or missing) - title generation may have failed previously
    # 2. Title looks like a fallback (truncated message)
    if has_messages:
        needs_title = not title  # No title at all
        try:
            first_message = await _get_first_user_message(request, str(thread_id))
            if first_message:
                is_fallback = title and _is_fallback_title(title, first_message)
                if needs_title or is_fallback:
                    logger.info("Generating title for %s (missing=%s, fallback=%s)",
                                thread_id, needs_title, is_fallback)
                    new_title = await _generate_title_with_llm(first_message)
                    # Update in database
                    update_result = (
                        client.client.table("conversations")
                        .update({"title": new_title})
                        .eq("thread_id", str(thread_id))
                        .execute()
                    )
                    if update_result.data:
                        row = update_result.data[0]
                        logger.info("Generated title for %s: %s", thread_id, new_title)
        except Exception as e:
            # Don't fail the request if title regeneration fails
            logger.warning("Failed to generate title for %s: %s", thread_id, e)

    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row.get("title"),
        vessel_id=row.get("vessel_id"),
        vessel_type=row.get("vessel_type"),
        vessel_class=row.get("vessel_class"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


@router.patch("/{thread_id}", response_model=ConversationResponse)
async def update_conversation(
    thread_id: UUID,
    data: ConversationUpdate,
    user: UserPayload = Depends(get_current_user),
):
    """Update conversation metadata (title, vessel_id/type/class, archive status)."""
    client = SupabaseClient()

    # Build update data from explicitly provided fields only
    # This allows setting vessel_id to null (clear filter) vs not providing it (don't change)
    update_data = data.model_dump(exclude_unset=True)

    # Convert vessel_id UUID to string for database (or None to clear)
    if "vessel_id" in update_data:
        update_data["vessel_id"] = str(update_data["vessel_id"]) if update_data["vessel_id"] else None

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    result = (
        client.client.table("conversations")
        .update(update_data)
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    row = result.data[0]
    logger.info("Updated conversation %s for user %s", thread_id, user.email or user.sub)

    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        vessel_id=row.get("vessel_id"),
        vessel_type=row.get("vessel_type"),
        vessel_class=row.get("vessel_class"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    thread_id: UUID,
    user: UserPayload = Depends(get_current_user),
):
    """Delete a conversation and its LangGraph checkpoints."""
    client = SupabaseClient()

    # First verify the conversation exists and belongs to user
    check_result = (
        client.client.table("conversations")
        .select("id")
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .maybe_single()
        .execute()
    )

    if not check_result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Delete LangGraph checkpoints for this thread
    # These tables are created by AsyncPostgresSaver
    thread_id_str = str(thread_id)
    try:
        client.client.table("checkpoints").delete().eq("thread_id", thread_id_str).execute()
        client.client.table("checkpoint_writes").delete().eq("thread_id", thread_id_str).execute()
        client.client.table("checkpoint_blobs").delete().eq("thread_id", thread_id_str).execute()
    except Exception as e:
        # Log but don't fail - checkpoints may not exist yet
        logger.warning("Failed to delete checkpoints for thread %s: %s", thread_id, e)

    # Delete the conversation metadata
    client.client.table("conversations").delete().eq("thread_id", thread_id_str).execute()

    logger.info("Deleted conversation %s for user %s", thread_id, user.email or user.sub)


@router.post("/{thread_id}/touch", response_model=ConversationResponse)
async def touch_conversation(
    thread_id: UUID,
    user: UserPayload = Depends(get_current_user),
):
    """Update last_message_at timestamp (called when a new message is sent)."""
    client = SupabaseClient()

    result = (
        client.client.table("conversations")
        .update({"last_message_at": datetime.utcnow().isoformat()})
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    row = result.data[0]
    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        vessel_id=row.get("vessel_id"),
        vessel_type=row.get("vessel_type"),
        vessel_class=row.get("vessel_class"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


@router.get("/{thread_id}/messages", response_model=MessagesListResponse)
async def get_messages(
    request: Request,
    thread_id: UUID,
    user: UserPayload = Depends(get_current_user),
):
    """Get all messages for a conversation from LangGraph checkpoints.

    Used by frontend to load conversation history on page mount.
    Returns messages in CopilotKit-compatible format (id, role, content).
    Only returns user and assistant messages with actual content.
    """
    client = SupabaseClient()

    # Verify conversation exists and belongs to user
    check_result = (
        client.client.table("conversations")
        .select("id")
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .maybe_single()
        .execute()
    )

    if not check_result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Get checkpointer from app state
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if not checkpointer:
        logger.warning("Checkpointer not available in app.state")
        return MessagesListResponse(messages=[])

    try:
        # Use LangGraph's API to get the checkpoint with deserialized messages
        config = {"configurable": {"thread_id": str(thread_id), "checkpoint_ns": ""}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)

        if not checkpoint_tuple:
            return MessagesListResponse(messages=[])

        # Get messages from the properly deserialized checkpoint
        lc_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])

        # Convert LangChain messages to response format
        messages: list[MessageResponse] = []
        for msg in lc_messages:
            msg_id = getattr(msg, "id", None) or str(len(messages))
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    # Extract vessel_id from additional_kwargs if present
                    vessel_id = msg.additional_kwargs.get("vessel_id") if hasattr(msg, "additional_kwargs") else None
                    messages.append(MessageResponse(id=msg_id, role="user", content=content, vessel_id=vessel_id))
            elif isinstance(msg, AIMessage):
                # Skip tool call messages (they have tool_calls but often empty content)
                if hasattr(msg, "tool_calls") and msg.tool_calls and not msg.content:
                    continue
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    messages.append(MessageResponse(id=msg_id, role="assistant", content=content))

        logger.debug("Returning %d messages for thread %s", len(messages), thread_id)
        return MessagesListResponse(messages=messages)

    except Exception as e:
        logger.error("Failed to get messages for thread %s: %s", thread_id, e)
        # Return empty list instead of failing - better UX
        return MessagesListResponse(messages=[])


def _is_fallback_title(title: str, content: str) -> bool:
    """Check if title looks like a fallback (truncated message) rather than LLM-generated."""
    if not title or not content:
        return False
    # Fallback titles end with "..." and match the start of the content
    if title.endswith("..."):
        title_prefix = title[:-3].strip()
        content_start = content.strip()[:len(title_prefix)]
        if title_prefix.lower() == content_start.lower():
            return True
    # Also check if title is exactly the content (short messages)
    if title.strip().lower() == content.strip()[:100].lower():
        return True
    return False


async def _generate_title_with_llm(content: str) -> str:
    """Generate a concise conversation title using LLM."""
    settings = get_settings()

    try:
        response = await litellm.acompletion(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short title (3-6 words) for this conversation. "
                        "Output ONLY the title, nothing else. No quotes, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": content[:300]},
            ],
            temperature=0.3,
        )
        title = (response.choices[0].message.content or "").strip()
        # Remove quotes if present
        title = title.strip('"\'')
        # Check for incomplete response (finish_reason='length')
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length" or not title:
            logger.warning("LLM title generation incomplete (finish_reason=%s), using fallback", finish_reason)
            raise ValueError("Incomplete LLM response")
        # Limit length
        if len(title) > 100:
            title = title[:97] + "..."
        return title
    except Exception as e:
        logger.warning("LLM title generation failed, using fallback: %s", e)
        # Fallback to simple truncation
        title = content.strip()[:60]
        if len(content) > 60:
            last_space = title.rfind(" ")
            if last_space > 30:
                title = title[:last_space]
            title += "..."
        return title


@router.post("/{thread_id}/generate-title", response_model=ConversationResponse)
async def generate_title(
    thread_id: UUID,
    data: GenerateTitleRequest = Body(...),
    user: UserPayload = Depends(get_current_user),
):
    """Generate and set conversation title from first message content using LLM."""
    client = SupabaseClient()

    # Check conversation exists
    check_result = (
        client.client.table("conversations")
        .select("id, title")
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .maybe_single()
        .execute()
    )

    if not check_result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    existing_title = check_result.data.get("title")

    # Determine if we should generate a new title
    should_generate = (
        not existing_title  # No title yet
        or data.force  # Force regeneration requested
        or _is_fallback_title(existing_title, data.content)  # Title looks like fallback
    )

    if not should_generate:
        # Return existing conversation without updating
        row = (
            client.client.table("conversations")
            .select("*")
            .eq("thread_id", str(thread_id))
            .single()
            .execute()
        ).data
    else:
        # Generate title using LLM
        title = await _generate_title_with_llm(data.content)

        result = (
            client.client.table("conversations")
            .update({"title": title})
            .eq("thread_id", str(thread_id))
            .eq("user_id", user.sub)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update title",
            )

        row = result.data[0]
        logger.info("Generated title for conversation %s: %s", thread_id, title)

    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        vessel_id=row.get("vessel_id"),
        vessel_type=row.get("vessel_type"),
        vessel_class=row.get("vessel_class"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )
