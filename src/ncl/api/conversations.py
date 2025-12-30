"""Conversations API for managing chat history.

Provides CRUD operations for conversation metadata.
Actual message history is managed by LangGraph's checkpointer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from .middleware.auth import UserPayload, get_current_user
from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


# ============================================
# Pydantic Models
# ============================================


class ConversationCreate(BaseModel):
    """Create a new conversation."""

    thread_id: Optional[UUID] = Field(default=None, description="Thread ID (auto-generated if not provided)")
    title: Optional[str] = Field(default=None, max_length=200)
    vessel_filter: Optional[str] = Field(default=None, max_length=100)


class ConversationUpdate(BaseModel):
    """Update conversation metadata."""

    title: Optional[str] = Field(default=None, max_length=200)
    vessel_filter: Optional[str] = Field(default=None, max_length=100)
    is_archived: Optional[bool] = None


class ConversationResponse(BaseModel):
    """Conversation response model."""

    id: UUID
    thread_id: UUID
    user_id: UUID
    title: Optional[str]
    vessel_filter: Optional[str]
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
            vessel_filter=row.get("vessel_filter"),
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
                "vessel_filter": data.vessel_filter,
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
        vessel_filter=row.get("vessel_filter"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


@router.get("/{thread_id}", response_model=ConversationResponse)
async def get_conversation(
    thread_id: UUID,
    user: UserPayload = Depends(get_current_user),
):
    """Get a specific conversation by thread_id."""
    client = SupabaseClient()

    result = (
        client.client.table("conversations")
        .select("*")
        .eq("thread_id", str(thread_id))
        .eq("user_id", user.sub)
        .maybe_single()
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    row = result.data
    return ConversationResponse(
        id=row["id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        vessel_filter=row.get("vessel_filter"),
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
    """Update conversation metadata (title, vessel_filter, archive status)."""
    client = SupabaseClient()

    # Build update data, excluding None values
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}

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
        vessel_filter=row.get("vessel_filter"),
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
        vessel_filter=row.get("vessel_filter"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


@router.post("/{thread_id}/generate-title", response_model=ConversationResponse)
async def generate_title(
    thread_id: UUID,
    data: GenerateTitleRequest = Body(...),
    user: UserPayload = Depends(get_current_user),
):
    """Generate and set conversation title from first message content."""
    client = SupabaseClient()

    # Check conversation exists and has no title yet
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

    # Only generate if no title exists
    if check_result.data.get("title"):
        # Return existing conversation without updating
        row = (
            client.client.table("conversations")
            .select("*")
            .eq("thread_id", str(thread_id))
            .single()
            .execute()
        ).data
    else:
        # Generate title from content (truncate to ~100 chars)
        title = data.content.strip()[:100]
        if len(data.content) > 100:
            # Try to break at word boundary
            last_space = title.rfind(" ")
            if last_space > 50:
                title = title[:last_space]
            title += "..."

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
        vessel_filter=row.get("vessel_filter"),
        is_archived=row["is_archived"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )
