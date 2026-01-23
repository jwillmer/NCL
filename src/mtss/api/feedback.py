"""User feedback API endpoint for Langfuse integration."""

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..observability import get_langfuse_client
from .middleware.auth import UserPayload, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Validated feedback request."""

    thread_id: UUID
    message_id: str = Field(max_length=100)
    value: Literal[0, 1]  # 0 = negative (thumbs down), 1 = positive (thumbs up)


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    status: str


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    user: UserPayload = Depends(get_current_user),
) -> FeedbackResponse:
    """Submit user feedback for a chat message.

    Feedback is stored as a score in Langfuse, linked to the conversation's trace.
    Uses deterministic trace IDs based on thread_id so feedback can be correlated
    with the conversation traces.

    Args:
        request: Feedback data including thread_id, message_id, and value
        user: Authenticated user from JWT token

    Returns:
        Status indicating success
    """
    langfuse = get_langfuse_client()

    if langfuse:
        try:
            # Use session_id (thread_id) to link feedback to the conversation
            # This links to all traces in the session without needing a specific trace_id
            langfuse.create_score(
                name="user_feedback",
                value=request.value,
                session_id=str(request.thread_id),
                comment=f"message_id: {request.message_id}",
            )
            logger.info(
                "Feedback submitted: thread=%s, message=%s, value=%d, user=%s",
                request.thread_id,
                request.message_id,
                request.value,
                user.email or user.sub,
            )
        except Exception as e:
            # Log but don't fail - feedback is non-critical
            logger.warning("Failed to submit feedback to Langfuse: %s", e)
    else:
        logger.debug("Langfuse not configured, feedback not recorded")

    return FeedbackResponse(status="ok")
