"""Langfuse v3 observability for LLM tracing."""

import atexit
import logging
import os
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Context variable for session_id propagation to LiteLLM calls
# This allows any LiteLLM call within the same async context to access the session_id
_session_id_var: ContextVar[str | None] = ContextVar("langfuse_session_id", default=None)

# Context variable for user_id propagation to LiteLLM calls
# Enables Langfuse user tracking for cost aggregation and usage analytics
# See: https://langfuse.com/docs/observability/features/users
_user_id_var: ContextVar[str | None] = ContextVar("langfuse_user_id", default=None)


def set_session_id(session_id: str | None) -> None:
    """Set the current session_id for LiteLLM metadata propagation.

    Call this at the start of a request to ensure all LiteLLM calls
    (embeddings, completions) within that request get the session_id.
    """
    _session_id_var.set(session_id)


def get_session_id() -> str | None:
    """Get the current session_id for LiteLLM metadata.

    Use this when making LiteLLM calls to include session_id in metadata.
    """
    return _session_id_var.get()


def set_user_id(user_id: str | None) -> None:
    """Set the current user_id for Langfuse user tracking.

    Call this at the start of a request to ensure all LiteLLM calls
    (embeddings, completions) within that request get the user_id.

    The user_id should be a stable identifier (e.g., Supabase user ID).
    """
    _user_id_var.set(user_id)


def get_user_id() -> str | None:
    """Get the current user_id for Langfuse user tracking.

    Use this when making LiteLLM calls to include user_id in metadata.
    """
    return _user_id_var.get()

# Global Langfuse client instance
_langfuse_client: "Optional[Langfuse]" = None

# Global handler instance for proper cleanup
_langfuse_handler: "CallbackHandler | None" = None


@lru_cache(maxsize=1)
def init_langfuse() -> bool:
    """Initialize Langfuse v3 with LiteLLM callbacks. Called once at startup."""
    from ..config import get_settings

    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.debug("Langfuse disabled")
        return False

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse enabled but credentials missing")
        return False

    try:
        import litellm

        # Set env vars for Langfuse
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url

        # Configure LiteLLM with Langfuse OTEL callback (required for Langfuse v3)
        # Note: Native "langfuse" callback is incompatible with Langfuse SDK v3
        # See: https://github.com/BerriAI/litellm/issues/13137
        # Known limitation: input/output may not display fully in Langfuse UI
        # See: https://github.com/langfuse/langfuse/issues/9474
        if "langfuse_otel" not in litellm.success_callback:
            litellm.success_callback.append("langfuse_otel")
        if "langfuse_otel" not in litellm.failure_callback:
            litellm.failure_callback.append("langfuse_otel")

        logger.debug("LiteLLM callbacks: %s", litellm.success_callback)
        logger.info("Langfuse initialized (host: %s)", settings.langfuse_base_url)
        return True

    except ImportError:
        logger.warning("Langfuse package not installed")
        return False
    except Exception as e:
        logger.error("Failed to initialize Langfuse: %s", e)
        return False


def get_langfuse_handler() -> "CallbackHandler | None":
    """Get LangChain/LangGraph callback handler for tracing.

    Returns a singleton handler instance that's reused across requests.
    The handler is automatically flushed on process exit.
    """
    global _langfuse_handler

    from ..config import get_settings

    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.debug("Langfuse credentials not configured")
        return None

    # Return existing handler if already created
    if _langfuse_handler is not None:
        return _langfuse_handler

    try:
        from langfuse.langchain import CallbackHandler

        logger.debug("Creating Langfuse CallbackHandler")
        _langfuse_handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.info("Langfuse callback handler created")

        # Register flush on exit to ensure traces are sent
        def _flush_langfuse() -> None:
            if _langfuse_handler is not None:
                try:
                    _langfuse_handler.flush()
                    logger.debug("Langfuse handler flushed on exit")
                except Exception as e:
                    logger.warning("Failed to flush Langfuse: %s", e)

        atexit.register(_flush_langfuse)

        return _langfuse_handler
    except ImportError:
        logger.debug("Langfuse callback not available")
        return None
    except Exception as e:
        logger.warning("Failed to create Langfuse handler: %s", e)
        return None


def get_langfuse_client() -> "Optional[Langfuse]":
    """Get or create a Langfuse client instance.

    Returns a singleton client that can be used for creating scores and other operations.
    Returns None if Langfuse is not configured.
    """
    global _langfuse_client

    from ..config import get_settings

    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.debug("Langfuse client created")
        return _langfuse_client
    except ImportError:
        logger.debug("Langfuse package not available")
        return None
    except Exception as e:
        logger.warning("Failed to create Langfuse client: %s", e)
        return None


def create_trace_id_for_thread(thread_id: str) -> str:
    """Create a deterministic Langfuse trace ID from a thread_id.

    This allows frontend to derive the same trace_id for feedback submission.
    Uses Langfuse.create_trace_id(seed=...) which generates a 32-char hex string.

    Args:
        thread_id: The conversation thread ID (UUID string)

    Returns:
        A deterministic 32-character hex trace ID
    """
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=f"ncl-thread-{thread_id}")
