"""Langfuse v3 observability for LLM tracing."""

import atexit
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse.callback import CallbackHandler

logger = logging.getLogger(__name__)

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
        from langfuse.callback import CallbackHandler

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
