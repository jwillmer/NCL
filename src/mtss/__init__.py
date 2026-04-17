"""MTSS - RAG pipeline for email processing with attachments."""

import logging
import os
from functools import lru_cache

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _init_api_keys() -> None:
    """Initialize API keys from settings.

    Called once at module load to set environment variables for LiteLLM.
    Uses lru_cache to ensure it only runs once.
    """
    from .config import get_settings

    settings = get_settings()

    # Set OpenRouter API key and base URL for LiteLLM
    if settings.openrouter_api_key:
        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    if settings.openrouter_base_url:
        os.environ.setdefault("OPENROUTER_API_BASE", settings.openrouter_base_url)


def init() -> None:
    """Explicitly initialize MTSS configuration and API keys.

    Call this at application startup if automatic initialization fails.
    """
    _init_api_keys()


# Initialize API keys when module is imported
try:
    _init_api_keys()
except Exception as e:
    # Config may not be available yet (e.g., during testing without .env)
    logger.debug("API key init deferred: %s", e)
