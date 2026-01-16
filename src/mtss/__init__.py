"""MTSS - RAG pipeline for email processing with attachments."""

import os
from functools import lru_cache

__version__ = "0.1.0"


@lru_cache(maxsize=1)
def _init_api_keys() -> None:
    """Initialize API keys from settings.

    Called once at module load to set environment variables for LiteLLM.
    Uses lru_cache to ensure it only runs once.
    """
    from .config import get_settings

    settings = get_settings()

    # Set OpenAI API key for LiteLLM
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    # Set Cohere API key for reranking
    if settings.cohere_api_key:
        os.environ.setdefault("COHERE_API_KEY", settings.cohere_api_key)


def init() -> None:
    """Explicitly initialize MTSS configuration and API keys.

    Call this at application startup if automatic initialization fails.
    """
    _init_api_keys()


# Initialize API keys when module is imported
try:
    _init_api_keys()
except Exception:
    # Config may not be available yet (e.g., during testing without .env)
    pass
