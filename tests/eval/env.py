"""Eval-time env loader.

Lets the eval CLI point the existing app at .env.test (or any other env file)
without forking config.py. We:

1. `load_dotenv(path, override=True)` so eval vars beat process env.
2. Clear the `@lru_cache` on `get_settings()` so the next call rebuilds Settings
   against the new env.
3. Re-enable the LiteLLM Langfuse callbacks that `mtss/cli/__init__.py`
   strips out for normal CLI ops — eval runs *want* tracing.

Call `setup_eval_env(...)` exactly once, before any module imports the agent
or RAG components.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_eval_env(
    env_file: Optional[str] = None,
    *,
    enable_langfuse: bool = True,
) -> Path:
    """Load env file + reset Settings cache + re-enable Langfuse tracing.

    Args:
        env_file: Path to .env.test (or any env file). If None, looks for
            $MTSS_ENV_FILE then defaults to .env.test in the project root.
        enable_langfuse: If True (default), re-add Langfuse callbacks that
            `mtss.cli.__init__` strips. Set False for offline eval.

    Returns:
        Resolved path of the env file that was loaded.
    """
    from dotenv import load_dotenv

    if env_file is None:
        env_file = os.environ.get("MTSS_ENV_FILE", ".env.test")
    path = Path(env_file).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Eval env file not found: {path}. "
            "Create it (copy from .env.template) or pass --env-file."
        )

    load_dotenv(path, override=True)
    logger.info("Loaded eval env from %s", path)

    # Force Settings rebuild
    from mtss.config import get_settings
    get_settings.cache_clear()

    # Re-enable Langfuse if requested + configured
    if enable_langfuse and os.environ.get("LANGFUSE_ENABLED", "").lower() == "true":
        try:
            from mtss.observability import init_langfuse
            init_langfuse()
            logger.info("Langfuse tracing enabled for eval run")
        except Exception as exc:
            logger.warning("Failed to init Langfuse for eval: %s", exc)

    return path


def settings_snapshot() -> dict:
    """Return the subset of Settings that affect retrieval/generation quality.

    Stored in each run's manifest.json so two runs can be compared with full
    knowledge of what changed.
    """
    from mtss.config import get_settings

    s = get_settings()
    return {
        "llm_model": s.llm_model,
        "embedding_model": s.embedding_model,
        "embedding_dimensions": s.embedding_dimensions,
        "rag_llm_model": getattr(s, "rag_llm_model", None),
        "retrieval_top_k": getattr(s, "retrieval_top_k", None),
        "rerank_enabled": getattr(s, "rerank_enabled", None),
        "rerank_model": getattr(s, "rerank_model", None),
        "rerank_top_n": getattr(s, "rerank_top_n", None),
        "hybrid_search_enabled": getattr(s, "hybrid_search_enabled", None),
        "topic_query_match_threshold": getattr(s, "topic_query_match_threshold", None),
    }
