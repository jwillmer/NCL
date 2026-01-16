"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_settings():
    """Get cached settings instance."""
    from ..config import get_settings as _get_settings

    return _get_settings()
