"""FastAPI dependency-injection providers.

Process-wide singletons created once at import time so request handlers
don't pay construction overhead (settings validation, PostgREST client
bootstrap, repository wiring) on every call.

SupabaseClient
--------------
The underlying supabase-py REST client and asyncpg pool are designed to be
long-lived — PostgREST uses a single keep-alive HTTP connection and the
asyncpg pool is lazily opened on first use. A per-request ``SupabaseClient()``
throws that away and rebuilds it every time, adding ~5-30ms of pure overhead
per handler invocation.

The singleton is closed from the FastAPI ``lifespan`` shutdown path; handlers
must NOT call ``client.close()`` themselves.
"""

from __future__ import annotations

from functools import lru_cache

from ..storage.supabase_client import SupabaseClient


@lru_cache(maxsize=1)
def get_supabase_client() -> SupabaseClient:
    """Return the process-wide SupabaseClient singleton.

    Safe to use as a FastAPI ``Depends`` provider: FastAPI caches the call
    result per-request by default, but ``lru_cache`` extends that to the
    whole process lifetime.
    """
    return SupabaseClient()
