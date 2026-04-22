"""FastAPI application with LangGraph Agent integration.

Exposes the LangGraph agent via Vercel AI SDK streaming protocol with
defense-in-depth JWT authentication.

Conversation history is persisted via LangGraph's AsyncPostgresSaver checkpointer.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_settings
from ..observability import flush_langfuse_traces, set_user_id
from ..storage.archive_storage import ArchiveStorage, ArchiveStorageError
from ..storage.supabase_client import SupabaseClient
from ..utils import CHUNK_ID_LENGTH
from ..version import APP_VERSION, GIT_SHA_SHORT
from .agent import create_graph
from .conversations import router as conversations_router
from .deps import get_supabase_client
from .feedback import router as feedback_router
from .middleware.auth import SupabaseJWTBearer
from .streaming import router as streaming_router

logger = logging.getLogger(__name__)

# Application-log level: default INFO so operational breadcrumbs
# (chat_node tool choice, set_filter resolution, langfuse init, etc.)
# surface to stdout. Override with LOG_LEVEL=DEBUG or WARNING as needed.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("mtss").setLevel(_log_level)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


_ARCHIVE_URI_PREFIX = "/archive/"


def _strip_archive_prefix(uri: str) -> str:
    """Turn a web-facing archive URI into a bucket-relative key.

    `archive_browse_uri` / `archive_download_uri` are stored with a leading
    `/archive/` so the UI can hit them via `/api/archive/...`; Supabase
    Storage keys are bare (`<folder>/attachments/<file>`).
    """
    return uri[len(_ARCHIVE_URI_PREFIX):] if uri.startswith(_ARCHIVE_URI_PREFIX) else uri


def _apply_spa_cache_headers(response: Response, path: str) -> None:
    """Set per-path cache headers for SPA assets.

    - Vite's ``assets/`` directory is content-hashed (``foo.abc123.js``) so
      those files are effectively immutable — cache them for a year.
    - ``index.html`` (and the SPA-fallback index) must revalidate every
      load so new deploys roll out to users immediately.
    - ``sw.js`` and ``registerSW.js`` are service-worker bootstrappers.
      Browsers already bypass the normal HTTP cache for these, but we
      set ``no-cache`` defensively so a misconfigured CDN can't pin an
      old version. ``Service-Worker-Allowed: /`` lets the SW control
      the entire origin even if it lives at ``/sw.js``.
    """
    # Normalise: Starlette's StaticFiles passes the path through
    # os.path.normpath, so on Windows we see backslashes. Collapse both
    # separators into a single forward-slash form for the prefix/suffix
    # checks below.
    lower = path.lower().replace("\\", "/").lstrip("/")
    if lower.startswith("assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif lower in ("sw.js", "registersw.js"):
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Service-Worker-Allowed"] = "/"
    elif lower == "" or lower == "." or lower.endswith("index.html"):
        response.headers["Cache-Control"] = "no-cache"


class SPAStaticFiles(StaticFiles):
    """StaticFiles with SPA fallback + per-path cache headers.

    SPA fallback: React Router uses BrowserRouter (HTML5 history API). Deep
    links like ``/chat?threadId=...`` must return ``index.html`` so the
    client-side router can resolve the route. Default StaticFiles returns
    404 for unknown paths.

    Cache headers: ``assets/*`` is Vite-hashed so it's cached forever;
    ``index.html`` and the SPA fallback must revalidate on every load;
    ``sw.js``/``registerSW.js`` get ``no-cache`` + ``Service-Worker-Allowed``
    for the PWA rollout.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
            _apply_spa_cache_headers(response, path)
            return response
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                index = Path(self.directory) / "index.html"
                if index.is_file():
                    fallback = FileResponse(index)
                    # The fallback IS index.html — make sure it revalidates.
                    fallback.headers["Cache-Control"] = "no-cache"
                    return fallback
            raise


class SSEAwareGZipMiddleware(GZipMiddleware):
    """GZip middleware that bypasses compression for SSE streams.

    Starlette's stock ``GZipMiddleware`` buffers enough of the response body
    to decide on compression, which breaks Server-Sent Events — the AI SDK
    client stalls waiting for tokens that are sitting in the compressor
    buffer. Vercel AI SDK v6 explicitly documents this as incompatible with
    transport-level compression.

    The streaming agent endpoint is the only SSE route we expose today
    (``POST /api/agent``). We bypass gzip whenever the request path matches
    that prefix — checking the path is cheaper and more reliable than
    peeking at response headers (which would require buffering + replaying
    the first ASGI ``http.response.start`` message).
    """

    # Any request path that starts with one of these prefixes is served
    # without gzip wrapping. Keep this list narrow so we don't accidentally
    # disable compression for the bulk of the API.
    _SSE_PATH_PREFIXES: tuple[str, ...] = ("/api/agent",)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            if any(path.startswith(prefix) for prefix in self._SSE_PATH_PREFIXES):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware for defense-in-depth security."""

    # Paths that don't require authentication
    PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/config.js"}

    # Path prefixes for static frontend files (served from web/dist)
    # /assets/ is Vite's default output directory for bundled files
    STATIC_PREFIXES = ("/assets/", "/icons/", "/images/", "/fonts/")

    # Static file extensions that don't require auth
    STATIC_EXTENSIONS = (".js", ".css", ".ico", ".png", ".svg", ".jpg", ".jpeg", ".woff", ".woff2", ".ttf", ".txt", ".json", ".map")

    async def dispatch(self, request: Request, call_next):
        """Validate JWT token for protected routes."""
        path = request.url.path

        # Skip auth for public paths
        if path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for static frontend files (Vite static build)
        # Allow root and .html files
        if path == "/" or path.endswith(".html"):
            return await call_next(request)
        if path.startswith(self.STATIC_PREFIXES):
            return await call_next(request)
        if any(path.endswith(ext) for ext in self.STATIC_EXTENSIONS):
            return await call_next(request)

        # Skip auth for CORS preflight requests (OPTIONS)
        # These don't include auth headers and are handled by CORSMiddleware
        if request.method == "OPTIONS":
            return await call_next(request)

        # SPA fallback: any non-/api/ path is a client-side React Router route.
        # SPAStaticFiles serves index.html for unknown paths, so auth is not
        # required here — the JWT is enforced on the /api/* XHR calls the SPA
        # makes after hydration. This avoids the need to maintain a hardcoded
        # list of frontend routes (e.g. /chat, /conversations, /settings, ...).
        if not path.startswith("/api/"):
            return await call_next(request)

        # Validate JWT for all other routes
        auth_header = request.headers.get("authorization", "")
        auth = SupabaseJWTBearer(auto_error=False)
        user = await auth(request)
        if not user:
            logger.warning(
                "Unauthorized request to %s from %s (auth header: %s)",
                request.url.path,
                request.client.host if request.client else "unknown",
                "present" if auth_header else "missing",
            )
            return JSONResponse(
                {"error": "Unauthorized", "detail": "Invalid or missing token"},
                status_code=401,
            )

        # Store user in request state for downstream use
        request.state.user = user

        # Set user_id for Langfuse tracking (uses Supabase user ID for aggregation)
        set_user_id(user.sub)

        logger.debug("Auth successful for user: %s", user.email or user.sub)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Only add HSTS in production (when not localhost)
        if request.url.hostname not in ("localhost", "127.0.0.1"):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        # Content Security Policy
        csp = "; ".join([
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self' https://*.supabase.co https://openrouter.ai https://cloud.langfuse.com",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ])
        response.headers["Content-Security-Policy"] = csp
        return response


class LangfuseFlushMiddleware(BaseHTTPMiddleware):
    """Flush Langfuse traces after agent requests.

    Langfuse v3 buffers traces in memory and sends them asynchronously.
    Without explicit flushing, traces from LangChain/LangGraph may never
    be sent in long-running server processes.

    This middleware flushes traces after each request to the /agent endpoint
    to ensure conversation traces appear in the Langfuse UI.
    """

    async def dispatch(self, request: Request, call_next):
        """Flush Langfuse traces after agent requests."""
        response = await call_next(request)

        # Flush after agent requests (where LangGraph conversations happen)
        if request.url.path.startswith("/api/agent"):
            try:
                flush_langfuse_traces()
            except Exception as e:
                logger.debug("Failed to flush Langfuse after request: %s", e)

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes the LangGraph PostgreSQL checkpointer for conversation persistence.
    The checkpointer is stored in app.state for access by the agent graph.
    """
    settings = get_settings()

    # Log startup banner with version info
    logger.info("=" * 60)
    logger.info("MTSS Email RAG API v%s (build: %s)", APP_VERSION, GIT_SHA_SHORT)
    logger.info("=" * 60)

    # Initialize Langfuse observability (if enabled)
    if settings.langfuse_enabled:
        from ..observability import init_langfuse

        logger.info("Initializing Langfuse with host: %s", settings.langfuse_base_url)
        if init_langfuse():
            logger.info("Langfuse observability enabled")
        else:
            logger.warning("Langfuse initialization failed - check credentials")
    else:
        logger.debug("Langfuse disabled")

    # Pre-warm process-wide caches (topics + vessels) so the first request
    # doesn't pay cold-load latency on the hot path. Uses the same singleton
    # client the request handlers will use, so the asyncpg pool is warm too.
    try:
        from ..processing.entity_cache import warm_caches

        await warm_caches(get_supabase_client())
        logger.info("Topic + vessel caches pre-warmed")
    except Exception as exc:
        logger.warning("Cache pre-warm failed (non-fatal): %s", exc)

    # Initialize AsyncPostgresSaver for LangGraph conversation persistence
    logger.info("Initializing LangGraph checkpointer...")
    async with AsyncPostgresSaver.from_conn_string(settings.supabase_db_url) as checkpointer:
        # Setup creates checkpoint tables if they don't exist
        await checkpointer.setup()
        logger.info("LangGraph checkpointer initialized successfully")

        # Store checkpointer in app.state for access by agent
        app.state.checkpointer = checkpointer

        # Create agent graph with checkpointer
        app.state.agent_graph = create_graph(checkpointer)

        logger.info("LangGraph agent endpoint registered at /api/agent")

        # Mount static frontend AFTER all API routes are registered
        # This ensures /api/* routes take priority over the catch-all static mount
        static_dir = Path(__file__).parent.parent.parent.parent / "web" / "dist"
        if static_dir.exists():
            app.mount("/", SPAStaticFiles(directory=static_dir, html=True), name="frontend")
            logger.info("Static frontend mounted from %s", static_dir)

        yield

        logger.info("Shutting down LangGraph checkpointer...")

        # Close the process-wide SupabaseClient singleton so the asyncpg
        # pool drains cleanly. Ignore errors — shutdown is best-effort.
        try:
            await get_supabase_client().close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error closing SupabaseClient singleton: %s", exc)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MTSS Email RAG API",
        description="LangGraph Agent for email document Q&A with streaming progress",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # GZip compression — SSE-aware. Stock GZipMiddleware would break the
    # /api/agent stream by buffering the response; our subclass short-circuits
    # on that path. Registered BEFORE CORS so it sits on the outside of the
    # middleware stack (add_middleware inserts at position 0), meaning it
    # compresses the response after CORS has set its headers.
    app.add_middleware(SSEAwareGZipMiddleware, minimum_size=500)

    # CORS middleware - allow requests from frontend and runtime
    # In production, this should be restricted appropriately
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security headers middleware (runs after CORS)
    app.add_middleware(SecurityHeadersMiddleware)

    # Authentication middleware (runs after security headers)
    app.add_middleware(AuthMiddleware)

    # Langfuse flush middleware (runs after auth to flush traces)
    # Note: Middleware execution order is reversed from registration order
    # This runs LAST after request completes, ensuring all traces are captured
    app.add_middleware(LangfuseFlushMiddleware)

    # Health check endpoint with rate limiting
    @app.get("/health")
    @limiter.limit("60/minute")
    async def health_check(request: Request):
        return {
            "status": "healthy",
            "service": "MTSS-api",
            "version": APP_VERSION,
            "git_sha": GIT_SHA_SHORT,
        }

    # Frontend runtime configuration endpoint
    # Serves env vars as JavaScript for static Vite build
    @app.get("/config.js")
    @limiter.limit("60/minute")
    async def frontend_config(request: Request):
        """Serve frontend runtime configuration as JavaScript.

        This endpoint allows runtime configuration of the static Vite frontend
        in Docker deployments. The frontend loads this script which sets
        window.__MTSS_CONFIG__ with the actual environment values.

        Reads VITE_* env vars first, falls back to NEXT_PUBLIC_* for backwards compatibility.
        """
        def env(vite_key: str, next_key: str) -> str:
            return os.environ.get(vite_key, "") or os.environ.get(next_key, "")

        config = {
            "SUPABASE_URL": env("VITE_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
            "SUPABASE_ANON_KEY": env("VITE_SUPABASE_ANON_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            "API_URL": env("VITE_API_URL", "NEXT_PUBLIC_API_URL"),
            "LANGFUSE_PUBLIC_KEY": env("VITE_LANGFUSE_PUBLIC_KEY", "NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY"),
            "LANGFUSE_BASE_URL": env("VITE_LANGFUSE_BASE_URL", "NEXT_PUBLIC_LANGFUSE_BASE_URL"),
        }

        # Generate JavaScript that sets window.__MTSS_CONFIG__
        js_content = f"window.__MTSS_CONFIG__ = {json.dumps(config)};"

        return Response(
            content=js_content,
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # Archive file serving endpoint
    # Uses same JWT auth as all other endpoints (via AuthMiddleware)
    # Proxies files from Supabase Storage bucket
    @app.get("/api/archive/{file_path:path}")
    @limiter.limit("100/minute")
    async def serve_archive(request: Request, file_path: str):
        """Serve archive files from Supabase Storage (markdown previews and original downloads).

        Security measures:
        - Uses same Supabase JWT auth as UI endpoints (via AuthMiddleware)
        - Path validation to prevent traversal attacks (rejects '..', absolute paths)
        - Rate limited to prevent abuse

        Args:
            request: FastAPI request object (user available in request.state.user)
            file_path: Relative path within the archive bucket

        Returns:
            Response with file content

        Raises:
            HTTPException: 400 for invalid path, 404 for file not found
        """
        # Security: Reject paths with traversal attempts or absolute paths
        if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
            user = getattr(request.state, "user", None)
            logger.warning(
                "Archive path traversal attempt: %s from user %s",
                file_path,
                getattr(user, "email", "unknown") if user else "unknown",
            )
            raise HTTPException(status_code=400, detail="Invalid path")

        # Download from Supabase Storage
        try:
            storage = ArchiveStorage()
            content = storage.download_file(file_path)
        except ArchiveStorageError:
            raise HTTPException(status_code=404, detail="File not found")

        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            # Default to text/markdown for .md, octet-stream for others
            if file_path.endswith(".md"):
                content_type = "text/markdown; charset=utf-8"
            else:
                content_type = "application/octet-stream"

        user = getattr(request.state, "user", None)
        logger.debug(
            "Serving archive file: %s to user %s",
            file_path,
            getattr(user, "email", "unknown") if user else "unknown",
        )

        # Get filename from path
        filename = Path(file_path).name

        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    # Citation details endpoint for fetching source content
    @app.get("/api/citations/{chunk_id}")
    @limiter.limit("100/minute")
    async def get_citation(
        request: Request,
        chunk_id: str,
        client: SupabaseClient = Depends(get_supabase_client),
    ):
        """Get citation details including metadata and markdown content.

        Used by the frontend to display source content when user clicks a citation.

        Args:
            request: FastAPI request object (user available in request.state.user)
            chunk_id: The chunk's hex ID (e.g., "8f3a2b1c")

        Returns:
            Citation details including source_title, page, lines, and content

        Raises:
            HTTPException: 400 for invalid chunk_id, 404 if not found
        """
        # Validate chunk_id format (hex string, defense-in-depth)
        if not re.match(r"^[a-f0-9]+$", chunk_id, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Invalid chunk ID format")

        # Validate chunk_id length (must be exactly 12 chars)
        if len(chunk_id) != CHUNK_ID_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunk ID length: {len(chunk_id)} (expected {CHUNK_ID_LENGTH})",
            )

        # Fetch chunk from database (synchronous call, no pool needed).
        # Uses the process-wide SupabaseClient singleton (see deps.py) to
        # avoid paying PostgREST bootstrap cost on every citation click.
        chunk = client.get_chunk_by_id(chunk_id)

        if not chunk:
            raise HTTPException(status_code=404, detail="Citation not found")

        # Fetch markdown content from Supabase Storage if archive exists.
        # Errors are surfaced as logged stack traces (previously they were
        # swallowed as warnings, which hid real misconfiguration).
        content = None
        storage: ArchiveStorage | None = None
        if chunk.archive_browse_uri:
            relative_path = _strip_archive_prefix(chunk.archive_browse_uri)
            try:
                storage = ArchiveStorage()
                content = storage.download_file(relative_path).decode("utf-8")
            except ArchiveStorageError:
                logger.exception("Archive .md not found for chunk %s at %s", chunk_id, relative_path)
            except Exception:
                logger.exception("Failed to read archive content for chunk %s at %s", chunk_id, relative_path)

        # Sign the download URL so the browser can open it in a new tab
        # without a Bearer header (link navigations can't send auth).
        archive_download_signed_url: str | None = None
        if chunk.archive_download_uri:
            download_path = _strip_archive_prefix(chunk.archive_download_uri)
            try:
                archive_download_signed_url = (storage or ArchiveStorage()).create_signed_url(
                    download_path, expires_in=300
                )
            except ArchiveStorageError:
                logger.exception("Failed to sign download URL for chunk %s at %s", chunk_id, download_path)

        user = getattr(request.state, "user", None)
        logger.debug(
            "Serving citation %s to user %s",
            chunk_id,
            getattr(user, "email", "unknown") if user else "unknown",
        )

        return {
            "chunk_id": chunk_id,
            "source_title": chunk.source_title,
            "page": chunk.page_number,
            "lines": [chunk.line_from, chunk.line_to]
            if chunk.line_from is not None and chunk.line_to is not None
            else None,
            "archive_browse_uri": chunk.archive_browse_uri,
            "archive_download_uri": chunk.archive_download_uri,
            "archive_download_signed_url": archive_download_signed_url,
            "content": content,
        }

    # ------------------------------------------------------------------
    # Vessels endpoints — served from the in-process VesselCache instead
    # of round-tripping to Supabase on every request. The cache has a 5-min
    # TTL and is pre-warmed on startup (see ``lifespan``). CLI write paths
    # call ``VesselCache.invalidate()`` after upsert/delete so within-process
    # writes propagate immediately; cross-process writes rely on the TTL.
    # ------------------------------------------------------------------
    _VESSEL_CACHE_CONTROL = "private, max-age=300, stale-while-revalidate=600"

    @app.get("/api/vessels")
    @limiter.limit("60/minute")
    async def list_vessels(
        request: Request,
        response: Response,
        client: SupabaseClient = Depends(get_supabase_client),
    ):
        """Get all vessels for dropdown selection.

        Returns minimal vessel info (id, name, type, class) for filtering.
        """
        from ..processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        await cache.ensure_loaded(client)
        vessels = sorted(cache.list_all(), key=lambda v: (v.name or "").lower())
        user = getattr(request.state, "user", None)
        logger.debug(
            "Serving %d vessels (cache) to user %s",
            len(vessels),
            getattr(user, "email", "unknown") if user else "unknown",
        )
        response.headers["Cache-Control"] = _VESSEL_CACHE_CONTROL
        return [
            {
                "id": str(v.id),
                "name": v.name,
                "vessel_type": v.vessel_type,
                "vessel_class": v.vessel_class,
            }
            for v in vessels
        ]

    @app.get("/api/vessel-types")
    @limiter.limit("60/minute")
    async def list_vessel_types(
        request: Request,
        response: Response,
        client: SupabaseClient = Depends(get_supabase_client),
    ):
        """Get distinct vessel types for filter dropdown."""
        from ..processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        await cache.ensure_loaded(client)
        types = sorted({v.vessel_type for v in cache.list_all() if v.vessel_type})
        response.headers["Cache-Control"] = _VESSEL_CACHE_CONTROL
        return types

    @app.get("/api/vessel-classes")
    @limiter.limit("60/minute")
    async def list_vessel_classes(
        request: Request,
        response: Response,
        client: SupabaseClient = Depends(get_supabase_client),
    ):
        """Get distinct vessel classes for filter dropdown."""
        from ..processing.entity_cache import get_vessel_cache

        cache = get_vessel_cache()
        await cache.ensure_loaded(client)
        classes = sorted({v.vessel_class for v in cache.list_all() if v.vessel_class})
        response.headers["Cache-Control"] = _VESSEL_CACHE_CONTROL
        return classes

    # Include routers under /api prefix to avoid collision with frontend routes
    app.include_router(conversations_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api/feedback")
    app.include_router(streaming_router, prefix="/api")

    # NOTE: Static frontend is mounted in lifespan() AFTER the agent endpoint
    # is registered to ensure proper route priority

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "mtss.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
