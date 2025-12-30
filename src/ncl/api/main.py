"""FastAPI application with LangGraph Agent integration.

This server exposes the LangGraph agent endpoint with defense-in-depth authentication.
JWT tokens are validated both at the CopilotKit runtime (Node.js) and here.
"""

from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_settings
from ..storage.supabase_client import SupabaseClient
from .agent import agent_graph
from .middleware.auth import SupabaseJWTBearer

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware for defense-in-depth security."""

    # Paths that don't require authentication
    PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        """Validate JWT token for protected routes."""
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
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
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="NCL Email RAG API",
        description="LangGraph Agent for email document Q&A with streaming progress",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware - allow requests from frontend and runtime
    # In production, this should be restricted appropriately
    origins = settings.cors_origins.split(",") if settings.cors_origins else []
    # Allow frontend (Next.js on 3000, legacy Vite on 5173)
    for origin in ["http://localhost:3000", "http://localhost:5173"]:
        if origin not in origins:
            origins.append(origin)
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

    # Health check endpoint with rate limiting
    @app.get("/health")
    @limiter.limit("60/minute")
    async def health_check(request: Request):
        return {"status": "healthy", "service": "ncl-api"}

    # Archive file serving endpoint
    # Uses same JWT auth as all other endpoints (via AuthMiddleware)
    @app.get("/archive/{file_path:path}")
    @limiter.limit("100/minute")
    async def serve_archive(request: Request, file_path: str):
        """Serve archive files (markdown previews and original downloads).

        Security measures:
        - Uses same Supabase JWT auth as UI endpoints (via AuthMiddleware)
        - Path validation to prevent traversal attacks (rejects '..', absolute paths)
        - Validates resolved path stays within ARCHIVE_DIR
        - Rate limited to prevent abuse

        Args:
            request: FastAPI request object (user available in request.state.user)
            file_path: Relative path within the archive directory

        Returns:
            FileResponse for the requested file

        Raises:
            HTTPException: 400 for invalid path, 404 for file not found
        """
        # Security: Reject paths with traversal attempts or absolute paths
        if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
            logger.warning(
                "Archive path traversal attempt: %s from user %s",
                file_path,
                getattr(request.state, "user", {}).get("email", "unknown"),
            )
            raise HTTPException(status_code=400, detail="Invalid path")

        # Resolve the full path
        archive_dir = settings.archive_dir.resolve()
        requested_path = (archive_dir / file_path).resolve()

        # Security: Ensure the resolved path is within archive_dir
        try:
            requested_path.relative_to(archive_dir)
        except ValueError:
            logger.warning(
                "Archive path escape attempt: %s resolved to %s from user %s",
                file_path,
                requested_path,
                getattr(request.state, "user", {}).get("email", "unknown"),
            )
            raise HTTPException(status_code=400, detail="Invalid path")

        # Check if file exists
        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not requested_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(requested_path))
        if content_type is None:
            # Default to text/plain for markdown, octet-stream for others
            if requested_path.suffix.lower() == ".md":
                content_type = "text/markdown; charset=utf-8"
            else:
                content_type = "application/octet-stream"

        logger.debug(
            "Serving archive file: %s to user %s",
            file_path,
            getattr(request.state, "user", {}).get("email", "unknown"),
        )

        return FileResponse(
            path=requested_path,
            media_type=content_type,
            filename=requested_path.name,
        )

    # Citation details endpoint for fetching source content
    @app.get("/citations/{chunk_id}")
    @limiter.limit("100/minute")
    async def get_citation(request: Request, chunk_id: str):
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
        import re

        # Validate chunk_id format (hex string, defense-in-depth)
        if not re.match(r"^[a-f0-9]+$", chunk_id, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Invalid chunk ID format")

        # Fetch chunk from database (synchronous call, no pool needed)
        client = SupabaseClient()
        chunk = client.get_chunk_by_id(chunk_id)

        if not chunk:
            raise HTTPException(status_code=404, detail="Citation not found")

        # Fetch markdown content if archive exists
        content = None
        if chunk.archive_browse_uri:
            # Strip /archive/ prefix if present (URI is for web, not filesystem)
            relative_path = chunk.archive_browse_uri
            if relative_path.startswith("/archive/"):
                relative_path = relative_path[len("/archive/"):]
            archive_path = settings.archive_dir / relative_path
            if archive_path.exists():
                try:
                    content = archive_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to read archive content: %s", e)

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
            "content": content,
        }

    # Add LangGraph agent endpoint via AG-UI protocol
    add_langgraph_fastapi_endpoint(
        app=app,
        agent=LangGraphAGUIAgent(
            name="default",
            description="NCL Email RAG Agent for document Q&A",
            graph=agent_graph,
        ),
        path="/copilotkit",
    )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ncl.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
