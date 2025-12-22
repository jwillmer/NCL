"""FastAPI application with CopilotKit integration."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from copilotkit import CopilotKitRemoteEndpoint
from copilotkit.integrations.fastapi import add_fastapi_endpoint

from ..config import get_settings
from .agent import NCLEmailAgent
from .middleware.auth import SupabaseJWTBearer

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global agent instance
_agent: NCLEmailAgent | None = None

# JWT auth instance (created once at module load)
_jwt_auth: SupabaseJWTBearer | None = None


def get_jwt_auth() -> SupabaseJWTBearer:
    """Get or create the JWT auth instance."""
    global _jwt_auth
    if _jwt_auth is None:
        _jwt_auth = SupabaseJWTBearer()
    return _jwt_auth


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate requests to protected endpoints."""

    def __init__(self, app, protected_paths: list[str]):
        super().__init__(app)
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next):
        # Check if path requires auth
        if any(request.url.path.startswith(p) for p in self.protected_paths):
            try:
                auth = get_jwt_auth()
                await auth(request)
            except Exception:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing authentication"},
                )
        return await call_next(request)


def get_agent() -> NCLEmailAgent:
    """Get the global agent instance."""
    global _agent
    if _agent is None:
        _agent = NCLEmailAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    yield
    # Cleanup on shutdown
    agent = get_agent()
    await agent.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="NCL Email RAG API",
        description="CopilotKit-powered API for email document Q&A",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware (must be added before auth middleware)
    origins = settings.cors_origins.split(",") if settings.cors_origins else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth middleware for protected endpoints
    app.add_middleware(AuthMiddleware, protected_paths=["/copilotkit", "/rag-state"])

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "ncl-api"}

    # System prompt endpoint (for frontend CopilotKit provider)
    @app.get("/system-prompt")
    async def get_system_prompt():
        """Return the system prompt for the CopilotKit provider."""
        agent = get_agent()
        return {"prompt": agent.system_prompt}

    # RAG state endpoint (for frontend state synchronization)
    @app.get("/rag-state")
    async def get_rag_state():
        """Return the current RAG state for frontend synchronization."""
        agent = get_agent()
        return agent.state.model_dump()

    # Initialize CopilotKit SDK with agent actions
    agent = get_agent()
    sdk = CopilotKitRemoteEndpoint(actions=agent.get_actions())

    # Add CopilotKit endpoint using official integration
    # Note: Auth is handled via middleware, rate limiting via slowapi
    add_fastapi_endpoint(app, sdk, "/copilotkit")

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
