# =============================================================================
# MTSS Dockerfile
# Multi-stage build: Frontend (Next.js) + Backend (FastAPI) in single image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build frontend (Next.js static export)
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/web

# Copy package files first for better caching
COPY web/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY web/ ./

# Build argument for Git SHA (injected into frontend at build time)
ARG GIT_SHA=development
ENV NEXT_PUBLIC_GIT_SHA=${GIT_SHA}

# Build static export
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Build backend (Python with uv)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS backend-builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python ".[api]"

# -----------------------------------------------------------------------------
# Stage 3: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Build argument for Git SHA (available at runtime)
ARG GIT_SHA=development
ENV GIT_SHA=${GIT_SHA}

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (security best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy Python environment from builder
COPY --from=backend-builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy backend source
COPY src/ ./src/
COPY pyproject.toml ./

# Copy frontend static build
COPY --from=frontend-builder /app/web/out ./web/out

# Set ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "mtss.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
