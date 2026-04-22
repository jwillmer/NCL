"""Tests for SPAStaticFiles cache-header behaviour.

Each path category must get the right Cache-Control:
- ``assets/*`` (Vite-hashed files) → ``public, max-age=31536000, immutable``
- ``index.html`` and the SPA fallback → ``no-cache``
- ``sw.js`` / ``registerSW.js`` → ``no-cache`` + ``Service-Worker-Allowed: /``
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def spa_dist(tmp_path: Path) -> Path:
    """Build a tiny fake dist/ with an index.html, an asset, and sw.js."""
    (tmp_path / "index.html").write_text("<html><body>root</body></html>", encoding="utf-8")
    (tmp_path / "sw.js").write_text("// service worker bootstrap", encoding="utf-8")
    (tmp_path / "registerSW.js").write_text("// register", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.abc123.js").write_text("console.log('hi')", encoding="utf-8")
    (assets / "app.abc123.css").write_text("body{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def spa_app(spa_dist: Path) -> FastAPI:
    """Minimal FastAPI app mounting SPAStaticFiles at /."""
    from mtss.api.main import SPAStaticFiles

    app = FastAPI()
    app.mount("/", SPAStaticFiles(directory=spa_dist, html=True), name="frontend")
    return app


@pytest.mark.asyncio
async def test_assets_get_immutable_cache_control(spa_app: FastAPI):
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/assets/app.abc123.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=31536000, immutable"


@pytest.mark.asyncio
async def test_assets_css_also_immutable(spa_app: FastAPI):
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/assets/app.abc123.css")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=31536000, immutable"


@pytest.mark.asyncio
async def test_root_index_gets_no_cache(spa_app: FastAPI):
    """Serving / (index.html) must revalidate on every load."""
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"


@pytest.mark.asyncio
async def test_spa_fallback_returns_no_cache_index(spa_app: FastAPI):
    """An unknown path falls back to index.html with no-cache."""
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/conversations/some-deep-link")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"


@pytest.mark.asyncio
async def test_service_worker_headers(spa_app: FastAPI):
    """/sw.js must be no-cache + Service-Worker-Allowed: /."""
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sw.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"
    assert response.headers.get("service-worker-allowed") == "/"


@pytest.mark.asyncio
async def test_register_sw_headers(spa_app: FastAPI):
    transport = ASGITransport(app=spa_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/registerSW.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"
    assert response.headers.get("service-worker-allowed") == "/"
