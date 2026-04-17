"""Tests for AuthMiddleware SPA-fallback bypass behavior.

The middleware must:
- Allow PUBLIC_PATHS (/health, /docs, /redoc, /openapi.json, /config.js) without auth.
- Allow /, *.html, STATIC_PREFIXES, and STATIC_EXTENSIONS without auth.
- Allow any non-/api/ path without auth (SPA fallback — React Router resolves
  client-side, so new frontend routes don't need middleware changes).
- Require JWT for every /api/* path.
- Never 401 an OPTIONS preflight.
"""

import pytest


class TestPublicPaths:
    """PUBLIC_PATHS stay reachable without auth."""

    @pytest.mark.asyncio
    async def test_health_public(self, client):
        """GET /health with no token returns 200."""
        response = await client.get("/health")
        assert response.status_code == 200


class TestApiRoutesRequireAuth:
    """/api/* routes must enforce JWT."""

    @pytest.mark.asyncio
    async def test_api_route_without_token_rejected(self, client):
        """GET /api/conversations with no Authorization header returns 401."""
        response = await client.get("/api/conversations")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_api_route_with_invalid_token_rejected(self, client):
        """GET /api/conversations with a malformed Authorization header returns 401.

        The test fixture's mock_jwt_call only accepts headers starting with
        "Bearer " — anything else resolves to no user and the middleware 401s.
        """
        response = await client.get(
            "/api/conversations",
            headers={"Authorization": "NotBearer garbage"},
        )
        assert response.status_code == 401


class TestSpaFallbackBypass:
    """Non-/api/ paths skip auth so React Router can resolve them client-side."""

    @pytest.mark.asyncio
    async def test_known_spa_route_still_allowed(self, client):
        """GET /chat with no token must not be 401.

        The TestClient doesn't mount the web/dist static directory, so a 404
        is acceptable — the important thing is that auth does not reject it.
        """
        response = await client.get("/chat")
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_known_spa_route_trailing_slash_allowed(self, client):
        """GET /conversations/ with no token must not be 401."""
        response = await client.get("/conversations/")
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_new_spa_route_allowed(self, client):
        """GET /settings (a route that was never hardcoded) must not be 401.

        This is the whole point of the fix: adding a new React Router route
        should not require touching AuthMiddleware.
        """
        response = await client.get("/settings")
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_deeply_nested_spa_route_allowed(self, client):
        """GET /admin/users/123 with no token must not be 401."""
        response = await client.get("/admin/users/123")
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_api_prefix_is_not_treated_as_spa(self, client):
        """A path starting with /api/ must still require auth (not treated as SPA)."""
        response = await client.get("/api/some-nonexistent-endpoint")
        # Either 401 (no auth) — must NOT be a silent pass-through.
        assert response.status_code == 401


class TestOptionsPreflight:
    """CORS preflight must never 401."""

    @pytest.mark.asyncio
    async def test_options_preflight_allowed(self, client):
        """OPTIONS on any path must not return 401.

        Preflights don't include auth headers; CORSMiddleware handles them.
        """
        response = await client.options(
            "/api/conversations",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code != 401
