"""Tests for API security: auth middleware, security headers, path traversal, input validation."""

import pytest


class TestAuthentication:
    """Verify auth middleware blocks unauthenticated requests to protected routes."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(self, client):
        """Protected endpoints must reject requests without auth."""
        response = await client.get("/api/conversations")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_public_health_skips_auth(self, client):
        """GET /health should work without auth."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_public_config_js_skips_auth(self, client):
        """GET /config.js should work without auth."""
        response = await client.get("/config.js")
        assert response.status_code == 200
        assert "application/javascript" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_authenticated_request_succeeds(self, client, auth_headers):
        """Authenticated request to protected endpoint should not return 401."""
        response = await client.get("/api/conversations", headers=auth_headers)
        # Should be 200 (or other non-401), not 401
        assert response.status_code != 401


class TestSecurityHeaders:
    """Verify security headers are present on responses."""

    @pytest.mark.asyncio
    async def test_security_headers_on_public_endpoint(self, client):
        """Security headers should be set on all responses including public ones."""
        response = await client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    @pytest.mark.asyncio
    async def test_csp_header_present(self, client):
        """Content-Security-Policy header must be set."""
        response = await client.get("/health")
        csp = response.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    @pytest.mark.asyncio
    async def test_csp_no_unsafe_eval(self, client):
        """CSP must not allow unsafe-eval for scripts."""
        response = await client.get("/health")
        csp = response.headers.get("Content-Security-Policy", "")
        assert "'unsafe-eval'" not in csp


class TestPathTraversal:
    """Verify archive endpoint blocks path traversal attempts."""

    @pytest.mark.asyncio
    async def test_dotdot_rejected(self, client, auth_headers):
        # URL-encode the dots to prevent client-side path normalization
        response = await client.get("/api/archive/docs/..%2F..%2Fetc/passwd", headers=auth_headers)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_backslash_rejected(self, client, auth_headers):
        response = await client.get("/api/archive/\\windows\\system32", headers=auth_headers)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, client, auth_headers):
        response = await client.get("/api/archive//etc/shadow", headers=auth_headers)
        # Leading slash after /api/archive/ should be rejected
        assert response.status_code == 400


class TestCitationValidation:
    """Verify citation endpoint validates chunk_id format."""

    @pytest.mark.asyncio
    async def test_non_hex_rejected(self, client, auth_headers):
        response = await client.get("/api/citations/not_hex_value", headers=auth_headers)
        assert response.status_code == 400
        assert "Invalid chunk ID format" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_wrong_length_rejected(self, client, auth_headers):
        response = await client.get("/api/citations/abc123", headers=auth_headers)
        assert response.status_code == 400
        assert "length" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_valid_format_passes_validation(self, client, auth_headers):
        """Valid hex chunk_id should pass validation (returns 404 since mock DB is empty)."""
        response = await client.get("/api/citations/aabbccddee12", headers=auth_headers)
        # 404 means validation passed but chunk not found (expected with mock)
        assert response.status_code == 404
