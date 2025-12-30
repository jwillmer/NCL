"""Supabase JWT authentication middleware using PyJWT."""

from __future__ import annotations

import logging
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel

from ...config import get_settings

logger = logging.getLogger(__name__)


class UserPayload(BaseModel):
    """Decoded JWT user payload."""

    sub: str  # User ID
    email: Optional[str] = None
    role: Optional[str] = None
    aud: Optional[str] = None


# Global JWKS client instance with caching
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> PyJWKClient:
    """Get or create global JWKS client with caching."""
    global _jwks_client
    if _jwks_client is None:
        settings = get_settings()
        # Construct JWKS URL from Supabase URL
        # e.g. https://<project>.supabase.co/auth/v1/.well-known/jwks.json
        base_url = settings.supabase_url.rstrip("/")
        jwks_url = f"{base_url}/auth/v1/.well-known/jwks.json"
        logger.info("Initializing JWKS client with URL: %s", jwks_url)
        # PyJWKClient handles caching and key rotation automatically
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
    return _jwks_client


# Use HTTPBearer directly - don't subclass it to avoid __call__ override issues
_bearer_scheme = HTTPBearer(auto_error=False)


def _verify_jwt(token: str) -> Optional[UserPayload]:
    """Verify JWT using Supabase's JWKS (ES256) or JWT secret (HS256)."""
    settings = get_settings()
    try:
        # Get unverified header to determine algorithm
        header = jwt.get_unverified_header(token)
        alg = header.get("alg")

        if alg == "ES256":
            # Use PyJWKClient for ES256 (asymmetric) verification
            jwks_client = get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        elif alg == "HS256":
            # Use JWT secret for HS256 (symmetric) verification
            jwt_secret = settings.supabase_jwt_secret
            if not jwt_secret:
                logger.warning("HS256 token but no JWT secret configured")
                return None

            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        else:
            logger.warning("Unsupported JWT algorithm: %s", alg)
            return None

        logger.debug(
            "JWT verified for user: %s",
            payload.get("email", payload.get("sub")),
        )
        return UserPayload(
            sub=payload.get("sub", ""),
            email=payload.get("email"),
            role=payload.get("role"),
            aud=payload.get("aud"),
        )
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        return None
    except jwt.InvalidAudienceError:
        logger.warning("JWT audience claim is invalid")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("JWT verification failed: %s", str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error during JWT verification: %s", str(e))
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> UserPayload:
    """Dependency to get the current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _verify_jwt(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Keep SupabaseJWTBearer for backwards compatibility with AuthMiddleware in main.py
class SupabaseJWTBearer:
    """JWT Bearer authentication for use in middleware.

    This class is used by AuthMiddleware which passes the Request directly.
    For route dependencies, use get_current_user instead.
    """

    def __init__(self, auto_error: bool = True):
        self.auto_error = auto_error
        self._settings = get_settings()

    async def __call__(self, request) -> Optional[UserPayload]:
        """Validate JWT from request headers."""
        from starlette.requests import Request

        if not isinstance(request, Request):
            return None

        auth_header = request.headers.get("authorization", "")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        token = parts[1]
        return _verify_jwt(token)
