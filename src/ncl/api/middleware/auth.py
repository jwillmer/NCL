"""Supabase JWT authentication middleware using PyJWT."""

from __future__ import annotations

import logging
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
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


class SupabaseJWTBearer(HTTPBearer):
    """JWT Bearer authentication using Supabase with PyJWT."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        settings = get_settings()
        self.jwt_secret = settings.supabase_jwt_secret

    async def __call__(self, request: Request) -> Optional[UserPayload]:
        credentials: Optional[HTTPAuthorizationCredentials] = await super().__call__(
            request
        )

        if not credentials:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authorization credentials",
                )
            return None

        if credentials.scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme",
                )
            return None

        payload = self._verify_jwt(credentials.credentials)
        if not payload:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or expired token",
                )
            return None

        return payload

    def _verify_jwt(self, token: str) -> Optional[UserPayload]:
        """Verify JWT using Supabase's JWKS (ES256) or JWT secret (HS256)."""
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
                if not self.jwt_secret:
                    logger.warning("HS256 token but no JWT secret configured")
                    return None

                payload = jwt.decode(
                    token,
                    self.jwt_secret,
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
    user: UserPayload = Depends(SupabaseJWTBearer()),
) -> UserPayload:
    """Dependency to get the current authenticated user."""
    return user
