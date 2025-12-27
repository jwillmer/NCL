"""Supabase JWT authentication middleware."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.request import urlopen

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from ...config import get_settings


class UserPayload(BaseModel):
    """Decoded JWT user payload."""

    sub: str  # User ID
    email: Optional[str] = None
    role: Optional[str] = None
    aud: Optional[str] = None


class JWKSClient:
    """Simple JWKS client with caching."""

    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url
        self._keys: Dict[str, Any] = {}

    def get_key(self, kid: str) -> Optional[Dict[str, Any]]:
        """Get key by ID, fetching if necessary."""
        if kid not in self._keys:
            self._refresh_keys()
        return self._keys.get(kid)

    def _refresh_keys(self):
        """Fetch keys from JWKS endpoint."""
        try:
            with urlopen(self.jwks_url) as response:
                jwks = json.loads(response.read())
                for key in jwks.get("keys", []):
                    self._keys[key["kid"]] = key
        except Exception:
            pass


# Global JWKS client instance
_jwks_client: Optional[JWKSClient] = None


def get_jwks_client() -> JWKSClient:
    """Get or create global JWKS client."""
    global _jwks_client
    if _jwks_client is None:
        settings = get_settings()
        # Construct JWKS URL from Supabase URL
        # e.g. https://<project>.supabase.co/auth/v1/.well-known/jwks.json
        base_url = settings.supabase_url.rstrip("/")
        jwks_url = f"{base_url}/auth/v1/.well-known/jwks.json"
        _jwks_client = JWKSClient(jwks_url)
    return _jwks_client


class SupabaseJWTBearer(HTTPBearer):
    """JWT Bearer authentication using Supabase."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        settings = get_settings()
        self.jwt_secret = settings.supabase_jwt_secret
        # We don't raise error for missing secret if we might use JWKS
        # but for now let's keep it simple and assume secret is there for HS256

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
        """Verify JWT using Supabase's JWT secret (HS256) or JWKS (ES256)."""
        try:
            # Check header to determine algorithm
            header = jwt.get_unverified_header(token)
            alg = header.get("alg")

            if alg == "ES256":
                # Asymmetric verification using JWKS
                kid = header.get("kid")
                if not kid:
                    return None
                
                client = get_jwks_client()
                key = client.get_key(kid)
                if not key:
                    return None
                    
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=["ES256"],
                    audience="authenticated",
                )
            else:
                # Fallback to Symmetric (HS256) with secret
                if not self.jwt_secret:
                    return None
                    
                payload = jwt.decode(
                    token,
                    self.jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )

            return UserPayload(
                sub=payload.get("sub", ""),
                email=payload.get("email"),
                role=payload.get("role"),
                aud=payload.get("aud"),
            )
        except (JWTError, Exception):
            return None


async def get_current_user(
    user: UserPayload = Depends(SupabaseJWTBearer())
) -> UserPayload:
    """Dependency to get the current authenticated user."""
    return user
