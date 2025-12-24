"""Supabase JWT authentication middleware."""

from __future__ import annotations

from typing import Optional

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


class SupabaseJWTBearer(HTTPBearer):
    """JWT Bearer authentication using Supabase."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        settings = get_settings()
        self.jwt_secret = settings.supabase_jwt_secret
        if not self.jwt_secret:
            raise ValueError(
                "SUPABASE_JWT_SECRET is required. "
                "Get it from Supabase Dashboard > Settings > API > JWT Secret"
            )

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
        """Verify JWT using Supabase's JWT secret."""
        try:
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
        except JWTError:
            return None


async def get_current_user(
    user: UserPayload = Depends(SupabaseJWTBearer())
) -> UserPayload:
    """Dependency to get the current authenticated user."""
    return user
