"""Topic models for categorization and filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Topic(BaseModel):
    """Full topic record from database."""

    id: UUID = Field(default_factory=uuid4)
    name: str  # Canonical lowercase
    display_name: str  # User-friendly
    description: Optional[str] = None
    embedding: Optional[List[float]] = None
    chunk_count: int = 0
    document_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("embedding", mode="before")
    @classmethod
    def parse_embedding(cls, v: Any) -> Optional[List[float]]:
        """Parse embedding from string if needed (pgvector returns string)."""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # pgvector format: "[0.1,0.2,...]"
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return None


class TopicSummary(BaseModel):
    """Minimal topic info for lists and dropdowns."""

    id: UUID
    name: str
    display_name: str
    chunk_count: int = 0


@dataclass
class ExtractedTopic:
    """Topic extracted from content by LLM."""

    name: str
    description: Optional[str] = None
