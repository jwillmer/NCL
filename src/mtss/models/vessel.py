"""Vessel data models for vessel filtering."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Vessel(BaseModel):
    """Represents a vessel from the vessel register."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    imo: Optional[str] = None  # 7-digit IMO number
    vessel_type: Optional[str] = None  # VLCC, Suezmax, Aframax, etc.
    dwt: Optional[int] = None  # Deadweight tonnage
    aliases: List[str] = Field(default_factory=list)  # Alternative names for matching
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class VesselSummary(BaseModel):
    """Minimal vessel info for dropdown lists."""

    id: UUID
    name: str
    imo: Optional[str] = None
    vessel_type: Optional[str] = None
