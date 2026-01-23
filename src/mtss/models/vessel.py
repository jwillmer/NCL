"""Vessel data models for vessel filtering."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Vessel(BaseModel):
    """Represents a vessel from the vessel register."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    vessel_type: str  # VLCC, SUEZMAX, AFRAMAX
    vessel_class: str  # Canopus Class, Plato Class, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class VesselSummary(BaseModel):
    """Minimal vessel info for dropdown lists."""

    id: UUID
    name: str
    vessel_type: str
    vessel_class: str
