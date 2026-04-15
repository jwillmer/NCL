"""Vessel data models and CSV loading for vessel filtering."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Vessel(BaseModel):
    """Represents a vessel from the vessel register."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    vessel_type: str  # VLCC, SUEZMAX, AFRAMAX
    vessel_class: str  # Canopus Class, Plato Class, etc.
    aliases: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class VesselSummary(BaseModel):
    """Minimal vessel info for dropdown lists."""

    id: UUID
    name: str
    vessel_type: str
    vessel_class: str


def load_vessels_from_csv(csv_file: Optional[Path] = None) -> List[Vessel]:
    """Load vessels from a CSV file without any database interaction.

    Args:
        csv_file: Path to CSV file. Defaults to settings.data_dir / "vessel-list.csv".

    Returns:
        List of Vessel objects. Empty list if file not found.
    """
    if csv_file is None:
        from ..config import get_settings

        csv_file = get_settings().data_dir / "vessel-list.csv"

    if not csv_file.exists():
        logger.warning(f"Vessel CSV not found: {csv_file} — vessel tagging disabled")
        return []

    with open(csv_file, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        delimiter = ";" if ";" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)

        required = {"NAME", "TYPE", "CLASS"}
        if reader.fieldnames:
            missing = required - set(reader.fieldnames)
            if missing:
                logger.error(f"Vessel CSV missing columns: {', '.join(missing)}")
                return []

        vessels: List[Vessel] = []
        for row in reader:
            name = row.get("NAME", "").strip()
            vessel_type = row.get("TYPE", "").strip()
            vessel_class = row.get("CLASS", "").strip()
            aliases_str = (row.get("ALIASES") or "").strip()
            aliases = [a.strip() for a in aliases_str.split(",") if a.strip()] if aliases_str else []

            if not name or not vessel_type or not vessel_class:
                continue

            vessels.append(Vessel(
                name=name,
                vessel_type=vessel_type,
                vessel_class=vessel_class,
                aliases=aliases,
            ))

    return vessels
