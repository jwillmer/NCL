"""Vessel matcher for tagging documents with vessel IDs during ingest."""

from __future__ import annotations

import re
from typing import Dict, List, Set
from uuid import UUID

from ..models.vessel import Vessel


class VesselMatcher:
    """Matches vessel names/aliases in document text.

    Builds a case-insensitive lookup from vessel names and aliases to vessel IDs.
    Used during ingest to tag documents with the vessels they reference.
    """

    def __init__(self, vessels: List[Vessel]):
        """Initialize matcher with vessel registry.

        Args:
            vessels: List of vessels with names and aliases.
        """
        self.lookup: Dict[str, UUID] = {}
        self._build_lookup(vessels)

    def _build_lookup(self, vessels: List[Vessel]) -> None:
        """Build case-insensitive lookup from names and aliases to IDs."""
        for vessel in vessels:
            # Add primary name (lowercased)
            name_lower = vessel.name.lower().strip()
            if name_lower:
                self.lookup[name_lower] = vessel.id

            # Add aliases (lowercased)
            for alias in vessel.aliases:
                alias_lower = alias.lower().strip()
                if alias_lower:
                    self.lookup[alias_lower] = vessel.id

    def find_vessels(self, text: str) -> Set[UUID]:
        """Find all vessel IDs mentioned in text.

        Uses word boundary matching to avoid false positives.

        Args:
            text: Text to search for vessel names.

        Returns:
            Set of vessel UUIDs found in the text.
        """
        if not text or not self.lookup:
            return set()

        found: Set[UUID] = set()
        text_lower = text.lower()

        for name, vessel_id in self.lookup.items():
            # Use word boundary matching to avoid partial matches
            # e.g., "MARAN" should not match in "AMARANTO"
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text_lower):
                found.add(vessel_id)

        return found

    def find_vessels_in_email(
        self,
        subject: str | None,
        body: str | None,
    ) -> Set[UUID]:
        """Find all vessel IDs mentioned in email subject and body.

        Args:
            subject: Email subject line.
            body: Email body text.

        Returns:
            Set of vessel UUIDs found in the email.
        """
        found: Set[UUID] = set()

        if subject:
            found.update(self.find_vessels(subject))
        if body:
            found.update(self.find_vessels(body))

        return found

    @property
    def vessel_count(self) -> int:
        """Number of unique vessels in the matcher."""
        return len(set(self.lookup.values()))

    @property
    def name_count(self) -> int:
        """Number of names/aliases in the lookup."""
        return len(self.lookup)
