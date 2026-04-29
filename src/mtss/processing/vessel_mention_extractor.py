"""Shared vessel-mention extractor — regex + noise filters.

Single source of truth for the candidate patterns used by:
  - ``scripts/audit_vessel_mentions.py`` (offline audit reports)
  - ``scripts/annotate_unknown_vessel_mentions.py`` (DB-side annotation)
  - ``mtss validate ingest`` (read-only validation check #36)

Keeping the regex here means a future audit/annotate update only touches
one file. The matcher itself (case-insensitive name + alias resolution)
lives in :class:`mtss.processing.vessel_matcher.VesselMatcher` — this
module only deals with surfacing candidate strings from raw text.
"""

from __future__ import annotations

import re
from typing import Iterable, Set

# Candidate regexes. Uppercase input. Order matters only for de-dupe.
CANDIDATE_PATTERNS = [
    re.compile(r"\bMARAN\s+([A-Z]{3,})\b"),
    re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})?)\s+VOYAGER\b"),
    re.compile(r"\bM[\.]?[TV][\.]?\s+([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\b"),
    re.compile(r"\b([A-Z]+\s+[A-Z]\.\s+ANGELICOUSSIS)\b"),
]

# Tokens that trail MARAN but name the org/department, not a vessel.
MARAN_NON_VESSEL_SECOND_TOKENS = frozenset({
    "TANKERS", "SHIP", "GAS", "SUPT", "DRY", "TSS", "CORP",
    "FLEET", "FINANCE", "OFFICE", "HQ", "GROUP", "TANKER",
})

# Recognised "<CITY> VOYAGER" prefixes — kept on the candidate list.
CITY_PREFIX_VOYAGERS = frozenset({
    "RICHMOND", "EL SEGUNDO", "PASCAGOULA", "SAN RAMON",
    "HOUSTON", "LONDON", "SINGAPORE", "GLASGOW",
})

# Confirmed-noise candidates from prior audits. Suppressed from output.
HARDCODED_NOISE = frozenset({
    "MARAN MARAN",   # duplication artifact in some forwarded headers
    "MARAN TANKER",  # company, not a vessel
    "SAP FLOW POST",
})


_WS_RUN = re.compile(r"\s+")


def _normalise(token: str) -> str:
    """Collapse internal whitespace runs (newlines included) into a single space.

    The MT/MV pattern's ``\\s+`` separator matches newlines, so an email like
    ``"M.T. MARAN ORPHEUS\\n\\n\\nKALISPERA..."`` would otherwise leak embedded
    newlines into the candidate string and explode the unique-mention count.
    """
    return _WS_RUN.sub(" ", token).strip()


def extract_vessel_mentions(text: str) -> Set[str]:
    """Surface vessel-like name candidates from raw text.

    Returns uppercase canonical-form mentions (e.g. ``"MARAN CANOPUS"``).
    The output is unfiltered against the canonical register — callers
    decide what counts as known vs. unknown.
    """
    if not text:
        return set()
    upper = text.upper()
    found: Set[str] = set()

    for m in CANDIDATE_PATTERNS[0].finditer(upper):
        second = _normalise(m.group(1))
        if second in MARAN_NON_VESSEL_SECOND_TOKENS:
            continue
        found.add(f"MARAN {second}")

    for m in CANDIDATE_PATTERNS[1].finditer(upper):
        prefix = _normalise(m.group(1))
        if prefix in CITY_PREFIX_VOYAGERS:
            found.add(f"{prefix} VOYAGER")
        elif len(prefix.split()) == 1 and len(prefix) >= 4:
            found.add(f"{prefix} VOYAGER")

    for m in CANDIDATE_PATTERNS[2].finditer(upper):
        found.add(_normalise(m.group(1)))

    for m in CANDIDATE_PATTERNS[3].finditer(upper):
        found.add(_normalise(m.group(1)))

    return found - HARDCODED_NOISE


def build_canonical_name_set(vessel_names: Iterable[str]) -> Set[str]:
    """Normalize a sequence of canonical vessel names + aliases to uppercase."""
    return {n.strip().upper() for n in vessel_names if n and n.strip()}
