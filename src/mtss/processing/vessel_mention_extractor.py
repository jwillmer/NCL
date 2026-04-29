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

# Confirmed-noise candidates. Suppressed from output.
#
# Sourced from ``reports/vessel-mismatch/extractor_noise.txt`` (regenerate
# via ``uv run python reports/vessel-mismatch/generate.py``). Includes every
# raw mention starting with "MARAN " in that report — both 2-token org names
# (MARAN HAS, MARAN VESSELS, MARAN TEAM, ...) and 3-token canonical-prefix
# concatenations leaked in via the M.[TV] pattern (MARAN APOLLO DURING,
# MARAN PLATO TEL, ...). The canonical-prefix-concat case is also caught by
# :func:`filter_canonical_concats`, but listing them here is belt-and-
# suspenders and makes the suppression visible to maintainers.
HARDCODED_NOISE = frozenset({
    # Pre-existing entries
    "MARAN MARAN",        # duplication artifact in some forwarded headers
    "MARAN TANKER",       # company, not a vessel
    "SAP FLOW POST",
    # 2-token MARAN org / business-phrase noise
    "MARAN VESSELS",
    "MARAN HAS",
    "MARAN SUPERINTENDENT",
    "MARAN TEAM",
    "MARAN ASIA",
    "MARAN VESSEL",
    "MARAN TRANSPORTER",
    "MARAN SHIPS",
    "MARAN WILL",
    "MARAN ORDER",
    # 3-token canonical-prefix concatenations from extractor_noise.txt
    # (these are also caught by filter_canonical_concats, but listing
    # them keeps the noise budget explicit)
    "MARAN ORPHEUS GOOD",
    "MARAN ORPHEUS KALISPERA",
    "MARAN PLATO TEL",
    "MARAN HERCULES CAPT",
    "MARAN LIBRA SHIPS",
    "MARAN HELEN ETA",
    "MARAN CANOPUS DESLOPPING",
    "MARAN POSEIDON AND",
    "MARAN APOLLO DURING",
    "MARAN HELEN FROM",
    "MARAN CLEO FROM",
    "MARAN HELIOS SHIP",
    "MARAN HELEN FRIDAY",
    "MARAN ORPHEUS FROM",
    "MARAN MIRA FROM",
    "MARAN LUPUS FROM",
    "MARAN PHOEBE FROM",
    "MARAN HERMIONE FROM",
    "MARAN PLATO CREW",
    "MARAN LYRA ETA",
    "MARAN THETIS TECHNICAL",
    "MARAN TAURUS DUE",
    "MARAN LYRA SERVICEREPORT",
    "MARAN MIRA CODES",
    "MARAN APOLLO SHIP",
    "MARAN PLATO FROM",
    "MARAN ARTEMIS FROM",
    "MARAN AJAX FROM",
    "MARAN ARETE FROM",
    "MARAN MARS FROM",
    "MARAN LIBRA FROM",
    "MARAN LEO AND",
    "MARAN PENELOPE SUPPLY",
    "MARAN TAURUS SHIP",
    "MARAN HERCULES SHIP",
    "MARAN ANTARES SHIPS",
    "MARAN ATHENA FROM",
    "MARAN SOLON FROM",
    "MARAN ANTIOPE FROM",
    "MARAN POSEIDON FROM",
    "MARAN ANTARES FROM",
    "MARAN ORPHEUS MAINTENANCE",
    "MARAN APOLLO FROM",
    "MARAN ASPASIA FROM",
    "MARAN CAPRICORN FROM",
    "MARAN CANOPUS FOR",
    "MARAN LEO FROM",
    "MARAN PLATO NON",
    "MARAN DIONE FOR",
    "MARAN ARIADNE FROM",
    "MARAN ARES FROM",
    "MARAN PYTHIA FROM",
    "MARAN ATALANTA FROM",
    "MARAN HERMES FROM",
    "MARAN THALEIA FROM",
    "MARAN LUPUS AND",
    "MARAN ARETE ARRIVED",
    "MARAN DANAE SINGAPORE",
    "MARAN DIONE FREEPORT",
    "MARAN CANOPUS SHIP",
    "MARAN CANOPUS SHANGHAI",
    "MARAN PLATO WILL",
    "MARAN PENELOPE PIRAEUS",
    "MARAN LIBRA SHIP",
    "MARAN HELIOS PIRAEUS",
    "MARAN PENELOPE FROM",
    "MARAN ATALANTA MNC",
    "MARAN ATALANTA SHIP",
    "MARAN CANOPUS ABT",
    "MARAN DANAE FREEPORT",
    "MARAN LUPUS MARAN",
    "MARAN DIONE AND",
    "MARAN HELEN DATE",
    "MARAN THETIS COMING",
    "MARAN LUPUS SHIP",
    "MARAN ARCTURUS SHIP",
    "MARAN CANOPUS MAXFREIGHT",
    "MARAN CANOPUS FROM",
    "MARAN HOMER FROM",
    "MARAN HERCULES FROM",
    "MARAN LYNX FROM",
    "MARAN HELIOS FROM",
    "MARAN MARS IMO",
    "MARAN ARCTURUS WAS",
    "MARAN ATLAS CONSIGNED",
    "MARAN PLATO INSTRUCTOR",
    "MARAN PLATO RECORD",
    "MARAN ANTARES MASTER",
    "MARAN ORPHEUS SWIFT",
    "MARAN ORPHEUS FORWARDING",
    "MARAN ARCTURUS PLEASE",
    "MARAN CLEO ETA",
    "MARAN PHOEBE DUE",
    "MARAN ARTEMIS IMO",
    "MARAN HELEN SITE",
    "MARAN THETIS FROM",
})

# Tokens that, when present anywhere in an M.[TV]-pattern capture, mean
# the match is a business-phrase / boilerplate, not a vessel name.
#
# Sourced from ``reports/vessel-mismatch/extractor_noise.txt`` — see
# entries like "PORT LNG TANK", "BEST REGARDS", "TOTAL LNG CONSUMPTION",
# "CALCULATED WEIGHT", "LOW SULPHUR", "SUMMER DRAFT", "HSFO PRICE",
# "ADDITIONAL PLEASE NOTE", "SEA WATER" — each of these decomposes into
# tokens that never appear in a real vessel name.
_BIZ_PHRASE_BLOCKLIST = frozenset({
    "PRICE", "WEIGHT", "CONSUMPTION", "SULPHUR", "DRAFT", "REGARDS", "TIME",
    "TANK", "LNG", "HSFO", "SUMMER", "WINTER", "TOTAL", "CALCULATED", "BEST",
    "LOW", "PORT", "NOON", "FUEL", "STOCK", "BUNKER", "WATER", "SEA", "DAY",
    "NOTE", "ADDITIONAL", "PLEASE",
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
        candidate = _normalise(m.group(1))
        # Reject business-phrase boilerplate ("MT TOTAL LNG CONSUMPTION",
        # "MT BEST REGARDS", "MV PORT LNG TANK", ...).
        tokens = candidate.split()
        if any(tok in _BIZ_PHRASE_BLOCKLIST for tok in tokens):
            continue
        found.add(candidate)

    for m in CANDIDATE_PATTERNS[3].finditer(upper):
        found.add(_normalise(m.group(1)))

    return found - HARDCODED_NOISE


def build_canonical_name_set(vessel_names: Iterable[str]) -> Set[str]:
    """Normalize a sequence of canonical vessel names + aliases to uppercase."""
    return {n.strip().upper() for n in vessel_names if n and n.strip()}


def filter_canonical_concats(
    mentions: Set[str], canonical_two_token: Set[str]
) -> Set[str]:
    """Drop mentions whose first two tokens form a canonical 2-token vessel name.

    These are concatenation artifacts — typically from the M.[TV] regex
    matching ``MV MARAN APOLLO DURING ITS VOYAGE`` or similar, where the
    real vessel name is the 2-token prefix and the trailing tokens are
    just sentence-continuation noise.

    ``canonical_two_token`` is the subset of the canonical vessel register
    consisting of names with exactly two whitespace-separated tokens
    (e.g. ``MARAN APOLLO``, ``RICHMOND VOYAGER``). Callers build it once
    per validation run and pass it through.
    """
    if not canonical_two_token:
        return mentions
    out: Set[str] = set()
    for m in mentions:
        parts = m.split()
        if len(parts) >= 3 and " ".join(parts[:2]) in canonical_two_token:
            continue
        out.add(m)
    return out
