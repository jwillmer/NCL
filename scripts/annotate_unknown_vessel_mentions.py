"""Stamp `metadata.unknown_vessel_names` on chunks where the source text
mentions a vessel-like name that doesn't resolve to the register.

Why this is a separate pass from the retag:
  - `vessel_ids` is the authoritative link to the register (used by filters).
    Only resolvable names go there — that's what the retag already writes.
  - When a corpus email mentions a vessel we do NOT yet recognise, dropping
    the name silently means we'd need to re-scan the full corpus later to
    find it again. Instead, preserve the raw name on the chunk so a future
    register update can pick up the signal cheaply.

Scope:
  - Candidate extractor mirrors scripts/audit_vessel_mentions.py — MARAN X,
    <CITY> VOYAGER, MT/MV NAME, ANGELICOUSSIS family.
  - For each root doc tree, walk the same text sources (documents.title /
    file_name / email_subject). Split candidates into:
      resolvable   → already in vessel_ids, ignore
      noise        → hardcoded blocklist, ignore
      preserve     → write to unknown_vessel_names
  - Writes touch chunks for the root doc + all its descendants.

Defaults to --dry-run. Pass --apply to write.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/annotate_unknown_vessel_mentions.py --dry-run
    uv run python scripts/annotate_unknown_vessel_mentions.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from typing import Iterable, List, Set
from uuid import UUID

from mtss.processing.vessel_matcher import VesselMatcher
from mtss.storage.supabase_client import SupabaseClient


# Keep in sync with scripts/audit_vessel_mentions.py — same regexes.
CANDIDATE_PATTERNS = [
    re.compile(r"\bMARAN\s+([A-Z]{3,})\b"),
    re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})?)\s+VOYAGER\b"),
    re.compile(r"\bM[\.]?[TV][\.]?\s+([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\b"),
    re.compile(r"\b([A-Z]+\s+[A-Z]\.\s+ANGELICOUSSIS)\b"),
]

# Tokens that trail MARAN but name the org, not a vessel. Never preserved.
MARAN_NON_VESSEL_SECOND_TOKENS = {
    "TANKERS", "SHIP", "GAS", "SUPT", "DRY", "TSS", "CORP",
    "FLEET", "FINANCE", "OFFICE", "HQ", "GROUP", "TANKER",
}

CITY_PREFIX_VOYAGERS = {"RICHMOND", "EL SEGUNDO", "PASCAGOULA", "SAN RAMON",
                        "HOUSTON", "LONDON", "SINGAPORE", "GLASGOW"}

# Confirmed-noise candidates from the 2026-04-23 audit. If a future audit
# reveals new noise, add here rather than bloating the regex.
HARDCODED_NOISE = {
    "MARAN MARAN",        # duplication artifact
    "MARAN TANKER",       # company
    "SAP FLOW POST",      # filename noise
}


def extract_candidates(text: str) -> Set[str]:
    if not text:
        return set()
    upper = text.upper()
    found: Set[str] = set()

    for m in CANDIDATE_PATTERNS[0].finditer(upper):
        second = m.group(1)
        if second in MARAN_NON_VESSEL_SECOND_TOKENS:
            continue
        found.add(f"MARAN {second}")

    for m in CANDIDATE_PATTERNS[1].finditer(upper):
        prefix = m.group(1).strip()
        if prefix in CITY_PREFIX_VOYAGERS:
            found.add(f"{prefix} VOYAGER")
        elif len(prefix.split()) == 1 and len(prefix) >= 4:
            found.add(f"{prefix} VOYAGER")

    for m in CANDIDATE_PATTERNS[2].finditer(upper):
        found.add(m.group(1).strip())

    for m in CANDIDATE_PATTERNS[3].finditer(upper):
        found.add(m.group(1).strip())

    if re.search(r"\bSOPHIA\b", upper):
        found.add("SOPHIA")
    return found


def partition_candidates(
    candidates: Iterable[str],
    register_names: Set[str],
) -> Set[str]:
    """Return the subset worth preserving.

    Skips:
      - names already in the register (resolvable → already on vessel_ids)
      - HARDCODED_NOISE
      - suffix-bleed: candidate starts with a register name + extra tokens
        (e.g. "MARAN LYRA SERVICEREPORT" — real vessel is MARAN LYRA, the
        rest is a file-name tail that the MV/MT 3-token pattern caught).
    """
    preserve: Set[str] = set()
    for cand in candidates:
        if cand in register_names:
            continue
        if cand in HARDCODED_NOISE:
            continue
        # Suffix-bleed filter: if any register name is a strict prefix of
        # the candidate (with a following space), the candidate is the
        # register vessel + garbage, not a new vessel.
        if any(
            cand.startswith(reg + " ") for reg in register_names
        ):
            continue
        preserve.add(cand)
    return preserve


async def _collect_doc_text(pool, root_id: UUID) -> str:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_title, file_name, email_subject
            FROM documents
            WHERE id = $1 OR root_id = $1
            """,
            root_id,
        )
    parts: List[str] = []
    for row in rows:
        for key in ("source_title", "file_name", "email_subject"):
            v = row[key]
            if v:
                parts.append(v)
    return "\n".join(parts)


async def _write_unknown_names(pool, root_id: UUID, names: List[str]) -> int:
    async with pool.acquire() as conn:
        if names:
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{unknown_vessel_names}',
                    to_jsonb($2::text[])
                )
                WHERE document_id IN (
                    SELECT id FROM documents WHERE id = $1 OR root_id = $1
                )
                """,
                root_id,
                names,
            )
        else:
            # Clear the key if previously set and now empty — keeps metadata clean.
            result = await conn.execute(
                """
                UPDATE chunks
                SET metadata = metadata - 'unknown_vessel_names'
                WHERE (metadata ? 'unknown_vessel_names')
                  AND document_id IN (
                      SELECT id FROM documents WHERE id = $1 OR root_id = $1
                  )
                """,
                root_id,
            )
    return int(result.split()[-1]) if result else 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    dry_run = not args.apply

    db = SupabaseClient()
    try:
        vessels = await db.get_all_vessels()
        register_names = {v.name.upper() for v in vessels}
        print(f"Loaded {len(vessels)} vessels from DB")

        pool = await db._domain.get_pool()
        async with pool.acquire() as conn:
            roots = await conn.fetch(
                """
                SELECT id, doc_id, file_name
                FROM documents
                WHERE depth = 0 AND status = 'completed'
                ORDER BY created_at DESC
                """
                + (f"\nLIMIT {int(args.limit)}" if args.limit else "")
            )
        print(f"Scanning {len(roots)} root documents (dry_run={dry_run}, concurrency={args.concurrency})")

        sem = asyncio.Semaphore(args.concurrency)
        counters: Counter = Counter()
        unknown_tally: Counter = Counter()

        async def process_one(row) -> None:
            async with sem:
                root_id: UUID = row["id"]
                try:
                    text = await _collect_doc_text(pool, root_id)
                    cands = extract_candidates(text)
                    preserve = partition_candidates(cands, register_names)
                    if preserve:
                        counters["docs_with_unknown"] += 1
                        for name in preserve:
                            unknown_tally[name] += 1
                        if not dry_run:
                            n = await _write_unknown_names(
                                pool, root_id, sorted(preserve)
                            )
                            counters["chunks_stamped"] += n
                    else:
                        # Still clear stale unknowns when applying, in case a
                        # prior run stamped something that the current text
                        # no longer produces.
                        if not dry_run:
                            n = await _write_unknown_names(pool, root_id, [])
                            counters["chunks_cleared"] += n
                    counters["docs_scanned"] += 1
                except Exception as e:
                    counters["docs_errored"] += 1
                    print(f"! {row['file_name']}: {e}", file=sys.stderr)

        batch_size = 500
        for start in range(0, len(roots), batch_size):
            batch = roots[start : start + batch_size]
            await asyncio.gather(*(process_one(r) for r in batch))
            print(
                f"  {start + len(batch)}/{len(roots)}  "
                f"with_unknown={counters['docs_with_unknown']} "
                f"stamped={counters['chunks_stamped']} "
                f"cleared={counters['chunks_cleared']}",
                flush=True,
            )

        print()
        print("=== SUMMARY ===")
        for k, v in counters.most_common():
            print(f"  {k:22s} {v}")
        print()
        print("Unknown vessels preserved (by doc count):")
        for name, n in unknown_tally.most_common(30):
            print(f"  {n:5d}  {name}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
