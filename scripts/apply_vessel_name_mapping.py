"""Resolve typo / alternate vessel names using a reviewed mapping CSV.

Input CSV columns (semicolon or comma delimited):
    detected_name   — the raw string that appeared in the corpus
    vessel_name     — authoritative register name, OR "(no vessel)" / blank

For every (detected_name -> vessel_name) row where:
  - `vessel_name` matches a row in the vessels table (case-insensitive), AND
  - `detected_name` is NOT already the same as `vessel_name`

we walk every chunk whose `metadata.unknown_vessel_names` contains
`detected_name` and:
  - add the resolved vessel's UUID to `vessel_ids`
  - add the resolved vessel's type/class to `vessel_types` / `vessel_classes`
  - remove `detected_name` from `unknown_vessel_names`
  - if `unknown_vessel_names` becomes empty, drop the key

This is a corpus-level retro-fix: the retag pass won't catch typos because
VesselMatcher only matches register names + aliases. The mapping CSV gives
us a reviewed override so previously-stranded references make it into the
canonical `vessel_ids` field without polluting the register itself.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/apply_vessel_name_mapping.py \
        --mapping "C:/Users/mail/Downloads/vessels_with_names.csv" --dry-run
    uv run python scripts/apply_vessel_name_mapping.py \
        --mapping "C:/Users/mail/Downloads/vessels_with_names.csv" --apply
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from mtss.storage.supabase_client import SupabaseClient


def load_mapping(csv_path: Path) -> List[Tuple[str, str]]:
    """Return [(detected_name_upper, target_register_name_upper), ...].

    Skips rows where target is empty, "(no vessel)", or identical to
    detected_name (those are already resolvable or pure noise).
    """
    with csv_path.open("r", encoding="utf-8-sig") as fh:
        sample = fh.read(1024)
        fh.seek(0)
        delim = ";" if ";" in sample else ","
        reader = csv.DictReader(fh, delimiter=delim)
        pairs: List[Tuple[str, str]] = []
        for row in reader:
            detected = (row.get("detected_name") or "").strip().upper()
            target = (row.get("vessel_name") or "").strip().upper()
            if not detected or not target:
                continue
            if target == "(NO VESSEL)" or target.startswith("("):
                continue
            if detected == target:
                continue
            pairs.append((detected, target))
    return pairs


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, required=True,
                    help="Reviewed detected->register mapping CSV")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry_run = not args.apply

    pairs = load_mapping(args.mapping)
    if not pairs:
        print(f"No actionable rows in {args.mapping}", file=sys.stderr)
        return 2
    print(f"Loaded {len(pairs)} detected->register mappings from {args.mapping}")

    db = SupabaseClient()
    try:
        pool = await db._domain.get_pool()

        # Register lookup: NAME (upper) -> (id, vessel_type, vessel_class).
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, vessel_type, vessel_class FROM vessels"
            )
        by_name: Dict[str, Tuple[str, str, str]] = {
            r["name"].upper(): (str(r["id"]), r["vessel_type"], r["vessel_class"])
            for r in rows
        }

        # Filter mapping against register — skip entries where the target isn't
        # actually in the vessels table yet (should be none after the CSV import).
        resolvable: List[Tuple[str, str, str, str, str]] = []
        skipped_no_target: List[Tuple[str, str]] = []
        for detected, target in pairs:
            if target not in by_name:
                skipped_no_target.append((detected, target))
                continue
            vid, vtype, vclass = by_name[target]
            resolvable.append((detected, target, vid, vtype, vclass))

        print(f"  {len(resolvable)} resolvable, "
              f"{len(skipped_no_target)} skipped (target not in vessels table)")
        if skipped_no_target:
            for d, t in skipped_no_target:
                print(f"    skip: {d!r} -> {t!r} (no such vessel)")

        stats: Counter = Counter()
        per_mapping: Counter = Counter()

        for detected, target, vid, vtype, vclass in resolvable:
            async with pool.acquire() as conn:
                # Scope: every chunk whose metadata.unknown_vessel_names
                # contains the typo.
                affected = await conn.fetchval(
                    """
                    SELECT count(*) FROM chunks
                    WHERE metadata ? 'unknown_vessel_names'
                      AND metadata->'unknown_vessel_names' @> to_jsonb(ARRAY[$1::text])
                    """,
                    detected,
                )
                if not affected:
                    stats["no_chunks_for_mapping"] += 1
                    continue

                per_mapping[f"{detected} -> {target}"] = int(affected)
                stats["mappings_with_chunks"] += 1

                if dry_run:
                    continue

                # Apply in one UPDATE: add vid to vessel_ids, vtype to
                # vessel_types, vclass to vessel_classes (dedup via jsonb
                # array concatenation with an anti-check), then strip the
                # detected name from unknown_vessel_names. If the resulting
                # array is empty, drop the key to keep metadata tidy.
                result = await conn.execute(
                    """
                    UPDATE chunks SET metadata = (
                      WITH base AS (
                        SELECT COALESCE(metadata, '{}'::jsonb) AS m
                      ),
                      -- strip typo from unknown_vessel_names
                      step1 AS (
                        SELECT jsonb_set(
                          m,
                          '{unknown_vessel_names}',
                          COALESCE(
                            (
                              SELECT jsonb_agg(v)
                              FROM jsonb_array_elements_text(m->'unknown_vessel_names') v
                              WHERE v <> $2
                            ),
                            '[]'::jsonb
                          )
                        ) AS m FROM base
                      ),
                      -- drop key if empty
                      step2 AS (
                        SELECT CASE
                          WHEN jsonb_array_length(COALESCE(m->'unknown_vessel_names','[]'::jsonb)) = 0
                            THEN m - 'unknown_vessel_names'
                          ELSE m
                        END AS m FROM step1
                      ),
                      -- append vessel_id if missing
                      step3 AS (
                        SELECT CASE
                          WHEN COALESCE(m->'vessel_ids','[]'::jsonb) @> to_jsonb(ARRAY[$1::text])
                            THEN m
                          ELSE jsonb_set(m, '{vessel_ids}',
                                COALESCE(m->'vessel_ids','[]'::jsonb)
                                  || to_jsonb(ARRAY[$1::text]))
                        END AS m FROM step2
                      ),
                      -- append vessel_type if missing
                      step4 AS (
                        SELECT CASE
                          WHEN COALESCE(m->'vessel_types','[]'::jsonb) @> to_jsonb(ARRAY[$3::text])
                            THEN m
                          ELSE jsonb_set(m, '{vessel_types}',
                                COALESCE(m->'vessel_types','[]'::jsonb)
                                  || to_jsonb(ARRAY[$3::text]))
                        END AS m FROM step3
                      ),
                      -- append vessel_class if missing (may be blank)
                      step5 AS (
                        SELECT CASE
                          WHEN $4 = '' THEN m
                          WHEN COALESCE(m->'vessel_classes','[]'::jsonb) @> to_jsonb(ARRAY[$4::text])
                            THEN m
                          ELSE jsonb_set(m, '{vessel_classes}',
                                COALESCE(m->'vessel_classes','[]'::jsonb)
                                  || to_jsonb(ARRAY[$4::text]))
                        END AS m FROM step4
                      )
                      SELECT m FROM step5
                    )
                    WHERE metadata ? 'unknown_vessel_names'
                      AND metadata->'unknown_vessel_names' @> to_jsonb(ARRAY[$2::text])
                    """,
                    vid, detected, vtype, vclass,
                )
                stats["chunks_updated"] += int(result.split()[-1]) if result else 0

        print()
        print("=== SUMMARY ===")
        for k, v in stats.most_common():
            print(f"  {k:28s} {v}")
        print()
        print("Per-mapping chunk counts:")
        for k, n in per_mapping.most_common():
            print(f"  {n:5d}  {k}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
