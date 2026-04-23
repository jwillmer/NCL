"""Emit a CSV of detected vessel names + proposed mapping to the register.

Reads the JSON report produced by audit_vessel_mentions.py and flattens
it into one row per detected candidate. proposed_mapping is left blank
when the candidate has no plausible match (bucket='unknown').

Usage:
    uv run python scripts/export_vessel_mapping_csv.py
    uv run python scripts/export_vessel_mapping_csv.py --in data/reports/vessel_audit.json \
        --out data/reports/vessel_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_IN = Path("data/reports/vessel_audit.json")
DEFAULT_OUT = Path("data/reports/vessel_mapping.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input JSON not found: {args.inp}. "
              f"Run scripts/audit_vessel_mentions.py first.", file=sys.stderr)
        return 2

    report = json.loads(args.inp.read_text(encoding="utf-8"))
    buckets = report.get("buckets", {})

    rows = []
    for bucket_name in ("known_exact", "known_fuzzy", "unknown"):
        for entry in buckets.get(bucket_name, []):
            rows.append({
                "detected_name": entry["candidate"],
                "proposed_mapping": entry.get("matched_register_name") or "",
                "bucket": bucket_name,
                "similarity": entry.get("similarity") if entry.get("similarity") is not None else "",
                "total_mentions": entry["total_mentions"],
                "sources": ", ".join(f"{k}={v}" for k, v in sorted(entry["sources"].items())),
            })

    # Stable ordering: unknown first (need human review), then fuzzy,
    # then exact — mentions desc within each bucket.
    bucket_rank = {"unknown": 0, "known_fuzzy": 1, "known_exact": 2}
    rows.sort(key=lambda r: (bucket_rank[r["bucket"]], -r["total_mentions"], r["detected_name"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["detected_name", "proposed_mapping", "bucket",
                        "similarity", "total_mentions", "sources"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  unknown:     {sum(1 for r in rows if r['bucket'] == 'unknown')}")
    print(f"  known_fuzzy: {sum(1 for r in rows if r['bucket'] == 'known_fuzzy')}")
    print(f"  known_exact: {sum(1 for r in rows if r['bucket'] == 'known_exact')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
