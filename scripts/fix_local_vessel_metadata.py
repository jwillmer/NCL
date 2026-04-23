"""Apply vessel retag + annotate + typo mapping to local SQLite ingest.db.

Mirrors the three Supabase-side passes in one local pass:

  1. Retag: rescan each document (title + file_name + email_subject +
     attachment titles) via VesselMatcher, rebuild
     chunks.metadata_json.vessel_ids / vessel_types / vessel_classes
     using UUIDs derived from the CSV load. Local UUIDs are ephemeral
     per load, but they ARE consistent within the local DB after this
     pass, which is all local needs (per current policy: local doesn't
     need to track either remote env).
  2. Annotate: for any remaining real-looking vessel mentions in the
     same text that don't resolve to the register (typos, unknown new
     vessels), stamp them into metadata.unknown_vessel_names.
  3. Mapping: consume the reviewed mapping CSV
     (typically data/reports/vessel_mapping_reviewed.csv or the user's
     Downloads copy). For each detected->register pair where detected
     appears in unknown_vessel_names, add the register UUID to
     vessel_ids and strip the typo from unknown_vessel_names.

Read-only by default. Pass --apply to commit.

Usage:
    uv run python scripts/fix_local_vessel_metadata.py --dry-run \
        --mapping "C:/Users/mail/Downloads/vessels_with_names.csv"
    uv run python scripts/fix_local_vessel_metadata.py --apply \
        --mapping "C:/Users/mail/Downloads/vessels_with_names.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from mtss.models.vessel import load_vessels_from_csv
from mtss.processing.vessel_matcher import VesselMatcher

# Keep in sync with scripts/annotate_unknown_vessel_mentions.py
CANDIDATE_PATTERNS = [
    re.compile(r"\bMARAN\s+([A-Z]{3,})\b"),
    re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})?)\s+VOYAGER\b"),
    re.compile(r"\bM[\.]?[TV][\.]?\s+([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\b"),
    re.compile(r"\b([A-Z]+\s+[A-Z]\.\s+ANGELICOUSSIS)\b"),
]
MARAN_NON_VESSEL_SECOND_TOKENS = {
    "TANKERS", "SHIP", "GAS", "SUPT", "DRY", "TSS", "CORP",
    "FLEET", "FINANCE", "OFFICE", "HQ", "GROUP", "TANKER",
}
CITY_PREFIX_VOYAGERS = {"RICHMOND", "EL SEGUNDO", "PASCAGOULA", "SAN RAMON",
                        "HOUSTON", "LONDON", "SINGAPORE", "GLASGOW"}
HARDCODED_NOISE = {"MARAN MARAN", "MARAN TANKER", "SAP FLOW POST"}


def extract_candidates(text: str) -> Set[str]:
    if not text:
        return set()
    upper = text.upper()
    found: Set[str] = set()
    for m in CANDIDATE_PATTERNS[0].finditer(upper):
        sec = m.group(1)
        if sec in MARAN_NON_VESSEL_SECOND_TOKENS:
            continue
        found.add(f"MARAN {sec}")
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


def load_mapping(csv_path: Path) -> List[Tuple[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as fh:
        sample = fh.read(1024)
        fh.seek(0)
        delim = ";" if ";" in sample else ","
        reader = csv.DictReader(fh, delimiter=delim)
        out: List[Tuple[str, str]] = []
        for row in reader:
            detected = (row.get("detected_name") or "").strip().upper()
            target = (row.get("vessel_name") or "").strip().upper()
            if not detected or not target:
                continue
            if target.startswith("("):
                continue
            if detected == target:
                continue
            out.append((detected, target))
    return out


def collect_doc_text(conn: sqlite3.Connection, root_id: str) -> str:
    cur = conn.execute(
        """
        SELECT title, source_title, file_name, metadata_json
        FROM documents
        WHERE id = ? OR root_id = ?
        """,
        (root_id, root_id),
    )
    parts: List[str] = []
    for title, source_title, file_name, meta_json in cur:
        if title:
            parts.append(title)
        if source_title and source_title != title:
            parts.append(source_title)
        if file_name:
            parts.append(file_name)
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                continue
            subj = meta.get("email_subject")
            if subj:
                parts.append(subj)
    return "\n".join(parts)


def update_chunks_for_doc(
    conn: sqlite3.Connection,
    root_id: str,
    vessel_ids: List[str],
    vessel_types: List[str],
    vessel_classes: List[str],
    unknown_names: List[str],
    apply: bool,
) -> int:
    """Rewrite metadata_json for every chunk under this root's tree.

    Always:
      - vessel_ids/vessel_types/vessel_classes set to computed arrays
        (empty list if no match; omit the key if empty).
      - unknown_vessel_names set to computed sorted list, or key dropped
        if empty.
    """
    cur = conn.execute(
        """
        SELECT c.id, c.metadata_json
        FROM chunks c
        WHERE c.document_id IN (
            SELECT id FROM documents WHERE id = ? OR root_id = ?
        )
        """,
        (root_id, root_id),
    )
    rows = cur.fetchall()
    updated = 0
    for chunk_id, meta_json in rows:
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = {}
        # Update fields (add or remove)
        def _set(key: str, values: List[str]) -> None:
            if values:
                meta[key] = values
            elif key in meta:
                del meta[key]

        _set("vessel_ids", vessel_ids)
        _set("vessel_types", vessel_types)
        _set("vessel_classes", vessel_classes)
        _set("unknown_vessel_names", unknown_names)
        new_json = json.dumps(meta, ensure_ascii=False) if meta else None
        if new_json != meta_json:
            updated += 1
            if apply:
                conn.execute(
                    "UPDATE chunks SET metadata_json = ? WHERE id = ?",
                    (new_json, chunk_id),
                )
    return updated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data/output/ingest.db"))
    ap.add_argument("--csv", type=Path, default=Path("data/vessel-list.csv"))
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry_run = not args.apply

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 2

    vessels = load_vessels_from_csv(args.csv)
    if not vessels:
        print("No vessels loaded", file=sys.stderr)
        return 2
    matcher = VesselMatcher(vessels)
    register_names = {v.name.upper() for v in vessels}
    name_to_info: Dict[str, Tuple[str, str, str]] = {
        v.name.upper(): (str(v.id), v.vessel_type, v.vessel_class)
        for v in vessels
    }
    print(f"Loaded {matcher.vessel_count} vessels ({matcher.name_count} names) from {args.csv}")

    mapping = load_mapping(args.mapping)
    # Keep only mappings where the target is in the register (otherwise
    # we can't resolve to a UUID).
    resolvable_mapping: Dict[str, str] = {}
    for detected, target in mapping:
        if target in register_names:
            resolvable_mapping[detected] = target
    print(f"Loaded {len(resolvable_mapping)} resolvable detected->register mappings")

    conn = sqlite3.connect(str(args.db))
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        roots = conn.execute(
            "SELECT id, doc_id, file_name FROM documents WHERE depth = 0"
        ).fetchall()
        print(f"Scanning {len(roots)} root documents (dry_run={dry_run})")

        stats: Counter = Counter()
        unknowns_tally: Counter = Counter()
        mapping_hits: Counter = Counter()

        for root_id, doc_id, file_name in roots:
            text = collect_doc_text(conn, root_id)
            matched_uuids = matcher.find_vessels(text)
            # Compute unknowns from candidates minus matched
            cands = extract_candidates(text)
            # Partition unknowns: suffix-bleed filter + apply mapping
            unknowns: Set[str] = set()
            extra_vessel_ids: Set[str] = set()
            extra_types: Set[str] = set()
            extra_classes: Set[str] = set()
            for cand in cands:
                if cand in register_names:
                    continue
                if cand in HARDCODED_NOISE:
                    continue
                if any(cand.startswith(r + " ") for r in register_names):
                    continue
                # Apply mapping: if typo resolves to a register vessel,
                # promote to vessel_ids instead of unknown_vessel_names.
                target = resolvable_mapping.get(cand)
                if target and target in name_to_info:
                    vid, vtype, vclass = name_to_info[target]
                    extra_vessel_ids.add(vid)
                    if vtype:
                        extra_types.add(vtype)
                    if vclass:
                        extra_classes.add(vclass)
                    mapping_hits[f"{cand} -> {target}"] += 1
                    continue
                unknowns.add(cand)

            # Build final arrays
            vessel_ids_final = sorted(
                {str(v) for v in matched_uuids} | extra_vessel_ids
            )
            vessel_types_final = sorted(
                set(matcher.get_types_for_ids(matched_uuids)) | extra_types
            )
            vessel_classes_final = sorted(
                set(matcher.get_classes_for_ids(matched_uuids)) | extra_classes
            )
            unknowns_list = sorted(unknowns)

            for u in unknowns_list:
                unknowns_tally[u] += 1

            n_updated = update_chunks_for_doc(
                conn, root_id,
                vessel_ids_final, vessel_types_final, vessel_classes_final,
                unknowns_list, apply=not dry_run,
            )
            if n_updated:
                stats["docs_changed"] += 1
                stats["chunks_updated"] += n_updated
            else:
                stats["docs_unchanged"] += 1

        if not dry_run:
            conn.commit()

        print()
        print("=== SUMMARY ===")
        for k, v in stats.most_common():
            print(f"  {k:22s} {v}")
        print()
        print("Unknown vessel names preserved (by doc count):")
        for name, n in unknowns_tally.most_common(30):
            print(f"  {n:5d}  {name}")
        print()
        print("Typo mappings applied (by doc count):")
        for k, n in mapping_hits.most_common():
            print(f"  {n:5d}  {k}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
