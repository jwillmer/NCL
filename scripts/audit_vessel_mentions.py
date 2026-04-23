"""Audit vessel mentions across the ingested corpus.

Scans documents.title, documents.source_title, documents.file_name,
and email_subject from documents.metadata_json for vessel-like tokens,
then classifies them against the vessel register in data/vessel-list.csv:

  - known_exact      : name/alias matches register verbatim (case-insensitive)
  - known_fuzzy      : close match to a register name (edit distance 1-2)
                       — likely a typo; review for alias addition
  - unknown          : no plausible match — either out-of-fleet vessel or noise

Read-only. Produces a JSON report and prints a summary.

Usage:
    uv run python scripts/audit_vessel_mentions.py
    uv run python scripts/audit_vessel_mentions.py --out data/reports/vessel_audit.json
    uv run python scripts/audit_vessel_mentions.py --scan-chunks   # also scan chunk body text (slow)
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_DB = Path("data/output/ingest.db")
DEFAULT_CSV = Path("data/vessel-list.csv")
DEFAULT_OUT = Path("data/reports/vessel_audit.json")

# Candidate extractors. Designed to catch how vessels are actually
# written in email subjects / file names / log headers. All patterns
# are applied on the UPPERCASED form of the text, so the register
# (which is all-caps) matches cleanly.
#
# We deliberately skip bare first-names ("HELEN", "MIRA") because
# those produce huge noise: people, places, report names. Vessel
# attribution should come from the full register name or an alias,
# not an unqualified token.
# Tokens that trail "MARAN" but are NOT vessels — company/org names.
# Candidates with only these second tokens are dropped.
MARAN_NON_VESSEL_SECOND_TOKENS = {
    "TANKERS", "SHIP", "GAS", "SUPT", "DRY", "TSS", "CORP",
    "FLEET", "FINANCE", "OFFICE", "HQ", "GROUP",
}

CANDIDATE_PATTERNS: List[re.Pattern[str]] = [
    # MARAN <WORD> — no hyphens in the token so file-name suffixes like
    # "MARAN DIONE-ENGINE" yield just "MARAN DIONE".
    re.compile(r"\bMARAN\s+([A-Z]{3,})\b"),
    # <CITY> VOYAGER — the Voyager class prefixes with a city name
    re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})?)\s+VOYAGER\b"),
    # MT / MV / M/V / M.V. <NAME>  (1 or 2 words). Require 2+ words
    # because single-token hits ("MV MARAN") overwhelmingly recapture
    # things we already catch with the MARAN pattern, and single bare
    # first names (e.g. "MV ANTONIS") are too ambiguous to keep.
    re.compile(r"\bM[\.]?[TV][\.]?\s+([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\b"),
    # ANTONIS I. ANGELICOUSSIS / MARIA A. ANGELICOUSSIS style
    re.compile(r"\b([A-Z]+\s+[A-Z]\.\s+ANGELICOUSSIS)\b"),
    re.compile(r"\b(SOPHIA)\b"),  # single explicit
]

CITY_PREFIX_VOYAGERS = {"RICHMOND", "EL SEGUNDO", "PASCAGOULA", "SAN RAMON",
                        "HOUSTON", "LONDON", "SINGAPORE", "GLASGOW"}


def load_register(csv_path: Path) -> Tuple[Dict[str, str], Set[str]]:
    """Return (name_to_key, all_known_tokens_upper).

    name_to_key maps every canonical name + alias (uppercased) to its
    canonical NAME column for reporting. all_known_tokens_upper is the
    flat set for quick membership tests.
    """
    name_to_key: Dict[str, str] = {}
    if not csv_path.exists():
        print(f"ERROR: vessel CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    import csv as _csv

    with csv_path.open("r", encoding="utf-8") as fh:
        sample = fh.read(1024)
        fh.seek(0)
        delim = ";" if ";" in sample else ","
        reader = _csv.DictReader(fh, delimiter=delim)
        for row in reader:
            name = (row.get("NAME") or "").strip().upper()
            if not name:
                continue
            name_to_key[name] = name
            aliases = (row.get("ALIASES") or "").strip()
            if aliases:
                for a in aliases.split(","):
                    a_up = a.strip().upper()
                    if a_up:
                        name_to_key.setdefault(a_up, name)
    return name_to_key, set(name_to_key.keys())


def iter_source_strings(conn: sqlite3.Connection, scan_chunks: bool) -> Iterable[Tuple[str, str]]:
    """Yield (source_kind, text) pairs from the DB.

    source_kind tags where the string came from so the final report
    can say "this candidate appeared in 12 email subjects and 3 file
    names" — useful for triaging unknowns.
    """
    cur = conn.cursor()

    cur.execute("SELECT title, source_title, file_name, metadata_json FROM documents")
    for title, source_title, file_name, meta_json in cur:
        if title:
            yield "doc_title", title
        if source_title and source_title != title:
            yield "doc_source_title", source_title
        if file_name:
            yield "doc_file_name", file_name
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                continue
            subj = meta.get("email_subject")
            if subj:
                yield "email_subject", subj

    if scan_chunks:
        # Chunk body scan is optional because it's O(43k rows * regex).
        # Only source_title + section_title are cheap enough by default.
        cur.execute("SELECT source_title, section_title FROM chunks")
        for source_title, section_title in cur:
            if source_title:
                yield "chunk_source_title", source_title
            if section_title:
                yield "chunk_section_title", section_title


def extract_candidates(text: str) -> Set[str]:
    """Return normalized candidate vessel names (uppercase) from text."""
    if not text:
        return set()
    upper = text.upper()
    found: Set[str] = set()

    # MARAN X → full "MARAN X", dropping obvious company/org tokens
    for m in CANDIDATE_PATTERNS[0].finditer(upper):
        second = m.group(1)
        if second in MARAN_NON_VESSEL_SECOND_TOKENS:
            continue
        found.add(f"MARAN {second}")

    # <X> VOYAGER → prefer the city-name form; register stores full phrase
    for m in CANDIDATE_PATTERNS[1].finditer(upper):
        prefix = m.group(1).strip()
        # Only keep the prefix if it's in the known city set OR looks
        # like a proper single token — avoids "THE VOYAGER", "MY VOYAGER".
        if prefix in CITY_PREFIX_VOYAGERS:
            found.add(f"{prefix} VOYAGER")
        elif len(prefix.split()) == 1 and len(prefix) >= 4:
            found.add(f"{prefix} VOYAGER")

    # MT/MV NAME
    for m in CANDIDATE_PATTERNS[2].finditer(upper):
        found.add(m.group(1).strip())

    # ANGELICOUSSIS family
    for m in CANDIDATE_PATTERNS[3].finditer(upper):
        found.add(m.group(1).strip())

    # SOPHIA (only useful when it stands alone; keep it coarse — will
    # be classified into known_exact if it matches).
    if re.search(r"\bSOPHIA\b", upper):
        found.add("SOPHIA")

    return found


def classify(
    candidate: str,
    known_tokens: Set[str],
    name_to_key: Dict[str, str],
    register_names: List[str],
) -> Tuple[str, Optional[str], Optional[float]]:
    """Return (bucket, matched_register_name, similarity).

    bucket ∈ {"known_exact", "known_fuzzy", "unknown"}.
    """
    if candidate in known_tokens:
        return "known_exact", name_to_key[candidate], 1.0

    # Fuzzy — close to a real name but not a direct hit. Cutoff 0.88
    # catches single-character typos on 10-18 letter names without
    # swallowing everything.
    matches = difflib.get_close_matches(candidate, register_names, n=1, cutoff=0.88)
    if matches:
        best = matches[0]
        score = difflib.SequenceMatcher(None, candidate, best).ratio()
        return "known_fuzzy", best, round(score, 3)

    return "unknown", None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scan-chunks", action="store_true",
                    help="Also scan chunks.source_title / section_title (slower)")
    ap.add_argument("--top-unknown", type=int, default=40,
                    help="How many unknown candidates to print in the summary")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 2

    name_to_key, known_tokens = load_register(args.csv)
    register_names = sorted({v for v in name_to_key.values()})
    print(f"Loaded {len(register_names)} vessels from {args.csv} "
          f"({len(known_tokens)} name+alias tokens)")

    # candidate → {source_kind: count}
    candidate_sources: Dict[str, Counter] = defaultdict(Counter)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        total_rows = 0
        for source_kind, text in iter_source_strings(conn, args.scan_chunks):
            total_rows += 1
            for cand in extract_candidates(text):
                candidate_sources[cand][source_kind] += 1
    finally:
        conn.close()

    print(f"Scanned {total_rows} source strings, "
          f"{len(candidate_sources)} distinct vessel-like candidates")

    buckets = {"known_exact": [], "known_fuzzy": [], "unknown": []}
    for cand in sorted(candidate_sources):
        bucket, matched, score = classify(cand, known_tokens, name_to_key, register_names)
        sources = dict(candidate_sources[cand])
        total = sum(sources.values())
        buckets[bucket].append({
            "candidate": cand,
            "total_mentions": total,
            "matched_register_name": matched,
            "similarity": score,
            "sources": sources,
        })

    # Sort each bucket by mentions desc for readability
    for b in buckets:
        buckets[b].sort(key=lambda x: -x["total_mentions"])

    # Fleet-coverage view: which register vessels were NOT seen at all
    seen_register_names = {e["matched_register_name"] for e in buckets["known_exact"]}
    seen_register_names |= {e["matched_register_name"] for e in buckets["known_fuzzy"]}
    unseen_register = sorted(set(register_names) - seen_register_names)

    report = {
        "db": str(args.db),
        "csv": str(args.csv),
        "counts": {
            "known_exact": len(buckets["known_exact"]),
            "known_fuzzy": len(buckets["known_fuzzy"]),
            "unknown": len(buckets["unknown"]),
            "register_total": len(register_names),
            "register_unseen": len(unseen_register),
        },
        "buckets": buckets,
        "register_unseen": unseen_register,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")

    # Console summary
    print()
    print("=== SUMMARY ===")
    for b in ("known_exact", "known_fuzzy", "unknown"):
        total = sum(e["total_mentions"] for e in buckets[b])
        print(f"  {b:14s} candidates={len(buckets[b]):4d}  mentions={total}")

    if buckets["known_fuzzy"]:
        print()
        print("Top known_fuzzy (likely typos — consider alias additions):")
        for e in buckets["known_fuzzy"][:20]:
            print(f"  {e['candidate']:30s} -> {e['matched_register_name']:25s} "
                  f"sim={e['similarity']:.3f}  mentions={e['total_mentions']}")

    if buckets["unknown"]:
        print()
        print(f"Top {args.top_unknown} unknown candidates:")
        for e in buckets["unknown"][:args.top_unknown]:
            src = ", ".join(f"{k}={v}" for k, v in e["sources"].items())
            print(f"  {e['candidate']:30s} mentions={e['total_mentions']:5d}  [{src}]")

    if unseen_register:
        print()
        print(f"Register vessels never seen in corpus ({len(unseen_register)}):")
        for n in unseen_register:
            print(f"  - {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
