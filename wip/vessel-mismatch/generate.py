"""Generate the vessel-mismatch action lists from a local ingest.db.

Reproducible companion to the files in this folder:
  * typo_mapping.csv          — detected_name → canonical vessel_name
  * untracked_parking.txt     — raw mentions to park in metadata.unknown_vessel_names
  * extractor_noise.txt       — false positives to feed into the regex blocklist (#20)
  * mismatch_report.md        — human-readable categorisation report

Re-run this on a fresh workstation to refresh the lists from the local ingest:

    cd <repo-root>
    OPENROUTER_API_KEY=test-key uv run python wip/vessel-mismatch/generate.py

(`OPENROUTER_API_KEY` only needs to satisfy `Settings()` — the value is not used
on this code path.)

The script is read-only against `data/ingest.db` and `data/vessel-list.csv`.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

# Make the repo's src package importable when run from any cwd.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from mtss.cli.validate_cmd import _load_canonical_vessel_names  # noqa: E402
from mtss.processing.vessel_mention_extractor import extract_vessel_mentions  # noqa: E402

# ---------------------------------------------------------------------------
# Categorisation heuristics — same as `mtss validate` check #36, plus extra
# refinement applied here for action-list generation.
# ---------------------------------------------------------------------------

NON_VESSEL_MARAN = {
    "MARAN HAS", "MARAN VESSELS", "MARAN VESSEL", "MARAN TEAM", "MARAN ASIA",
    "MARAN SUPERINTENDENT", "MARAN LIBRA SHIPS", "MARAN MARAN", "MARAN TANKERS",
    "MARAN TANKER", "MARAN GAS", "MARAN SHIPS", "MARAN ORDER", "MARAN WILL",
    "MARAN OFFICE", "MARAN CORP", "MARAN GROUP", "MARAN DRY", "MARAN HQ",
    "MARAN FLEET", "MARAN FINANCE", "MARAN TRANSPORTER",
}

BIZ_PHRASE_TOKENS = {
    "PRICE", "WEIGHT", "CONSUMPTION", "SULPHUR", "DRAFT", "REGARDS", "TIME",
    "TANK", "LNG", "HSFO", "SUMMER", "WINTER", "TOTAL", "CALCULATED", "BEST",
    "LOW", "PORT", "NOON", "FUEL", "STOCK", "BUNKER", "WATER", "SEA", "DAY",
    "NOTE", "ADDITIONAL", "PLEASE",
}


def main() -> int:
    db = _REPO / "data" / "ingest.db"
    out_dir = Path(__file__).resolve().parent
    if not db.exists():
        print(f"ERROR: {db} not found — run from a workstation with a populated ingest.")
        return 2

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    docs = [dict(r) for r in conn.execute("SELECT * FROM documents")]
    chunks = [
        dict(r)
        for r in conn.execute(
            "SELECT id, document_id, content, source_title, metadata_json FROM chunks"
        )
    ]
    canonical = _load_canonical_vessel_names()

    unknown: dict[str, int] = defaultdict(int)
    for d in docs:
        for field in ("title", "source_title", "file_name"):
            v = d.get(field)
            if v:
                for m in extract_vessel_mentions(str(v)):
                    if m not in canonical:
                        unknown[m] += 1
    for c in chunks:
        if c.get("content"):
            for m in extract_vessel_mentions(str(c["content"])):
                if m not in canonical:
                    unknown[m] += 1

    canonical_two_token = {n for n in canonical if len(n.split()) == 2}

    def is_canonical_concat(m: str) -> bool:
        parts = m.split()
        return len(parts) >= 3 and " ".join(parts[:2]) in canonical_two_token

    def is_extractor_noise(m: str) -> bool:
        if m in NON_VESSEL_MARAN:
            return True
        parts = m.split()
        if not (m.startswith("MARAN ") or m.endswith(" VOYAGER")):
            if any(p in BIZ_PHRASE_TOKENS for p in parts):
                return True
        return is_canonical_concat(m)

    def likely_typo(m: str):
        if not (m.startswith("MARAN ") or m.endswith(" VOYAGER")):
            return None
        matches = get_close_matches(m, list(canonical), n=1, cutoff=0.85)
        return matches[0] if matches else None

    typos: dict[str, tuple[str, int]] = {}
    untracked: dict[str, int] = {}
    noise: dict[str, int] = {}

    for m, count in unknown.items():
        if is_extractor_noise(m):
            noise[m] = count
            continue
        canon = likely_typo(m)
        if canon:
            typos[m] = (canon, count)
            continue
        untracked[m] = count

    # ── write outputs ────────────────────────────────────────────────
    typo_csv = out_dir / "typo_mapping.csv"
    with typo_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detected_name", "vessel_name", "occurrences"])
        for m, (canon, c) in sorted(typos.items(), key=lambda kv: -kv[1][1]):
            w.writerow([m, canon, c])

    untracked_txt = out_dir / "untracked_parking.txt"
    with untracked_txt.open("w", encoding="utf-8") as f:
        f.write(
            f"# {len(untracked)} untracked vessel mentions to park in "
            "metadata.unknown_vessel_names\n"
        )
        f.write("# Format: <raw mention>\\t<occurrences>\n")
        for m, c in sorted(untracked.items(), key=lambda kv: -kv[1]):
            f.write(f"{m}\t{c}\n")

    noise_txt = out_dir / "extractor_noise.txt"
    with noise_txt.open("w", encoding="utf-8") as f:
        f.write(
            f"# {len(noise)} extractor false-positives — feed into HARDCODED_NOISE / "
            "regex tightening (task #20)\n"
        )
        for m, c in sorted(noise.items(), key=lambda kv: -kv[1]):
            f.write(f"{m}\t{c}\n")

    print(
        "Generated lists at",
        out_dir.relative_to(_REPO),
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
    )
    print(f"  typos:      {len(typos):>4}  ({sum(c for _, c in typos.values())} occ)")
    print(f"  untracked:  {len(untracked):>4}  ({sum(untracked.values())} occ)")
    print(f"  noise:      {len(noise):>4}  ({sum(noise.values())} occ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
