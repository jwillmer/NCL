"""Rename archive folders from 16-char ``doc_id[:16]`` to 32-char ``compute_folder_id``.

Background
----------
Archive folders were previously named ``archive/<doc_id[:16]>/``. The folder
name is now a separately-keyed 32-char hash derived from ``doc_id`` via
``compute_folder_id`` (see ``src/mtss/utils.py``). Existing archives must be
renamed in-place and every ``archive_browse_uri`` / ``archive_download_uri``
reference in ``documents.jsonl`` and ``chunks.jsonl`` must be rewritten.

Inference rules
---------------
For each ``archive/<old_folder>/``:

1. Treat ``<old_folder>`` as a doc_id prefix; find the unique email document in
   ``documents.jsonl`` whose ``doc_id`` starts with that prefix AND whose
   ``parent_id`` is empty (emails are the folder roots).
2. Compute the new folder name via ``compute_folder_id(email_doc_id)``.
3. Plan a rename and record every URI rewrite needed in
   ``documents.jsonl`` / ``chunks.jsonl``.

If no unique email match is found, the folder is skipped and reported.

Usage
-----
    # Dry-run (default). Reports counts and sample rewrites, no writes.
    python scripts/migrate_folder_prefix.py --output-dir data/output

    # Real run. Only run after inspecting the dry-run output.
    python scripts/migrate_folder_prefix.py --output-dir data/output --apply

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.utils import compute_folder_id  # noqa: E402

logger = logging.getLogger("migrate_folder_prefix")


@dataclass
class FolderPlan:
    old_folder: str
    new_folder: str
    email_doc_id: str
    doc_rewrites: int = 0
    chunk_rewrites: int = 0


@dataclass
class MigrationResult:
    plans: list[FolderPlan] = field(default_factory=list)
    skipped_no_match: list[str] = field(default_factory=list)
    skipped_ambiguous: list[str] = field(default_factory=list)
    skipped_target_exists: list[str] = field(default_factory=list)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed %s line: %s", path.name, exc)


def _build_prefix_index(documents_jsonl: Path) -> dict[str, list[str]]:
    """Index email doc_ids by the old 16-char prefix.

    Only emails (parent_id empty / missing) are root folders, so attachments
    are excluded from the index.
    """
    index: dict[str, list[str]] = {}
    for doc in _iter_jsonl(documents_jsonl):
        parent_id = doc.get("parent_id")
        if parent_id:
            continue
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        prefix = doc_id[:16]
        index.setdefault(prefix, []).append(doc_id)
    return index


def _rewrite_uri(value: Optional[str], old_folder: str, new_folder: str) -> Optional[str]:
    """Rewrite ``/archive/<old>/...`` URI segment.

    Matches the folder segment exactly so we don't corrupt URIs that happen to
    contain the old hex string elsewhere. Returns the rewritten string, or the
    original if no change applied.
    """
    if not value or not isinstance(value, str):
        return value
    prefix = f"/archive/{old_folder}/"
    if value.startswith(prefix):
        return f"/archive/{new_folder}/" + value[len(prefix):]
    # Exact bare folder (no trailing slash) also possible.
    if value == f"/archive/{old_folder}":
        return f"/archive/{new_folder}"
    return value


def _plan_folder_rename(
    old_folder: str, index: dict[str, list[str]], archive_dir: Path
) -> tuple[Optional[FolderPlan], Optional[str]]:
    """Return (plan, skip_reason). Exactly one of the two is not None."""
    matches = index.get(old_folder, [])
    if not matches:
        return None, "no_match"
    if len(matches) > 1:
        return None, "ambiguous"
    email_doc_id = matches[0]
    new_folder = compute_folder_id(email_doc_id)
    if old_folder == new_folder:
        # No-op (shouldn't happen — 16 vs 32 char — but be safe).
        return None, "identical"
    target = archive_dir / new_folder
    if target.exists():
        return None, "target_exists"
    return FolderPlan(old_folder=old_folder, new_folder=new_folder, email_doc_id=email_doc_id), None


def plan_migration(output_dir: Path) -> MigrationResult:
    """Produce a migration plan for every folder in ``archive/``."""
    archive_dir = output_dir / "archive"
    documents_jsonl = output_dir / "documents.jsonl"

    result = MigrationResult()
    if not archive_dir.exists():
        logger.warning("No archive directory at %s", archive_dir)
        return result

    index = _build_prefix_index(documents_jsonl)

    for folder in sorted(archive_dir.iterdir()):
        if not folder.is_dir():
            continue
        old_name = folder.name
        # Skip folders that already look like 32-char names (already migrated).
        if len(old_name) == 32:
            continue
        if len(old_name) != 16:
            logger.warning("Unexpected archive folder name length: %s", old_name)
            continue
        plan, skip_reason = _plan_folder_rename(old_name, index, archive_dir)
        if plan is None:
            if skip_reason == "no_match":
                result.skipped_no_match.append(old_name)
            elif skip_reason == "ambiguous":
                result.skipped_ambiguous.append(old_name)
            elif skip_reason == "target_exists":
                result.skipped_target_exists.append(old_name)
            continue
        result.plans.append(plan)

    _count_uri_rewrites(result, output_dir)
    return result


def _count_uri_rewrites(result: MigrationResult, output_dir: Path) -> None:
    """Scan documents.jsonl / chunks.jsonl to count rewrites per plan."""
    old_to_plan = {p.old_folder: p for p in result.plans}

    for doc in _iter_jsonl(output_dir / "documents.jsonl"):
        for field_name in ("archive_browse_uri", "archive_download_uri"):
            value = doc.get(field_name)
            if not isinstance(value, str):
                continue
            for old, plan in old_to_plan.items():
                if f"/archive/{old}/" in value or value == f"/archive/{old}":
                    plan.doc_rewrites += 1
                    break

    for chunk in _iter_jsonl(output_dir / "chunks.jsonl"):
        for field_name in ("archive_browse_uri", "archive_download_uri"):
            value = chunk.get(field_name)
            if not isinstance(value, str):
                continue
            for old, plan in old_to_plan.items():
                if f"/archive/{old}/" in value or value == f"/archive/{old}":
                    plan.chunk_rewrites += 1
                    break


def _rewrite_jsonl(path: Path, plans: list[FolderPlan]) -> int:
    """Rewrite archive_browse_uri / archive_download_uri in place. Returns row count."""
    if not path.exists():
        return 0
    old_to_new = {p.old_folder: p.new_folder for p in plans}
    tmp_path = path.with_suffix(path.suffix + ".migrate.tmp")
    changed = 0
    total = 0
    with open(path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write(line)
                continue
            total += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                fout.write(line)
                continue
            modified = False
            for field_name in ("archive_browse_uri", "archive_download_uri"):
                value = row.get(field_name)
                if not isinstance(value, str):
                    continue
                for old, new in old_to_new.items():
                    rewritten = _rewrite_uri(value, old, new)
                    if rewritten != value:
                        row[field_name] = rewritten
                        modified = True
                        break
            if modified:
                changed += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                fout.write(line)
    tmp_path.replace(path)
    logger.info("Rewrote %s: %d/%d rows changed", path.name, changed, total)
    return changed


def _update_metadata_json(archive_dir: Path, plan: FolderPlan) -> None:
    """Update the ``folder_id`` field inside the renamed folder's metadata.json."""
    metadata_path = archive_dir / plan.new_folder / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Malformed metadata.json at %s — leaving as-is", metadata_path)
        return
    if data.get("folder_id") == plan.new_folder:
        return
    data["folder_id"] = plan.new_folder
    metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_migration(output_dir: Path, result: MigrationResult) -> None:
    """Apply rename + URI rewrite. Call plan_migration first."""
    archive_dir = output_dir / "archive"
    # Rename folders first — then rewrite JSONL, so a crash mid-way leaves
    # pointers to non-existent folders (visible in validate) rather than
    # orphaned folders with no pointers at all.
    for plan in result.plans:
        src = archive_dir / plan.old_folder
        dst = archive_dir / plan.new_folder
        if not src.exists():
            logger.warning("Source folder vanished: %s", src)
            continue
        if dst.exists():
            logger.warning("Target folder appeared since plan: %s — skipping", dst)
            continue
        shutil.move(str(src), str(dst))
        _update_metadata_json(archive_dir, plan)

    _rewrite_jsonl(output_dir / "documents.jsonl", result.plans)
    _rewrite_jsonl(output_dir / "chunks.jsonl", result.plans)


def _print_report(result: MigrationResult, apply: bool) -> None:
    total_doc = sum(p.doc_rewrites for p in result.plans)
    total_chunk = sum(p.chunk_rewrites for p in result.plans)
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n[{mode}] migrate_folder_prefix summary")
    print(f"  folders to rename: {len(result.plans)}")
    print(f"  document URI rewrites: {total_doc}")
    print(f"  chunk URI rewrites: {total_chunk}")
    if result.skipped_no_match:
        print(f"  SKIPPED (no matching email doc): {len(result.skipped_no_match)}")
        for name in result.skipped_no_match[:5]:
            print(f"    - {name}")
        if len(result.skipped_no_match) > 5:
            print(f"    ... and {len(result.skipped_no_match) - 5} more")
    if result.skipped_ambiguous:
        print(f"  SKIPPED (multiple doc_id prefix collisions): {len(result.skipped_ambiguous)}")
        for name in result.skipped_ambiguous[:5]:
            print(f"    - {name}")
    if result.skipped_target_exists:
        print(f"  SKIPPED (target folder already exists): {len(result.skipped_target_exists)}")
        for name in result.skipped_target_exists[:5]:
            print(f"    - {name}")
    if result.plans:
        print("\n  sample plan entries:")
        for plan in result.plans[:3]:
            print(
                f"    {plan.old_folder} -> {plan.new_folder}  "
                f"(doc={plan.doc_rewrites}, chunk={plan.chunk_rewrites})"
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Output directory containing archive/, documents.jsonl, chunks.jsonl",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the rename + rewrite. Default: dry-run.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    output_dir = args.output_dir
    if not output_dir.exists():
        logger.error("Output directory does not exist: %s", output_dir)
        return 1

    result = plan_migration(output_dir)
    _print_report(result, apply=args.apply)
    if not args.apply:
        print("\n  (dry-run) re-run with --apply to execute.")
        return 0
    if not result.plans:
        print("\n  nothing to apply.")
        return 0
    apply_migration(output_dir, result)
    print("\n  migration applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
