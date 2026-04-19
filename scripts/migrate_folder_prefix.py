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


@dataclass
class InnerRewriteResult:
    """Stats from the per-folder metadata.json URI rewrite pass.

    Runs separately from the folder-rename pass so it also catches folders
    that were already renamed by an earlier run (the rename pass records
    zero plans on a re-run, but the inner URIs can still be stale).
    """

    folders_touched: int = 0
    folders_skipped_malformed: int = 0
    total_refs_rewritten: int = 0
    samples: list[tuple[str, str, str]] = field(default_factory=list)  # (folder, old_prefix, new_prefix)


@dataclass
class MarkdownRewriteResult:
    """Stats from rewriting stale ``](doc_id/...`` links inside archive *.md."""

    folders_touched: int = 0
    files_touched: int = 0
    total_links_rewritten: int = 0


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


def _rewrite_prefix_str(value: object, old_prefix: str, new_prefix: str) -> tuple[object, bool]:
    """If ``value`` is a string starting with ``<old_prefix>/`` or equals
    ``<old_prefix>``, return the rewritten version + True. Else untouched + False."""
    if not isinstance(value, str):
        return value, False
    if value.startswith(f"{old_prefix}/"):
        return f"{new_prefix}/" + value[len(old_prefix) + 1:], True
    if value == old_prefix:
        return new_prefix, True
    return value, False


def _rewrite_metadata_inner_uris(data: dict, old_prefix: str, new_prefix: str) -> int:
    """Rewrite every ``<old_prefix>/...`` reference inside ``data`` to use
    ``<new_prefix>/...`` . Mutates ``data`` in place. Returns ref count changed.

    Rewrites:
      - ``email_browse_uri``, ``email_download_uri``
      - ``attachments`` keys (rebuilt)
      - ``attachments[*].download_uri``, ``attachments[*].browse_uri``
    """
    changed = 0
    for field_name in ("email_browse_uri", "email_download_uri"):
        new_value, was_changed = _rewrite_prefix_str(data.get(field_name), old_prefix, new_prefix)
        if was_changed:
            data[field_name] = new_value
            changed += 1

    attachments = data.get("attachments")
    if isinstance(attachments, dict) and attachments:
        new_attachments: dict = {}
        for key, att in attachments.items():
            new_key, key_changed = _rewrite_prefix_str(key, old_prefix, new_prefix)
            if key_changed:
                changed += 1
            if isinstance(att, dict):
                new_att = dict(att)
                for uri_field in ("download_uri", "browse_uri"):
                    new_value, was_changed = _rewrite_prefix_str(new_att.get(uri_field), old_prefix, new_prefix)
                    if was_changed:
                        new_att[uri_field] = new_value
                        changed += 1
                new_attachments[new_key] = new_att
            else:
                new_attachments[new_key] = att
        data["attachments"] = new_attachments

    return changed


def _rewrite_folder_markdown_links(
    folder: Path, doc_id: str, folder_name: str, apply: bool
) -> tuple[int, int]:
    """Replace ``](<doc_id>/`` with ``](<folder_name>/`` across every *.md
    file under ``folder``. Returns (files_touched, total_links_rewritten).

    Scoped to the markdown link-target prefix (``](``) so plain-text
    occurrences of the doc_id are left alone.
    """
    old_token = f"]({doc_id}/"
    new_token = f"]({folder_name}/"
    files_touched = 0
    total = 0
    for md in folder.rglob("*.md"):
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", md, exc)
            continue
        count = content.count(old_token)
        if count == 0:
            continue
        files_touched += 1
        total += count
        if apply:
            md.write_text(content.replace(old_token, new_token), encoding="utf-8")
    return files_touched, total


def rewrite_all_markdown_links(output_dir: Path, apply: bool) -> MarkdownRewriteResult:
    """Third pass: rewrite stale ``](<doc_id>/...`` links inside every
    folder's markdown files. Safe to re-run."""
    archive_dir = output_dir / "archive"
    result = MarkdownRewriteResult()
    if not archive_dir.exists():
        return result

    for folder in sorted(archive_dir.iterdir()):
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        doc_id = data.get("doc_id")
        folder_name = folder.name
        if not isinstance(doc_id, str) or not doc_id or doc_id == folder_name:
            continue

        files_touched, links = _rewrite_folder_markdown_links(
            folder, doc_id, folder_name, apply=apply
        )
        if links:
            result.folders_touched += 1
            result.files_touched += files_touched
            result.total_links_rewritten += links
    return result


def rewrite_all_inner_metadata_uris(output_dir: Path, apply: bool) -> InnerRewriteResult:
    """Second pass: rewrite stale ``<doc_id>/...`` URIs inside every folder's
    ``metadata.json`` to use the folder's actual name.

    Runs independently of the folder-rename pass. Safe to re-run — folders
    whose inner URIs already match the folder name are skipped.
    """
    archive_dir = output_dir / "archive"
    result = InnerRewriteResult()
    if not archive_dir.exists():
        return result

    for folder in sorted(archive_dir.iterdir()):
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result.folders_skipped_malformed += 1
            logger.warning("Malformed metadata.json at %s — skipping", metadata_path)
            continue
        doc_id = data.get("doc_id")
        folder_name = folder.name
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if doc_id == folder_name:
            # Legacy layout where folder_id == doc_id; no rewrite needed.
            continue

        refs_changed = _rewrite_metadata_inner_uris(data, doc_id, folder_name)
        # Top-level folder_id should always match on-disk folder name.
        if data.get("folder_id") != folder_name:
            data["folder_id"] = folder_name
            refs_changed += 1

        if refs_changed:
            result.folders_touched += 1
            result.total_refs_rewritten += refs_changed
            if len(result.samples) < 3:
                result.samples.append((folder_name, doc_id, folder_name))
            if apply:
                metadata_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    return result


def apply_migration(output_dir: Path, result: MigrationResult) -> None:
    """Apply rename + URI rewrite. Call plan_migration first.

    The per-folder metadata.json inner-URI pass is invoked separately via
    ``rewrite_all_inner_metadata_uris`` so it also heals folders that were
    renamed by a previous partial run.
    """
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


def _print_report(
    result: MigrationResult,
    inner: InnerRewriteResult,
    md_rewrite: MarkdownRewriteResult,
    apply: bool,
) -> None:
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

    print("\n  per-folder metadata.json inner-URI pass:")
    print(f"    folders to rewrite: {inner.folders_touched}")
    print(f"    total stale refs rewritten: {inner.total_refs_rewritten}")
    if inner.folders_skipped_malformed:
        print(f"    malformed (skipped): {inner.folders_skipped_malformed}")

    print("\n  markdown link-target pass (*.md inside archive/):")
    print(f"    folders to rewrite: {md_rewrite.folders_touched}")
    print(f"    files to rewrite: {md_rewrite.files_touched}")
    print(f"    total link rewrites: {md_rewrite.total_links_rewritten}")


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
    if args.apply and result.plans:
        apply_migration(output_dir, result)
    # Always run the inner-URI + markdown-link passes — they also heal
    # folders renamed by a previous run. In dry-run mode they only count.
    inner = rewrite_all_inner_metadata_uris(output_dir, apply=args.apply)
    md_rewrite = rewrite_all_markdown_links(output_dir, apply=args.apply)
    _print_report(result, inner, md_rewrite, apply=args.apply)
    if not args.apply:
        print("\n  (dry-run) re-run with --apply to execute.")
        return 0
    if (
        not result.plans
        and inner.folders_touched == 0
        and md_rewrite.folders_touched == 0
    ):
        print("\n  nothing to apply.")
        return 0
    print("\n  migration applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
