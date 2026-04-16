"""Migrate archive file keys from URL-encoded to underscore-based naming.

Renames files in archive/ folder, updates URIs in documents.jsonl,
and fixes internal references in markdown and metadata.json files.
Run with --dry-run first to preview changes.

Usage:
    python scripts/migrate_archive_keys.py --output-dir data/output --dry-run
    python scripts/migrate_archive_keys.py --output-dir data/output
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote


def _new_key(old_name: str) -> str:
    """Convert a URL-encoded filename to underscore-based naming.

    Applies the same transformations as the updated _sanitize_storage_key:
    - URL-decode first (%20 -> space, %27 -> ', etc.)
    - Spaces -> underscores
    - Special chars (', &, #, ,) -> underscores
    - Collapse multiple underscores
    - Strip leading/trailing underscores (preserve extension)
    """
    decoded = unquote(old_name)
    result = decoded.replace(" ", "_")
    result = re.sub(r"[',&#]+", "_", result)
    result = re.sub(r"_+", "_", result)
    # Strip leading/trailing underscores but preserve extension
    name_part, dot, ext_part = result.rpartition(".")
    if name_part and dot:
        result = f"{name_part.strip('_')}.{ext_part}"
    return result


_ENCODED_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def _fix_encoded_refs(text: str) -> tuple[str, int]:
    """Replace all URL-encoded path segments in text with underscore versions.

    Handles both markdown links like (folder/file%20name.pdf) and
    bare path strings in JSON values.

    Returns:
        (updated_text, number_of_replacements)
    """
    count = 0

    def _replace_match(m: re.Match) -> str:
        nonlocal count
        full = m.group(0)
        # Decode and re-sanitize the segment
        new = _new_key(full)
        if new != full:
            count += 1
        return new

    # Match path-like strings containing %XX encoding
    # Covers: folder/attachments/file%20name.pdf and similar
    result = re.sub(
        r"[A-Za-z0-9_./()\-]+%[0-9A-Fa-f]{2}[A-Za-z0-9_./()\-%]*",
        _replace_match,
        text,
    )
    return result, count


def migrate(output_dir: Path, dry_run: bool = True) -> None:
    archive_dir = output_dir / "archive"
    docs_path = output_dir / "documents.jsonl"

    if not archive_dir.exists():
        print(f"No archive directory at {archive_dir}")
        return
    if not docs_path.exists():
        print(f"No documents.jsonl at {docs_path}")
        return

    # === Phase 1: Rename archive files on disk ===
    renames: list[tuple[Path, Path]] = []
    for f in sorted(archive_dir.rglob("*")):
        if f.is_file() and "%" in f.name:
            new_name = _new_key(f.name)
            if new_name != f.name:
                new_path = f.parent / new_name
                renames.append((f, new_path))

    print(f"Phase 1 — Archive files to rename: {len(renames)}")
    for old, new in renames[:10]:
        print(f"  {old.name} -> {new.name}")
    if len(renames) > 10:
        print(f"  ... and {len(renames) - 10} more")

    # === Phase 2: Update URIs in documents.jsonl ===
    lines = docs_path.read_text(encoding="utf-8").splitlines()
    uri_updates = 0
    updated_doc_lines = []
    for line in lines:
        if not line.strip():
            updated_doc_lines.append(line)
            continue
        d = json.loads(line)
        changed = False
        for key in ("archive_browse_uri", "archive_download_uri"):
            uri = d.get(key)
            if uri and "%" in uri:
                parts = uri.split("/")
                new_parts = []
                for part in parts:
                    if "%" in part:
                        new_parts.append(_new_key(part))
                    else:
                        new_parts.append(part)
                new_uri = "/".join(new_parts)
                if new_uri != uri:
                    d[key] = new_uri
                    changed = True
                    uri_updates += 1
        if changed:
            updated_doc_lines.append(json.dumps(d, default=str))
        else:
            updated_doc_lines.append(line)

    print(f"Phase 2 — URI fields in documents.jsonl to update: {uri_updates}")

    # === Phase 3: Fix internal references in markdown files ===
    md_fixes = 0
    md_files_fixed = 0
    md_files_to_update: list[tuple[Path, str]] = []
    for md_file in archive_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        if not _ENCODED_RE.search(content):
            continue
        new_content, fixes = _fix_encoded_refs(content)
        if fixes:
            md_fixes += fixes
            md_files_fixed += 1
            md_files_to_update.append((md_file, new_content))

    print(f"Phase 3 — Markdown files with stale refs: {md_files_fixed} ({md_fixes} links)")

    # === Phase 4: Fix internal references in metadata.json files ===
    meta_fixes = 0
    meta_files_fixed = 0
    meta_files_to_update: list[tuple[Path, str]] = []
    for meta_file in archive_dir.rglob("metadata.json"):
        content = meta_file.read_text(encoding="utf-8")
        if not _ENCODED_RE.search(content):
            continue
        new_content, fixes = _fix_encoded_refs(content)
        if fixes:
            meta_fixes += fixes
            meta_files_fixed += 1
            meta_files_to_update.append((meta_file, new_content))

    print(f"Phase 4 — metadata.json files with stale refs: {meta_files_fixed} ({meta_fixes} refs)")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply.")
        return

    # === Apply Phase 1: file renames ===
    renamed = 0
    errors = 0
    for old, new in renames:
        try:
            if new.exists():
                print(f"  SKIP (exists): {new}")
                continue
            old.rename(new)
            renamed += 1
        except Exception as e:
            print(f"  ERROR renaming {old.name}: {e}")
            errors += 1
    print(f"Renamed: {renamed} files ({errors} errors)")

    # === Apply Phase 2: documents.jsonl ===
    docs_path.write_text("\n".join(updated_doc_lines) + "\n", encoding="utf-8")
    print(f"Updated documents.jsonl ({uri_updates} URI fields)")

    # === Apply Phase 3: markdown files ===
    for md_file, new_content in md_files_to_update:
        md_file.write_text(new_content, encoding="utf-8")
    print(f"Updated {md_files_fixed} markdown files ({md_fixes} links)")

    # === Apply Phase 4: metadata.json files ===
    for meta_file, new_content in meta_files_to_update:
        meta_file.write_text(new_content, encoding="utf-8")
    print(f"Updated {meta_files_fixed} metadata.json files ({meta_fixes} refs)")

    print("\nDone. Run 'mtss validate ingest' to verify, then clean Supabase and re-import.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate archive keys from URL-encoded to underscores")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (e.g., data/output)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    migrate(args.output_dir, dry_run=args.dry_run)
