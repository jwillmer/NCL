"""Migrate archive file keys from URL-encoded to underscore-based naming.

Renames files in archive/ folder and updates URIs in documents.jsonl.
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

    print(f"Archive files to rename: {len(renames)}")
    for old, new in renames[:20]:
        print(f"  {old.name} -> {new.name}")
    if len(renames) > 20:
        print(f"  ... and {len(renames) - 20} more")

    # === Phase 2: Update URIs in documents.jsonl ===
    lines = docs_path.read_text(encoding="utf-8").splitlines()
    uri_updates = 0
    updated_lines = []
    for line in lines:
        if not line.strip():
            updated_lines.append(line)
            continue
        d = json.loads(line)
        changed = False
        for key in ("archive_browse_uri", "archive_download_uri"):
            uri = d.get(key)
            if uri and "%" in uri:
                # Split URI into segments, decode and re-sanitize each filename part
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
            updated_lines.append(json.dumps(d, default=str))
        else:
            updated_lines.append(line)

    print(f"URI fields to update: {uri_updates}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply.")
        return

    # === Apply file renames ===
    renamed = 0
    errors = 0
    for old, new in renames:
        try:
            if new.exists():
                # Target already exists (shouldn't happen), skip to be safe
                print(f"  SKIP (exists): {new}")
                continue
            old.rename(new)
            renamed += 1
        except Exception as e:
            print(f"  ERROR renaming {old.name}: {e}")
            errors += 1

    print(f"Renamed: {renamed} files ({errors} errors)")

    # === Apply JSONL updates ===
    docs_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Updated documents.jsonl ({uri_updates} URI fields)")

    print("\nDone. Run 'mtss validate ingest' to verify, then clean Supabase and re-import.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate archive keys from URL-encoded to underscores")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (e.g., data/output)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    migrate(args.output_dir, dry_run=args.dry_run)
