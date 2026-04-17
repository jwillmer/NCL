"""Relabel attachment docs that should be ATTACHMENT_IMAGE but got ATTACHMENT_OTHER.

Why this exists
---------------
``AttachmentProcessor.MIME_TO_DOC_TYPE`` originally omitted ``image/gif`` and
``image/webp`` even though ``ImageProcessor.SUPPORTED_TYPES`` and the lane
classifier already handled them. A ``.gif`` attachment therefore ran through
the vision pipeline (producing a valid ``image_description`` chunk) but its
document row stayed labelled ``attachment_other``. Check 7 of
``mtss validate ingest`` skips ``attachment_image`` chunks when looking for
missing ``context_summary``/``embedding_text`` (image chunks legitimately
lack those), so the mis-typed docs leaked through as false-positive warnings.

The pipeline fix lives in ``src/mtss/parsers/attachment_processor.py`` plus
the parity test in ``tests/test_attachment_processor.py::TestImageMimeDocTypeParity``.
This script cleans up the already-ingested docs so we don't need to re-parse
the affected emails.

Scope
-----
- Metadata-only: flips ``document_type`` on matched rows.
- Does NOT touch chunks, embeddings, archive files, or manifests.
- Matches on either ``content_type`` (preferred) or ``file_name`` suffix
  (``.gif``/``.webp``/``.png`` — the last only when ``content_type`` is
  the legacy ``image/x-png`` alias).

Usage
-----
    python scripts/migrate_image_attachment_types.py --output-dir data/output --dry-run
    python scripts/migrate_image_attachment_types.py --output-dir data/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


# MIME types the image pipeline handles but MIME_TO_DOC_TYPE previously missed
# (or still aliases to the canonical form).
_IMAGE_MIMES = {"image/gif", "image/webp", "image/x-png"}

# File-name suffixes to catch rows where content_type is missing/None.
_IMAGE_SUFFIXES = {".gif", ".webp"}


def _needs_relabel(doc: dict) -> bool:
    if doc.get("document_type") != "attachment_other":
        return False
    ct = (doc.get("content_type") or "").lower()
    if ct in _IMAGE_MIMES:
        return True
    name = (doc.get("file_name") or "").lower()
    return any(name.endswith(suffix) for suffix in _IMAGE_SUFFIXES)


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        yield from f


def migrate(output_dir: Path, dry_run: bool) -> int:
    docs_path = output_dir / "documents.jsonl"
    if not docs_path.exists():
        print(f"error: {docs_path} not found", file=sys.stderr)
        return 2

    to_fix: list[dict] = []
    total = 0
    for line in _iter_lines(docs_path):
        line = line.strip()
        if not line:
            continue
        total += 1
        doc = json.loads(line)
        if _needs_relabel(doc):
            to_fix.append(doc)

    print(f"scanned {total} docs — {len(to_fix)} need relabel")
    for doc in to_fix:
        print(
            f"  {doc.get('id', '?')[:8]}..  "
            f"file={doc.get('file_name', '?')}  "
            f"ct={doc.get('content_type')!r}  "
            f"type={doc.get('document_type')} -> attachment_image"
        )

    if not to_fix:
        return 0
    if dry_run:
        print("dry-run: no changes written")
        return 0

    fix_ids = {doc["id"] for doc in to_fix}
    tmp_path = docs_path.with_suffix(docs_path.suffix + ".tmp")
    rewritten = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for line in _iter_lines(docs_path):
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("id") in fix_ids:
                doc["document_type"] = "attachment_image"
                rewritten += 1
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp_path, docs_path)
    print(f"rewrote {rewritten} rows in {docs_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Path to the ingest output dir (default: data/output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned changes without writing",
    )
    args = parser.parse_args()
    return migrate(args.output_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
