"""One-off: rebuild missing attachment .md files from chunks.jsonl.

Targets documents whose ``archive_browse_uri`` points to a .md that is no
longer on disk. For image attachments the chunk ``content`` field already
holds the vision description byte-for-byte, so reconstruction is lossless.

Usage:
    uv run python scripts/regen_missing_md_from_chunks.py --dry-run
    uv run python scripts/regen_missing_md_from_chunks.py
    uv run python scripts/regen_missing_md_from_chunks.py --types attachment_image
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_TYPES = ("attachment_image",)  # lossless reconstruction


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _render_md(filename: str, content_type: str, size_bytes: int,
               download_rel: str, content: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {filename}",
        "",
        f"**Type:** {content_type}",
        f"**Size:** {_format_size(size_bytes)}",
        f"**Extracted:** {ts}",
        "",
        f"[Download Original]({download_rel})",
        "",
        "---",
        "",
        "## Content",
        "",
        content,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(DEFAULT_TYPES),
        help="Document types to repair. Default: attachment_image (lossless).",
    )
    args = parser.parse_args()

    docs_path: Path = args.output_dir / "documents.jsonl"
    chunks_path: Path = args.output_dir / "chunks.jsonl"
    archive_root: Path = args.output_dir / "archive"

    if not docs_path.exists() or not chunks_path.exists():
        print(f"missing documents.jsonl or chunks.jsonl under {args.output_dir}")
        return 1

    allowed_types = set(args.types)

    broken: list[dict] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("document_type") not in allowed_types:
                continue
            bu = d.get("archive_browse_uri")
            if not bu:
                continue
            rel = bu.removeprefix("/archive/")
            if (archive_root / rel).is_file():
                continue
            broken.append(d)

    if not broken:
        print("no broken .md URIs in scope — nothing to do.")
        return 0

    broken_ids = {d["id"] for d in broken}
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("document_id") in broken_ids:
                chunks_by_doc[c["document_id"]].append(c)

    written = 0
    skipped = 0
    by_type: dict[str, int] = defaultdict(int)
    for d in broken:
        chunks = sorted(
            chunks_by_doc.get(d["id"], []),
            key=lambda c: (c.get("char_start") or 0, c.get("char_end") or 0),
        )
        if not chunks:
            skipped += 1
            continue
        content = "\n\n".join(
            (c.get("content") or "").strip() for c in chunks if (c.get("content") or "").strip()
        )
        if not content:
            skipped += 1
            continue

        bu_rel = d["archive_browse_uri"].removeprefix("/archive/")
        dest = archive_root / bu_rel
        du = d.get("archive_download_uri") or ""
        download_rel = du.removeprefix("/archive/") or dest.name.removesuffix(".md")

        md = _render_md(
            filename=d.get("file_name") or dest.stem,
            content_type=d.get("attachment_content_type") or "application/octet-stream",
            size_bytes=int(d.get("attachment_size_bytes") or 0),
            download_rel=download_rel,
            content=content,
        )

        if args.dry_run:
            print(f"  would write: {dest.relative_to(args.output_dir)} ({len(content)} chars)")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(md, encoding="utf-8")
        written += 1
        by_type[d.get("document_type") or "?"] += 1

    action = "would write" if args.dry_run else "wrote"
    print(f"\n{action} {written} .md files, skipped {skipped} (no chunk content).")
    for k, v in sorted(by_type.items()):
        print(f"  {v:5d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
