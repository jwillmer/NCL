"""Retroactively repair archives for ZIP-extracted attachments.

Background
----------
Before the fix in `process_zip_attachment`, extracted ZIP members never had
their "original" file uploaded into the email's archive folder. The
`update_attachment_markdown` call then short-circuited on a `file_exists`
guard and no `.md` preview was written. Net effect: docs exist in
documents.jsonl with chunks, but `archive_browse_uri` / `archive_download_uri`
are None and the archive folder has neither the member file nor its
`.md` preview.

This script repairs existing output without a full re-ingest:

For each ZIP-extracted doc missing archive URIs:
  1. Locate the parent ZIP attachment on disk (the ZIP was uploaded).
  2. Extract just the needed member from the ZIP (by basename match).
  3. Write the extracted member into `<archive>/<folder>/attachments/<safe_name>`.
  4. Reconstruct parsed_content from the doc's chunks in chunks.jsonl.
  5. Generate and write the `.md` preview using the same format as live ingest.
  6. Update the doc's `archive_browse_uri` and `archive_download_uri` in-place.

Nothing is re-parsed — no LlamaParse, no LLM, no vision calls. Existing
chunks are trusted as the source of truth for parsed text.

Usage
-----
    python scripts/repair_zip_archives.py --output-dir data/output --dry-run
    python scripts/repair_zip_archives.py --output-dir data/output

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Repo import
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.ingest.archive_generator import (  # noqa: E402
    ArchiveGenerator,
    _sanitize_storage_key,
)

logger = logging.getLogger("repair_zip_archives")


@dataclass
class RepairCandidate:
    doc: Dict
    source_zip_path: Path
    folder_id: str
    member_basename: str
    safe_member_name: str
    parsed_content: str


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl_atomic(path: Path, rows: List[Dict]) -> None:
    """Write rows to `path` via tmp-file rename. Preserves order."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------


def _is_missing_archive(doc: Dict) -> bool:
    """True if doc should have archive URIs but doesn't."""
    if doc.get("status") == "failed":
        return False
    if doc.get("document_type") == "email":
        return False
    # Images never had .md previews before either; skip here.
    if doc.get("document_type") == "attachment_image":
        return False
    return not doc.get("archive_browse_uri") and not doc.get("archive_download_uri")


def _find_source_zip_on_disk(
    doc: Dict,
    docs_by_id: Dict[str, Dict],
    archive_dir: Path,
    zip_namelist_cache: Dict[str, "tuple[Path, List[str]]"],
) -> Optional[Path]:
    """Return the on-disk ZIP containing `doc`'s basename as a member.

    Context: in the current pipeline, process_zip_attachment never creates
    a document row for the ZIP itself — only for its extracted members
    (with parent_id = email). So the ZIP isn't a sibling doc we can find
    via documents.jsonl. We instead enumerate the email's archive folder
    directly for `.zip` files and match by namelist.

    Strategy:
      1. Resolve the email's folder_id from the doc's root_id.
      2. Glob `<archive>/<folder_id>/attachments/*.zip` on disk.
      3. For each, read + cache its namelist, test basename match.
      4. Return the first ZIP that contains our doc's basename, or None.
    """
    # Skip non-candidates: emails, images, the ZIP itself.
    if doc.get("document_type") == "email":
        return None
    filename = (doc.get("file_name") or "").lower()
    if filename.endswith(".zip"):
        return None

    root_id = doc.get("root_id")
    if not root_id:
        return None
    email_doc = docs_by_id.get(str(root_id))
    if not email_doc:
        return None
    folder_id = (email_doc.get("doc_id") or "")[:16]
    if not folder_id:
        return None

    target_basename = Path(doc.get("file_name") or "").name.lower()
    if not target_basename:
        return None

    attachments_dir = archive_dir / folder_id / "attachments"
    if not attachments_dir.exists():
        return None

    for zip_path in sorted(attachments_dir.glob("*.zip")):
        cache_key = str(zip_path)
        if cache_key not in zip_namelist_cache:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = [
                        Path(info.filename).name.lower()
                        for info in zf.infolist()
                        if not info.is_dir()
                    ]
                zip_namelist_cache[cache_key] = (zip_path, names)
            except zipfile.BadZipFile:
                zip_namelist_cache[cache_key] = (zip_path, [])
        _, names = zip_namelist_cache[cache_key]
        if target_basename in names:
            return zip_path
    return None


def _find_zip_on_disk(archive_dir: Path, folder_id: str, zip_doc: Dict) -> Optional[Path]:
    """Locate the archived ZIP file.

    Prefer the doc's own archive_download_uri if set (trimmed of /archive/).
    Otherwise look for {folder_id}/attachments/{sanitized(zip_filename)}.
    """
    download_uri = zip_doc.get("archive_download_uri") or ""
    if download_uri.startswith("/archive/"):
        rel = download_uri[len("/archive/"):]
        candidate = archive_dir / rel
        if candidate.exists():
            return candidate
    zip_name = zip_doc.get("file_name") or ""
    if zip_name:
        safe = _sanitize_storage_key(zip_name)
        candidate = archive_dir / folder_id / "attachments" / safe
        if candidate.exists():
            return candidate
    # Fallback: scan the folder for any .zip
    folder = archive_dir / folder_id / "attachments"
    if folder.exists():
        zips = sorted(folder.glob("*.zip"))
        if len(zips) == 1:
            return zips[0]
    return None


def _extract_member_bytes(zip_path: Path, member_basename: str) -> Optional[bytes]:
    """Extract a ZIP member by basename match.

    ZIP entries may be nested under subfolders; we match on the final path
    component (case-insensitive). Returns None if not found or on ambiguous
    multi-match (logged and skipped).
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            matches = [
                info
                for info in zf.infolist()
                if not info.is_dir()
                and Path(info.filename).name.lower() == member_basename.lower()
            ]
            if not matches:
                return None
            if len(matches) > 1:
                logger.warning(
                    "Ambiguous member %s in %s (%d matches) — skipping",
                    member_basename, zip_path, len(matches),
                )
                return None
            return zf.read(matches[0])
    except zipfile.BadZipFile:
        logger.warning("Bad zip file: %s", zip_path)
        return None


# ---------------------------------------------------------------------------
# Main repair
# ---------------------------------------------------------------------------


def build_candidates(
    output_dir: Path, verbose: bool
) -> tuple[List[RepairCandidate], List[Dict]]:
    """Scan output and return repairable candidates plus the full docs list."""
    archive_dir = output_dir / "archive"
    docs = _read_jsonl(output_dir / "documents.jsonl")
    chunks = _read_jsonl(output_dir / "chunks.jsonl")

    docs_by_id: Dict[str, Dict] = {d["id"]: d for d in docs if "id" in d}

    # Group chunks by document_id in insertion order for content reconstruction.
    chunks_by_doc: Dict[str, List[Dict]] = {}
    for c in chunks:
        did = c.get("document_id")
        if did:
            chunks_by_doc.setdefault(did, []).append(c)

    candidates: List[RepairCandidate] = []
    skipped_no_parent = 0
    skipped_no_chunks = 0
    skipped_not_zip = 0

    # Cache ZIP namelists across candidate detection to avoid reopening each ZIP.
    zip_namelist_cache: Dict[str, "tuple[Path, List[str]]"] = {}

    for doc in docs:
        if not _is_missing_archive(doc):
            continue
        root_id = doc.get("root_id")
        root_doc = docs_by_id.get(str(root_id)) if root_id else None
        if not root_doc:
            skipped_no_parent += 1
            continue
        folder_id = (root_doc.get("doc_id") or "")[:16]
        if not folder_id:
            skipped_no_parent += 1
            continue
        source_zip = _find_source_zip_on_disk(
            doc, docs_by_id, archive_dir, zip_namelist_cache
        )
        if not source_zip:
            skipped_not_zip += 1
            continue
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        if not doc_chunks:
            skipped_no_chunks += 1
            continue

        parsed_content = "\n\n".join(
            (c.get("content") or "").strip() for c in doc_chunks if c.get("content")
        ).strip()
        if not parsed_content:
            skipped_no_chunks += 1
            continue

        member_basename = doc.get("file_name") or Path(doc.get("source_id", "")).name
        if not member_basename:
            continue
        safe_member_name = _sanitize_storage_key(member_basename)

        candidates.append(
            RepairCandidate(
                doc=doc,
                source_zip_path=source_zip,
                folder_id=folder_id,
                member_basename=member_basename,
                safe_member_name=safe_member_name,
                parsed_content=parsed_content,
            )
        )

    if verbose:
        logger.info(
            "Candidate scan: %d to repair; skipped %d non-zip, %d no-chunks, %d no-parent",
            len(candidates), skipped_not_zip, skipped_no_chunks, skipped_no_parent,
        )
    return candidates, docs


def repair(
    output_dir: Path,
    dry_run: bool,
    limit: int,
    verbose: bool,
) -> int:
    archive_dir = output_dir / "archive"
    if not archive_dir.exists():
        logger.error("Archive dir not found: %s", archive_dir)
        return 1

    candidates, docs = build_candidates(output_dir, verbose)
    if limit > 0:
        candidates = candidates[:limit]

    logger.info("Found %d ZIP-member docs to repair", len(candidates))
    if not candidates:
        return 0

    # Reuse the ingest-side markdown generator so format matches new ingests.
    class _NullStorage:
        """No-op storage used only so ArchiveGenerator instantiates."""
        def upload_file(self, *a, **kw): return ""
        def upload_text(self, *a, **kw): return ""
        def file_exists(self, *a, **kw): return True

    generator = ArchiveGenerator(ingest_root=output_dir, storage=_NullStorage())

    docs_by_id = {d["id"]: d for d in docs if "id" in d}

    repaired = 0
    skipped_zip_missing = 0
    skipped_member_missing = 0
    failed = 0

    for idx, cand in enumerate(candidates, 1):
        doc = cand.doc
        zip_path = cand.source_zip_path
        if not zip_path.exists():
            # Detected at candidate time but gone now — skip without failing.
            skipped_zip_missing += 1
            if verbose:
                logger.info(
                    "[%d/%d] ZIP missing on disk for %s at %s",
                    idx, len(candidates), cand.member_basename, zip_path,
                )
            continue

        member_bytes = _extract_member_bytes(zip_path, cand.member_basename)
        if member_bytes is None:
            skipped_member_missing += 1
            if verbose:
                logger.info(
                    "[%d/%d] Member %s not in %s",
                    idx, len(candidates), cand.member_basename, zip_path.name,
                )
            continue

        folder = archive_dir / cand.folder_id / "attachments"
        original_target = folder / cand.safe_member_name
        md_target = folder / f"{cand.safe_member_name}.md"

        # Content type from doc metadata (best-effort)
        attach_meta = doc.get("attachment_metadata") or {}
        content_type = (
            attach_meta.get("content_type") if isinstance(attach_meta, dict) else None
        ) or "application/octet-stream"

        size_bytes = len(member_bytes)
        md_content = generator._generate_content_markdown(
            filename=cand.member_basename,
            content_type=content_type,
            size_bytes=size_bytes,
            parsed_content=cand.parsed_content,
            folder_id=cand.folder_id,
        )

        browse_uri = f"/archive/{cand.folder_id}/attachments/{cand.safe_member_name}.md"
        download_uri = f"/archive/{cand.folder_id}/attachments/{cand.safe_member_name}"

        if dry_run:
            if verbose:
                logger.info(
                    "[%d/%d] would write %s (%d bytes) + %s (%d chars); set URIs on doc_id=%s",
                    idx, len(candidates),
                    original_target.relative_to(archive_dir),
                    size_bytes,
                    md_target.relative_to(archive_dir),
                    len(md_content),
                    doc.get("doc_id", "?")[:16],
                )
            repaired += 1
            continue

        try:
            folder.mkdir(parents=True, exist_ok=True)
            # Do not overwrite an existing original blindly — only write if absent.
            if not original_target.exists():
                original_target.write_bytes(member_bytes)
            md_target.write_text(md_content, encoding="utf-8")

            # Update doc row in-place
            doc["archive_browse_uri"] = browse_uri
            doc["archive_download_uri"] = download_uri
            repaired += 1
        except Exception as e:
            failed += 1
            logger.warning(
                "Failed to repair %s: %s", cand.member_basename, e
            )

    if not dry_run and repaired:
        # Rewrite documents.jsonl atomically; docs list already mutated in place.
        _write_jsonl_atomic(output_dir / "documents.jsonl", docs)

    logger.info(
        "Summary: repaired=%d dry_run=%s zip_missing=%d member_missing=%d failed=%d",
        repaired, dry_run, skipped_zip_missing, skipped_member_missing, failed,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing documents.jsonl / chunks.jsonl / archive/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Scan and report without writing. Safe default for first run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max candidates to process (0 = all). Useful for a subset test.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Per-doc progress logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return repair(args.output_dir, args.dry_run, args.limit, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
