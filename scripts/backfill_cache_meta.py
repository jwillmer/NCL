"""Backfill ``.meta.json`` sidecars for existing attachment ``.md`` caches.

Background
----------
The cache read path in ``attachment_handler._process_non_zip_attachment`` now
validates a ``<safe_filename>.meta.json`` sidecar before trusting a cached
``<safe_filename>.md``. When the sidecar reports a different parser than the
router's current choice, the cache is treated as stale and re-parsed.

Existing archives written before the sidecar shipped have no ``.meta.json``.
The read path tolerates that (legacy caches are still honoured), but we lose
the parser-change detection unless we backfill. This script walks the archive
and writes a sidecar for every cached ``.md`` by inferring the parser from
the ``documents`` table in ``ingest.db`` — no re-parse, no LLM calls, zero
API cost.

Inference rules
---------------
For each ``archive/<folder_id>/attachments/<safe_filename>.md``:

1. Locate the matching document in the ``documents`` table by
   ``parent_id -> folder_id`` and ``file_name == <original_filename>``.
2. Read the document's ``processing_trail.steps[*].parse.parser`` / ``.model``
   stamp (the trail is the source of truth for which parser produced the
   content).
3. Write ``<safe_filename>.meta.json`` with
   ``{"parser": ..., "model": ..., "parsed_at": <doc.processed_at>,
     "backfilled": true}``.

If the trail is missing or lacks a parse step (very old records), the sidecar
is skipped and a warning is logged — the read path will continue trusting the
legacy cache.

Usage
-----
    # Dry-run (default). Reports counts and sample entries, no writes.
    python scripts/backfill_cache_meta.py --output-dir data/output

    # Real run. Only run after inspecting the dry-run output.
    python scripts/backfill_cache_meta.py --output-dir data/output --apply

Exit codes: 0 = success (including dry-run), 1 = unrecoverable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtss.ingest.archive_generator import _sanitize_storage_key  # noqa: E402
from mtss.utils import compute_folder_id  # noqa: E402

logger = logging.getLogger("backfill_cache_meta")


@dataclass
class SidecarPlan:
    md_path: Path
    meta_path: Path
    parser_name: str
    parser_model: Optional[str]
    parsed_at: Optional[str]
    source: str  # How parser was inferred (trail | attachment_meta | unknown)


def _iter_documents(output_dir: Path) -> Iterator[dict]:
    """Yield documents from ``ingest.db``."""
    from mtss.storage.sqlite_client import SqliteStorageClient

    db_path = output_dir / "ingest.db"
    if not db_path.exists():
        logger.error("ingest.db not found in %s", output_dir)
        return
    client = SqliteStorageClient(output_dir=output_dir)
    try:
        yield from client.iter_documents()
    finally:
        try:
            client._conn.close()
        except Exception:
            pass


def _read_metadata_json(folder: Path) -> Optional[dict]:
    metadata_path = folder / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", metadata_path, exc)
        return None


def _parser_from_trail(metadata: dict, attachment_filename: str) -> tuple[Optional[str], Optional[str]]:
    """Return (parser_name, parser_model) from the stored processing_trail."""
    if not metadata:
        return None, None
    trail = metadata.get("processing_trail") or metadata.get("trail") or {}
    attachments = trail.get("attachments") or {}
    # Keys can be the filename or "<zip_filename>/<member>" — try both.
    entry = attachments.get(attachment_filename)
    if entry is None:
        for key, value in attachments.items():
            if key.endswith("/" + attachment_filename) or key == attachment_filename:
                entry = value
                break
    if not isinstance(entry, dict):
        return None, None
    steps = entry.get("steps") or entry.get("step_stamps") or {}
    parse_step = steps.get("parse") or steps.get("STEP_PARSE") or {}
    if not isinstance(parse_step, dict):
        return None, None
    return parse_step.get("parser"), parse_step.get("model")


def _plan_sidecars(output_dir: Path) -> list[SidecarPlan]:
    archive_dir = output_dir / "archive"
    if not archive_dir.exists():
        logger.error("No archive/ directory under %s", output_dir)
        return []

    # First pass: build UUID → doc_id index so attachments (whose parent_id
    # references the email's UUID in SQLite) can resolve back to the email's
    # stable doc_id. ``compute_folder_id`` must be called on the doc_id, not
    # the UUID, or the resulting folder won't match what ingest wrote.
    all_docs = list(_iter_documents(output_dir))
    doc_id_by_uuid: dict[str, str] = {}
    for doc in all_docs:
        uid = doc.get("id")
        did = doc.get("doc_id")
        if uid and did:
            doc_id_by_uuid[str(uid)] = did

    # Build lookup: folder_id → [documents]. Attachments inherit the parent
    # email's folder.
    by_folder: dict[str, list[dict]] = {}
    for doc in all_docs:
        doc_id = doc.get("doc_id")
        parent_id = doc.get("parent_id")
        file_name = doc.get("file_name")
        if not file_name:
            continue
        folder_doc_id = doc_id_by_uuid.get(str(parent_id)) if parent_id else doc_id
        if not folder_doc_id:
            continue
        folder_id = compute_folder_id(folder_doc_id)
        by_folder.setdefault(folder_id, []).append(doc)

    plans: list[SidecarPlan] = []
    for folder in sorted(archive_dir.iterdir()):
        if not folder.is_dir():
            continue
        attachments_dir = folder / "attachments"
        if not attachments_dir.exists():
            continue
        metadata = _read_metadata_json(folder) or {}

        folder_id = folder.name
        docs_in_folder = by_folder.get(folder_id, [])
        # Build a safe_filename → original filename lookup for this folder.
        name_by_safe: dict[str, str] = {}
        for doc in docs_in_folder:
            original = doc.get("file_name")
            if not original:
                continue
            name_by_safe[_sanitize_storage_key(original)] = original

        for md_path in sorted(attachments_dir.glob("*.md")):
            # The .md suffix was added to the sanitized filename, so strip it
            # to recover the storage key. Runtime writes the sidecar as
            # ``<storage_key>.meta.json`` (no intermediate ``.md``), so we
            # must mirror that convention here.
            stem = md_path.name.removesuffix(".md")
            meta_path = md_path.with_name(stem + ".meta.json")
            if meta_path.exists():
                continue  # already has sidecar
            original_filename = name_by_safe.get(stem)
            if not original_filename:
                logger.debug(
                    "No documents-table entry for %s (folder %s); skipping",
                    md_path.name,
                    folder_id,
                )
                continue

            parser_name, parser_model = _parser_from_trail(metadata, original_filename)
            if not parser_name:
                logger.debug(
                    "No trail parser entry for %s in folder %s; skipping",
                    original_filename,
                    folder_id,
                )
                continue

            plans.append(
                SidecarPlan(
                    md_path=md_path,
                    meta_path=meta_path,
                    parser_name=parser_name,
                    parser_model=parser_model,
                    parsed_at=None,
                    source="trail",
                )
            )

    return plans


def _apply_plan(plans: list[SidecarPlan]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for plan in plans:
        payload = {
            "parser": plan.parser_name,
            "model": plan.parser_model,
            "parsed_at": plan.parsed_at or now,
            "backfilled": True,
        }
        plan.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize(plans: list[SidecarPlan], apply: bool) -> None:
    if not plans:
        print("No sidecars to backfill — all .md files already have .meta.json or no inferable parser.")
        return
    by_parser: dict[str, int] = {}
    for p in plans:
        by_parser[p.parser_name] = by_parser.get(p.parser_name, 0) + 1
    prefix = "Wrote" if apply else "Would write"
    print(f"{prefix} {len(plans)} sidecars:")
    for parser, count in sorted(by_parser.items(), key=lambda kv: -kv[1]):
        print(f"  {parser}: {count}")
    print("\nSample entries:")
    for p in plans[:5]:
        print(f"  {p.meta_path.relative_to(p.meta_path.parents[3])} <- {p.parser_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/output",
        type=Path,
        help="Path to the ingest output directory (default: data/output).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write sidecars. Without this flag the script is a dry-run.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print debug-level inference reasoning.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir: Path = args.output_dir
    if not output_dir.exists():
        logger.error("Output dir does not exist: %s", output_dir)
        return 1

    plans = _plan_sidecars(output_dir)
    if args.apply:
        _apply_plan(plans)
    _summarize(plans, apply=args.apply)

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to write the sidecars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
