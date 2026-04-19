#!/usr/bin/env python
"""One-time backfill of per-step model/timestamp metadata into archive folders.

Populates ``archive/<folder>/metadata.json`` with a ``processing`` key for
existing ingests that pre-date the ProcessingTrail feature. Uses:

- **Timestamps**: filesystem creation/modification time of the archive folder
  (and per-attachment file where available).
- **Models**: pre-uncommitted-changes pipeline rules (LlamaParse default,
  local parsers for modern Office, vision for images, current settings for
  LLM steps).

Usage::

    uv run python scripts/backfill_processing_metadata.py            # dry-run
    uv run python scripts/backfill_processing_metadata.py --apply    # writes

Defaults to ``data/output/``. Idempotent: archives already carrying a
``processing`` key are skipped unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

# Allow running as a plain script (scripts/) without `uv run` wiring paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mtss.config import get_settings  # noqa: E402
from mtss.ingest.archive_generator import _sanitize_storage_key  # noqa: E402
from mtss.models.document import DocumentType  # noqa: E402
from mtss.parsers.pdf_classifier import PDFComplexity  # noqa: E402

logger = logging.getLogger("backfill_processing_metadata")

# ── pre-branch rules ────────────────────────────────────────────────────

# Types that always went through LlamaParse on the old pipeline (PDFs are
# handled separately via pdf_classifier).
_LLAMAPARSE_TYPES = {
    DocumentType.ATTACHMENT_DOC.value,
    DocumentType.ATTACHMENT_XLS.value,
    DocumentType.ATTACHMENT_PPT.value,
    DocumentType.ATTACHMENT_RTF.value,
    DocumentType.ATTACHMENT_OTHER.value,
}
_PDF_TYPE = DocumentType.ATTACHMENT_PDF.value
# Types handled by local parsers (deterministic, no model field)
_LOCAL_PARSE_TYPES = {
    DocumentType.ATTACHMENT_DOCX.value: "local_docx",
    DocumentType.ATTACHMENT_XLSX.value: "local_xlsx",
    DocumentType.ATTACHMENT_CSV.value: "local_csv",
}
_IMAGE_TYPE = DocumentType.ATTACHMENT_IMAGE.value
_EMAIL_TYPE = DocumentType.EMAIL.value


# ── timestamp helpers ───────────────────────────────────────────────────

def _iso_z(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_ts(path: Path) -> float | None:
    """Creation/modification time for a file or folder, seconds since epoch.

    Windows exposes creation as ``st_ctime``; on POSIX systems ``st_ctime``
    is metadata-change time, which is still a reasonable proxy for ingest
    time. Falls back to ``st_mtime`` if the file vanished between scan and
    stat. Returns None when the path doesn't exist.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    # Prefer birthtime when available (macOS / newer Python)
    ts = getattr(st, "st_birthtime", None)
    return ts or st.st_ctime or st.st_mtime


# ── core backfill ───────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ChunkAggregate:
    """Per-document summary of chunks, built by streaming chunks.jsonl.

    Holds only what ``build_processing_for_folder`` needs — count, presence
    of context summary, presence of topics and their IDs, and the set of
    ``metadata.type`` values seen. Parsing the full embedding arrays for
    ~46K chunks takes ~20s of pure CPU; building aggregates on the fly
    without materializing the embedding fields drops startup to <1s.
    """

    __slots__ = ("count", "has_context", "topic_ids", "types")

    def __init__(self) -> None:
        self.count: int = 0
        self.has_context: bool = False
        self.topic_ids: set[str] = set()
        self.types: set[str] = set()


# Chunks.jsonl has a 1024-dim embedding array per line (~15 KB each).
# json.loads spends most of its time parsing those float arrays. We don't
# need the embedding for this migration, so strip it out of each line with
# a regex before handing the remaining JSON to the parser — cuts startup
# from ~19 s to ~1 s on a 42K-chunk corpus.
_EMBEDDING_RE = re.compile(r'"embedding"\s*:\s*(\[[^\]]*\]|null)')


def _stream_chunk_aggregates(path: Path) -> dict[str, ChunkAggregate]:
    """One streaming pass over chunks.jsonl → per-document aggregates."""
    out: dict[str, ChunkAggregate] = defaultdict(ChunkAggregate)
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Drop the embedding payload before json.loads so we don't pay
            # the parse cost for floats we never read.
            stripped = _EMBEDDING_RE.sub('"embedding":null', line, count=1)
            d = json.loads(stripped)
            did = d.get("document_id")
            if not did:
                continue
            agg = out[did]
            agg.count += 1
            if d.get("context_summary"):
                agg.has_context = True
            md = d.get("metadata") or {}
            for tid in md.get("topic_ids") or ():
                agg.topic_ids.add(tid)
            t = md.get("type")
            if t:
                agg.types.add(t)
    return out


def _folder_id_of(doc: dict) -> str | None:
    doc_id = doc.get("doc_id")
    return doc_id[:16] if doc_id else None


def _classify_pdf_sampled(file_path: Path, sample_pages: int = 2) -> PDFComplexity:
    """Cheap per-PDF classification for migration backfill.

    Mirrors ``classify_pdf`` from the parsers module but only inspects the
    first ``sample_pages`` pages. The production classifier scans every page
    with pure-Python pypdf, which blows up on 200-page PDFs and stalls under
    the GIL. Two pages is enough to detect scanned docs (image-only) and
    form-field PDFs, which is the signal the old pipeline used.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(file_path))
    except Exception:
        return PDFComplexity.COMPLEX
    if not reader.pages:
        return PDFComplexity.COMPLEX
    try:
        if reader.get_fields():
            return PDFComplexity.COMPLEX
    except (PyPdfError, KeyError, AttributeError, TypeError):
        return PDFComplexity.COMPLEX
    for page in reader.pages[:sample_pages]:
        try:
            text = page.extract_text() or ""
        except (PyPdfError, KeyError, AttributeError, TypeError, ValueError):
            return PDFComplexity.COMPLEX
        if len(text.strip()) < 50:
            return PDFComplexity.COMPLEX
    return PDFComplexity.SIMPLE


def _classify_pdf_parser(att_file: Path) -> tuple[str, str | None, str | None]:
    """Return (parser_name, model, note) for a stored PDF."""
    if not att_file.exists():
        return "pdf_file_missing", None, "archive file not on disk"
    try:
        complexity = _classify_pdf_sampled(att_file)
    except Exception as e:  # noqa: BLE001
        return "pdf_classify_failed", None, f"sampled classify raised: {e}"
    if complexity == PDFComplexity.SIMPLE:
        return "local_pdf", None, None
    return "llamaparse", "llamaparse:agentic", None


def _build_attachment_parse_entry(
    doc: dict,
    ts_iso: str,
    settings: Any,
    att_file: Path,
    undetermined: list[str],
) -> Dict[str, Any] | None:
    """Return the ``parse`` entry for an attachment, or None for images."""
    doc_type = doc.get("document_type")
    if doc_type == _IMAGE_TYPE:
        return None
    if doc_type == _PDF_TYPE:
        parser_name, model, note = _classify_pdf_parser(att_file)
        entry: Dict[str, Any] = {"ran_at": ts_iso, "parser": parser_name}
        if model is not None:
            entry["model"] = model
        if note is not None:
            undetermined.append(f"{doc.get('file_name', '?')}: {note}")
        return entry
    if doc_type in _LLAMAPARSE_TYPES:
        return {"ran_at": ts_iso, "model": "llamaparse:agentic", "parser": "llamaparse"}
    if doc_type in _LOCAL_PARSE_TYPES:
        return {"ran_at": ts_iso, "parser": _LOCAL_PARSE_TYPES[doc_type]}
    # Shouldn't happen with current DocumentType enum, but record a marker
    undetermined.append(f"{doc.get('file_name', '?')}: unknown document_type {doc_type!r}")
    return {"ran_at": ts_iso, "parser": "legacy_unknown"}


_EMPTY_AGG = ChunkAggregate()


def build_processing_for_folder(
    *,
    email_doc: dict,
    attachment_docs: list[dict],
    chunk_aggs: Dict[str, ChunkAggregate],
    folder_path: Path,
    archive_root: Path,
    settings: Any,
    undetermined: list[str],
) -> Dict[str, Any]:
    """Assemble the ``processing`` dict for one archive folder."""
    folder_ts = _file_ts(folder_path) or datetime.now(timezone.utc).timestamp()
    folder_iso = _iso_z(folder_ts)

    email_agg = chunk_aggs.get(email_doc["id"], _EMPTY_AGG)
    email_entry: Dict[str, Dict[str, Any]] = {
        "parse": {"ran_at": folder_iso, "parser": "eml_local"},
    }
    # The old pipeline always ran the LLM body cleaner before chunking
    # bodies, so presence of any email_body chunk implies content_cleanup ran.
    if "email_body" in email_agg.types:
        email_entry["content_cleanup"] = {
            "ran_at": folder_iso,
            "model": settings.get_model(settings.email_cleaner_model),
        }
    if email_agg.has_context:
        email_entry["context"] = {
            "ran_at": folder_iso,
            "model": settings.get_model(settings.context_llm_model),
        }
    if email_agg.topic_ids:
        email_entry["topics"] = {
            "ran_at": folder_iso,
            "model": settings.get_model(settings.context_llm_model),
            "topic_count": len(email_agg.topic_ids),
        }
    if "thread_digest" in email_agg.types:
        email_entry["thread_digest"] = {
            "ran_at": folder_iso,
            "model": settings.get_model(settings.thread_digest_model),
        }
    if email_agg.count:
        email_entry["embed"] = {
            "ran_at": folder_iso,
            "model": settings.embedding_model,
            "chunk_count": email_agg.count,
        }

    attachments_entry: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for att in attachment_docs:
        filename = att.get("file_name") or ""
        if not filename:
            continue
        safe_name = _sanitize_storage_key(filename)
        att_file = archive_root / folder_path.name / "attachments" / safe_name
        att_ts = _file_ts(att_file)
        att_iso = _iso_z(att_ts) if att_ts is not None else folder_iso

        entry: Dict[str, Dict[str, Any]] = {}
        att_agg = chunk_aggs.get(att["id"], _EMPTY_AGG)
        doc_type = att.get("document_type")

        parse = _build_attachment_parse_entry(att, att_iso, settings, att_file, undetermined)
        if parse is not None:
            entry["parse"] = parse

        if doc_type == _IMAGE_TYPE:
            entry["vision"] = {
                "ran_at": att_iso,
                "model": settings.get_model(settings.image_llm_model),
            }
        elif att_agg.has_context:
            entry["context"] = {
                "ran_at": att_iso,
                "model": settings.get_model(settings.context_llm_model),
            }

        if att_agg.count:
            entry["embed"] = {
                "ran_at": att_iso,
                "model": settings.embedding_model,
                "chunk_count": att_agg.count,
            }

        if entry:
            attachments_entry[filename] = entry

    return {"email": email_entry, "attachments": attachments_entry}


# ── driver ──────────────────────────────────────────────────────────────

def _process_one_folder(
    folder_path: Path,
    emails: Dict[str, dict],
    atts_by_folder: Dict[str, list[dict]],
    chunk_aggs: Dict[str, ChunkAggregate],
    archive_root: Path,
    settings: Any,
    force: bool,
    apply: bool,
) -> tuple[str, list[str]]:
    """Process one folder. Returns (status_key, folder_undetermined).

    Designed to run under a ThreadPoolExecutor — reads shared dicts
    read-only, writes only to its own metadata.json.
    """
    fid = folder_path.name
    meta_path = folder_path / "metadata.json"
    if not meta_path.exists():
        return "skipped_no_metadata", []
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "skipped_malformed", []

    if "processing" in metadata and not force:
        return "skipped_has_processing", []

    email_doc = emails.get(fid)
    if not email_doc:
        return "skipped_no_email_row", []

    undet: list[str] = []
    processing = build_processing_for_folder(
        email_doc=email_doc,
        attachment_docs=atts_by_folder.get(fid, []),
        chunk_aggs=chunk_aggs,
        folder_path=folder_path,
        archive_root=archive_root,
        settings=settings,
        undetermined=undet,
    )

    if apply:
        metadata["processing"] = processing
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp.replace(meta_path)
        return "written", undet
    return "would_write", undet


def run(
    output_dir: Path,
    *,
    apply: bool,
    force: bool,
    verbose: bool,
    limit: int | None = None,
    workers: int = 8,
) -> tuple[Counter, list[str]]:
    settings = get_settings()
    archive_root = output_dir / "archive"
    if not archive_root.exists():
        logger.error("archive/ not found under %s", output_dir)
        return Counter(), []
    undetermined: list[str] = []

    docs = _load_jsonl(output_dir / "documents.jsonl")
    chunk_aggs = _stream_chunk_aggregates(output_dir / "chunks.jsonl")
    total_chunks = sum(a.count for a in chunk_aggs.values())
    logger.info("Loaded %d docs, %d chunks (aggregated)", len(docs), total_chunks)

    # Group docs by folder_id (email doc_id prefix)
    emails: Dict[str, dict] = {}
    attachments_by_folder: Dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        fid = _folder_id_of(d)
        if not fid:
            continue
        if d.get("document_type") == _EMAIL_TYPE:
            emails[fid] = d
        else:
            # Attachments inherit the email's doc_id prefix via parent chain;
            # we attribute them by resolving their root's folder_id.
            root_id = d.get("root_id")
            if not root_id:
                continue
            # root_id is a UUID, not a doc_id — need to map via email docs
            # which we haven't fully indexed yet. Do a second pass.
            attachments_by_folder[root_id].append(d)

    # Build UUID→folder_id map from email docs
    uuid_to_folder: Dict[str, str] = {}
    for fid, e in emails.items():
        uuid_to_folder[e["id"]] = fid

    # Re-key attachments by folder_id
    atts_by_folder: Dict[str, list[dict]] = defaultdict(list)
    for root_uuid, atts in attachments_by_folder.items():
        fid = uuid_to_folder.get(root_uuid)
        if not fid:
            continue
        atts_by_folder[fid].extend(atts)

    # Collect folders up front so we know the total for progress reporting
    all_folders = [p for p in sorted(archive_root.iterdir()) if p.is_dir()]
    if limit is not None:
        all_folders = all_folders[:limit]
    total = len(all_folders)
    logger.info("Scanning %d folders with %d workers", total, workers)

    stats: Counter = Counter()
    start = time.monotonic()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _process_one_folder,
                folder_path,
                emails,
                atts_by_folder,
                chunk_aggs,
                archive_root,
                settings,
                force,
                apply,
            ): folder_path
            for folder_path in all_folders
        }
        for fut in as_completed(futures):
            status, folder_undet = fut.result()
            stats[status] += 1
            undetermined.extend(folder_undet)
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(
                    f"  [{done}/{total}] {rate:.0f}/s  eta {eta:.0f}s",
                    flush=True,
                )

    return stats, undetermined


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("data/output"))
    ap.add_argument("--apply", action="store_true", help="Write the processing key (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing processing keys")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N folders (dry-run sanity check)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel folder workers (default 8)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats, undetermined = run(
        args.output_dir.resolve(),
        apply=args.apply,
        force=args.force,
        verbose=args.verbose,
        limit=args.limit,
        workers=args.workers,
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] backfill summary:")
    for k in (
        "written",
        "would_write",
        "skipped_has_processing",
        "skipped_no_email_row",
        "skipped_no_metadata",
        "skipped_malformed",
    ):
        if stats[k]:
            print(f"  {k}: {stats[k]}")

    if undetermined:
        print(f"\nCould not fully determine parse model for {len(undetermined)} attachment(s):")
        for line in undetermined[:40]:
            print(f"  - {line}")
        if len(undetermined) > 40:
            print(f"  … and {len(undetermined) - 40} more")

    if not args.apply:
        print("\nRun with --apply to persist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
