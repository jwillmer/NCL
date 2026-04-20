"""Validate ingest output and Supabase import for data integrity issues."""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import typer
from rich.table import Table

from ..utils import compute_folder_id
from ._common import console, make_progress

logger = logging.getLogger(__name__)

# Hidden/system files to exclude from archive file counts
_HIDDEN_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})

_ARCHIVE_URI_FIELDS = ("archive_path", "archive_browse_uri", "archive_download_uri")
_ENCODED_FILENAME_RE = _re.compile(r"%[0-9A-Fa-f]{2}")

# Human-readable explanation for each ingest-event reason category. Keys are
# the canonical category (text before the first ':' in the raw reason string).
# Unknown categories render with a blank description column.
_REASON_DESCRIPTIONS: Dict[str, str] = {
    "partial_download": "file incomplete on disk",
    "attachment_too_large": "exceeds max attachment size",
    "filtered_by_heuristic": "signature / boilerplate image filter",
    "pdf_too_large_unreadable": "PDF over page ceiling, no text layer",
    "unsupported_format": "no parser for this file type",
    "Classification failed": "image vision classifier errored",
    "File not found": "expected file missing on disk",
    "extraction_failed": "parser raised after opening file",
    "corrupted": "file could not be opened / read",
    "triage_failed": "LLM triage errored, defaulted to summary",
    "triage_prose": "LLM classified as prose, full embed",
    "triage_noise": "LLM classified as noise, metadata_only",
    "triage_dense": "dense tabular, summary embed",
    "New document": "first-time ingest",
    "Content changed since last ingest": "file hash differs from prior run",
    "Already processed with current version": "skipped, idempotent",
    "Ingest logic upgraded": "ingest version bumped, reprocessing",
}


def _reason_category(reason: str) -> str:
    """Collapse a raw event reason into its category key for grouping/lookup.

    Strips any `": <detail>"` suffix so high-cardinality reasons like
    ``partial_download: foo.pdf`` collapse together. Also normalizes
    ``Ingest logic upgraded from vN to vM`` to its prefix so every version
    pair doesn't produce a separate row.
    """
    category = reason.split(":", 1)[0].strip()
    if category.startswith("Ingest logic upgraded"):
        category = "Ingest logic upgraded"
    return category


def _compute_last_run_cutoff(run_history: List[Dict[str, Any]]) -> Optional[str]:
    """Return the ISO-8601 start timestamp of the most recent run, or None.

    Uses ``timestamp`` (completion) minus ``elapsed_seconds``. Events with a
    timestamp >= this cutoff are considered part of the last run.
    """
    if not run_history:
        return None
    from datetime import datetime, timedelta, timezone
    latest = run_history[-1]
    ts = latest.get("timestamp")
    elapsed = latest.get("elapsed_seconds") or 0
    if not ts:
        return None
    try:
        end = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return (end - timedelta(seconds=float(elapsed))).isoformat()
    except Exception:
        return None


def _event_timestamp(e: Dict[str, Any], doc_ts_by_id: Dict[str, str]) -> Optional[str]:
    """Resolve an event's timestamp for last-run bucketing.

    Prefer the event's own ``timestamp`` field; fall back to the parent
    document's ``updated_at`` so pre-fix events (missing their own timestamp)
    still bucket correctly.
    """
    ts = e.get("timestamp")
    if ts:
        return ts
    pid = e.get("parent_document_id") or e.get("document_id")
    if pid and pid in doc_ts_by_id:
        return doc_ts_by_id[pid]
    return None


def _format_examples(names: List[str], limit: int = 3) -> str:
    """Format a short 'e.g. a, b, c' suffix from up to ``limit`` unique names."""
    seen: List[str] = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
        if len(seen) >= limit:
            break
    if not seen:
        return ""
    # Trim overly long filenames so the Note cell stays readable.
    trimmed = [n if len(n) <= 40 else n[:37] + "..." for n in seen]
    return "e.g. " + ", ".join(trimmed)


def _parser_attribution_for_failed_docs(
    failed_docs: List[Dict[str, Any]], archive_dir: Path
) -> "Counter[str]":
    """Aggregate parse-step parser/model names for a set of failed docs.

    Reads ``archive/<folder_id>/metadata.json`` and pulls
    ``processing.attachments[<file_name>].parse.model`` for each failed
    attachment. Returns Counter keyed on model name (or 'unknown' when the
    trail is missing).
    """
    counts: "Counter[str]" = Counter()
    if not archive_dir.exists():
        return counts
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    for d in failed_docs:
        fname = d.get("file_name") or ""
        # Root-email folder id: compute from root_id so attachments map back.
        root_id = d.get("root_id") or d.get("doc_id")
        if not root_id:
            continue
        folder_id = compute_folder_id(root_id) if len(root_id) != 32 else root_id
        mpath = archive_dir / folder_id / "metadata.json"
        if folder_id not in metadata_cache:
            try:
                metadata_cache[folder_id] = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                metadata_cache[folder_id] = {}
        meta = metadata_cache[folder_id]
        trail = (meta.get("processing") or {}).get("attachments") or {}
        parse_entry = (trail.get(fname) or {}).get("parse") or {}
        model = parse_entry.get("model") or "unknown"
        counts[model] += 1
    return counts


validate_app = typer.Typer(
    help="Validate ingest output and Supabase import integrity",
    invoke_without_command=True,
    context_settings={"help_option_names": ["--help", "-h", "-?", "/?"]},
)


@validate_app.callback()
def validate_callback(ctx: typer.Context):
    """Validate ingest output and Supabase import integrity."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def register(app: typer.Typer):
    """Register validate command group on the app."""
    app.add_typer(validate_app, name="validate")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_folder_to_email_map(docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map archive folder id -> source email identifier.

    Keyed on archive_path when present (what folders are named), else a
    freshly computed folder_id derived from doc_id.
    Value prefers source_id (eml filename) so warnings point users at a real file.
    """
    mapping: Dict[str, str] = {}
    for d in docs:
        if d.get("depth", 0) != 0:
            continue
        doc_id = d.get("doc_id") or ""
        key = d.get("archive_path") or (compute_folder_id(doc_id) if doc_id else "")
        if not key:
            continue
        mapping[key] = (
            d.get("source_id") or d.get("file_name") or d.get("source_title") or "?"
        )
    return mapping


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load rows from the SQLite ``ingest.db``. Path is parsed for its parent
    directory (output dir) and stem (table name) — the JSONL file itself is
    never read; it no longer exists post-cutover.
    """
    from ..storage.sqlite_client import SqliteStorageClient

    output_dir = path.parent
    table = path.stem  # "documents" / "chunks" / …
    db_path = output_dir / "ingest.db"
    if not db_path.exists():
        raise FileNotFoundError(f"ingest.db not found in {output_dir}")

    client = SqliteStorageClient(output_dir=output_dir)
    try:
        if table == "documents":
            return list(_adapt_documents(client))
        if table == "chunks":
            return list(_adapt_chunks(client))
        if table == "topics":
            return list(_adapt_topics(client))
        if table == "ingest_events":
            return list(_adapt_events(client))
        if table == "processing_log":
            return list(_adapt_processing_log(client))
        if table == "run_history":
            return list(_adapt_run_history(client))
        return []
    finally:
        try:
            client._conn.close()
        except Exception:
            pass


def _adapt_documents(client) -> "Iterable[Dict[str, Any]]":
    for row in client.iter_documents():
        meta = row.get("metadata_json")
        if meta:
            try:
                meta_obj = json.loads(meta)
                if isinstance(meta_obj, dict):
                    for k, v in meta_obj.items():
                        row.setdefault(k, v)
            except (TypeError, ValueError):
                pass
        row.pop("metadata_json", None)
        yield row


def _adapt_chunks(client) -> "Iterable[Dict[str, Any]]":
    for row in client.iter_chunks():
        # iter_chunks already decodes embedding, metadata, section_path, topic_ids.
        row.pop("metadata_json", None)
        row.pop("section_path_json", None)
        row.pop("embedding_dim", None)
        yield row


def _adapt_topics(client) -> "Iterable[Dict[str, Any]]":
    for row in client.iter_topics():
        row.pop("keywords_json", None)
        row.pop("embedding_dim", None)
        yield row


def _adapt_events(client) -> "Iterable[Dict[str, Any]]":
    for row in client.iter_events():
        yield row


def _adapt_processing_log(client) -> "Iterable[Dict[str, Any]]":
    for row in client._conn.execute("SELECT * FROM processing_log"):
        yield dict(row)


def _adapt_run_history(client) -> "Iterable[Dict[str, Any]]":
    for row in client._conn.execute("SELECT * FROM run_history"):
        yield dict(row)


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Resolve the output directory, falling back to settings if None."""
    if output_dir:
        return output_dir.resolve()
    try:
        from ..config import get_settings
        settings = get_settings()
        return (settings.eml_source_dir.parent / "output").resolve()
    except Exception:
        return Path("data/output").resolve()


def _count_archive_files(doc_dir: Path) -> int:
    """Count real files in an archive directory, excluding hidden/system files."""
    return sum(
        1 for f in doc_dir.rglob("*")
        if f.is_file() and f.name not in _HIDDEN_FILES
    )


# ---------------------------------------------------------------------------
# mtss validate ingest
# ---------------------------------------------------------------------------


@validate_app.command("ingest")
def validate_ingest(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory to validate (default: data/output)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed per-document info"
    ),
):
    """Validate local ingest output for data integrity issues."""
    resolved = _resolve_output_dir(output_dir)
    _run_ingest_validation(resolved, verbose)


# ---------------------------------------------------------------------------
# Per-check functions (extracted from _run_ingest_validation)
#
# Each function corresponds to one numbered check (1..22). All share the same
# signature convention: take only the pre-loaded, pre-indexed data they need,
# return (issues, warnings) — two lists of user-facing strings.
#
# The strings are part of the public contract (regression coverage:
# tests/test_sanitize_migration.py::TestValidateNewChecks and
# tests/test_validate_checks.py) — do not edit message wording casually.
# ---------------------------------------------------------------------------


def _check_duplicate_uuids(
    docs: List[Dict[str, Any]], doc_by_uuid: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Check 1: Duplicate UUID detection."""
    issues: List[str] = []
    warnings: List[str] = []
    if len(doc_by_uuid) != len(docs):
        issues.append(
            f"Duplicate document UUIDs: {len(docs)} rows but {len(doc_by_uuid)} unique ids"
        )
    return issues, warnings


def _check_processing_log(
    proc_by_file: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 2: Processing log — every file should be COMPLETED."""
    issues: List[str] = []
    warnings: List[str] = []
    failed = [p for p in proc_by_file.values() if p.get("status") != "COMPLETED"]
    if failed:
        issues.append(f"{len(failed)} files not COMPLETED in processing log")
        for f_entry in failed:
            issues.append(
                f"  {f_entry.get('file_path', '?')}: "
                f"status={f_entry.get('status')}, error={f_entry.get('error')}"
            )
    return issues, warnings


def _check_document_types(
    docs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 3: Document type breakdown.

    No issues/warnings emitted — this is a purely informational summary,
    surfaced in the summary table. Preserved as a dedicated check so the
    numbering remains aligned with the documented 22-check contract.
    """
    return [], []


def _check_orphan_chunks(
    chunks: List[Dict[str, Any]], doc_by_uuid: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Check 4: Chunk -> Document linkage."""
    issues: List[str] = []
    warnings: List[str] = []
    orphan_chunks = [c for c in chunks if c["document_id"] not in doc_by_uuid]
    if orphan_chunks:
        issues.append(
            f"{len(orphan_chunks)} chunks reference non-existent documents"
        )
    return issues, warnings


def _check_embedding_completeness(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 5: Embedding completeness + dimension consistency."""
    issues: List[str] = []
    warnings: List[str] = []
    no_embedding = [c for c in chunks if not c.get("embedding")]
    if no_embedding:
        issues.append(f"{len(no_embedding)}/{len(chunks)} chunks missing embeddings")

    dims: Set[int] = set()
    for c in chunks:
        emb = c.get("embedding")
        if emb:
            dims.add(len(emb))
    if len(dims) > 1:
        issues.append(f"Inconsistent embedding dimensions: {dims}")
    return issues, warnings


def _check_empty_content(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 6: Empty content in chunks."""
    issues: List[str] = []
    warnings: List[str] = []
    empty_content = [
        c for c in chunks if not c.get("content") or not c["content"].strip()
    ]
    if empty_content:
        issues.append(f"{len(empty_content)} chunks have empty content")
    return issues, warnings


def _check_context_summary(
    chunks: List[Dict[str, Any]],
    docs: List[Dict[str, Any]],
    doc_by_uuid: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 7: Context summary / embedding_text presence.

    Skipped by design: image chunks, and chunks from docs in `summary` or
    `metadata_only` embedding mode — those modes produce a single synthesized
    chunk whose content already serves as its own summary.
    """
    issues: List[str] = []
    warnings: List[str] = []
    skipped_doc_uuids = {
        d["id"] for d in docs
        if d.get("document_type") == "attachment_image"
        or d.get("embedding_mode") in {"summary", "metadata_only"}
    }
    text_chunks = [c for c in chunks if c["document_id"] not in skipped_doc_uuids]
    incomplete_chunks = [
        c for c in text_chunks if not c.get("context_summary") or not c.get("embedding_text")
    ]
    if incomplete_chunks:
        # Group by source email for actionable output
        affected_docs: Dict[str, int] = {}
        for c in incomplete_chunks:
            did = c["document_id"]
            affected_docs[did] = affected_docs.get(did, 0) + 1
        affected_emails: Dict[str, int] = {}
        for d in docs:
            if d["id"] in affected_docs:
                root = d.get("root_id", d["id"])
                email = doc_by_uuid.get(root, d)
                eml = email.get("source_id", "?")
                affected_emails[eml] = affected_emails.get(eml, 0) + affected_docs[d["id"]]
        warnings.append(
            f"{len(incomplete_chunks)}/{len(text_chunks)} text chunks missing context_summary/embedding_text "
            f"({len(affected_emails)} emails)"
        )
        for eml, cnt in sorted(affected_emails.items()):
            warnings.append(f"    {eml} ({cnt} chunks)")
    return issues, warnings


def _check_docs_without_chunks(
    docs: List[Dict[str, Any]],
    chunk_doc_ids: "Counter[str]",
    filtered_doc_uuids: Set[str],
) -> Tuple[List[str], List[str]]:
    """Check 8: Documents without chunks (excluding filtered / images / failed)."""
    issues: List[str] = []
    warnings: List[str] = []
    docs_without_chunks: List[Dict[str, Any]] = []
    for d in docs:
        if d["id"] not in chunk_doc_ids:
            # Skip emails explicitly filtered (forwarding/cover emails)
            if d["id"] in filtered_doc_uuids:
                continue
            # Images may legitimately have no chunks (non-content)
            if d.get("document_type") == "attachment_image":
                continue
            # Failed documents have 0 chunks by definition — already surfaced by status=failed check
            if d.get("status") == "failed":
                continue
            docs_without_chunks.append(d)

    if docs_without_chunks:
        issues.append(
            f"{len(docs_without_chunks)} document(s) have no chunks (not explained by events)"
        )
        for d in docs_without_chunks:
            issues.append(
                f"  {d.get('document_type')}: {d.get('file_path', '?')[:70]}"
            )
    return issues, warnings


def _check_orphan_attachments(
    docs: List[Dict[str, Any]], email_uuids: Set[str]
) -> Tuple[List[str], List[str]]:
    """Check 9: Attachment -> Email parent chain."""
    issues: List[str] = []
    warnings: List[str] = []
    orphan_attachments = []
    for d in docs:
        if d.get("document_type") != "email" and d.get("root_id"):
            if d["root_id"] not in email_uuids:
                orphan_attachments.append(d)
    if orphan_attachments:
        warnings.append(
            f"{len(orphan_attachments)} attachments have root_id not matching any email"
        )
    return issues, warnings


def _check_failed_documents(
    docs: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    """Check 10: Failed documents still in output.

    Cross-references `extraction_failed` events so that docs with a matching
    event (same parent email UUID + same file_name as the event) are treated
    as expected. Only failed docs *without* a matching event are warned about.
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Build lookup: (parent_email_uuid, filename_lower) for extraction_failed events.
    failed_attachment_keys: Set[Tuple[str, str]] = set()
    for e in events:
        if e.get("event_type") != "extraction_failed":
            continue
        parent = e.get("parent_document_id")
        fname = e.get("file_name")
        if parent and fname:
            failed_attachment_keys.add((str(parent), fname.lower()))

    def _is_explained(d: Dict[str, Any]) -> bool:
        root = d.get("root_id")
        src = d.get("source_id") or ""
        # attachment source_id is "<email.eml>/<attachment_filename>"
        tail = src.rsplit("/", 1)[-1] if src else ""
        return bool(root and tail and (str(root), tail.lower()) in failed_attachment_keys)

    failed_docs = [d for d in docs if d.get("status") == "failed"]
    unexplained = [d for d in failed_docs if not _is_explained(d)]

    if unexplained:
        warnings.append(
            f"{len(unexplained)} document(s) have status='failed' in the documents table "
            f"(no matching extraction_failed event)"
        )
        if verbose:
            for d in unexplained:
                err = d.get("error_message") or "no error message"
                warnings.append(f"  {d.get('doc_id', '?')[:16]}: {err[:80]}")
    return issues, warnings


def _check_trailing_dot_filenames(
    docs: List[Dict[str, Any]], verbose: bool
) -> Tuple[List[str], List[str]]:
    """Check 11: Trailing-dot filenames (may cause parsing issues)."""
    issues: List[str] = []
    warnings: List[str] = []
    trailing_dot_docs = [
        d for d in docs if d.get("source_id") and d["source_id"].endswith(".")
    ]
    if trailing_dot_docs:
        warnings.append(
            f"{len(trailing_dot_docs)} document(s) have source_id ending in '.' "
            f"(potential parsing issues)"
        )
        if verbose:
            for d in trailing_dot_docs:
                warnings.append(
                    f"  {d.get('document_type')}: {d.get('source_id', '')[:70]}"
                )
    return issues, warnings


def _check_topic_health(
    topics: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 12: Topic embeddings + document_count sanity."""
    issues: List[str] = []
    warnings: List[str] = []
    no_topic_embedding = sum(1 for t in topics if not t.get("embedding"))
    if no_topic_embedding:
        warnings.append(f"{no_topic_embedding}/{len(topics)} topics missing embeddings")

    zero_count_topics = [t for t in topics if (t.get("document_count") or 0) == 0]
    if zero_count_topics:
        warnings.append(
            f"{len(zero_count_topics)}/{len(topics)} topics have document_count=0"
        )
    return issues, warnings


def _check_archive_uris(
    docs: List[Dict[str, Any]],
    email_uuids: Set[str],
    archive_folders_on_disk: Set[str],
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    """Check 13: Archive directory and URI checks."""
    issues: List[str] = []
    warnings: List[str] = []
    emails_missing_archive: List[Tuple[Dict[str, Any], List[str]]] = []
    emails_without_folder: List[Dict[str, Any]] = []
    for d in docs:
        if d.get("document_type") != "email":
            continue
        missing_fields = [f for f in _ARCHIVE_URI_FIELDS if not d.get(f)]
        if missing_fields:
            emails_missing_archive.append((d, missing_fields))
        if archive_folders_on_disk:
            # `archive_path` is a cached value that was stamped at ingest time
            # and can be stale (older docs stored the 16-char doc_id prefix
            # before the 32-char folder_id scheme). Match against either the
            # cached path or the computed folder — whichever is on disk wins.
            computed = compute_folder_id(d["doc_id"])
            cached = d.get("archive_path")
            if (
                computed not in archive_folders_on_disk
                and (not cached or cached not in archive_folders_on_disk)
            ):
                emails_without_folder.append(d)

    if emails_missing_archive:
        issues.append(
            f"{len(emails_missing_archive)}/{len(email_uuids)} "
            f"emails missing archive URIs (UI will show broken links)"
        )
        if verbose:
            for d, fields in emails_missing_archive:
                issues.append(
                    f"  {d.get('doc_id', '?')[:16]}: missing {', '.join(fields)}"
                )

    if emails_without_folder:
        warnings.append(
            f"{len(emails_without_folder)} emails have no archive folder on disk"
        )
    return issues, warnings


def _check_stale_topic_refs(
    chunks: List[Dict[str, Any]], topics: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Check 14: Stale topic references in chunk metadata."""
    issues: List[str] = []
    warnings: List[str] = []
    topic_id_set = {t["id"] for t in topics}
    stale_topic_chunks = 0
    stale_topic_ids: Set[str] = set()
    for c in chunks:
        tids = (c.get("metadata") or {}).get("topic_ids", [])
        for tid in tids:
            if tid not in topic_id_set:
                stale_topic_chunks += 1
                stale_topic_ids.add(tid)
    if stale_topic_chunks:
        issues.append(
            f"{stale_topic_chunks} chunk topic_ids reference {len(stale_topic_ids)} "
            f"non-existent topics (stale from merges)"
        )
    return issues, warnings


def _check_topic_count_accuracy(
    chunks: List[Dict[str, Any]], topics: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Check 15: Topic count accuracy (JSONL counts vs actual chunk refs)."""
    issues: List[str] = []
    warnings: List[str] = []
    topic_cc: "Counter[str]" = Counter()
    topic_ds: Dict[str, set] = {}
    for c in chunks:
        for tid in (c.get("metadata") or {}).get("topic_ids", []):
            topic_cc[tid] += 1
            topic_ds.setdefault(tid, set()).add(c.get("document_id", ""))
    stale_count_topics = 0
    for t in topics:
        tid = t["id"]
        actual_cc = topic_cc.get(tid, 0)
        actual_dc = len(topic_ds.get(tid, set()))
        if (t.get("chunk_count", 0) or 0) != actual_cc or (
            t.get("document_count", 0) or 0
        ) != actual_dc:
            stale_count_topics += 1
    if stale_count_topics:
        issues.append(
            f"{stale_count_topics}/{len(topics)} topics have stale counts "
            f"(JSONL counts don't match actual chunk references)"
        )
    return issues, warnings


def _check_broken_archive_uris(
    docs: List[Dict[str, Any]], archive_dir: Path
) -> Tuple[List[str], List[str]]:
    """Check 16: Archive URIs point to existing files on disk."""
    issues: List[str] = []
    warnings: List[str] = []
    broken_uris: List[Tuple[str, str, str]] = []
    for d in docs:
        for key in ("archive_browse_uri", "archive_download_uri"):
            uri = d.get(key)
            if uri:
                rel = uri.removeprefix("/archive/")
                if not (archive_dir / rel).exists():
                    broken_uris.append((d.get("file_name", "?"), key, rel))
    if broken_uris:
        issues.append(
            f"{len(broken_uris)} archive URIs point to missing files on disk"
        )
        for fn, key, rel in broken_uris[:5]:
            issues.append(f"  {fn}: {rel}")
        if len(broken_uris) > 5:
            issues.append(f"  ... and {len(broken_uris) - 5} more")
    return issues, warnings


def _check_duplicate_ids(
    docs: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    """Check 17: Duplicate doc_ids and chunk_ids."""
    issues: List[str] = []
    warnings: List[str] = []
    doc_id_counts = Counter(d.get("doc_id") for d in docs if d.get("doc_id"))
    dup_doc_ids = {did: cnt for did, cnt in doc_id_counts.items() if cnt > 1}
    if dup_doc_ids:
        issues.append(f"{len(dup_doc_ids)} duplicate doc_ids in documents table")
        if verbose:
            for did, cnt in list(dup_doc_ids.items())[:5]:
                issues.append(f"  {did[:16]}: {cnt}x")

    chunk_id_counts = Counter(c.get("chunk_id") for c in chunks if c.get("chunk_id"))
    dup_chunk_ids = {cid: cnt for cid, cnt in chunk_id_counts.items() if cnt > 1}
    if dup_chunk_ids:
        issues.append(f"{len(dup_chunk_ids)} duplicate chunk_ids in chunks table")
    return issues, warnings


def _check_encoded_filenames(
    archive_dir: Path, verbose: bool
) -> Tuple[List[str], List[str]]:
    """Check 18: Encoded filenames on disk (should use underscores, not %XX)."""
    issues: List[str] = []
    warnings: List[str] = []
    encoded_disk_files: List[str] = []
    if archive_dir.exists():
        for f in archive_dir.rglob("*"):
            if f.is_file() and _ENCODED_FILENAME_RE.search(f.name):
                encoded_disk_files.append(f.name)
    if encoded_disk_files:
        issues.append(
            f"{len(encoded_disk_files)} archive files have URL-encoded names "
            f"(run migration script to fix)"
        )
        if verbose:
            for name in encoded_disk_files[:5]:
                issues.append(f"  {name}")
    return issues, warnings


def _check_encoded_uris(
    docs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 19: Encoded URIs in documents.jsonl."""
    issues: List[str] = []
    warnings: List[str] = []
    encoded_uris: List[Tuple[str, str, str]] = []
    for d in docs:
        for key in ("archive_browse_uri", "archive_download_uri"):
            uri = d.get(key) or ""
            if _ENCODED_FILENAME_RE.search(uri):
                encoded_uris.append((d.get("file_name", "?"), key, uri))
    if encoded_uris:
        issues.append(
            f"{len(encoded_uris)} archive URIs contain URL-encoding "
            f"(run migration script to fix)"
        )
    return issues, warnings


def _looks_like_prose_match(text: str, target: str) -> bool:
    """Return True when a `[text](target)` pair is prose that accidentally
    matches markdown link syntax, not a real link to a missing file.

    Two real-world patterns from maritime technical emails:

    - ``[cid:17336...@fleet.marantankers.com](sample)`` — Outlook's plain-text
      rendering of an inline ``<img>`` followed by a parenthesised caption.
      The ``cid:`` signals an email content-id reference, never a filesystem
      path.
    - ``[PC-JB1](17&18)`` — engineer prose: component label in brackets,
      terminal numbers in parens. The target has no URL scheme, no path
      separator, and no extension.

    Caller should only consult this for *link-form* (``[...]``) matches;
    image-form (``![...]``) broken targets are always worth reporting, since
    they should have been stripped by ``strip_llamaparse_image_refs``.
    """
    if target.startswith("cid:") or text.startswith("cid:"):
        return True
    # Bare-token target: no URL scheme, no path separator, no file extension.
    if (
        "://" not in target
        and "/" not in target
        and "\\" not in target
        and not Path(target).suffix
    ):
        return True
    return False


def _check_broken_markdown_links(
    docs: List[Dict[str, Any]], archive_dir: Path
) -> Tuple[List[str], List[str]]:
    """Check 20: Broken markdown internal links."""
    issues: List[str] = []
    warnings: List[str] = []
    if not archive_dir.exists():
        return issues, warnings

    broken_md_links: Dict[str, List[str]] = {}
    for md_file in archive_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for img_mark, link_text, link in _re.findall(
            r"(!?)\[(.*?)\]\(([^)]+)\)", content
        ):
            if link.startswith(("http", "#", "mailto:")):
                continue
            # Skip LlamaParse page images (not stored locally)
            if _re.match(r"page_\d+_(?:image|chart|seal)_\d+", link):
                continue
            # Link-form only: filter accidental prose matches so engineer
            # notation like "[PC-JB1](17&18)" or Outlook-flattened cid refs
            # don't masquerade as broken links.
            if not img_mark and _looks_like_prose_match(link_text, link):
                continue
            target = archive_dir / link
            rel_target = md_file.parent / link
            if not target.exists() and not rel_target.exists():
                folder = str(md_file.relative_to(archive_dir)).split("/")[0].split("\\")[0]
                broken_md_links.setdefault(folder, []).append(link)

    total_broken = sum(len(v) for v in broken_md_links.values())
    if total_broken:
        folder_to_email = build_folder_to_email_map(docs)
        truncated = sum(
            1 for links in broken_md_links.values() for l in links if not Path(l).suffix
        )
        unicode_broken = sum(
            1 for links in broken_md_links.values() for l in links if not l.isascii()
        )
        other = total_broken - truncated - unicode_broken
        parts = []
        if truncated:
            parts.append(f"{truncated} truncated by parens")
        if unicode_broken:
            parts.append(f"{unicode_broken} unicode-mangled")
        if other:
            parts.append(f"{other} other")
        detail = ", ".join(parts)
        warnings.append(
            f"{total_broken} broken markdown links in {len(broken_md_links)} archive folders "
            f"({detail})"
        )
        # Sort by broken-link count descending so worst offenders surface first.
        for folder_id, links in sorted(
            broken_md_links.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )[:10]:
            email = folder_to_email.get(folder_id, "unknown email")
            warnings.append(f"    {folder_id} ({email}): {len(links)} broken")
        if len(broken_md_links) > 10:
            warnings.append(f"    ... and {len(broken_md_links) - 10} more folders")
    return issues, warnings


def _check_chunk_positions(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 21: Chunk position validity.

    Thread digest chunks legitimately use (-1, -1) — skip those.
    """
    issues: List[str] = []
    warnings: List[str] = []
    invalid_positions: List[str] = []
    for c in chunks:
        cs = c.get("char_start")
        ce = c.get("char_end")
        if cs is not None and ce is not None and not (cs == -1 and ce == -1):
            if cs < 0 or ce < 0 or cs > ce:
                invalid_positions.append(c.get("chunk_id", "?"))
    if invalid_positions:
        warnings.append(
            f"{len(invalid_positions)} chunks have invalid char positions "
            f"(negative or start > end)"
        )
    return issues, warnings


def _check_email_metadata(
    docs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 22: Email metadata consistency.

    JSONL keys use email_ prefix (see serializers.py doc_to_dict).
    """
    issues: List[str] = []
    warnings: List[str] = []
    bad_dates = 0
    missing_participants = 0
    for d in docs:
        if d.get("document_type") != "email":
            continue
        ds = d.get("email_date_start")
        de = d.get("email_date_end")
        if ds and de and ds > de:
            bad_dates += 1
        if not d.get("email_participants"):
            missing_participants += 1
    if bad_dates:
        warnings.append(f"{bad_dates} emails have date_start > date_end")
    if missing_participants:
        warnings.append(f"{missing_participants} emails have no participants")
    return issues, warnings


# ---------------------------------------------------------------------------
# Extended checks (23+) — SQLite-level integrity + data-quality invariants
# added after the original 22-check contract. These surface issues that
# pre-SQLite validate couldn't observe: FK drift, schema drift, embedding
# mode inheritance, archive-disk drift, residual parser artifacts, stale
# PROCESSING rows, ingest-version regression.
# ---------------------------------------------------------------------------

# Expected columns for critical tables — compared against live schema to
# catch drift between tracker/client schemas. Mirrors sqlite_client.py DDL.
_EXPECTED_COLUMNS: Dict[str, Set[str]] = {
    "documents": {
        "id", "doc_id", "source_id", "document_type", "status", "error_message",
        "file_hash", "file_name", "file_path", "parent_id", "root_id", "depth",
        "content_version", "ingest_version", "archive_path", "title",
        "source_title", "mime_type", "content_type", "size_bytes",
        "embedding_mode", "archive_browse_uri", "archive_download_uri",
        "metadata_json", "processed_at", "created_at", "updated_at",
    },
    "chunks": {
        "id", "chunk_id", "document_id", "source_id", "content", "chunk_index",
        "char_start", "char_end", "line_from", "line_to", "page_number",
        "section_title", "section_path_json", "context_summary",
        "embedding_text", "embedding", "embedding_dim", "embedding_mode",
        "source_title", "archive_browse_uri", "archive_download_uri",
        "metadata_json", "created_at",
    },
    "processing_log": {
        "file_path", "file_hash", "status", "started_at", "completed_at",
        "duration_seconds", "attempts", "error", "ingest_version",
    },
    "chunk_topics": {"chunk_id", "topic_id"},
    "topics": {
        "id", "name", "display_name", "description", "keywords_json",
        "embedding", "embedding_dim", "chunk_count", "document_count",
        "created_at", "updated_at",
    },
}

# Residual parser image-ref patterns — any hit means `strip_llamaparse_image_refs`
# missed a form. Image-form (`![...]`) only; link-form is reported by check 20.
_RESIDUAL_IMAGE_PATTERNS = [
    _re.compile(r"<img\s[^>]*>", _re.IGNORECASE),
    _re.compile(r"!\[[^\]]*\]\(page_\d+_(?:image|chart|seal|table|layout)\w*[^)]*\)"),
    _re.compile(r"!\[[^\]]*\]\(image_\d+\.(?:png|jpe?g)\)"),
    _re.compile(r"!\[[^\]]*\]\(layout(?:_\w+)*\)"),
    _re.compile(r"!\[[^\]]*\]\(image\)"),
]


def _check_sqlite_integrity(conn) -> Tuple[List[str], List[str]]:
    """Check 23: SQLite built-in integrity + foreign-key checks.

    `PRAGMA foreign_key_check` returns one row per FK violation (table,
    rowid, referenced table, fk id). `PRAGMA integrity_check` returns
    the literal string 'ok' when clean.
    """
    issues: List[str] = []
    warnings: List[str] = []
    try:
        fk_rows = list(conn.execute("PRAGMA foreign_key_check"))
    except Exception as e:
        warnings.append(f"foreign_key_check failed to run: {e}")
        fk_rows = []
    if fk_rows:
        by_table: "Counter[str]" = Counter(row[0] for row in fk_rows)
        summary = ", ".join(f"{t}={n}" for t, n in by_table.most_common())
        issues.append(
            f"{len(fk_rows)} foreign-key violations ({summary}) "
            f"— run `mtss repair` or drop the offending rows"
        )

    try:
        ic_rows = list(conn.execute("PRAGMA integrity_check"))
    except Exception as e:
        warnings.append(f"integrity_check failed to run: {e}")
        ic_rows = []
    if ic_rows and not (len(ic_rows) == 1 and ic_rows[0][0] == "ok"):
        issues.append(
            f"SQLite integrity_check failed ({len(ic_rows)} issues) — DB is corrupt"
        )
        for row in ic_rows[:5]:
            issues.append(f"  {row[0]}")
    return issues, warnings


def _check_schema_parity(conn) -> Tuple[List[str], List[str]]:
    """Check 24: Live schema matches the expected column set per table.

    Drift here is the root cause of commit 9c70622 / fc38daf — tracker
    and client used to produce two slightly-different processing_log
    tables, silently dropping writes from whichever lost the race.
    """
    issues: List[str] = []
    warnings: List[str] = []
    for table, expected in _EXPECTED_COLUMNS.items():
        try:
            actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        except Exception as e:
            issues.append(f"Could not read schema for {table}: {e}")
            continue
        if not actual:
            issues.append(f"Table {table!r} is missing")
            continue
        missing = expected - actual
        extra = actual - expected
        if missing:
            issues.append(
                f"Table {table!r} missing columns: {sorted(missing)}"
            )
        if extra:
            warnings.append(
                f"Table {table!r} has unexpected columns (schema drift): {sorted(extra)}"
            )
    return issues, warnings


def _check_embedding_mode_coverage(
    docs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 25: Every non-failed, non-image document has a valid embedding_mode.

    Tied to Bug B-prime (commit 63e608d): empty-parse path used to skip
    stamping embedding_mode, leaving downstream re-embed heuristics with
    no hint whether to chunk or synthesize.
    """
    issues: List[str] = []
    warnings: List[str] = []
    valid = {"full", "summary", "metadata_only"}
    missing: List[Dict[str, Any]] = []
    invalid: List[Tuple[Dict[str, Any], str]] = []
    for d in docs:
        if d.get("status") == "failed":
            continue
        if d.get("document_type") == "attachment_image":
            continue
        mode = d.get("embedding_mode")
        if not mode:
            missing.append(d)
        elif mode not in valid:
            invalid.append((d, mode))
    if missing:
        issues.append(
            f"{len(missing)} document(s) missing embedding_mode "
            f"(re-embed cannot classify — fix with `mtss re-embed`)"
        )
    if invalid:
        issues.append(
            f"{len(invalid)} document(s) have unknown embedding_mode values"
        )
        for d, mode in invalid[:5]:
            issues.append(f"  {d.get('doc_id', '?')[:16]}: {mode!r}")
    return issues, warnings


def _check_embedding_mode_inheritance(
    chunks: List[Dict[str, Any]],
    doc_by_uuid: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 26: Chunks inherit embedding_mode from their parent document.

    Per CLAUDE.md: mode is stamped on Document and inherited by every Chunk.
    Mismatch means the mode changed after chunks were written.
    """
    issues: List[str] = []
    warnings: List[str] = []
    mismatches = 0
    missing_on_chunk = 0
    for c in chunks:
        parent = doc_by_uuid.get(c.get("document_id") or "")
        if not parent:
            continue  # reported by check 4
        parent_mode = parent.get("embedding_mode")
        if not parent_mode:
            continue  # reported by check 25
        chunk_mode = c.get("embedding_mode")
        if not chunk_mode:
            missing_on_chunk += 1
        elif chunk_mode != parent_mode:
            mismatches += 1
    if missing_on_chunk:
        warnings.append(
            f"{missing_on_chunk} chunk(s) missing embedding_mode inherited from document"
        )
    if mismatches:
        issues.append(
            f"{mismatches} chunk(s) have embedding_mode that disagrees with parent document"
        )
    return issues, warnings


def _check_single_chunk_modes(
    docs: List[Dict[str, Any]],
    chunk_doc_ids: "Counter[str]",
) -> Tuple[List[str], List[str]]:
    """Check 27: `summary` / `metadata_only` docs have exactly one chunk.

    The whole point of those modes is one synthesized chunk per doc. A
    count other than 1 (for successful docs) means chunking diverged.
    """
    issues: List[str] = []
    warnings: List[str] = []
    offenders: List[Tuple[Dict[str, Any], int]] = []
    for d in docs:
        if d.get("status") == "failed":
            continue
        if d.get("embedding_mode") not in {"summary", "metadata_only"}:
            continue
        n = chunk_doc_ids.get(d["id"], 0)
        if n != 1:
            offenders.append((d, n))
    if offenders:
        issues.append(
            f"{len(offenders)} summary/metadata_only document(s) "
            f"do not have exactly 1 chunk"
        )
        for d, n in offenders[:5]:
            issues.append(
                f"  {d.get('embedding_mode')}: {d.get('file_name', '?')[:50]} ({n} chunks)"
            )
    return issues, warnings


def _check_orphan_archive_folders(
    docs: List[Dict[str, Any]],
    archive_folders_on_disk: Set[str],
) -> Tuple[List[str], List[str]]:
    """Check 28: Archive folders on disk with no corresponding email document.

    Inverse of check 13. Leftovers from deleted emails or interrupted
    runs waste disk. Warning only — safe to leave; fix with an
    archive-sweep script if they accumulate.
    """
    issues: List[str] = []
    warnings: List[str] = []
    if not archive_folders_on_disk:
        return issues, warnings
    expected: Set[str] = set()
    for d in docs:
        if d.get("depth", 0) != 0:
            continue
        doc_id = d.get("doc_id") or ""
        if doc_id:
            expected.add(compute_folder_id(doc_id))
        # Also accept the cached archive_path (legacy 16-char form still on disk).
        ap = d.get("archive_path")
        if ap:
            expected.add(ap)
    orphans = sorted(archive_folders_on_disk - expected)
    if orphans:
        warnings.append(
            f"{len(orphans)} archive folder(s) on disk with no matching email document"
        )
        for folder in orphans[:10]:
            warnings.append(f"    {folder}")
        if len(orphans) > 10:
            warnings.append(f"    ... and {len(orphans) - 10} more")
    return issues, warnings


def _check_residual_image_refs(
    archive_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Check 29: Residual LlamaParse/Gemini image refs in archive markdown.

    `strip_llamaparse_image_refs` should have scrubbed these; presence means
    the regex missed a form. Run `mtss clean-archive-md` to re-apply.
    """
    issues: List[str] = []
    warnings: List[str] = []
    if not archive_dir.exists():
        return issues, warnings
    offenders: Dict[str, int] = {}
    total = 0
    for md in archive_dir.rglob("*.md"):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = 0
        for pattern in _RESIDUAL_IMAGE_PATTERNS:
            hits += len(pattern.findall(text))
        if hits:
            folder = str(md.relative_to(archive_dir)).split("/")[0].split("\\")[0]
            offenders[folder] = offenders.get(folder, 0) + hits
            total += hits
    if total:
        warnings.append(
            f"{total} residual image ref(s) in {len(offenders)} archive folder(s) "
            f"(run `mtss clean-archive-md`)"
        )
        for folder, n in sorted(offenders.items(), key=lambda kv: -kv[1])[:5]:
            warnings.append(f"    {folder}: {n}")
    return issues, warnings


def _check_duplicate_file_hashes(
    docs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 30: Same file_hash on multiple documents.

    The tracker dedupes pending files by hash, so this should never happen
    during normal ingest. `--retry-failed` with a stale hash or a manual
    SQL insert could theoretically produce one.
    """
    issues: List[str] = []
    warnings: List[str] = []
    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for d in docs:
        h = d.get("file_hash")
        if not h:
            continue
        if d.get("document_type") != "email":
            continue  # attachment dupes are expected (same PDF in many emails)
        by_hash.setdefault(h, []).append(d)
    dupes = {h: ds for h, ds in by_hash.items() if len(ds) > 1}
    if dupes:
        warnings.append(
            f"{len(dupes)} file_hash value(s) map to multiple email documents"
        )
        for h, ds in list(dupes.items())[:5]:
            names = ", ".join(
                d.get("source_id", "?")[:40] for d in ds[:3]
            )
            warnings.append(f"    {h[:12]}: {names}")
    return issues, warnings


def _check_embedding_vector_sanity(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check 31: Embedding vectors are finite and non-zero.

    All-zero / NaN / Inf vectors slip through the embedding API once in a
    while (rate-limit retries, truncated responses). They poison retrieval
    silently — cosine against a zero vector is undefined.
    """
    issues: List[str] = []
    warnings: List[str] = []
    import math
    zero = 0
    nan_inf = 0
    for c in chunks:
        emb = c.get("embedding")
        if not emb:
            continue
        try:
            total_sq = 0.0
            bad = False
            for v in emb:
                if not math.isfinite(v):
                    bad = True
                    break
                total_sq += v * v
        except (TypeError, ValueError):
            continue
        if bad:
            nan_inf += 1
        elif total_sq == 0.0:
            zero += 1
    if nan_inf:
        issues.append(f"{nan_inf} chunk embedding(s) contain NaN or Inf values")
    if zero:
        issues.append(f"{zero} chunk embedding(s) are all zeros (useless for retrieval)")
    return issues, warnings


def _check_outdated_ingest_version(
    docs: List[Dict[str, Any]],
    current_version: Optional[int],
) -> Tuple[List[str], List[str]]:
    """Check 32: Documents stamped with an older ingest_version.

    Informational: `mtss ingest --reprocess-outdated` exists specifically
    to catch these. Warning (not issue) — nothing is broken, just old.
    """
    issues: List[str] = []
    warnings: List[str] = []
    if current_version is None:
        return issues, warnings
    by_version: "Counter[int]" = Counter()
    for d in docs:
        v = d.get("ingest_version")
        if isinstance(v, int) and v < current_version:
            by_version[v] += 1
    if by_version:
        total = sum(by_version.values())
        parts = ", ".join(f"v{v}={n}" for v, n in sorted(by_version.items()))
        warnings.append(
            f"{total} document(s) below current ingest_version (v{current_version}): {parts}"
        )
    return issues, warnings


def _check_thread_root_consistency(
    docs: List[Dict[str, Any]],
    email_uuids: Set[str],
) -> Tuple[List[str], List[str]]:
    """Check 33: Thread-root + attachment-root invariants.

    - Email docs at depth=0: root_id must equal id (they ARE the root).
    - Non-email docs: root_id must reference an existing email document.
    """
    issues: List[str] = []
    warnings: List[str] = []
    self_ref_violations = 0
    missing_root_parent = 0
    for d in docs:
        if d.get("document_type") == "email" and d.get("depth", 0) == 0:
            if d.get("root_id") and d["root_id"] != d["id"]:
                self_ref_violations += 1
        elif d.get("document_type") != "email":
            root = d.get("root_id")
            if root and root not in email_uuids:
                missing_root_parent += 1
    if self_ref_violations:
        issues.append(
            f"{self_ref_violations} email document(s) have root_id != id (thread root mis-stamp)"
        )
    if missing_root_parent:
        warnings.append(
            f"{missing_root_parent} non-email document(s) have root_id pointing to a non-email / missing document"
        )
    return issues, warnings


def _check_stale_processing_entries(conn) -> Tuple[List[str], List[str]]:
    """Check 34: PROCESSING rows older than the stale threshold.

    Covers crashed ingest runs that left rows stuck in PROCESSING. The
    main ingest's `--retry-failed` flow resets these automatically, but
    surfacing them here saves users from discovering the backlog only
    when the next ingest quietly skips over them.
    """
    from datetime import datetime, timedelta, timezone
    from ._common import STALE_PROCESSING_THRESHOLD_MINUTES

    issues: List[str] = []
    warnings: List[str] = []
    try:
        rows = list(conn.execute(
            "SELECT file_path, started_at FROM processing_log "
            "WHERE status = 'PROCESSING' AND started_at IS NOT NULL"
        ))
    except Exception as e:
        warnings.append(f"stale-processing check failed: {e}")
        return issues, warnings
    if not rows:
        return issues, warnings
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=STALE_PROCESSING_THRESHOLD_MINUTES
    )
    stale: List[str] = []
    for row in rows:
        try:
            started = datetime.fromisoformat(row[1])
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if started < cutoff:
                stale.append(row[0])
        except (TypeError, ValueError):
            continue
    if stale:
        warnings.append(
            f"{len(stale)} file(s) stuck in PROCESSING > {STALE_PROCESSING_THRESHOLD_MINUTES} min "
            f"(run `mtss reset-stale` if no ingest is active)"
        )
        for fp in stale[:5]:
            warnings.append(f"    {fp}")
    return issues, warnings


def _run_ingest_validation(output_dir: Path, verbose: bool):
    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        raise typer.Exit(1)

    # Load all data files
    docs = _load_jsonl(output_dir / "documents.jsonl")
    chunks = _load_jsonl(output_dir / "chunks.jsonl")
    topics = _load_jsonl(output_dir / "topics.jsonl")
    events = _load_jsonl(output_dir / "ingest_events.jsonl")
    proc_log = _load_jsonl(output_dir / "processing_log.jsonl")
    run_history = _load_jsonl(output_dir / "run_history.jsonl")

    # Dedicated read-only connection for SQLite-level checks (PRAGMAs,
    # schema parity, stale-row queries). Kept separate from _load_jsonl's
    # transient client connections so it stays open for the whole run.
    import sqlite3 as _sqlite3
    db_path = output_dir / "ingest.db"
    ro_conn: Optional[_sqlite3.Connection] = None
    try:
        ro_conn = _sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro", uri=True, isolation_level=None
        )
    except _sqlite3.OperationalError:
        ro_conn = None

    try:
        current_ingest_version: Optional[int] = None
        try:
            from ..config import get_settings
            current_ingest_version = int(get_settings().current_ingest_version)
        except Exception:
            current_ingest_version = None
        _run_ingest_validation_with_conn(
            output_dir, verbose, docs, chunks, topics, events, proc_log,
            run_history, ro_conn, current_ingest_version,
        )
    finally:
        if ro_conn is not None:
            try:
                ro_conn.close()
            except Exception:
                pass


def _run_ingest_validation_with_conn(
    output_dir: Path,
    verbose: bool,
    docs: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    topics: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    proc_log: List[Dict[str, Any]],
    run_history: List[Dict[str, Any]],
    ro_conn,
    current_ingest_version: Optional[int],
):

    if not docs and not chunks:
        console.print("[yellow]No data found in output directory.[/yellow]")
        raise typer.Exit(0)

    issues: List[str] = []
    warnings: List[str] = []

    # === Index building ===
    doc_by_uuid = {d["id"]: d for d in docs}
    email_uuids = {d["id"] for d in docs if d.get("document_type") == "email"}
    chunk_doc_ids = Counter(c["document_id"] for c in chunks)

    # Event source tracking (for expected missing chunks)
    filtered_doc_uuids: Set[str] = set()
    for e in events:
        if e.get("event_type") in ("message_filtered", "no_body_chunks"):
            if e.get("document_id"):
                filtered_doc_uuids.add(e["document_id"])

    # Processing log — deduplicate by file_path, keep last entry
    proc_by_file: Dict[str, Dict[str, Any]] = {}
    for p in proc_log:
        proc_by_file[p.get("file_path", "")] = p
    failed = [p for p in proc_by_file.values() if p.get("status") != "COMPLETED"]

    # Document type breakdown (for summary table)
    doc_types = Counter(d.get("document_type", "unknown") for d in docs)

    # Embedding presence (for summary table)
    no_embedding = [c for c in chunks if not c.get("embedding")]
    dims: Set[int] = set()
    for c in chunks:
        emb = c.get("embedding")
        if emb:
            dims.add(len(emb))

    # Archive directory + folders on disk (shared across several checks)
    archive_dir = output_dir / "archive"
    archive_folders_on_disk: Set[str] = set()
    total_archive_files = 0
    if archive_dir.exists():
        for p in archive_dir.iterdir():
            if p.is_dir():
                archive_folders_on_disk.add(p.name)
                total_archive_files += _count_archive_files(p)

    # Emails missing archive URIs — also needed for the summary table
    emails_missing_archive: List[Tuple[Dict[str, Any], List[str]]] = []
    for d in docs:
        if d.get("document_type") != "email":
            continue
        missing_fields = [f for f in _ARCHIVE_URI_FIELDS if not d.get(f)]
        if missing_fields:
            emails_missing_archive.append((d, missing_fields))

    # Per-check results carry their number + short name so every emitted
    # message can be cross-referenced back to the specific check that
    # produced it (`[#N] ...` prefix at render time).
    check_runs: List[Tuple[int, str, List[str], List[str]]] = []

    def _run(num: int, name: str, result: Tuple[List[str], List[str]]) -> None:
        ci, cw = result
        check_runs.append((num, name, ci, cw))
        issues.extend(ci)
        warnings.extend(cw)

    _run(1, "duplicate_uuids", _check_duplicate_uuids(docs, doc_by_uuid))
    _run(2, "processing_log", _check_processing_log(proc_by_file))
    _run(3, "document_types", _check_document_types(docs))
    _run(4, "orphan_chunks", _check_orphan_chunks(chunks, doc_by_uuid))
    _run(5, "embedding_completeness", _check_embedding_completeness(chunks))
    _run(6, "empty_content", _check_empty_content(chunks))
    _run(7, "context_summary", _check_context_summary(chunks, docs, doc_by_uuid))
    _run(8, "docs_without_chunks",
         _check_docs_without_chunks(docs, chunk_doc_ids, filtered_doc_uuids))
    _run(9, "orphan_attachments", _check_orphan_attachments(docs, email_uuids))
    _run(10, "failed_documents", _check_failed_documents(docs, events, verbose))
    _run(11, "trailing_dot_filenames", _check_trailing_dot_filenames(docs, verbose))
    _run(12, "topic_health", _check_topic_health(topics))
    _run(13, "archive_uris",
         _check_archive_uris(docs, email_uuids, archive_folders_on_disk, verbose))
    _run(14, "stale_topic_refs", _check_stale_topic_refs(chunks, topics))
    _run(15, "topic_count_accuracy", _check_topic_count_accuracy(chunks, topics))
    _run(16, "broken_archive_uris", _check_broken_archive_uris(docs, archive_dir))
    _run(17, "duplicate_ids", _check_duplicate_ids(docs, chunks, verbose))
    _run(18, "encoded_filenames", _check_encoded_filenames(archive_dir, verbose))
    _run(19, "encoded_uris", _check_encoded_uris(docs))
    _run(20, "broken_markdown_links", _check_broken_markdown_links(docs, archive_dir))
    _run(21, "chunk_positions", _check_chunk_positions(chunks))
    _run(22, "email_metadata", _check_email_metadata(docs))

    # Extended checks (23+) — SQLite integrity + data-quality invariants.
    # Skipped gracefully when the RO connection couldn't be opened.
    if ro_conn is not None:
        _run(23, "sqlite_integrity", _check_sqlite_integrity(ro_conn))
        _run(24, "schema_parity", _check_schema_parity(ro_conn))
    _run(25, "embedding_mode_coverage", _check_embedding_mode_coverage(docs))
    _run(26, "embedding_mode_inheritance",
         _check_embedding_mode_inheritance(chunks, doc_by_uuid))
    _run(27, "single_chunk_modes", _check_single_chunk_modes(docs, chunk_doc_ids))
    _run(28, "orphan_archive_folders",
         _check_orphan_archive_folders(docs, archive_folders_on_disk))
    _run(29, "residual_image_refs", _check_residual_image_refs(archive_dir))
    _run(30, "duplicate_file_hashes", _check_duplicate_file_hashes(docs))
    _run(31, "embedding_vector_sanity", _check_embedding_vector_sanity(chunks))
    _run(32, "outdated_ingest_version",
         _check_outdated_ingest_version(docs, current_ingest_version))
    _run(33, "thread_root_consistency",
         _check_thread_root_consistency(docs, email_uuids))
    if ro_conn is not None:
        _run(34, "stale_processing_entries",
             _check_stale_processing_entries(ro_conn))

    # === Display results ===
    summary = Table(title="Ingest Validation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")

    summary.add_row("Output directory", str(output_dir))
    summary.add_row("Runs", str(len(run_history)))
    summary.add_row("Files processed", str(len(proc_by_file)))
    summary.add_row("Files failed", str(len(failed)))
    summary.add_section()

    summary.add_row("Documents", str(len(docs)))
    for dt, count in sorted(doc_types.items()):
        summary.add_row(f"  {dt}", str(count))
    summary.add_section()

    summary.add_row("Chunks", str(len(chunks)))
    if dims:
        summary.add_row("Embedding dim", str(next(iter(dims))))
    summary.add_row("Embeddings present", f"{len(chunks) - len(no_embedding)}/{len(chunks)}")
    summary.add_section()

    summary.add_row("Topics", str(len(topics)))
    summary.add_row("Ingest events", str(len(events)))
    summary.add_row("Archive folders", str(len(archive_folders_on_disk)))
    summary.add_row("Archive files", str(total_archive_files))
    archive_uri_ok = len(email_uuids) - len(emails_missing_archive)
    summary.add_row("Archive URIs", f"{archive_uri_ok}/{len(email_uuids)}")

    if chunk_doc_ids:
        counts = list(chunk_doc_ids.values())
        summary.add_section()
        summary.add_row("Chunks/doc min", str(min(counts)))
        summary.add_row("Chunks/doc max", str(max(counts)))
        summary.add_row("Chunks/doc avg", str(sum(counts) // len(counts)))

    console.print(summary)
    console.print()

    if verbose:
        detail = Table(title="Document Detail")
        detail.add_column("Type", style="cyan")
        detail.add_column("Source", style="dim", max_width=50)
        detail.add_column("Chunks", justify="right")
        detail.add_column("Subject / File", max_width=50)

        for d in sorted(docs, key=lambda x: (x.get("document_type", ""), x.get("source_id", ""))):
            n_chunks = chunk_doc_ids.get(d["id"], 0)
            label = d.get("email_subject") or d.get("file_name") or d.get("file_path", "")
            if len(label) > 50:
                label = label[:47] + "..."
            chunk_str = str(n_chunks) if n_chunks > 0 else "[dim]0[/dim]"
            detail.add_row(
                d.get("document_type", "?"),
                d.get("source_id", "")[:50],
                chunk_str,
                label,
            )

        console.print(detail)
        console.print()

    if events:
        # Normalize event_type via _reason_category — legacy local_client writes
        # the full skip_reason (e.g. "partial_download: foo.pdf") into
        # event_type, exploding the row count. Collapse to the category.
        event_types: "Counter[str]" = Counter(
            _reason_category(e.get("event_type") or "unknown") for e in events
        )

        # Last-run cutoff and per-doc timestamp lookup (falls back to
        # updated_at when an event pre-dates the timestamp field).
        last_run_cutoff = _compute_last_run_cutoff(run_history)
        doc_ts_by_id: Dict[str, str] = {}
        for d in docs:
            did = d.get("id")
            ts = d.get("updated_at") or d.get("created_at")
            if did and ts:
                doc_ts_by_id[str(did)] = str(ts)

        ev_table = Table(title="Ingest Events")
        ev_table.add_column("Event", style="cyan")
        ev_table.add_column("Count", justify="right", style="green")
        if last_run_cutoff:
            ev_table.add_column("Last run", justify="right", style="magenta")
        ev_table.add_column("Note", style="dim")

        def _emit_row(label: str, total: int, last: int, note: str) -> None:
            if last_run_cutoff:
                ev_table.add_row(label, str(total), str(last) if last else "", note)
            else:
                ev_table.add_row(label, str(total), note)

        for et, count in event_types.most_common():
            # Collect this event's matching records once for cheap reuse.
            matching = [
                e for e in events
                if _reason_category(e.get("event_type") or "") == et
            ]
            last_count = 0
            if last_run_cutoff:
                last_count = sum(
                    1 for e in matching
                    if (_event_timestamp(e, doc_ts_by_id) or "") >= last_run_cutoff
                )
            _emit_row(et, count, last_count, _REASON_DESCRIPTIONS.get(et, ""))

            # Children for this event row. Two modes:
            #  - unsupported_format: break down by mime_type so the user can
            #    decide whether to add a parser for a common format.
            #  - everything else: break down by reason category (collapsing
            #    high-cardinality filenames like "partial_download: foo.pdf").
            child_counts: "Counter[str]" = Counter()
            child_last: "Counter[str]" = Counter()
            child_examples: Dict[str, List[str]] = {}
            child_note_default = ""
            if et == "unsupported_format":
                child_note_default = "candidate for new parser"

            def _child_key(e: Dict[str, Any]) -> str:
                if et == "unsupported_format":
                    mime = e.get("mime_type")
                    if not mime:
                        for src in (e.get("event_type"), e.get("reason")):
                            if src and ":" in src:
                                mime = src.split(":", 1)[1].strip()
                                break
                    return mime or "unknown"
                reason = e.get("reason")
                if not reason:
                    return ""
                cat = _reason_category(reason)
                return "" if cat == et else cat

            for e in matching:
                key = _child_key(e)
                if not key:
                    continue
                child_counts[key] += 1
                if last_run_cutoff and (_event_timestamp(e, doc_ts_by_id) or "") >= last_run_cutoff:
                    child_last[key] += 1
                child_examples.setdefault(key, []).append(e.get("file_name") or "")

            ordered = child_counts.most_common()
            for idx, (label, rc) in enumerate(ordered):
                connector = "\u2514\u2500" if idx == len(ordered) - 1 else "\u251c\u2500"
                base_note = child_note_default or _REASON_DESCRIPTIONS.get(label, "")
                examples = _format_examples(child_examples.get(label, []))
                note = f"{base_note} — {examples}" if base_note and examples else (base_note or examples)
                _emit_row(f" {connector} {label}", rc, child_last.get(label, 0), note)

        console.print(ev_table)

        # Parser attribution for extraction_failed — tells you which parser
        # produced the most failures so you know where to focus. Reads
        # processing trail from each affected email's metadata.json.
        failed_docs_for_attribution = [
            d for d in docs
            if d.get("status") == "failed" and d.get("document_type") != "email"
        ]
        if failed_docs_for_attribution:
            parser_counts = _parser_attribution_for_failed_docs(
                failed_docs_for_attribution, archive_dir
            )
            if parser_counts:
                parts = [f"{p}={c}" for p, c in parser_counts.most_common()]
                console.print(
                    f"[dim]  failed docs by parser:[/dim] {', '.join(parts)}"
                )

        # Verbose: show individual event messages
        if verbose:
            msg_events = [e for e in events if e.get("message")]
            if msg_events:
                console.print()
                msg_table = Table(title="Event Details")
                msg_table.add_column("Event Type", style="cyan")
                msg_table.add_column("Source", style="dim", max_width=40)
                msg_table.add_column("Message", max_width=60)
                for e in msg_events:
                    source = e.get("source_eml_path") or e.get("file_name") or ""
                    msg_table.add_row(
                        e.get("event_type", ""),
                        source[:40],
                        e.get("message", ""),
                    )
                console.print(msg_table)

        console.print()

    # ── Check-level overview ────────────────────────────────────────
    total_checks = len(check_runs)
    failed_checks = sum(1 for _, _, ci, _ in check_runs if ci)
    warn_checks = sum(1 for _, _, ci, cw in check_runs if not ci and cw)
    passed_checks = total_checks - failed_checks - warn_checks

    banner = (
        f"[bold]Checks:[/bold] "
        f"[green]{passed_checks} passed[/green] \u00b7 "
        f"[yellow]{warn_checks} warn[/yellow] \u00b7 "
        f"[red]{failed_checks} fail[/red] "
        f"[dim](of {total_checks})[/dim]"
    )
    console.print(banner)
    console.print()

    # Per-check status table — lists every check that flagged something so
    # the reader can jump straight from `[#N]` references in the detail
    # lists below back to the check that produced them.
    flagged = [
        (num, name, ci, cw) for num, name, ci, cw in check_runs if ci or cw
    ]
    if flagged:
        status_table = Table(title="Check Results (non-clean only)")
        status_table.add_column("#", justify="right", style="dim")
        status_table.add_column("Check", style="cyan")
        status_table.add_column("Status", justify="center")
        status_table.add_column("Issues", justify="right", style="red")
        status_table.add_column("Warnings", justify="right", style="yellow")
        for num, name, ci, cw in flagged:
            # Only the parent (non-indented) messages count toward the
            # headline number; sub-lines are context for the parent.
            n_issues = sum(1 for m in ci if not m.startswith("  "))
            n_warns = sum(1 for m in cw if not m.startswith("  "))
            status = "[red]FAIL[/red]" if n_issues else "[yellow]WARN[/yellow]"
            status_table.add_row(
                str(num), name, status,
                str(n_issues) if n_issues else "",
                str(n_warns) if n_warns else "",
            )
        console.print(status_table)
        console.print()

    # Map issue/warning text → the check that produced it, so every
    # top-level message gets a `[#N]` prefix.
    def _prefix_for(msg: str, runs) -> str:
        for num, _, ci, cw in runs:
            if msg in ci or msg in cw:
                return f"[dim][#{num}][/dim] "
        return ""

    if issues:
        console.print(f"[bold red]Issues ({len(issues)}):[/bold red]")
        for issue in issues:
            if issue.startswith("  "):
                console.print(f"    [dim]{issue.strip()}[/dim]")
            else:
                console.print(f"  [red]\u2717[/red] {_prefix_for(issue, check_runs)}{issue}")
        console.print()

    if warnings:
        console.print(f"[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for warning in warnings:
            if warning.startswith("  "):
                console.print(f"    [dim]{warning.strip()}[/dim]")
            else:
                console.print(f"  [yellow]![/yellow] {_prefix_for(warning, check_runs)}{warning}")
        console.print()

    if not issues and not warnings:
        console.print("[bold green]\u2713 All checks passed[/bold green]")
    elif not issues:
        console.print("[green]\u2713 No errors found[/green] (warnings above are informational)")
    else:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# mtss validate import
# ---------------------------------------------------------------------------


@validate_app.command("import")
def validate_import(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Local output directory to compare against (default: data/output)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-document comparison detail"
    ),
):
    """Compare local ingest output against Supabase to verify import integrity."""
    asyncio.run(_run_import_validation(output_dir, verbose))


async def _run_import_validation(output_dir: Optional[Path], verbose: bool):
    from ..storage.supabase_client import SupabaseClient

    resolved = _resolve_output_dir(output_dir)

    db_path = resolved / "ingest.db"
    if not db_path.exists():
        console.print(f"[red]No ingest.db in {resolved}[/red]")
        console.print("[dim]Run 'mtss ingest' first.[/dim]")
        raise typer.Exit(1)

    # Load local data from the SQLite store.
    local_docs = _load_jsonl(resolved / "documents.jsonl")
    local_chunks = _load_jsonl(resolved / "chunks.jsonl")
    local_topics = _load_jsonl(resolved / "topics.jsonl")

    if not local_docs:
        console.print("[yellow]No local documents to validate.[/yellow]")
        raise typer.Exit(0)

    # Local indexes
    local_doc_ids = {d["doc_id"] for d in local_docs}
    local_topic_names = {t["name"] for t in local_topics}
    local_chunks_by_doc = Counter(c["document_id"] for c in local_chunks)

    console.print(f"Validating import from: {resolved}")
    console.print(f"Local: {len(local_docs)} docs, {len(local_chunks)} chunks, {len(local_topics)} topics")
    console.print()

    # Connect to Supabase
    try:
        db = SupabaseClient()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    issues: List[str] = []
    warnings: List[str] = []

    # Safe defaults for display variables (prevents UnboundLocalError on partial failure)
    remote_doc_map: Dict[str, Dict] = {}
    missing_docs: set = set()
    field_mismatches: List[str] = []
    remote_cc: Dict[str, int] = {}
    chunk_count_mismatches: List[tuple] = []
    null_embedding_count = 0
    remote_embedding_count = 0
    missing_topics: set = set()
    remote_topic_map: Dict[str, Dict] = {}
    orphan_chunk_count = 0
    hierarchy_issues: List[str] = []
    local_archive_files = 0
    archive_missing: List[str] = []
    archive_file_mismatches: List[tuple] = []
    archive_orphans: List[str] = []
    orphan_folders: List[str] = []
    remote_archive_files = 0
    remote_archive_folders = 0
    orphan_docs = 0
    total_remote_docs = 0
    total_remote_chunks = 0
    total_remote_topics = 0
    total_remote_vessels = 0
    local_vessel_count = 0
    remote_vessel_count = 0
    vessel_status_msg = ""

    try:
        pool = await db.get_pool()

        # === 1. Document presence and field checks ===
        async with pool.acquire() as conn:
            remote_docs = await conn.fetch(
                "SELECT id, doc_id, source_id, document_type, file_hash, status, "
                "email_subject, root_id, parent_id, depth, ingest_version "
                "FROM documents WHERE doc_id = ANY($1::text[])",
                list(local_doc_ids),
            )

        remote_doc_map = {r["doc_id"]: dict(r) for r in remote_docs}
        missing_docs = local_doc_ids - set(remote_doc_map.keys())

        if missing_docs:
            issues.append(f"{len(missing_docs)} documents missing from Supabase")
            if verbose:
                for d in local_docs:
                    if d["doc_id"] in missing_docs:
                        issues.append(f"  missing: {d['doc_id']} ({d.get('document_type')}: {d.get('file_name', '?')[:50]})")

        # Field-level checks for present documents
        for local_d in local_docs:
            remote_d = remote_doc_map.get(local_d["doc_id"])
            if not remote_d:
                continue
            for field in ("document_type", "file_hash", "source_id", "ingest_version"):
                local_val = local_d.get(field)
                remote_val = remote_d.get(field)
                if local_val and remote_val and str(local_val) != str(remote_val):
                    field_mismatches.append(
                        f"  {local_d['doc_id']}: {field} local={local_val} vs remote={remote_val}"
                    )

        if field_mismatches:
            issues.append(f"{len(field_mismatches)} document field mismatches")
            if verbose:
                issues.extend(field_mismatches)

        # === 2. Document hierarchy FK integrity (parent_id / root_id) ===
        if remote_docs:
            remote_uuid_set = {r["id"] for r in remote_docs}
            async with pool.acquire() as conn:
                # Check all documents whose parent_id or root_id points to a non-existent document
                broken_parents = await conn.fetchval(
                    "SELECT COUNT(*)::int FROM documents d "
                    "WHERE d.id = ANY($1::uuid[]) "
                    "  AND d.parent_id IS NOT NULL "
                    "  AND NOT EXISTS (SELECT 1 FROM documents p WHERE p.id = d.parent_id)",
                    [r["id"] for r in remote_docs],
                ) or 0
                broken_roots = await conn.fetchval(
                    "SELECT COUNT(*)::int FROM documents d "
                    "WHERE d.id = ANY($1::uuid[]) "
                    "  AND d.root_id IS NOT NULL "
                    "  AND NOT EXISTS (SELECT 1 FROM documents p WHERE p.id = d.root_id)",
                    [r["id"] for r in remote_docs],
                ) or 0

            if broken_parents:
                issues.append(f"{broken_parents} documents have broken parent_id FK in Supabase")
            if broken_roots:
                issues.append(f"{broken_roots} documents have broken root_id FK in Supabase")

        # === 3. Chunk counts per document ===
        if remote_docs:
            async with pool.acquire() as conn:
                remote_chunk_counts = await conn.fetch(
                    "SELECT document_id::text, COUNT(*)::int AS cnt "
                    "FROM chunks WHERE document_id = ANY($1::uuid[]) "
                    "GROUP BY document_id",
                    [r["id"] for r in remote_docs],
                )
            remote_cc = {str(r["document_id"]): r["cnt"] for r in remote_chunk_counts}

            for local_d in local_docs:
                if local_d["doc_id"] not in remote_doc_map:
                    continue
                local_uuid = local_d["id"]
                local_count = local_chunks_by_doc.get(local_uuid, 0)
                remote_uuid = str(remote_doc_map[local_d["doc_id"]]["id"])
                remote_count = remote_cc.get(remote_uuid, 0)
                if local_count != remote_count:
                    chunk_count_mismatches.append(
                        (local_d["doc_id"], local_d.get("file_name", "?"), local_count, remote_count)
                    )

        if chunk_count_mismatches:
            issues.append(f"{len(chunk_count_mismatches)} documents have chunk count mismatches")
            if verbose:
                for doc_id, fname, lc, rc in chunk_count_mismatches:
                    issues.append(f"  {doc_id} ({fname[:40]}): local={lc} remote={rc}")

        # === 4. Embedding presence in Supabase ===
        if remote_docs:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT "
                    "  COUNT(*)::int AS total, "
                    "  COUNT(*) FILTER (WHERE embedding IS NOT NULL)::int AS with_emb "
                    "FROM chunks WHERE document_id = ANY($1::uuid[])",
                    [r["id"] for r in remote_docs],
                )
                remote_embedding_count = row["with_emb"] if row else 0
                null_embedding_count = (row["total"] - row["with_emb"]) if row else 0

        if null_embedding_count:
            issues.append(f"{null_embedding_count} chunks in Supabase have NULL embeddings")

        # === 5. Topic checks (presence, embeddings, stale remote topics) ===
        topic_count_mismatches: List[tuple] = []
        extra_remote_topics: List[str] = []

        async with pool.acquire() as conn:
            all_remote_topics = await conn.fetch(
                "SELECT id, name, document_count, chunk_count, "
                "  (embedding IS NOT NULL) AS has_embedding "
                "FROM topics"
            )

        all_remote_topic_map = {r["name"]: dict(r) for r in all_remote_topics}

        if local_topic_names:
            remote_topic_map = {n: t for n, t in all_remote_topic_map.items() if n in local_topic_names}
            missing_topics = local_topic_names - set(remote_topic_map.keys())

            if missing_topics:
                issues.append(f"{len(missing_topics)} topics missing from Supabase")
                if verbose:
                    for name in sorted(missing_topics):
                        issues.append(f"  missing topic: {name}")

            no_emb_topics = [r["name"] for r in all_remote_topics if not r["has_embedding"] and r["name"] in local_topic_names]
            if no_emb_topics:
                warnings.append(f"{len(no_emb_topics)} topics in Supabase missing embeddings")

            # Compare counts for matched topics
            local_topic_map = {t["name"]: t for t in local_topics}
            for name in local_topic_names & set(remote_topic_map.keys()):
                lt = local_topic_map[name]
                rt = remote_topic_map[name]
                local_cc = lt.get("chunk_count", 0) or 0
                remote_cc_val = rt.get("chunk_count", 0) or 0
                local_dc = lt.get("document_count", 0) or 0
                remote_dc = rt.get("document_count", 0) or 0
                if local_cc != remote_cc_val or local_dc != remote_dc:
                    topic_count_mismatches.append((name, local_dc, remote_dc, local_cc, remote_cc_val))

            if topic_count_mismatches:
                warnings.append(f"{len(topic_count_mismatches)} topics have count mismatches (local vs remote)")
                if verbose:
                    for name, ld, rd, lc, rc in topic_count_mismatches:
                        warnings.append(f"  {name}: docs {ld}/{rd}, chunks {lc}/{rc}")

        # Check for remote topics that don't exist locally (stale/absorbed)
        extra_remote_topics = sorted(set(all_remote_topic_map.keys()) - local_topic_names)
        if extra_remote_topics:
            warnings.append(f"{len(extra_remote_topics)} topics in Supabase not in local data (stale/absorbed)")
            if verbose:
                for name in extra_remote_topics:
                    warnings.append(f"  extra: {name}")

        # === 6. Vessel check ===
        try:
            from ..models.vessel import load_vessels_from_csv
            local_vessels = load_vessels_from_csv()
            local_vessel_count = len(local_vessels) if local_vessels else 0
        except Exception:
            local_vessel_count = 0

        async with pool.acquire() as conn:
            remote_vessel_count = await conn.fetchval(
                "SELECT COUNT(*)::int FROM vessels"
            ) or 0

        if local_vessel_count and remote_vessel_count == 0:
            warnings.append(f"{local_vessel_count} vessels in CSV but 0 in Supabase")
        elif local_vessel_count and remote_vessel_count < local_vessel_count:
            warnings.append(
                f"Vessel count: {local_vessel_count} in CSV, {remote_vessel_count} in Supabase"
            )

        # === 7. Archive storage — full file-level check ===
        archive_dir = resolved / "archive"
        # Build local file keys per folder
        local_keys_by_folder: Dict[str, set[str]] = {}
        if archive_dir.exists():
            for doc_dir in archive_dir.iterdir():
                if doc_dir.is_dir():
                    keys: set[str] = set()
                    for f in doc_dir.rglob("*"):
                        if f.is_file() and f.name not in _HIDDEN_FILES:
                            rel = str(f.relative_to(archive_dir)).replace("\\", "/")
                            keys.add(rel)
                    local_keys_by_folder[doc_dir.name] = keys
                    local_archive_files += len(keys)

        try:
            from ..storage.archive_storage import ArchiveStorage
            storage = ArchiveStorage()

            # List all root-level folders in the bucket (paginated + retried)
            all_remote_folders: set[str] = set()
            try:
                for f in storage.list_folder("", files_only=False):
                    name = f.get("name")
                    if name and not f.get("id"):  # folders have id=null
                        all_remote_folders.add(name)
                remote_archive_folders = len(all_remote_folders)
            except Exception as e:
                warnings.append(f"Could not list root archive folders: {e}")

            root_docs = [d for d in local_docs if d.get("depth", 0) == 0]
            local_folder_ids = {compute_folder_id(d["doc_id"]) for d in root_docs}
            orphan_folders = sorted(all_remote_folders - local_folder_ids)

            folder_ids = [compute_folder_id(d["doc_id"]) for d in root_docs]
            remote_keys_by_folder: Dict[str, set[str]] = {}
            incomplete_folders: List[str] = []
            with make_progress() as progress:
                task_id = progress.add_task(f"Checking {len(folder_ids)} archive folders", total=len(folder_ids))
                for folder_id in folder_ids:
                    keys: set[str] = set()
                    folder_ok = True
                    for subfolder in (folder_id, f"{folder_id}/attachments"):
                        try:
                            for f in storage.list_folder(subfolder):
                                name = f.get("name")
                                if name:
                                    keys.add(f"{subfolder}/{name}")
                        except Exception as e:
                            folder_ok = False
                            logger.warning(
                                f"Could not list remote archive {subfolder!r}: {e}"
                            )
                    if not folder_ok:
                        incomplete_folders.append(folder_id)
                    remote_keys_by_folder[folder_id] = keys
                    progress.advance(task_id)
            if incomplete_folders:
                sample = ", ".join(incomplete_folders[:3])
                more = f" (+{len(incomplete_folders) - 3} more)" if len(incomplete_folders) > 3 else ""
                warnings.append(
                    f"{len(incomplete_folders)} archive folders could not be fully listed "
                    f"after retries — counts may be understated: {sample}{more}"
                )

            for folder_id, remote_keys in remote_keys_by_folder.items():
                remote_count = len(remote_keys)
                remote_archive_files += remote_count
                local_keys = local_keys_by_folder.get(folder_id, set())

                if remote_count == 0:
                    archive_missing.append(folder_id)
                else:
                    orphans = remote_keys - local_keys
                    if orphans:
                        archive_orphans.extend(orphans)

                    if len(local_keys) != remote_count:
                        archive_file_mismatches.append(
                            (folder_id, len(local_keys), remote_count)
                        )
        except Exception as e:
            warnings.append(f"Archive check skipped: {e}")

        if archive_missing:
            issues.append(
                f"{len(archive_missing)}/{len(root_docs)} documents have no archive files in storage"
            )
            if verbose:
                for doc_id in archive_missing:
                    issues.append(f"  no archive files for: {doc_id}")

        if archive_orphans:
            warnings.append(
                f"{len(archive_orphans)} orphan archive files in storage (not in local data)"
            )
            if verbose:
                for key in sorted(archive_orphans)[:20]:
                    warnings.append(f"  orphan: {key}")
                if len(archive_orphans) > 20:
                    warnings.append(f"  ... and {len(archive_orphans) - 20} more")

        if orphan_folders:
            warnings.append(
                f"{len(orphan_folders)} orphan folders in storage (not in local data) — will be cleaned up on next import run"
            )
            if verbose:
                for folder in orphan_folders[:20]:
                    warnings.append(f"  orphan folder: {folder}")
                if len(orphan_folders) > 20:
                    warnings.append(f"  ... and {len(orphan_folders) - 20} more")

        if archive_file_mismatches:
            # Map folder_id -> source email for readable output
            folder_to_email = {
                compute_folder_id(d["doc_id"]): d.get("file_name") or d.get("source_title", "?")
                for d in local_docs if d.get("depth", 0) == 0
            }
            warnings.append(
                f"{len(archive_file_mismatches)} documents have different archive file counts (local vs remote)"
            )
            for doc_id, lc, rc in archive_file_mismatches:
                email_name = folder_to_email.get(doc_id, "?")
                warnings.append(f"  {doc_id} ({email_name}): local={lc} remote={rc}")

        # === 8. Foreign key integrity (chunks -> documents) ===
        if remote_docs:
            async with pool.acquire() as conn:
                orphan_chunk_count = await conn.fetchval(
                    "SELECT COUNT(*)::int FROM chunks c "
                    "LEFT JOIN documents d ON c.document_id = d.id "
                    "WHERE d.id IS NULL AND c.document_id = ANY($1::uuid[])",
                    [r["id"] for r in remote_docs],
                ) or 0

        if orphan_chunk_count:
            issues.append(f"{orphan_chunk_count} chunks in Supabase reference non-existent documents")

        # === 9. Remote totals and orphan documents ===
        async with pool.acquire() as conn:
            total_remote_docs = await conn.fetchval("SELECT COUNT(*)::int FROM documents") or 0
            total_remote_chunks = await conn.fetchval("SELECT COUNT(*)::int FROM chunks") or 0
            total_remote_topics = await conn.fetchval("SELECT COUNT(*)::int FROM topics") or 0

            # Documents in Supabase not present in local JSONL (orphans)
            if local_doc_ids:
                orphan_docs = await conn.fetchval(
                    "SELECT COUNT(*)::int FROM documents "
                    "WHERE doc_id IS NOT NULL AND doc_id != '' "
                    "AND doc_id NOT IN (SELECT unnest($1::text[]))",
                    list(local_doc_ids),
                ) or 0

    finally:
        await db.close()

    # === Display results ===
    summary = Table(title="Import Validation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Local", justify="right", style="green")
    summary.add_column("Remote", justify="right", style="blue")
    summary.add_column("Status", justify="center")

    # Documents — show matched/total to disambiguate when remote has more
    doc_status = "[green]\u2713[/green]" if not missing_docs else f"[red]{len(missing_docs)} missing[/red]"
    remote_doc_label = str(total_remote_docs)
    if total_remote_docs > len(local_docs) and not missing_docs:
        remote_doc_label = f"{len(local_docs)}/{total_remote_docs}"
    summary.add_row("Documents", str(len(local_docs)), remote_doc_label, doc_status)

    # Chunks — matched count / total
    matched_remote_chunks = sum(
        remote_cc.get(str(remote_doc_map[d["doc_id"]]["id"]), 0)
        for d in local_docs if d["doc_id"] in remote_doc_map
    )
    chunk_status = "[green]\u2713[/green]" if not chunk_count_mismatches else f"[red]{len(chunk_count_mismatches)} mismatched[/red]"
    remote_chunk_label = str(total_remote_chunks)
    if total_remote_chunks > len(local_chunks) and not chunk_count_mismatches:
        remote_chunk_label = f"{matched_remote_chunks}/{total_remote_chunks}"
    summary.add_row("Chunks", str(len(local_chunks)), remote_chunk_label, chunk_status)

    # Topics
    topic_problems = len(missing_topics) + len(extra_remote_topics) + len(topic_count_mismatches)
    if missing_topics:
        topic_status = f"[red]{len(missing_topics)} missing[/red]"
    elif extra_remote_topics or topic_count_mismatches:
        parts = []
        if extra_remote_topics:
            parts.append(f"{len(extra_remote_topics)} stale")
        if topic_count_mismatches:
            parts.append(f"{len(topic_count_mismatches)} counts differ")
        topic_status = f"[yellow]{', '.join(parts)}[/yellow]"
    else:
        topic_status = "[green]\u2713[/green]"
    summary.add_row("Topics", str(len(local_topics)), str(total_remote_topics), topic_status)

    summary.add_section()
    emb_status = "[green]\u2713[/green]" if not null_embedding_count else f"[red]{null_embedding_count} null[/red]"
    summary.add_row("Embeddings", f"{len(local_chunks)}", str(remote_embedding_count), emb_status)

    fk_status = "[green]\u2713[/green]" if not orphan_chunk_count else f"[red]{orphan_chunk_count} orphans[/red]"
    summary.add_row("FK integrity", "", "", fk_status)

    hierarchy_ok = not any(
        i for i in issues if "parent_id FK" in i or "root_id FK" in i
    )
    hier_status = "[green]\u2713[/green]" if hierarchy_ok else "[red]broken[/red]"
    summary.add_row("Hierarchy FKs", "", "", hier_status)

    # Vessels
    if local_vessel_count or remote_vessel_count:
        v_ok = remote_vessel_count >= local_vessel_count
        v_status = "[green]\u2713[/green]" if v_ok else f"[yellow]{remote_vessel_count}/{local_vessel_count}[/yellow]"
        summary.add_row("Vessels", str(local_vessel_count), str(remote_vessel_count), v_status)

    # Orphan documents
    if orphan_docs:
        summary.add_row("Orphan docs", "", str(orphan_docs), f"[yellow]{orphan_docs} in remote only[/yellow]")

    # Archive folders
    local_folder_count = len(local_keys_by_folder) if archive_dir.exists() else 0
    if orphan_folders:
        folder_status = f"[yellow]{len(orphan_folders)} orphans[/yellow]"
    else:
        folder_status = "[green]\u2713[/green]"
    summary.add_row("Archive folders", str(local_folder_count), str(remote_archive_folders), folder_status)

    # Archive files
    arc_parts = []
    if archive_missing:
        arc_parts.append(f"[red]{len(archive_missing)} missing[/red]")
    if archive_orphans:
        arc_parts.append(f"[yellow]{len(archive_orphans)} orphans[/yellow]")
    if archive_file_mismatches and not archive_orphans:
        arc_parts.append(f"[yellow]{len(archive_file_mismatches)} mismatched[/yellow]")
    arc_status = ", ".join(arc_parts) if arc_parts else "[green]\u2713[/green]"
    summary.add_row("Archive files", str(local_archive_files), str(remote_archive_files), arc_status)

    console.print(summary)
    console.print()

    # Field mismatch detail
    if verbose and field_mismatches:
        fm_table = Table(title="Field Mismatches")
        fm_table.add_column("Detail", style="red")
        for fm in field_mismatches:
            fm_table.add_row(fm.strip())
        console.print(fm_table)
        console.print()

    # Chunk count mismatch detail
    if verbose and chunk_count_mismatches:
        cc_table = Table(title="Chunk Count Mismatches")
        cc_table.add_column("doc_id", style="cyan")
        cc_table.add_column("File", style="dim", max_width=40)
        cc_table.add_column("Local", justify="right", style="green")
        cc_table.add_column("Remote", justify="right", style="blue")
        for doc_id, fname, lc, rc in chunk_count_mismatches:
            cc_table.add_row(doc_id[:16], fname[:40], str(lc), str(rc))
        console.print(cc_table)
        console.print()

    if issues:
        console.print(f"[bold red]Issues ({len(issues)}):[/bold red]")
        for issue in issues:
            if issue.startswith("  "):
                console.print(f"    [dim]{issue.strip()}[/dim]")
            else:
                console.print(f"  [red]\u2717[/red] {issue}")
        console.print()

    if warnings:
        console.print(f"[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for warning in warnings:
            if warning.startswith("  "):
                console.print(f"    [dim]{warning.strip()}[/dim]")
            else:
                console.print(f"  [yellow]![/yellow] {warning}")
        console.print()

    if not issues and not warnings:
        console.print("[bold green]\u2713 All checks passed \u2014 local and remote are in sync[/bold green]")
    elif not issues:
        console.print("[green]\u2713 No errors found[/green] (warnings above are informational)")
    else:
        raise typer.Exit(1)
