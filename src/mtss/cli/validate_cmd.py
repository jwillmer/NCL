"""Validate ingest output and Supabase import for data integrity issues."""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import typer
from rich.table import Table

from ._common import console, make_progress

logger = logging.getLogger(__name__)

# Hidden/system files to exclude from archive file counts
_HIDDEN_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})

_ARCHIVE_URI_FIELDS = ("archive_path", "archive_browse_uri", "archive_download_uri")
_ENCODED_FILENAME_RE = _re.compile(r"%[0-9A-Fa-f]{2}")

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

    Keyed on archive_path when present (what folders are named), else doc_id[:16].
    Value prefers source_id (eml filename) so warnings point users at a real file.
    """
    mapping: Dict[str, str] = {}
    for d in docs:
        if d.get("depth", 0) != 0:
            continue
        key = d.get("archive_path") or (d.get("doc_id") or "")[:16]
        if not key:
            continue
        mapping[key] = (
            d.get("source_id") or d.get("file_name") or d.get("source_title") or "?"
        )
    return mapping


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning empty list if missing."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    Image chunks skip LLM context generation by design — only flag non-image chunks.
    """
    issues: List[str] = []
    warnings: List[str] = []
    image_doc_uuids = {
        d["id"] for d in docs if d.get("document_type") == "attachment_image"
    }
    text_chunks = [c for c in chunks if c["document_id"] not in image_doc_uuids]
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
    docs: List[Dict[str, Any]], verbose: bool
) -> Tuple[List[str], List[str]]:
    """Check 10: Failed documents still in output."""
    issues: List[str] = []
    warnings: List[str] = []
    failed_docs = [d for d in docs if d.get("status") == "failed"]
    if failed_docs:
        warnings.append(
            f"{len(failed_docs)} document(s) have status='failed' in documents.jsonl"
        )
        if verbose:
            for d in failed_docs:
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
            folder_id = d.get("archive_path") or d["doc_id"][:16]
            if folder_id not in archive_folders_on_disk:
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
        issues.append(f"{len(dup_doc_ids)} duplicate doc_ids in documents.jsonl")
        if verbose:
            for did, cnt in list(dup_doc_ids.items())[:5]:
                issues.append(f"  {did[:16]}: {cnt}x")

    chunk_id_counts = Counter(c.get("chunk_id") for c in chunks if c.get("chunk_id"))
    dup_chunk_ids = {cid: cnt for cid, cnt in chunk_id_counts.items() if cnt > 1}
    if dup_chunk_ids:
        issues.append(f"{len(dup_chunk_ids)} duplicate chunk_ids in chunks.jsonl")
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
        for link in _re.findall(r"\[.*?\]\(([^)]+)\)", content):
            if link.startswith(("http", "#", "mailto:")):
                continue
            # Skip LlamaParse page images (not stored locally)
            if _re.match(r"page_\d+_(?:image|chart|seal)_\d+", link):
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
        for folder_id, links in sorted(broken_md_links.items())[:10]:
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

    # === Run all 22 checks in order ===
    # Each check appends to the shared issues/warnings lists.
    _check_results: List[Tuple[List[str], List[str]]] = [
        # === 1. Duplicate UUID detection ===
        _check_duplicate_uuids(docs, doc_by_uuid),
        # === 2. Processing log checks ===
        _check_processing_log(proc_by_file),
        # === 3. Document type breakdown ===
        _check_document_types(docs),
        # === 4. Chunk -> Document linkage ===
        _check_orphan_chunks(chunks, doc_by_uuid),
        # === 5. Embedding completeness ===
        _check_embedding_completeness(chunks),
        # === 6. Empty content ===
        _check_empty_content(chunks),
        # === 7. Context summary / embedding_text presence ===
        _check_context_summary(chunks, docs, doc_by_uuid),
        # === 8. Documents without chunks ===
        _check_docs_without_chunks(docs, chunk_doc_ids, filtered_doc_uuids),
        # === 9. Attachment -> Email parent chain ===
        _check_orphan_attachments(docs, email_uuids),
        # === 10. Failed documents still in output ===
        _check_failed_documents(docs, verbose),
        # === 11. Trailing-dot filenames (may cause parsing issues) ===
        _check_trailing_dot_filenames(docs, verbose),
        # === 12. Topic health ===
        _check_topic_health(topics),
        # === 13. Archive directory and URI checks ===
        _check_archive_uris(docs, email_uuids, archive_folders_on_disk, verbose),
        # === 14. Stale topic references in chunk metadata ===
        _check_stale_topic_refs(chunks, topics),
        # === 15. Topic count accuracy (JSONL counts vs actual chunk refs) ===
        _check_topic_count_accuracy(chunks, topics),
        # === 16. Archive URIs point to existing files on disk ===
        _check_broken_archive_uris(docs, archive_dir),
        # === 17. Duplicate doc_ids and chunk_ids ===
        _check_duplicate_ids(docs, chunks, verbose),
        # === 18. Encoded filenames on disk (should use underscores, not %XX) ===
        _check_encoded_filenames(archive_dir, verbose),
        # === 19. Encoded URIs in documents.jsonl ===
        _check_encoded_uris(docs),
        # === 20. Broken markdown internal links ===
        _check_broken_markdown_links(docs, archive_dir),
        # === 21. Chunk position validity ===
        _check_chunk_positions(chunks),
        # === 22. Email metadata consistency ===
        _check_email_metadata(docs),
    ]
    for check_issues, check_warnings in _check_results:
        issues.extend(check_issues)
        warnings.extend(check_warnings)

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
        event_types = Counter(e.get("event_type") or "unknown" for e in events)

        ev_table = Table(title="Ingest Events")
        ev_table.add_column("Event", style="cyan")
        ev_table.add_column("Count", justify="right", style="green")
        for et, count in event_types.most_common():
            ev_table.add_row(et, str(count))
            # Show reason breakdown indented below (skip if reason == event_type)
            reason_counts = Counter(
                e.get("reason") for e in events
                if e.get("event_type") == et and e.get("reason") and e.get("reason") != et
            )
            for reason, rc in reason_counts.most_common():
                ev_table.add_row(f"  {reason}", str(rc))

        console.print(ev_table)

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

    docs_path = resolved / "documents.jsonl"
    if not docs_path.exists():
        console.print(f"[red]No documents.jsonl in {resolved}[/red]")
        console.print("[dim]Run 'mtss ingest' first.[/dim]")
        raise typer.Exit(1)

    # Load local data
    local_docs = _load_jsonl(docs_path)
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
            local_folder_ids = {d["doc_id"][:16] for d in root_docs}
            orphan_folders = sorted(all_remote_folders - local_folder_ids)

            folder_ids = [d["doc_id"][:16] for d in root_docs]
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
                d["doc_id"][:16]: d.get("file_name") or d.get("source_title", "?")
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
