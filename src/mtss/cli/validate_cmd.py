"""Validate ingest output and Supabase import for data integrity issues."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.table import Table

from ._common import console

# Hidden/system files to exclude from archive file counts
_HIDDEN_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})

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

    # === 1. Duplicate detection ===
    if len(doc_by_uuid) != len(docs):
        issues.append(f"Duplicate document UUIDs: {len(docs)} rows but {len(doc_by_uuid)} unique ids")

    doc_id_counts = Counter(d["doc_id"] for d in docs)
    dup_doc_ids = {k: v for k, v in doc_id_counts.items() if v > 1}
    if dup_doc_ids:
        issues.append(f"{len(dup_doc_ids)} duplicate doc_id values")

    chunk_id_counts = Counter(c.get("chunk_id", "") for c in chunks if c.get("chunk_id"))
    dup_chunk_ids = {k: v for k, v in chunk_id_counts.items() if v > 1}
    if dup_chunk_ids:
        issues.append(f"{len(dup_chunk_ids)} duplicate chunk_id values")

    # Event source tracking (for expected missing chunks)
    filtered_doc_uuids = set()
    for e in events:
        if e.get("event_type") in ("message_filtered", "no_body_chunks"):
            if e.get("document_id"):
                filtered_doc_uuids.add(e["document_id"])

    # === 2. Processing log checks (deduplicate by file_path, keep last entry) ===
    proc_by_file: Dict[str, Dict] = {}
    for p in proc_log:
        proc_by_file[p.get("file_path", "")] = p
    failed = [p for p in proc_by_file.values() if p.get("status") != "COMPLETED"]
    if failed:
        issues.append(f"{len(failed)} files not COMPLETED in processing log")
        for f_entry in failed:
            issues.append(
                f"  {f_entry.get('file_path', '?')}: "
                f"status={f_entry.get('status')}, error={f_entry.get('error')}"
            )

    # === 3. Document type breakdown ===
    doc_types = Counter(d.get("document_type", "unknown") for d in docs)

    # === 4. Chunk -> Document linkage ===
    orphan_chunks = [c for c in chunks if c["document_id"] not in doc_by_uuid]
    if orphan_chunks:
        issues.append(
            f"{len(orphan_chunks)} chunks reference non-existent documents"
        )

    # === 5. Embedding completeness ===
    no_embedding = [c for c in chunks if not c.get("embedding")]
    if no_embedding:
        issues.append(f"{len(no_embedding)}/{len(chunks)} chunks missing embeddings")

    # Embedding dimension consistency
    dims = set()
    for c in chunks:
        emb = c.get("embedding")
        if emb:
            dims.add(len(emb))
    if len(dims) > 1:
        issues.append(f"Inconsistent embedding dimensions: {dims}")

    # === 6. Empty content ===
    empty_content = [
        c for c in chunks if not c.get("content") or not c["content"].strip()
    ]
    if empty_content:
        issues.append(f"{len(empty_content)} chunks have empty content")

    # === 7. Context summary / embedding_text presence ===
    # Image chunks skip LLM context generation by design — only flag non-image chunks
    image_doc_uuids = {d["id"] for d in docs if d.get("document_type") == "attachment_image"}
    text_chunks = [c for c in chunks if c["document_id"] not in image_doc_uuids]
    no_context = sum(1 for c in text_chunks if not c.get("context_summary"))
    no_emb_text = sum(1 for c in text_chunks if not c.get("embedding_text"))
    if no_context:
        warnings.append(f"{no_context}/{len(text_chunks)} text chunks missing context_summary")
    if no_emb_text:
        warnings.append(f"{no_emb_text}/{len(text_chunks)} text chunks missing embedding_text")

    # === 8. Documents without chunks ===
    docs_without_chunks = []
    for d in docs:
        if d["id"] not in chunk_doc_ids:
            # Skip emails explicitly filtered (forwarding/cover emails)
            if d["id"] in filtered_doc_uuids:
                continue
            # Images may legitimately have no chunks (non-content)
            if d.get("document_type") == "attachment_image":
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

    # === 9. Attachment -> Email parent chain ===
    orphan_attachments = []
    for d in docs:
        if d.get("document_type") != "email" and d.get("root_id"):
            if d["root_id"] not in email_uuids:
                orphan_attachments.append(d)
    if orphan_attachments:
        warnings.append(
            f"{len(orphan_attachments)} attachments have root_id not matching any email"
        )

    # === 10. Topic health ===
    no_topic_embedding = sum(1 for t in topics if not t.get("embedding"))
    if no_topic_embedding:
        warnings.append(f"{no_topic_embedding}/{len(topics)} topics missing embeddings")

    zero_count_topics = [t for t in topics if (t.get("document_count") or 0) == 0]
    if zero_count_topics:
        warnings.append(
            f"{len(zero_count_topics)}/{len(topics)} topics have document_count=0"
        )

    # === 11. Archive directory and URI checks ===
    archive_dir = output_dir / "archive"
    _ARCHIVE_URI_FIELDS = ("archive_path", "archive_browse_uri", "archive_download_uri")
    archive_folders_on_disk: set[str] = set()
    total_archive_files = 0
    if archive_dir.exists():
        for p in archive_dir.iterdir():
            if p.is_dir():
                archive_folders_on_disk.add(p.name)
                total_archive_files += _count_archive_files(p)

    # URI + folder checks in a single pass over email docs
    emails_missing_archive = []
    emails_without_folder = []
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
                issues.append(f"  {d.get('doc_id', '?')[:16]}: missing {', '.join(fields)}")

    if emails_without_folder:
        warnings.append(
            f"{len(emails_without_folder)} emails have no archive folder on disk"
        )

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
    remote_archive_files = 0
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
        local_files_by_doc: Dict[str, int] = {}
        if archive_dir.exists():
            for doc_dir in archive_dir.iterdir():
                if doc_dir.is_dir():
                    count = _count_archive_files(doc_dir)
                    local_files_by_doc[doc_dir.name] = count
                    local_archive_files += count

        try:
            from ..storage.archive_storage import ArchiveStorage
            storage = ArchiveStorage()

            root_docs = [d for d in local_docs if d.get("depth", 0) == 0]
            for d in root_docs:
                # Archive folders use first 16 chars of doc_id
                folder_id = d["doc_id"][:16]
                files = [f for f in storage.list_files(folder_id) if f.get("id")]
                remote_count = len(files)
                remote_archive_files += remote_count
                local_count = local_files_by_doc.get(folder_id, 0)

                if remote_count == 0:
                    archive_missing.append(folder_id)
                elif local_count != remote_count:
                    archive_file_mismatches.append(
                        (folder_id, local_count, remote_count)
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

        if archive_file_mismatches:
            warnings.append(
                f"{len(archive_file_mismatches)} documents have different archive file counts (local vs remote)"
            )
            if verbose:
                for doc_id, lc, rc in archive_file_mismatches:
                    warnings.append(f"  {doc_id}: local={lc} remote={rc}")

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

        # === 9. Remote totals for context ===
        async with pool.acquire() as conn:
            total_remote_docs = await conn.fetchval("SELECT COUNT(*)::int FROM documents") or 0
            total_remote_chunks = await conn.fetchval("SELECT COUNT(*)::int FROM chunks") or 0
            total_remote_topics = await conn.fetchval("SELECT COUNT(*)::int FROM topics") or 0

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

    # Archives
    if archive_missing:
        arc_status = f"[red]{len(archive_missing)} missing[/red]"
    elif archive_file_mismatches:
        arc_status = f"[yellow]{len(archive_file_mismatches)} mismatched[/yellow]"
    else:
        arc_status = "[green]\u2713[/green]"
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
