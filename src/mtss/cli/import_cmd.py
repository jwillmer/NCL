"""Import command — push local ingest output to Supabase."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from uuid import UUID

import typer

from ._common import console, make_progress
from .._io import read_bytes_async, retry_with_backoff
from ..models.serializers import dict_to_chunk as _dict_to_chunk
from ..models.serializers import dict_to_document as _dict_to_document
from ..models.serializers import dict_to_topic as _dict_to_topic

logger = logging.getLogger(__name__)

# Retry policy for archive uploads. Supabase Storage occasionally returns
# non-JSON bodies (gateway errors, transient 5xx) which surface as
# JSONDecodeError in storage3. Retry with exponential backoff.
_UPLOAD_MAX_ATTEMPTS = 3
_UPLOAD_BACKOFF_BASE = 1.0  # seconds; delays: 1s, 2s

# Upper bound on concurrent archive uploads. Bounded to avoid saturating
# the Supabase Storage endpoint or the local disk while still overlapping
# network I/O. Exposed as a module-level constant so tests can monkeypatch
# and operators can tune at runtime.
_ARCHIVE_UPLOAD_CONCURRENCY = 8


def register(app: typer.Typer):
    """Register the import command on the app."""

    @app.command("import")
    def import_data(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Local output directory to import from (default: data/../output)",
        ),
        skip_archives: bool = typer.Option(
            False,
            "--skip-archives",
            help="Skip uploading archive files to Supabase Storage",
        ),
        skip_vessels: bool = typer.Option(
            False,
            "--skip-vessels",
            help="Skip importing vessels from CSV",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            "-n",
            help="Show what would be imported without making changes",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ):
        """Push local ingest output to Supabase.

        Reads the local output directory (JSONL files + archive/) and
        imports everything into Supabase. Safe to re-run (idempotent).
        """
        asyncio.run(_import_data(output_dir, skip_archives, skip_vessels, dry_run, verbose))


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read ingest rows from ``ingest.db``. ``path`` is parsed for its parent
    dir and table name — the JSONL file itself is no longer consulted."""
    from .validate_cmd import _load_jsonl as _validate_load

    return _validate_load(path)


# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------


async def _import_data(
    output_dir: Optional[Path],
    skip_archives: bool,
    skip_vessels: bool,
    dry_run: bool,
    verbose: bool,
):
    """Async implementation of import command."""
    from ..config import get_settings
    from ..models.vessel import load_vessels_from_csv
    from ..storage.supabase_client import SupabaseClient

    settings = get_settings()
    resolved_output = (output_dir or settings.eml_source_dir.parent / "output").resolve()

    db_path = resolved_output / "ingest.db"
    if not db_path.exists():
        console.print(f"[red]ingest.db not found in {resolved_output}[/red]")
        console.print("[dim]Run 'MTSS ingest' first to generate local output.[/dim]")
        raise typer.Exit(1)

    console.print(f"Importing from: {resolved_output}")

    db = SupabaseClient()
    totals = {"vessels": 0, "topics": 0, "documents": 0, "chunks": 0, "archive folders": 0, "archive files": 0}
    changes = {"new_documents": 0, "new_chunks": 0, "new_archive files": 0,
               "topics_removed": 0, "orphans_removed": 0, "failed": 0}

    try:
        if not skip_vessels:
            await _import_vessels(db, load_vessels_from_csv(), totals, dry_run, verbose)

        await _import_topics(db, resolved_output, totals, changes, dry_run, verbose)
        await _import_documents(db, resolved_output, totals, changes, dry_run, verbose)

        await _remove_db_orphans(db, resolved_output, changes, dry_run, verbose)

        if not skip_archives:
            archive_dir = resolved_output / "archive"
            if archive_dir.exists():
                docs_data = _read_jsonl(resolved_output / "documents.jsonl")
                local_doc_folder_ids = {
                    d["doc_id"][:16] for d in docs_data
                    if d.get("doc_id") and d.get("document_type") == "email"
                }
                await _import_archives(archive_dir, local_doc_folder_ids, totals, changes, dry_run, verbose)
            elif verbose:
                console.print("[dim]No archive/ directory found, skipping.[/dim]")

        console.print()
        if dry_run:
            console.print("[yellow](dry run — no changes made)[/yellow]")
        from rich.table import Table
        summary = Table(title="Import Summary")
        summary.add_column("Resource", style="cyan")
        summary.add_column("Total", justify="right", style="green")
        summary.add_column("New", justify="right", style="dim")
        for key in ("vessels", "topics", "documents", "chunks", "archive folders", "archive files"):
            total = totals.get(key, 0)
            new = changes.get(f"new_{key}", 0)
            summary.add_row(key, str(total) if total else "", str(new) if new else "")
        # Changes row for removals/failures
        for key in ("topics_removed", "orphans_removed", "failed"):
            count = changes.get(key, 0)
            if count > 0:
                label = key.replace("_", " ")
                color = "red" if key == "failed" else "yellow"
                summary.add_row(f"[{color}]{label}[/{color}]", "", f"[{color}]{count}[/{color}]")
        console.print(summary)

    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


async def _import_vessels(db, vessels, totals, dry_run, verbose):
    """Import vessels from CSV, skipping unchanged."""
    if not vessels:
        if verbose:
            console.print("[dim]No vessels to import.[/dim]")
        return

    console.print(f"Syncing {len(vessels)} vessels...")
    if dry_run:
        totals["vessels"] = len(vessels)
        return

    # Batch-fetch existing vessels to skip unchanged ones
    remote_vessels = await db.get_all_vessels()
    remote_by_name = {
        v.name: (v.vessel_type, v.vessel_class, sorted(v.aliases))
        for v in remote_vessels
    }

    to_upsert = []
    for vessel in vessels:
        local_key = (vessel.vessel_type, vessel.vessel_class, sorted(vessel.aliases))
        if remote_by_name.get(vessel.name) == local_key:
            continue
        to_upsert.append(vessel)

    totals["vessels"] = len(vessels)
    if not to_upsert:
        console.print(f"Vessels up to date ({len(vessels)} already in sync)")
        return

    console.print(f"Upserting {len(to_upsert)} vessels ({len(vessels) - len(to_upsert)} unchanged)...")
    with make_progress() as progress:
        task_id = progress.add_task("Vessels", total=len(to_upsert))
        for vessel in to_upsert:
            try:
                await db.upsert_vessel(vessel)
            except Exception as e:
                logger.warning(f"Failed to import vessel {vessel.name}: {e}")
            progress.update(task_id, advance=1)


async def _import_topics(db, output_dir, totals, changes, dry_run, verbose):
    """Sync topics from topics.jsonl to Supabase (upsert + prune stale)."""
    topics_data = _read_jsonl(output_dir / "topics.jsonl")
    if not topics_data:
        if verbose:
            console.print("[dim]No topics to import.[/dim]")
        return

    local_names = {td["name"] for td in topics_data}
    totals["topics"] = len(topics_data)
    console.print(f"Syncing {len(topics_data)} topics...")

    if dry_run:
        return

    # Batch-fetch all remote topics upfront (1 query instead of N)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        remote_rows = await conn.fetch(
            "SELECT name, chunk_count, document_count, description FROM topics"
        )
    remote_by_name = {r["name"]: r for r in remote_rows}

    # Separate into new, changed, and unchanged topics
    to_insert = []
    to_update = []
    for td in topics_data:
        remote = remote_by_name.get(td["name"])
        if not remote:
            to_insert.append(td)
        elif (remote["chunk_count"] != (td.get("chunk_count", 0) or 0)
              or remote["document_count"] != (td.get("document_count", 0) or 0)
              or remote["description"] != td.get("description")):
            to_update.append(td)

    if not to_insert and not to_update:
        console.print(f"Topics up to date ({len(topics_data)} unchanged)")
    else:
        console.print(f"  {len(to_insert)} new, {len(to_update)} changed")
        work = [(td, "insert") for td in to_insert] + [(td, "update") for td in to_update]
        with make_progress() as progress:
            task_id = progress.add_task("Topics", total=len(work))
            for td, action in work:
                try:
                    if action == "update":
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE topics SET display_name=$2, description=$3, "
                                "embedding=$4, chunk_count=$5, document_count=$6, "
                                "updated_at=NOW() WHERE name=$1",
                                td["name"],
                                td.get("display_name", td["name"]),
                                td.get("description"),
                                td.get("embedding"),
                                td.get("chunk_count", 0) or 0,
                                td.get("document_count", 0) or 0,
                            )
                    else:
                        topic = _dict_to_topic(td)
                        await db.insert_topic(topic)
                except Exception as e:
                    logger.warning(f"Failed to import topic {td.get('name')}: {e}")
                    changes["failed"] += 1
                progress.update(task_id, advance=1)

    # Remove stale topics from Supabase that no longer exist locally
    stale_names = [name for name in remote_by_name if name not in local_names]
    if stale_names:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM topics WHERE name = ANY($1::text[])", stale_names
                )
            changes["topics_removed"] = len(stale_names)
            if verbose:
                for name in stale_names:
                    console.print(f"  [dim]Removed stale topic: {name}[/dim]")
        except Exception as e:
            logger.warning(f"Failed to prune stale topics: {e}")


async def _import_documents(db, output_dir, totals, changes, dry_run, verbose):
    """Import documents + chunks from JSONL files."""
    docs_data = _read_jsonl(output_dir / "documents.jsonl")
    chunks_data = _read_jsonl(output_dir / "chunks.jsonl")

    if not docs_data:
        console.print("[dim]No documents to import.[/dim]")
        return

    # Index chunks by document_id
    chunks_by_doc: Dict[str, List[Dict]] = {}
    for cd in chunks_data:
        chunks_by_doc.setdefault(cd.get("document_id", ""), []).append(cd)

    # Group documents by root (depth=0)
    roots = [d for d in docs_data if d.get("depth", 0) == 0]
    children_by_root: Dict[str, List[Dict]] = {}
    for d in docs_data:
        if d.get("depth", 0) > 0:
            children_by_root.setdefault(d.get("root_id", ""), []).append(d)

    totals["documents"] = len(docs_data)
    totals["chunks"] = len(chunks_data)
    console.print(f"Importing {len(roots)} emails ({len(docs_data)} documents, {len(chunks_data)} chunks)...")
    if dry_run:
        return

    # Batch-fetch existing doc_ids for all documents (roots + attachments)
    all_local_doc_ids = [d.get("doc_id", "") for d in docs_data if d.get("doc_id")]
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        existing_rows = await conn.fetch(
            "SELECT doc_id FROM documents WHERE doc_id = ANY($1::text[])",
            all_local_doc_ids,
        )
    existing_doc_ids = {r["doc_id"] for r in existing_rows}

    new_roots = [r for r in roots if r.get("doc_id", "") not in existing_doc_ids]

    if not new_roots:
        console.print(f"Documents up to date ({len(roots)} already imported)")
        return

    console.print(f"  {len(new_roots)} new, {len(roots) - len(new_roots)} already imported")

    with make_progress() as progress:
        task_id = progress.add_task("Documents", total=len(new_roots))
        for root_dict in new_roots:
            try:
                email_doc = _dict_to_document(root_dict)
                child_dicts = children_by_root.get(str(root_dict["id"]), [])
                # Deduplicate children by doc_id, skip existing and self-references
                skip_doc_ids = existing_doc_ids | {root_dict.get("doc_id")}
                seen_doc_ids: set = set()
                unique_children = []
                for cd in child_dicts:
                    did = cd.get("doc_id")
                    if did in skip_doc_ids or did in seen_doc_ids:
                        continue
                    seen_doc_ids.add(did)
                    unique_children.append(cd)
                attachment_docs = [_dict_to_document(cd) for cd in unique_children]

                # Gather chunks for all NEW documents in this email tree
                all_doc_ids = [str(email_doc.id)] + [str(d.id) for d in attachment_docs]
                all_chunks = []
                for did in all_doc_ids:
                    all_chunks.extend(_dict_to_chunk(cd) for cd in chunks_by_doc.get(did, []))

                await db.persist_ingest_result(
                    email_doc=email_doc,
                    attachment_docs=attachment_docs,
                    chunks=all_chunks,
                    topic_ids=None,
                    chunk_delta=0,
                )
                changes["new_documents"] += 1 + len(attachment_docs)
                changes["new_chunks"] += len(all_chunks)

                if verbose:
                    console.print(f"  [dim]Imported: {root_dict.get('source_title', root_dict.get('file_name', '?'))}[/dim]")

            except Exception as e:
                logger.warning(f"Failed to import document {root_dict.get('doc_id', '?')}: {e}")
                changes["failed"] += 1

            progress.update(task_id, advance=1)


async def _remove_db_orphans(db, output_dir: Path, changes, dry_run, verbose):
    """Remove documents and chunks from Supabase that no longer exist locally."""
    local_doc_ids = set()
    for d in _read_jsonl(output_dir / "documents.jsonl"):
        if d.get("doc_id"):
            local_doc_ids.add(d["doc_id"])

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        remote_rows = await conn.fetch("SELECT id, doc_id FROM documents")

    remote_by_doc_id = {r["doc_id"]: r["id"] for r in remote_rows if r["doc_id"]}
    orphan_doc_ids = set(remote_by_doc_id.keys()) - local_doc_ids

    if not orphan_doc_ids:
        return

    orphan_uuids = [remote_by_doc_id[did] for did in orphan_doc_ids]
    if dry_run:
        console.print(f"[yellow]Would remove {len(orphan_doc_ids)} orphan documents from database[/yellow]")
        return

    console.print(f"Removing {len(orphan_doc_ids)} orphan documents from database...")
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM chunks WHERE document_id = ANY($1::uuid[])", orphan_uuids)
        await conn.execute("DELETE FROM documents WHERE id = ANY($1::uuid[])", orphan_uuids)
    console.print(f"  [green]Removed {len(orphan_doc_ids)} orphan documents and their chunks[/green]")
    changes["orphans_removed"] += len(orphan_doc_ids)


def _upload_with_retry(storage, rel_key: str, payload: bytes, content_type: str) -> bool:
    """Upload one archive file with exponential backoff retry.

    Returns True on success, False after all attempts exhausted. Handles
    transient Supabase Storage failures (non-JSON gateway responses,
    temporary 5xx) that surface as JSONDecodeError in storage3.
    """
    def _upload() -> None:
        storage.upload_file(rel_key, payload, content_type)

    def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
        logger.warning(
            f"Upload failed for {rel_key} (attempt {attempt}/{_UPLOAD_MAX_ATTEMPTS}): {exc}. "
            f"Retrying in {delay:.1f}s..."
        )

    try:
        retry_with_backoff(
            _upload,
            max_attempts=_UPLOAD_MAX_ATTEMPTS,
            backoff_base=_UPLOAD_BACKOFF_BASE,
            on_retry=_log_retry,
            # Route through the module's `time` attribute so tests that
            # patch `import_cmd.time` continue to observe sleeps.
            sleep=lambda d: time.sleep(d),
        )
        return True
    except Exception as last_error:
        logger.warning(
            f"Failed to upload {rel_key} after {_UPLOAD_MAX_ATTEMPTS} attempts: {last_error}"
        )
        return False


async def _import_archives(archive_dir: Path, local_doc_folder_ids: set, totals, changes, dry_run, verbose):
    """Upload archive files to Supabase Storage, skipping files that already exist."""
    from ..storage.archive_storage import ArchiveStorage

    resolved_archive = archive_dir.resolve()

    # Collect local files grouped by folder (doc_id[:16])
    local_by_folder: Dict[str, List[tuple[Path, str]]] = {}
    for file_path in archive_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            resolved = file_path.resolve()
            if not resolved.is_relative_to(resolved_archive):
                logger.warning(f"Skipping symlink outside archive: {file_path}")
                continue
        except (OSError, ValueError):
            continue

        rel_key = str(file_path.relative_to(archive_dir)).replace("\\", "/")

        # Check for actual path traversal (.. as a path segment, not in filenames like "S.A..pdf")
        if any(seg == ".." for seg in rel_key.split("/")):
            logger.warning(f"Skipping path with traversal: {rel_key}")
            continue
        folder = rel_key.split("/", 1)[0]
        local_by_folder.setdefault(folder, []).append((file_path, rel_key))

    total_local = sum(len(files) for files in local_by_folder.values())
    if not total_local:
        if verbose:
            console.print("[dim]No archive files to upload.[/dim]")
        return

    totals["archive folders"] = len(local_by_folder)
    totals["archive files"] = total_local
    if dry_run:
        console.print(f"Would upload up to {total_local} archive files...")
        return

    storage = ArchiveStorage()

    # Pre-fetch existing files per folder to skip already-uploaded files.
    # list_folder handles pagination (>100 files) and retries transient
    # gateway errors — silent truncation here causes phantom re-uploads.
    existing_keys: set[str] = set()
    subfolders = []
    for folder_id in local_by_folder:
        subfolders.append(folder_id)
        subfolders.append(f"{folder_id}/attachments")

    with make_progress() as progress:
        task_id = progress.add_task(f"Checking {len(local_by_folder)} folders", total=len(subfolders))
        for subfolder in subfolders:
            try:
                for f in storage.list_folder(subfolder):
                    name = f.get("name")
                    if name:
                        existing_keys.add(f"{subfolder}/{name}")
            except Exception as e:
                logger.warning(f"Could not list existing archive files in {subfolder!r}: {e}")
            progress.advance(task_id)

    # Build set of expected local keys for orphan detection
    local_keys = {rel_key for files in local_by_folder.values() for _, rel_key in files}

    files_to_upload = [
        (local_path, rel_key)
        for files in local_by_folder.values()
        for local_path, rel_key in files
        if rel_key not in existing_keys
    ]

    # Detect orphan files: in remote but not in local
    orphan_keys = existing_keys - local_keys

    # Detect orphan folders: list all root-level folders in bucket, compare to local
    orphan_folder_ids: List[str] = []
    all_remote_folders: set[str] = set()
    try:
        for f in storage.list_folder("", files_only=False):
            name = f.get("name")
            if name and not f.get("id"):  # folders have id=null
                all_remote_folders.add(name)
    except Exception as e:
        logger.warning(f"Could not list root archive folders: {e}")
    # Compare against documents.jsonl doc_ids, not just disk folders
    orphan_folder_ids = sorted(all_remote_folders - local_doc_folder_ids)

    if not files_to_upload and not orphan_keys and not orphan_folder_ids:
        console.print(f"Archives up to date ({total_local} files already in storage)")
        return

    if files_to_upload:
        skipped = total_local - len(files_to_upload)
        console.print(f"Uploading {len(files_to_upload)} new archive files ({skipped} already exist)...")
        with make_progress() as progress:
            task_id = progress.add_task("Archives", total=len(files_to_upload))

            # Bound concurrent uploads to avoid saturating Supabase Storage.
            # Read and upload happen on threads via asyncio.to_thread so the
            # event loop is never blocked on disk or network I/O.
            semaphore = asyncio.Semaphore(_ARCHIVE_UPLOAD_CONCURRENCY)

            async def _upload_one(local_path: Path, rel_key: str) -> bool:
                async with semaphore:
                    content_type = (
                        mimetypes.guess_type(str(local_path))[0]
                        or "application/octet-stream"
                    )
                    payload = await read_bytes_async(local_path)
                    success = await asyncio.to_thread(
                        _upload_with_retry, storage, rel_key, payload, content_type
                    )
                    progress.update(task_id, advance=1)
                    return success

            results = await asyncio.gather(
                *(_upload_one(lp, rk) for lp, rk in files_to_upload),
                return_exceptions=True,
            )

            for result in results:
                if result is True:
                    changes["new_archive files"] += 1
                else:
                    # False or an exception (gather captured it because of
                    # return_exceptions=True) — both count as a failed upload.
                    changes["failed"] += 1
                    if isinstance(result, BaseException):
                        logger.warning(f"Archive upload raised: {result!r}")

    if orphan_keys:
        console.print(f"Removing {len(orphan_keys)} orphan archive files...")
        removed = 0
        try:
            storage.bucket.remove(list(orphan_keys))
            removed = len(orphan_keys)
        except Exception as e:
            logger.warning(f"Failed to remove orphan files: {e}")
        if removed:
            console.print(f"  [green]Removed {removed} orphan files[/green]")
        changes["orphans_removed"] += removed

    if orphan_folder_ids:
        console.print(f"Removing {len(orphan_folder_ids)} orphan folders...")
        removed_folders = 0
        for folder_id in orphan_folder_ids:
            try:
                storage.delete_folder(folder_id)
                removed_folders += 1
            except Exception as e:
                logger.warning(f"Failed to remove folder {folder_id}: {e}")
        if removed_folders:
            console.print(f"  [green]Removed {removed_folders} orphan folders[/green]")
        changes["orphans_removed"] += removed_folders
