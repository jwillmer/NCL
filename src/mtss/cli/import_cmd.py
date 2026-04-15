"""Import command — push local ingest output to Supabase."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import typer

from ._common import console, make_progress

logger = logging.getLogger(__name__)


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
    """Read all records from a JSONL file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Model reconstruction from JSONL dicts
# ---------------------------------------------------------------------------


def _dict_to_document(d: Dict[str, Any]) -> "Document":
    """Reconstruct a Document model from a JSONL dict."""
    from ..models.document import (
        AttachmentMetadata,
        Document,
        DocumentType,
        EmailMetadata,
        ProcessingStatus,
    )

    email_metadata = None
    if d.get("email_subject") or d.get("email_participants"):
        email_metadata = EmailMetadata(
            subject=d.get("email_subject"),
            participants=d.get("email_participants") or [],
            initiator=d.get("email_initiator"),
            date_start=datetime.fromisoformat(d["email_date_start"]) if d.get("email_date_start") else None,
            date_end=datetime.fromisoformat(d["email_date_end"]) if d.get("email_date_end") else None,
            message_count=d.get("email_message_count") or 1,
        )

    attachment_metadata = None
    if d.get("attachment_content_type"):
        attachment_metadata = AttachmentMetadata(
            content_type=d["attachment_content_type"],
            size_bytes=d.get("attachment_size_bytes") or 0,
            original_filename=d.get("file_name", ""),
        )

    return Document(
        id=UUID(d["id"]),
        source_id=d.get("source_id"),
        doc_id=d.get("doc_id"),
        content_version=d.get("content_version", 1),
        ingest_version=d.get("ingest_version", 1),
        document_type=DocumentType(d["document_type"]),
        file_path=d["file_path"],
        file_name=d["file_name"],
        file_hash=d.get("file_hash"),
        parent_id=UUID(d["parent_id"]) if d.get("parent_id") else None,
        root_id=UUID(d["root_id"]) if d.get("root_id") else None,
        depth=d.get("depth", 0),
        path=d.get("path") or [],
        source_title=d.get("source_title"),
        archive_path=d.get("archive_path"),
        archive_browse_uri=d.get("archive_browse_uri"),
        archive_download_uri=d.get("archive_download_uri"),
        email_metadata=email_metadata,
        attachment_metadata=attachment_metadata,
        status=ProcessingStatus(d.get("status", "completed")),
        error_message=d.get("error_message"),
        processed_at=datetime.fromisoformat(d["processed_at"]) if d.get("processed_at") else None,
    )


def _dict_to_chunk(d: Dict[str, Any]) -> "Chunk":
    """Reconstruct a Chunk model from a JSONL dict."""
    from ..models.chunk import Chunk

    return Chunk(
        id=UUID(d["id"]),
        document_id=UUID(d["document_id"]),
        chunk_id=d.get("chunk_id"),
        content=d["content"],
        chunk_index=d.get("chunk_index", 0),
        context_summary=d.get("context_summary"),
        embedding_text=d.get("embedding_text"),
        section_path=d.get("section_path") or [],
        section_title=d.get("section_title"),
        source_title=d.get("source_title"),
        source_id=d.get("source_id"),
        page_number=d.get("page_number"),
        line_from=d.get("line_from"),
        line_to=d.get("line_to"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
        archive_browse_uri=d.get("archive_browse_uri"),
        archive_download_uri=d.get("archive_download_uri"),
        embedding=d.get("embedding"),
        metadata=d.get("metadata") or {},
    )


def _dict_to_topic(d: Dict[str, Any]) -> "Topic":
    """Reconstruct a Topic model from a JSONL dict."""
    from ..models.topic import Topic

    return Topic(
        id=UUID(d["id"]),
        name=d["name"],
        display_name=d.get("display_name", d["name"]),
        description=d.get("description"),
        embedding=d.get("embedding"),
        chunk_count=d.get("chunk_count", 0),
        document_count=d.get("document_count", 0),
    )


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

    docs_path = resolved_output / "documents.jsonl"
    if not docs_path.exists():
        console.print(f"[red]No documents.jsonl found in {resolved_output}[/red]")
        console.print("[dim]Run 'MTSS ingest' first to generate local output.[/dim]")
        raise typer.Exit(1)

    console.print(f"Importing from: {resolved_output}")

    db = SupabaseClient()
    counts = {"vessels": 0, "topics": 0, "documents": 0, "chunks": 0, "skipped": 0, "failed": 0, "archives": 0}

    try:
        if not skip_vessels:
            await _import_vessels(db, load_vessels_from_csv(), counts, dry_run, verbose)

        await _import_topics(db, resolved_output, counts, dry_run, verbose)
        await _import_documents(db, resolved_output, counts, dry_run, verbose)

        if not skip_archives:
            archive_dir = resolved_output / "archive"
            if archive_dir.exists():
                await _import_archives(archive_dir, counts, dry_run, verbose)
            elif verbose:
                console.print("[dim]No archive/ directory found, skipping.[/dim]")

        console.print()
        console.print("[bold]Import Summary[/bold]")
        if dry_run:
            console.print("[yellow](dry run — no changes made)[/yellow]")
        for key, count in counts.items():
            if count > 0:
                color = "red" if key == "failed" else "yellow" if key == "skipped" else "green"
                console.print(f"  [{color}]{key}: {count}[/{color}]")

    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


async def _import_vessels(db, vessels, counts, dry_run, verbose):
    """Import vessels from CSV."""
    if not vessels:
        if verbose:
            console.print("[dim]No vessels to import.[/dim]")
        return

    console.print(f"Importing {len(vessels)} vessels...")
    if dry_run:
        counts["vessels"] = len(vessels)
        return

    with make_progress() as progress:
        task_id = progress.add_task("Vessels", total=len(vessels))
        for vessel in vessels:
            try:
                await db.upsert_vessel(vessel)
                counts["vessels"] += 1
            except Exception as e:
                logger.warning(f"Failed to import vessel {vessel.name}: {e}")
                counts["failed"] += 1
            progress.update(task_id, advance=1)


async def _import_topics(db, output_dir, counts, dry_run, verbose):
    """Import topics from topics.jsonl."""
    topics_data = _read_jsonl(output_dir / "topics.jsonl")
    if not topics_data:
        if verbose:
            console.print("[dim]No topics to import.[/dim]")
        return

    console.print(f"Importing {len(topics_data)} topics...")
    if dry_run:
        counts["topics"] = len(topics_data)
        return

    with make_progress() as progress:
        task_id = progress.add_task("Topics", total=len(topics_data))
        for td in topics_data:
            try:
                existing = await db.get_topic_by_name(td["name"])
                if existing:
                    counts["skipped"] += 1
                else:
                    topic = _dict_to_topic(td)
                    await db.insert_topic(topic)
                    counts["topics"] += 1
            except Exception as e:
                logger.warning(f"Failed to import topic {td.get('name')}: {e}")
                counts["failed"] += 1
            progress.update(task_id, advance=1)


async def _import_documents(db, output_dir, counts, dry_run, verbose):
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

    console.print(f"Importing {len(roots)} emails ({len(docs_data)} documents, {len(chunks_data)} chunks)...")
    if dry_run:
        counts["documents"] = len(docs_data)
        counts["chunks"] = len(chunks_data)
        return

    with make_progress() as progress:
        task_id = progress.add_task("Documents", total=len(roots))
        for root_dict in roots:
            try:
                existing = await db.get_document_by_doc_id(root_dict.get("doc_id", ""))
                if existing:
                    counts["skipped"] += 1
                    progress.update(task_id, advance=1)
                    continue

                email_doc = _dict_to_document(root_dict)
                child_dicts = children_by_root.get(str(root_dict["id"]), [])
                attachment_docs = [_dict_to_document(cd) for cd in child_dicts]

                # Gather chunks for all documents in this email tree
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
                counts["documents"] += 1 + len(attachment_docs)
                counts["chunks"] += len(all_chunks)

                if verbose:
                    console.print(f"  [dim]Imported: {root_dict.get('source_title', root_dict.get('file_name', '?'))}[/dim]")

            except Exception as e:
                logger.warning(f"Failed to import document {root_dict.get('doc_id', '?')}: {e}")
                counts["failed"] += 1

            progress.update(task_id, advance=1)


async def _import_archives(archive_dir: Path, counts, dry_run, verbose):
    """Upload archive files to Supabase Storage."""
    from ..storage.archive_storage import ArchiveStorage

    resolved_archive = archive_dir.resolve()
    files_to_upload: List[tuple[Path, str]] = []
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
        if ".." in rel_key:
            logger.warning(f"Skipping path with traversal: {rel_key}")
            continue

        files_to_upload.append((file_path, rel_key))

    if not files_to_upload:
        if verbose:
            console.print("[dim]No archive files to upload.[/dim]")
        return

    console.print(f"Uploading {len(files_to_upload)} archive files...")
    if dry_run:
        counts["archives"] = len(files_to_upload)
        return

    storage = ArchiveStorage()

    async def _upload_one(local_path: Path, rel_key: str):
        content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        await asyncio.to_thread(
            lambda: storage.upload_file(rel_key, local_path.read_bytes(), content_type)
        )

    with make_progress() as progress:
        task_id = progress.add_task("Archives", total=len(files_to_upload))
        batch_size = 10
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i : i + batch_size]
            results = await asyncio.gather(
                *[_upload_one(lp, rk) for lp, rk in batch],
                return_exceptions=True,
            )
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to upload {batch[j][1]}: {result}")
                    counts["failed"] += 1
                else:
                    counts["archives"] += 1
            progress.update(task_id, advance=len(batch))
