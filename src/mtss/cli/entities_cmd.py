"""Vessels and topics CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID as UUIDType

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from . import _common
from ._common import console, vprint


def register(app: typer.Typer, vessels_app: typer.Typer, topics_app: typer.Typer):
    """Register vessel and topic commands on their sub-apps."""

    # ==================== Vessel Commands ====================

    @vessels_app.command("import")
    def vessels_import(
        csv_file: Optional[Path] = typer.Argument(
            None,
            help="Path to vessel CSV file (semicolon-delimited)",
        ),
        clear: bool = typer.Option(
            False,
            "--clear",
            "-c",
            help="Clear existing vessels before import",
        ),
    ):
        """Import vessel register from CSV file.

        Default file: data/vessel-list.csv

        CSV format (semicolon-delimited, 3 required columns):
            NAME;TYPE;CLASS
            MARAN THALEIA;VLCC;Canopus Class
        """
        asyncio.run(_vessels_import(csv_file, clear))

    @vessels_app.command("retag")
    def vessels_retag(
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            "-n",
            help="Preview changes without updating the database",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Show detailed output per document",
        ),
        limit: int = typer.Option(
            None,
            "--limit",
            "-l",
            help="Limit number of documents to process",
        ),
    ):
        """Re-tag existing chunks with vessel IDs from archive content.

        Scans the markdown files in the archive bucket, matches vessel names
        using the current vessel registry, and updates chunk metadata.

        Use this after importing new vessels to tag existing documents
        without re-ingesting from scratch.

        Examples:
            MTSS vessels retag              # Re-tag all documents
            MTSS vessels retag --dry-run    # Preview changes
            MTSS vessels retag --limit 100  # Process first 100 documents
        """
        _common._verbose = verbose
        asyncio.run(_vessels_retag(dry_run, limit))

    @vessels_app.command("list")
    def vessels_list():
        """List all vessels in the registry."""
        asyncio.run(_vessels_list())

    # ==================== Topics Commands ====================

    @topics_app.command("list")
    def topics_list(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory containing ingest.db (default: data/output)",
        ),
        sort: str = typer.Option(
            "chunks",
            "--sort",
            "-s",
            help="Sort key: chunks | docs | name (default: chunks).",
        ),
        filter_substring: Optional[str] = typer.Option(
            None,
            "--filter",
            "-f",
            help="Case-insensitive substring match against name + display_name.",
        ),
        min_chunks: int = typer.Option(
            0,
            "--min-chunks",
            help="Only show topics with at least this many chunks.",
        ),
        limit: int = typer.Option(
            50,
            "--limit",
            "-l",
            help="Max rows to show (0 = show all). Default 50.",
        ),
        json_out: Optional[Path] = typer.Option(
            None,
            "--json",
            help="Write the full filtered list to this JSON file.",
        ),
    ):
        """List topics from local ingest.db with sort/filter/paginate.

        Reads `data/output/ingest.db` (WAL mode — safe during active ingests).
        """
        _topics_list_local(output_dir, sort, filter_substring, min_chunks, limit, json_out)

    @topics_app.command("consolidate")
    def topics_consolidate(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory containing ingest.db (default: data/output)",
        ),
        strategy: str = typer.Option(
            "pairwise",
            "--strategy",
            help="Merge strategy: pairwise | cluster | name. Default pairwise (greedy single-pair merges). `cluster` does single-linkage transitive merges. `name` groups topics whose normalized name is identical (case/punct-insensitive).",
        ),
        threshold: float = typer.Option(
            0.80,
            "--threshold",
            "-t",
            help="Cosine similarity threshold for pairwise/cluster. Ignored by --strategy name.",
        ),
        apply: bool = typer.Option(
            False,
            "--apply",
            help="Actually perform the merge. Without this, the command is dry-run only.",
        ),
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip the interactive confirmation on --apply.",
        ),
        limit: int = typer.Option(
            50,
            "--limit",
            "-l",
            help="Max merges to show in the preview table (0 = show all).",
        ),
        json_out: Optional[Path] = typer.Option(
            None,
            "--json",
            help="Write the full merge plan to this JSON file.",
        ),
    ):
        """Consolidate near-duplicate topics.

        Dry-run by default. Prints the greedy merge plan — which topic
        absorbs which — and exits without touching ingest.db. Re-run with
        ``--apply`` to actually perform the merge; you'll be prompted once
        unless ``--yes`` is set.

        Strategies:
          pairwise — merges every (A, B) pair with cosine >= threshold (greedy).
          cluster  — single-linkage: A-B-C chains collapse even if A-C < threshold.
          name     — merges topics whose normalized name is identical (case/punct).

        WARNING: a local merge leaves orphan topic UUIDs inside the remote
        ``chunks.metadata.topic_ids`` JSONB array in Supabase. Plan a
        re-import or topic-rewrite RPC before pushing.
        """
        _topics_consolidate(output_dir, strategy, threshold, apply, yes, limit, json_out)

    @topics_app.command("audit")
    def topics_audit(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory containing ingest.db (default: data/output)",
        ),
        lower: float = typer.Option(
            0.75,
            "--lower",
            help="Lower cosine threshold (inclusive). Default 0.75.",
        ),
        upper: float = typer.Option(
            0.85,
            "--upper",
            help="Upper cosine threshold (exclusive). Default 0.85 — the ingest dedup line.",
        ),
        limit: int = typer.Option(
            50,
            "--limit",
            "-l",
            help="Max pairs to show in the table (0 = show all).",
        ),
        json_out: Optional[Path] = typer.Option(
            None,
            "--json",
            help="Write the full pair list to this JSON file.",
        ),
    ):
        """Audit topic dedup: flag near-duplicate pairs in the [lower, upper) band.

        Read-only against local ingest.db. Pairs above `upper` were already
        merged; pairs below `lower` are considered distinct. The band in
        between is the grey zone worth a human look before lowering the
        merge threshold and running a consolidation pass.
        """
        _topics_audit(output_dir, lower, upper, limit, json_out)


async def _vessels_import(csv_file: Optional[Path], clear: bool):
    """Async implementation of vessels import command."""
    from ..models.vessel import load_vessels_from_csv
    from ..processing.entity_cache import get_vessel_cache
    from ..storage.supabase_client import SupabaseClient

    vessels_to_import = load_vessels_from_csv(csv_file)
    if not vessels_to_import:
        console.print("[yellow]No vessels found in CSV file[/yellow]")
        return

    db = SupabaseClient()

    try:
        # Clear existing vessels if requested
        if clear:
            deleted = await db.delete_all_vessels()
            console.print(f"[yellow]Cleared {deleted} existing vessels[/yellow]")

        # Import vessels with progress
        imported_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Importing vessels...", total=len(vessels_to_import))

            for vessel in vessels_to_import:
                try:
                    await db.upsert_vessel(vessel)
                    imported_count += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to import {vessel.name}: {e}[/yellow]")
                progress.update(task, advance=1)

        console.print(f"[green]Imported {imported_count} vessels[/green]")

        # Invalidate the in-process VesselCache so the next cache consumer
        # (e.g. an agent tool call or the REST /api/vessels endpoint) sees
        # the new rows instead of serving stale data up to the 5-min TTL.
        get_vessel_cache().invalidate()

    finally:
        await db.close()


async def _vessels_retag(dry_run: bool, limit: int | None):
    """Async implementation of vessels retag command."""
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.archive_storage import ArchiveStorage, ArchiveStorageError
    from ..storage.supabase_client import SupabaseClient

    db = SupabaseClient()

    try:
        # Load vessel registry
        vessels = await db.get_all_vessels()
        if not vessels:
            console.print("[yellow]No vessels in registry - nothing to tag[/yellow]")
            return

        vessel_matcher = VesselMatcher(vessels)
        console.print(f"Loaded {vessel_matcher.vessel_count} vessels ({vessel_matcher.name_count} names)")

        # Initialize archive storage
        try:
            storage = ArchiveStorage()
        except ArchiveStorageError as e:
            console.print(f"[red]Failed to initialize archive storage: {e}[/red]")
            return

        # Get documents to process
        documents = await db.get_root_documents_for_retagging(limit)
        if not documents:
            console.print("[yellow]No documents found to retag[/yellow]")
            return

        console.print(f"Found {len(documents)} documents to scan")
        if dry_run:
            console.print("[dim]Dry run mode - no changes will be made[/dim]")

        # Track statistics
        docs_scanned = 0
        docs_updated = 0
        chunks_updated = 0
        vessels_added = 0
        vessels_removed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning documents...", total=len(documents))

            for doc in documents:
                doc_id = doc.get("doc_id")
                document_uuid = UUIDType(doc["id"])
                file_name = doc.get("file_name", "unknown")

                if not doc_id:
                    progress.update(task, advance=1)
                    continue

                # Download markdown content from archive
                content_parts: list[str] = []

                # Get email markdown
                try:
                    email_md_path = f"{doc_id}/email.eml.md"
                    email_content = storage.download_file(email_md_path)
                    content_parts.append(email_content.decode("utf-8", errors="ignore"))
                except ArchiveStorageError:
                    vprint(f"No email.md found for {doc_id}", file_name)

                # Get attachment markdowns
                try:
                    files = storage.list_files(doc_id)
                    for f in files:
                        fname = f.get("name", "")
                        if fname.endswith(".md") and fname != "email.eml.md":
                            try:
                                # Files in attachments subfolder
                                att_path = f"{doc_id}/attachments/{fname}"
                                att_content = storage.download_file(att_path)
                                content_parts.append(att_content.decode("utf-8", errors="ignore"))
                            except ArchiveStorageError:
                                pass
                except ArchiveStorageError:
                    pass

                if not content_parts:
                    vprint(f"No content found for {doc_id}", file_name)
                    progress.update(task, advance=1)
                    continue

                # Match vessels in combined content
                combined_content = "\n\n".join(content_parts)
                matched_vessels = vessel_matcher.find_vessels(combined_content)
                new_vessel_ids = sorted([str(v) for v in matched_vessels])
                new_vessel_types = vessel_matcher.get_types_for_ids(matched_vessels)
                new_vessel_classes = vessel_matcher.get_classes_for_ids(matched_vessels)

                # Get current vessel IDs
                current_vessel_ids = await db.get_current_vessel_ids(document_uuid)

                # Check if update is needed (always update to add types/classes for backfill)
                if set(new_vessel_ids) != set(current_vessel_ids) or new_vessel_types or new_vessel_classes:
                    added = set(new_vessel_ids) - set(current_vessel_ids)
                    removed = set(current_vessel_ids) - set(new_vessel_ids)

                    if added or removed or new_vessel_types or new_vessel_classes:
                        vprint(
                            f"{file_name}: +{len(added)} -{len(removed)} vessels, "
                            f"{len(new_vessel_types)} types, {len(new_vessel_classes)} classes",
                            doc_id[:8],
                        )

                        if not dry_run:
                            updated = await db.update_chunks_vessel_metadata(
                                document_uuid, new_vessel_ids, new_vessel_types, new_vessel_classes
                            )
                            chunks_updated += updated

                        docs_updated += 1
                        vessels_added += len(added)
                        vessels_removed += len(removed)

                docs_scanned += 1
                progress.update(task, advance=1)

        # Show summary
        console.print()
        table = Table(title="Vessel Retagging Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Documents scanned", str(docs_scanned))
        table.add_row("Documents updated", str(docs_updated))
        if not dry_run:
            table.add_row("Chunks updated", str(chunks_updated))
        table.add_row("Vessel tags added", str(vessels_added))
        table.add_row("Vessel tags removed", str(vessels_removed))

        console.print(table)

        if dry_run and docs_updated > 0:
            console.print("\n[dim]Run without --dry-run to apply changes[/dim]")

    finally:
        await db.close()


async def _vessels_list():
    """Async implementation of vessels list command."""
    from ..storage.supabase_client import SupabaseClient

    db = SupabaseClient()

    try:
        vessels = await db.get_all_vessels()

        if not vessels:
            console.print("[dim]No vessels in registry[/dim]")
            return

        table = Table(title=f"Vessel Registry ({len(vessels)} vessels)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Class", style="green")
        table.add_column("Aliases", style="dim")

        for vessel in vessels:
            aliases_str = ", ".join(vessel.aliases) if vessel.aliases else "-"
            table.add_row(
                vessel.name,
                vessel.vessel_type or "-",
                vessel.vessel_class or "-",
                aliases_str,
            )

        console.print(table)

    finally:
        await db.close()


def _topics_list_local(
    output_dir: Optional[Path],
    sort: str,
    filter_substring: Optional[str],
    min_chunks: int,
    limit: int,
    json_out: Optional[Path],
) -> None:
    """Local-SQLite implementation of `mtss topics list`."""
    import json as _json

    from ..storage.sqlite_client import SqliteStorageClient

    sort_columns = {
        "chunks": "chunk_count DESC, document_count DESC, name ASC",
        "docs": "document_count DESC, chunk_count DESC, name ASC",
        "name": "name ASC",
    }
    if sort not in sort_columns:
        console.print(f"[red]--sort must be one of: {', '.join(sort_columns)}[/red]")
        raise typer.Exit(2)

    resolved_output = output_dir or Path("data/output")
    db_path = resolved_output / "ingest.db"
    if not db_path.exists():
        console.print(f"[red]ingest.db not found in {resolved_output}[/red]")
        raise typer.Exit(1)

    try:
        db = SqliteStorageClient(output_dir=resolved_output)
    except Exception as exc:
        console.print(f"[red]Failed to open ingest.db: {exc}[/red]")
        raise typer.Exit(1)

    query_parts = [
        "SELECT id, name, display_name, chunk_count, document_count",
        "FROM topics",
    ]
    where = ["chunk_count >= ?"]
    params: list[object] = [min_chunks]
    if filter_substring:
        where.append("(LOWER(name) LIKE ? OR LOWER(COALESCE(display_name, '')) LIKE ?)")
        needle = f"%{filter_substring.lower()}%"
        params.extend([needle, needle])
    query_parts.append("WHERE " + " AND ".join(where))
    query_parts.append(f"ORDER BY {sort_columns[sort]}")
    rows = list(db._conn.execute(" ".join(query_parts), params))
    total = db._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]

    console.print(
        f"[cyan]Topics:[/cyan] {total} total, [bold]{len(rows)}[/bold] match filter "
        f"(min_chunks={min_chunks}"
        + (f", filter={filter_substring!r}" if filter_substring else "")
        + f", sort={sort})"
    )

    if not rows:
        console.print("[yellow]No topics match the filter.[/yellow]")
        if json_out:
            json_out.write_text(_json.dumps([], indent=2), encoding="utf-8")
        return

    display = rows if limit == 0 else rows[:limit]
    table = Table(title=f"Topics ({len(display)} shown of {len(rows)})")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Docs", justify="right", style="green")
    table.add_column("Name", style="cyan")
    for r in display:
        table.add_row(
            str(r["chunk_count"] or 0),
            str(r["document_count"] or 0),
            r["display_name"] or r["name"],
        )
    console.print(table)

    if limit != 0 and len(rows) > limit:
        console.print(
            f"[dim]{len(rows) - limit} more row(s) hidden. Use --limit 0 to show all, "
            f"or --filter to narrow.[/dim]"
        )

    if json_out:
        payload = [
            {
                "id": r["id"],
                "name": r["name"],
                "display_name": r["display_name"] or r["name"],
                "chunk_count": r["chunk_count"] or 0,
                "document_count": r["document_count"] or 0,
            }
            for r in rows
        ]
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]Full list written to {json_out}[/dim]")


def _topics_consolidate(
    output_dir: Optional[Path],
    strategy: str,
    threshold: float,
    apply: bool,
    yes: bool,
    limit: int,
    json_out: Optional[Path],
) -> None:
    """Local implementation of `mtss topics consolidate`."""
    import json as _json

    from ..storage.sqlite_client import SqliteStorageClient

    valid_strategies = {"pairwise", "cluster", "name"}
    if strategy not in valid_strategies:
        console.print(
            f"[red]--strategy must be one of: {', '.join(sorted(valid_strategies))}[/red]"
        )
        raise typer.Exit(2)
    if strategy != "name" and not 0 < threshold < 1:
        console.print("[red]--threshold must be in (0, 1)[/red]")
        raise typer.Exit(2)

    resolved_output = output_dir or Path("data/output")
    db_path = resolved_output / "ingest.db"
    if not db_path.exists():
        console.print(f"[red]ingest.db not found in {resolved_output}[/red]")
        raise typer.Exit(1)

    try:
        db = SqliteStorageClient(output_dir=resolved_output)
    except Exception as exc:
        console.print(f"[red]Failed to open ingest.db: {exc}[/red]")
        raise typer.Exit(1)

    before_total = db._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    with_embeddings = db._conn.execute(
        "SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    if strategy == "pairwise":
        plan = db.compute_merge_plan(threshold)
        header = f"strategy=pairwise threshold={threshold:.2f}"
    elif strategy == "cluster":
        plan = db.compute_cluster_merge_plan(threshold)
        header = f"strategy=cluster threshold={threshold:.2f}"
    else:  # name
        plan = db.compute_name_merge_plan()
        header = "strategy=name (normalized-name buckets)"

    mode_label = "[bold yellow]DRY-RUN[/bold yellow]" if not apply else "[bold red]APPLY[/bold red]"
    console.print(
        f"{mode_label} {header} · "
        f"{before_total} topics ({with_embeddings} with embeddings) → "
        f"plan: [bold]{len(plan)}[/bold] merge(s)"
    )
    if strategy != "name" and before_total != with_embeddings:
        missing = before_total - with_embeddings
        console.print(
            f"[yellow]Warning:[/yellow] {missing} topic(s) lack embeddings — excluded from consolidation"
        )

    if not plan:
        console.print("[green]No merges at this threshold. Nothing to do.[/green]")
        if json_out:
            json_out.write_text(_json.dumps([], indent=2), encoding="utf-8")
        return

    display = plan if limit == 0 else plan[:limit]
    table = Table(title=f"Merge plan ({len(display)} shown of {len(plan)})")
    table.add_column("Sim", justify="right", style="magenta")
    table.add_column("Keeper", style="cyan")
    table.add_column("K·chunks", justify="right", style="green")
    table.add_column("Absorbed →", style="cyan")
    table.add_column("A·chunks", justify="right", style="green")
    for p in display:
        table.add_row(
            f"{p['similarity']:.3f}",
            p["keeper_display_name"],
            str(p["keeper_chunks"]),
            p["absorbed_display_name"],
            str(p["absorbed_chunks"]),
        )
    console.print(table)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(plan, indent=2), encoding="utf-8")
        console.print(f"[dim]Full plan written to {json_out}[/dim]")

    if not apply:
        console.print(
            "[dim]Dry-run only. Re-run with --apply to execute these merges.[/dim]"
        )
        return

    console.print(
        "[yellow]About to mutate ingest.db:[/yellow] "
        f"{len(plan)} topic row(s) will be deleted, chunk_topics remapped, counts transferred."
    )
    console.print(
        "[yellow]Remote Supabase footgun:[/yellow] any already-imported chunks will retain the absorbed "
        "topic UUIDs inside metadata.topic_ids until those documents are re-imported."
    )

    if not yes:
        if not typer.confirm("Proceed with merge?", default=False):
            console.print("[dim]Aborted. No changes written.[/dim]")
            raise typer.Exit(0)

    merges = db.apply_merge_plan(plan)
    after_total = db._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    console.print(
        f"[green]Merged {len(merges)} topic pair(s).[/green] "
        f"Topics {before_total} → {after_total} (-{before_total - after_total})."
    )


def _topics_audit(
    output_dir: Optional[Path],
    lower: float,
    upper: float,
    limit: int,
    json_out: Optional[Path],
) -> None:
    """Local (read-only) implementation of `mtss topics audit`."""
    import json as _json

    from ..storage.sqlite_client import SqliteStorageClient

    resolved_output = output_dir or Path("data/output")
    db_path = resolved_output / "ingest.db"
    if not db_path.exists():
        console.print(f"[red]ingest.db not found in {resolved_output}[/red]")
        raise typer.Exit(1)

    try:
        db = SqliteStorageClient(output_dir=resolved_output)
    except Exception as exc:
        console.print(f"[red]Failed to open ingest.db: {exc}[/red]")
        raise typer.Exit(1)

    total_topics = db._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    with_embeddings = db._conn.execute(
        "SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    try:
        pairs = db.audit_similar_topics(lower=lower, upper=upper)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    console.print(
        f"[cyan]Topics:[/cyan] {total_topics} total, "
        f"{with_embeddings} with embeddings; "
        f"band [{lower:.2f}, {upper:.2f}) → [bold]{len(pairs)}[/bold] pair(s)"
    )
    if total_topics != with_embeddings:
        missing = total_topics - with_embeddings
        console.print(
            f"[yellow]Warning:[/yellow] {missing} topic(s) lack embeddings — excluded from audit"
        )

    if not pairs:
        console.print("[green]No near-duplicate pairs in the audit band.[/green]")
        if json_out:
            json_out.write_text(_json.dumps([], indent=2), encoding="utf-8")
        return

    display = pairs if limit == 0 else pairs[:limit]
    table = Table(title=f"Near-duplicate topic pairs ({len(display)} shown of {len(pairs)})")
    table.add_column("Sim", justify="right", style="magenta")
    table.add_column("Keeper", style="cyan")
    table.add_column("K·chunks", justify="right", style="green")
    table.add_column("Absorbed", style="cyan")
    table.add_column("A·chunks", justify="right", style="green")
    for p in display:
        table.add_row(
            f"{p['similarity']:.3f}",
            p["keeper_display_name"],
            str(p["keeper_chunks"]),
            p["absorbed_display_name"],
            str(p["absorbed_chunks"]),
        )
    console.print(table)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(pairs, indent=2), encoding="utf-8")
        console.print(f"[dim]Full report written to {json_out}[/dim]")
