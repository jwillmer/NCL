#!/usr/bin/env python3
"""Fix UTF-8 BOM corruption in email documents.

This script detects and cleans up email documents that were corrupted by
UTF-8 BOM (Byte Order Mark) at the start of EML files. The BOM causes
Python's email parser to misidentify multipart emails as text/plain,
resulting in thousands of garbage chunks containing raw headers and
base64-encoded attachment data.

Detection criteria:
- Document type is 'email'
- Document has more than CHUNK_THRESHOLD chunks (default: 100)

Cleanup actions:
1. Delete all chunks for affected documents
2. Reset document status to 'pending' and ingest_version to 0
3. Documents will be re-processed on next `mtss ingest --resume`

Usage:
    python scripts/fix_bom_corruption.py              # Dry-run mode (report only)
    python scripts/fix_bom_corruption.py --execute    # Actually delete and reset

Requirements:
    - SUPABASE_URL and SUPABASE_KEY environment variables must be set
    - Or .env file in project root with these values
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import NamedTuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table


# Default threshold for detecting BOM corruption
# Normal emails have 10-50 chunks; corrupted ones have 1000+
DEFAULT_CHUNK_THRESHOLD = 100


class AffectedDocument(NamedTuple):
    """Document affected by BOM corruption."""
    id: str
    file_name: str
    file_hash: str
    source_id: str
    chunk_count: int
    status: str


async def get_affected_documents(db, threshold: int) -> list[AffectedDocument]:
    """Find email documents with suspiciously high chunk counts.

    Args:
        db: SupabaseClient instance.
        threshold: Minimum chunk count to consider as corrupted.

    Returns:
        List of affected documents with chunk counts.
    """
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                d.id,
                d.file_name,
                d.file_hash,
                d.source_id,
                d.status,
                COUNT(c.id) as chunk_count
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            WHERE d.document_type = 'email'
            GROUP BY d.id, d.file_name, d.file_hash, d.source_id, d.status
            HAVING COUNT(c.id) > $1
            ORDER BY COUNT(c.id) DESC
        """, threshold)

    return [
        AffectedDocument(
            id=str(row["id"]),
            file_name=row["file_name"],
            file_hash=row["file_hash"],
            source_id=row["source_id"] or "",
            chunk_count=row["chunk_count"],
            status=row["status"],
        )
        for row in rows
    ]


async def delete_chunks_for_documents(db, doc_ids: list[str]) -> int:
    """Delete all chunks for the specified documents.

    Args:
        db: SupabaseClient instance.
        doc_ids: List of document UUIDs.

    Returns:
        Total number of chunks deleted.
    """
    if not doc_ids:
        return 0

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        # Count chunks first
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM chunks
            WHERE document_id = ANY($1::uuid[])
        """, doc_ids)

        # Delete chunks
        await conn.execute("""
            DELETE FROM chunks
            WHERE document_id = ANY($1::uuid[])
        """, doc_ids)

    return count or 0


async def reset_document_status(db, doc_ids: list[str]) -> int:
    """Reset document status to pending for re-processing.

    Args:
        db: SupabaseClient instance.
        doc_ids: List of document UUIDs.

    Returns:
        Number of documents reset.
    """
    if not doc_ids:
        return 0

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE documents
            SET status = 'pending',
                ingest_version = 0,
                updated_at = NOW()
            WHERE id = ANY($1::uuid[])
        """, doc_ids)

    # Parse "UPDATE N" to get count
    return int(result.split()[-1]) if result else 0


async def reset_processing_log(db, file_hashes: list[str]) -> int:
    """Reset processing_log entries to allow re-processing.

    The processing_log table is what the ingest command checks to determine
    which files need processing. We need to either delete or reset these entries.

    Args:
        db: SupabaseClient instance.
        file_hashes: List of file hashes to reset.

    Returns:
        Number of processing_log entries deleted.
    """
    if not file_hashes:
        return 0

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        # Delete the processing_log entries so ingest sees these files as "new"
        result = await conn.execute("""
            DELETE FROM processing_log
            WHERE file_hash = ANY($1::text[])
        """, file_hashes)

    # Parse "DELETE N" to get count
    return int(result.split()[-1]) if result else 0


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect and fix UTF-8 BOM corruption in email documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete chunks and reset documents. Without this flag, only reports affected documents.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_CHUNK_THRESHOLD,
        help=f"Chunk count threshold for detecting corruption (default: {DEFAULT_CHUNK_THRESHOLD})",
    )
    args = parser.parse_args()

    threshold = args.threshold
    console = Console()

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required if env vars are set

    # Initialize database client
    try:
        from mtss.storage.supabase_client import SupabaseClient
        db = SupabaseClient()
    except Exception as e:
        console.print(f"[red]Failed to connect to database: {e}[/red]")
        console.print("[yellow]Make sure SUPABASE_URL and SUPABASE_KEY are set[/yellow]")
        sys.exit(1)

    try:
        # Find affected documents
        console.print(f"\n[bold]Scanning for BOM-corrupted documents (>{threshold} chunks)...[/bold]\n")
        affected = await get_affected_documents(db, threshold)

        if not affected:
            console.print("[green]No corrupted documents found![/green]")
            return

        # Display affected documents
        table = Table(title=f"Affected Documents ({len(affected)} found)")
        table.add_column("File Name", style="cyan")
        table.add_column("Chunks", justify="right", style="red")
        table.add_column("Status", style="yellow")
        table.add_column("Document ID", style="dim")

        total_chunks = 0
        for doc in affected:
            table.add_row(
                doc.file_name[:50] + "..." if len(doc.file_name) > 50 else doc.file_name,
                f"{doc.chunk_count:,}",
                doc.status,
                doc.id[:8] + "...",
            )
            total_chunks += doc.chunk_count

        console.print(table)
        console.print(f"\n[bold]Total:[/bold] {len(affected)} documents, {total_chunks:,} chunks to delete")

        if not args.execute:
            console.print("\n[yellow]Dry-run mode:[/yellow] No changes made.")
            console.print("Run with [bold]--execute[/bold] to delete chunks and reset documents.")
            return

        # Confirm execution
        console.print("\n[bold red]WARNING:[/bold red] This will delete chunks and reset document status!")
        confirm = console.input("[yellow]Type 'yes' to proceed: [/yellow]")
        if confirm.lower() != "yes":
            console.print("[yellow]Aborted.[/yellow]")
            return

        # Execute cleanup
        doc_ids = [doc.id for doc in affected]
        file_hashes = [doc.file_hash for doc in affected]

        console.print("\n[bold]Deleting chunks...[/bold]")
        deleted = await delete_chunks_for_documents(db, doc_ids)
        console.print(f"  Deleted {deleted:,} chunks")

        console.print("[bold]Resetting document status...[/bold]")
        reset = await reset_document_status(db, doc_ids)
        console.print(f"  Reset {reset} documents to 'pending'")

        console.print("[bold]Resetting processing log...[/bold]")
        log_reset = await reset_processing_log(db, file_hashes)
        console.print(f"  Deleted {log_reset} processing_log entries")

        console.print("\n[green]Cleanup complete![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Run [cyan]uv run mtss ingest --source ./data/emails --resume[/cyan]")
        console.print("  2. Documents will be re-parsed with BOM stripping and re-chunked")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
