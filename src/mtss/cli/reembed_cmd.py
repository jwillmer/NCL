"""mtss re-embed command — re-run the embedding pipeline against archived markdown.

Reads documents + archived attachment markdown from ``ingest.db``, re-classifies
each document via the embedding decider (or a forced --mode), rebuilds chunks
per the chosen mode, re-embeds, and rewrites the ``chunks`` table for the
affected documents. No re-parse — operates only on what's already in the local
archive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from ..config import get_settings
from ..models.document import EmbeddingMode
from ..models.serializers import chunk_to_dict, dict_to_document
from ..utils import strip_archive_prefix

logger = logging.getLogger(__name__)


@dataclass
class ReembedStats:
    docs_considered: int = 0
    docs_committed: int = 0
    docs_skipped_idempotent: int = 0
    docs_skipped_no_archive: int = 0
    docs_failed: int = 0
    total_chunks_written: int = 0
    mode_distribution: Dict[str, int] = field(default_factory=dict)


def _extract_content_from_md(md: str) -> str:
    """Return the ``## Content`` section body, or the full markdown if the
    sentinel is absent (re-embed runs against user-visible markdown either way)."""
    marker = "## Content\n"
    idx = md.find(marker)
    if idx == -1:
        return md
    return md[idx + len(marker):].strip()


async def reembed_run(
    output_dir: Path,
    *,
    doc_id: Optional[str] = None,
    mode: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
    verbose: bool = False,
    include_images: bool = False,
) -> ReembedStats:
    """Core implementation. CLI wrapper calls this via asyncio.run."""
    from ..ingest.embedding_decider import decide_embedding_mode
    from ..parsers.chunker import ContextGenerator, DocumentChunker, build_chunks_for_mode
    from ..parsers.llamaparse_parser import strip_llamaparse_image_refs
    from ..processing.embeddings import EmbeddingGenerator
    from ..storage.sqlite_client import SqliteStorageClient

    db_path = output_dir / "ingest.db"
    archive_root = output_dir / "archive"
    if not db_path.exists():
        raise FileNotFoundError(f"ingest.db not found in {output_dir}")

    stats = ReembedStats()
    db = SqliteStorageClient(output_dir=output_dir)

    # Flatten docs into raw dicts matching the legacy JSONL shape so the
    # processing loop below is unchanged — iter_documents already surfaces
    # metadata fields under the top-level key set.
    docs_raw: List[Dict[str, Any]] = []
    for row in db.iter_documents():
        meta = row.get("metadata_json")
        if meta:
            try:
                parsed = json.loads(meta)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        row.setdefault(k, v)
            except (TypeError, ValueError):
                pass
        row.pop("metadata_json", None)
        docs_raw.append(row)

    if doc_id:
        docs_raw = [d for d in docs_raw if d.get("doc_id") == doc_id or d.get("id") == doc_id]

    # Image attachments are stored as their vision-API description text. Re-embedding
    # them just re-runs the embedding model on text that hasn't changed since the
    # original ingest — wasted spend and noise. Skip by default.
    if not include_images:
        docs_raw = [d for d in docs_raw if d.get("document_type") != "attachment_image"]

    candidates = [d for d in docs_raw if d.get("archive_browse_uri")]
    stats.docs_skipped_no_archive = len(docs_raw) - len(candidates)

    # Apply limit before fan-out so dry-run previews and live runs both honor
    # the same cap. Idempotent skips count toward stats.docs_considered but
    # not toward the limit (they're free).
    if limit is not None:
        candidates = candidates[:limit]
    stats.docs_considered = len(candidates)

    if not candidates:
        return stats

    chunker = DocumentChunker()
    context_generator = ContextGenerator()
    embed_gen = EmbeddingGenerator()

    forced_mode: Optional[EmbeddingMode] = None
    if mode and mode != "auto":
        forced_mode = EmbeddingMode(mode)

    doc_ids_to_replace: set[str] = set()
    new_chunks_serialized: List[str] = []
    updated_docs_by_id: Dict[str, Dict[str, Any]] = {}

    sem = asyncio.Semaphore(max(1, get_settings().max_concurrent_files))

    async def _process_one(row: Dict[str, Any]) -> None:
        async with sem:
            current_mode = row.get("embedding_mode")
            archive_uri = strip_archive_prefix(row["archive_browse_uri"])

            local_path = archive_root / archive_uri
            try:
                markdown = local_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.warning(
                    "Archive file missing for %s: %s", row.get("doc_id"), local_path
                )
                stats.docs_failed += 1
                return
            except OSError as e:
                logger.warning("Cannot read archive for %s: %s", row.get("doc_id"), e)
                stats.docs_failed += 1
                return

            # Strip dead LlamaParse image refs in-memory; older archives still
            # carry them and the embedder shouldn't see noise.
            markdown = strip_llamaparse_image_refs(_extract_content_from_md(markdown))
            if not markdown.strip():
                stats.docs_failed += 1
                return

            doc_obj = dict_to_document(row)

            if forced_mode is not None:
                decided_mode = forced_mode
                reason = f"forced:{forced_mode.value}"
            else:
                decision = await decide_embedding_mode(markdown, doc_obj)
                decided_mode = decision.mode
                reason = decision.reason

            stats.mode_distribution[decided_mode.value] = (
                stats.mode_distribution.get(decided_mode.value, 0) + 1
            )

            if verbose or dry_run:
                label = row.get("source_title") or row.get("file_name") or row.get("doc_id") or row.get("id")
                typer.echo(
                    f"  {label}: {current_mode or '(none)'} -> {decided_mode.value}"
                    f" ({reason})"
                )

            if not force and current_mode == decided_mode.value:
                stats.docs_skipped_idempotent += 1
                return
            if dry_run:
                return

            try:
                chunks = await build_chunks_for_mode(
                    mode=decided_mode,
                    document=doc_obj,
                    markdown=markdown,
                    chunker=chunker,
                    context_generator=context_generator,
                    source_file=archive_uri,
                )
            except Exception as e:
                logger.warning("build_chunks_for_mode failed for %s: %s", row.get("doc_id"), e)
                stats.docs_failed += 1
                return

            if not chunks:
                stats.docs_failed += 1
                return

            try:
                chunks = await embed_gen.embed_chunks(chunks)
            except Exception as e:
                logger.warning("embed failed for %s: %s", row.get("doc_id"), e)
                stats.docs_failed += 1
                return

            doc_ids_to_replace.add(str(doc_obj.id))
            for ch in chunks:
                new_chunks_serialized.append(
                    json.dumps(chunk_to_dict(ch), ensure_ascii=False)
                )
            row["embedding_mode"] = decided_mode.value
            updated_docs_by_id[str(doc_obj.id)] = row

            stats.docs_committed += 1
            stats.total_chunks_written += len(chunks)

    await asyncio.gather(*(_process_one(row) for row in candidates))

    if dry_run or not doc_ids_to_replace:
        try:
            await db.close()
        except Exception:  # noqa: BLE001
            pass
        return stats

    await _apply_reembed_writes(db, doc_ids_to_replace, new_chunks_serialized, updated_docs_by_id)
    try:
        await db.close()
    except Exception:  # noqa: BLE001
        pass
    return stats


async def _apply_reembed_writes(
    db,
    doc_ids_to_replace: set[str],
    new_chunks_serialized: List[str],
    updated_docs_by_id: Dict[str, Dict[str, Any]],
) -> None:
    """Apply re-embed results to SQLite atomically.

    1. DELETE chunks WHERE document_id IN (…) — FK CASCADE clears chunk_topics.
    2. INSERT new chunks via ``insert_chunks`` (roundtrips through the normal
       ``Chunk`` model so metadata stamping and topic sync match the ingest path).
    3. UPDATE documents SET embedding_mode = ? for each re-embedded doc.
    """
    from ..models.serializers import dict_to_chunk

    new_chunks = [dict_to_chunk(json.loads(line)) for line in new_chunks_serialized]

    conn = db._conn
    with conn:
        conn.execute("BEGIN")
        if doc_ids_to_replace:
            placeholders = ",".join(["?"] * len(doc_ids_to_replace))
            conn.execute(
                f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
                tuple(doc_ids_to_replace),
            )
        for did, row in updated_docs_by_id.items():
            conn.execute(
                "UPDATE documents SET embedding_mode = ?, updated_at = datetime('now') WHERE id = ?",
                (row.get("embedding_mode"), did),
            )
    # Insert rebuilt chunks after the DELETE commits so any chunk_id collisions
    # resolve cleanly via INSERT OR REPLACE inside insert_chunks.
    if new_chunks:
        await db.insert_chunks(new_chunks)


def register(app: "typer.Typer") -> None:
    """Register the re-embed command on the given Typer app."""

    @app.command("re-embed")
    def re_embed(
        doc_id: Optional[str] = typer.Option(
            None, "--doc-id", help="Process a single doc by doc_id or UUID"
        ),
        all_docs: bool = typer.Option(
            False, "--all", help="Process every doc with archive_browse_uri"
        ),
        mode: str = typer.Option(
            "auto",
            "--mode",
            help="auto (decider) | full | summary | metadata_only",
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Print decisions, write nothing"
        ),
        limit: Optional[int] = typer.Option(
            None, "--limit", help="Process at most N docs"
        ),
        force: bool = typer.Option(
            False, "--force", help="Re-process even when mode already matches"
        ),
        verbose: bool = typer.Option(
            False, "--verbose", "-v", help="Verbose per-doc output"
        ),
        include_images: bool = typer.Option(
            False,
            "--include-images",
            help="Also re-embed image attachments (skipped by default — their "
                 "vision-API descriptions don't change between runs)",
        ),
        output_dir: Path = typer.Option(
            Path("./data/output"),
            "--output-dir",
            help="Directory holding ingest.db (and the archive/ tree)",
        ),
    ) -> None:
        """Re-embed archived documents against the embedding decider."""
        if not doc_id and not all_docs:
            typer.echo("Either --doc-id or --all must be provided.")
            raise typer.Exit(code=2)
        if mode not in {"auto", "full", "summary", "metadata_only"}:
            typer.echo(f"Invalid --mode: {mode}")
            raise typer.Exit(code=2)

        stats = asyncio.run(
            reembed_run(
                output_dir=output_dir,
                doc_id=doc_id,
                mode=mode,
                dry_run=dry_run,
                limit=limit,
                force=force,
                verbose=verbose,
                include_images=include_images,
            )
        )

        typer.echo("")
        typer.echo(f"Docs considered:      {stats.docs_considered}")
        typer.echo(f"Docs committed:       {stats.docs_committed}")
        typer.echo(f"Skipped (idempotent): {stats.docs_skipped_idempotent}")
        typer.echo(f"Skipped (no archive): {stats.docs_skipped_no_archive}")
        typer.echo(f"Failed:               {stats.docs_failed}")
        typer.echo(f"Chunks written:       {stats.total_chunks_written}")
        if stats.mode_distribution:
            dist = ", ".join(f"{k}={v}" for k, v in sorted(stats.mode_distribution.items()))
            typer.echo(f"Mode distribution:    {dist}")
