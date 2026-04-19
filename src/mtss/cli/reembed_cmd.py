"""mtss re-embed command — re-run the embedding pipeline against archived markdown.

Reads existing ``documents.jsonl`` + archived attachment markdown, re-classifies
each document via the embedding decider (or a forced --mode), rebuilds chunks
per the chosen mode, re-embeds, and rewrites ``chunks.jsonl`` for the affected
documents. No re-parse — operates only on what's already in the local archive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

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
    from ..storage.archive_storage import ArchiveStorage

    docs_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"documents.jsonl not found in {output_dir}")

    stats = ReembedStats()

    docs_raw: List[Dict[str, Any]] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs_raw.append(json.loads(line))

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

    storage = ArchiveStorage()
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

            try:
                md_bytes = storage.download_file(archive_uri)
                markdown = md_bytes.decode("utf-8") if isinstance(md_bytes, bytes) else md_bytes
            except Exception as e:
                logger.warning("Cannot load archive for %s: %s", row.get("doc_id"), e)
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
        return stats

    _rewrite_chunks_jsonl(chunks_path, doc_ids_to_replace, new_chunks_serialized)
    _rewrite_documents_jsonl(docs_path, updated_docs_by_id)

    return stats


def _rewrite_chunks_jsonl(
    chunks_path: Path, doc_ids_to_drop: set[str], new_lines: List[str]
) -> None:
    """Drop chunks belonging to re-embedded docs; append new ones. Atomic replace."""
    kept: List[str] = []
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if row.get("document_id") in doc_ids_to_drop:
                    continue
                kept.append(stripped)

    tmp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
        for line in new_lines:
            f.write(line + "\n")
    tmp_path.replace(chunks_path)


def _rewrite_documents_jsonl(
    docs_path: Path, updated_by_id: Dict[str, Dict[str, Any]]
) -> None:
    """Rewrite documents.jsonl applying embedding_mode updates. Atomic replace."""
    rows: List[Dict[str, Any]] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))

    for row in rows:
        rid = str(row.get("id"))
        if rid in updated_by_id:
            row["embedding_mode"] = updated_by_id[rid].get("embedding_mode")

    tmp_path = docs_path.with_suffix(docs_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(docs_path)


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
            help="Directory holding documents.jsonl + chunks.jsonl",
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
