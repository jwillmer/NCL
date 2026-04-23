"""Shared implementation for vessel retag on Supabase.

Single source of truth used by:
  - `scripts/retag_vessel_ids_from_db.py` (manual / full-corpus passes)
  - `mtss import` (auto-heal after every wave so newly-pushed chunks
    pick up the vessel UUIDs from the current `vessels` table instead
    of the ephemeral uuid4 values minted by `load_vessels_from_csv`
    during ingest)

Contract:
  - Vessels loaded from the DB. CSV-minted UUIDs are NEVER used here
    because they drift per load; the vessels table is canonical.
  - Text sources: documents.source_title + file_name + email_subject
    across the root doc and all descendants. No archive I/O.
  - Writes go through `DomainRepository.update_chunks_vessel_metadata`
    which touches all chunks under a root doc's tree.
  - Idempotent: re-running the retag produces no diff when nothing has
    changed in the source text or the register.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from uuid import UUID

from ..processing.vessel_matcher import VesselMatcher
from .supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


@dataclass
class RetagStats:
    docs_scanned: int = 0
    docs_changed: int = 0
    docs_unchanged: int = 0
    docs_errored: int = 0
    vessel_tags_added: int = 0
    vessel_tags_removed: int = 0
    chunks_updated: int = 0
    per_vessel_deltas: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "docs_scanned": self.docs_scanned,
            "docs_changed": self.docs_changed,
            "docs_unchanged": self.docs_unchanged,
            "docs_errored": self.docs_errored,
            "vessel_tags_added": self.vessel_tags_added,
            "vessel_tags_removed": self.vessel_tags_removed,
            "chunks_updated": self.chunks_updated,
        }


async def _collect_doc_text(pool, root_id: UUID) -> str:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_title, file_name, email_subject
            FROM documents
            WHERE id = $1 OR root_id = $1
            """,
            root_id,
        )
    parts: list[str] = []
    for row in rows:
        for key in ("source_title", "file_name", "email_subject"):
            v = row[key]
            if v:
                parts.append(v)
    return "\n".join(parts)


async def _current_vessel_ids(pool, root_id: UUID) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT jsonb_array_elements_text(metadata->'vessel_ids') AS vid
            FROM chunks c
            WHERE c.document_id IN (
                SELECT id FROM documents WHERE id = $1 OR root_id = $1
            )
            """,
            root_id,
        )
    return {r["vid"] for r in rows if r["vid"]}


async def retag_vessels(
    db: SupabaseClient,
    *,
    root_ids: Optional[Sequence[UUID]] = None,
    concurrency: int = 3,
    dry_run: bool = False,
    progress_every: int = 500,
) -> RetagStats:
    """Retag chunks under the given root docs (or every completed root).

    Args:
        db: Open ``SupabaseClient``.
        root_ids: Scope. ``None`` = every completed root email.
            Pass the wave slice from ``_import_documents`` for post-import
            healing so a full-corpus pass isn't needed on every push.
        concurrency: Parallel doc workers. Supabase pooler session-mode
            caps client count; 3 is safe when paired with another
            concurrent script, go up to 8 solo.
        dry_run: Compute diffs but don't UPDATE.
        progress_every: Emit a log line every N docs processed.

    Returns:
        ``RetagStats`` — also logged at INFO when non-trivial changes land.
    """
    vessels = await db.get_all_vessels()
    if not vessels:
        logger.warning("vessel_retag: vessels table is empty; nothing to retag")
        return RetagStats()
    matcher = VesselMatcher(vessels)

    pool = await db._domain.get_pool()

    if root_ids is None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM documents
                WHERE depth = 0 AND status = 'completed'
                ORDER BY created_at DESC
                """
            )
        roots: list[UUID] = [r["id"] for r in rows]
    else:
        roots = list(root_ids)

    if not roots:
        return RetagStats()

    logger.info(
        "vessel_retag: scanning %d root docs (dry_run=%s, concurrency=%d)",
        len(roots), dry_run, concurrency,
    )

    stats = RetagStats()
    sem = asyncio.Semaphore(concurrency)

    async def process_one(root_id: UUID) -> None:
        async with sem:
            try:
                text = await _collect_doc_text(pool, root_id)
                matched = matcher.find_vessels(text)
                new_ids = sorted(str(v) for v in matched)
                new_types = matcher.get_types_for_ids(matched)
                new_classes = matcher.get_classes_for_ids(matched)

                current = await _current_vessel_ids(pool, root_id)
                added = set(new_ids) - current
                removed = current - set(new_ids)

                if added or removed:
                    stats.docs_changed += 1
                    stats.vessel_tags_added += len(added)
                    stats.vessel_tags_removed += len(removed)
                    for v in added:
                        stats.per_vessel_deltas[f"+{v}"] += 1
                    for v in removed:
                        stats.per_vessel_deltas[f"-{v}"] += 1
                    if not dry_run:
                        n = await db._domain.update_chunks_vessel_metadata(
                            root_id, new_ids, new_types, new_classes
                        )
                        stats.chunks_updated += n
                else:
                    stats.docs_unchanged += 1
                stats.docs_scanned += 1
            except Exception as exc:
                stats.docs_errored += 1
                logger.warning("vessel_retag: root_id=%s failed: %s", root_id, exc)

    # Process in fixed-size batches so progress logs are useful on big runs.
    for start in range(0, len(roots), progress_every):
        batch = roots[start : start + progress_every]
        await asyncio.gather(*(process_one(r) for r in batch))
        logger.info(
            "vessel_retag: %d/%d  changed=%d  +tags=%d  -tags=%d  chunks_updated=%d",
            stats.docs_scanned, len(roots),
            stats.docs_changed, stats.vessel_tags_added,
            stats.vessel_tags_removed, stats.chunks_updated,
        )

    if stats.chunks_updated or stats.vessel_tags_added or stats.vessel_tags_removed:
        logger.info("vessel_retag summary: %s", stats.as_dict())
    return stats
