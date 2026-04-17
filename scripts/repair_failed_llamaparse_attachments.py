"""Repair attachment docs that failed LlamaParse with the cost_optimizer SDK bug.

Why this exists
---------------
Commit d5a7fd6 fixed a signature bug where
``AttachmentProcessor``/``LlamaParseParser`` passed ``cost_optimizer`` at the
top level of ``client.parsing.parse(...)``. The llama-cloud 2.x SDK only
accepts that option inside ``processing_options``, so every call raised
``TypeError: AsyncParsingResource.parse() got an unexpected keyword argument
'cost_optimizer'``. The exception was caught by ``process_attachment`` and
emitted an ``extraction_failed`` event, leaving 964 attachment docs in
``status=failed`` with 0 chunks across the 2026-04-17 1000-email run.

The fix is live. This script repairs the already-ingested records so we don't
pay a full re-parse for emails whose body/vision/topics already completed
correctly. It is targeted to docs whose archived file is still on disk at
``archive/<root_doc_id_16>/attachments/<sanitized_filename>``; ZIP-member
docs whose ``_extracted`` temp path is gone are handled by
``mtss mark-failed`` + ``mtss ingest --retry-failed`` instead (much smaller
set — 9 emails).

What it does per candidate
--------------------------
1. Re-parse the archived attachment file via ``LlamaParseParser`` (now
   correctly passing ``cost_optimizer`` inside ``processing_options``).
2. Chunk the markdown via ``DocumentChunker.chunk_text``.
3. Enrich chunk citation metadata via
   ``enrich_chunks_with_document_metadata``.
4. Generate the document's ``context_summary`` via
   ``ContextGenerator.generate_context`` and populate each chunk's
   ``context_summary`` + ``embedding_text``.
5. Embed chunks via ``EmbeddingGenerator.embed_chunks``.
6. Append chunks to ``chunks.jsonl`` (atomic per-line append).
7. Regenerate the archived ``.md`` preview via
   ``ArchiveGenerator.update_attachment_markdown``.
8. Record the updated doc fields (status/error_message/archive URIs/
   updated_at) for the final atomic rewrite of ``documents.jsonl``.

Idempotency
-----------
Candidates whose document_id already has chunks in ``chunks.jsonl`` are
skipped (the earlier half of a partial run would have reached the append
step only after embedding completed). A failed run can therefore safely be
re-executed.

Usage
-----
    uv run python scripts/repair_failed_llamaparse_attachments.py \\
        --output-dir data/output --dry-run
    uv run python scripts/repair_failed_llamaparse_attachments.py \\
        --output-dir data/output [--concurrency 3] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import UUID

logger = logging.getLogger("repair_llp")

# Skip reasons (exported for tests)
SKIP_ZIP_MEMBER = "zip_member_ephemeral_path"
SKIP_MISSING_FILE = "archive_file_missing"
SKIP_NO_ROOT = "no_root_doc_id"
SKIP_ALREADY_HAS_CHUNKS = "already_has_chunks"
SKIP_OVER_PAGE_LIMIT = "over_page_limit"
SKIP_UNDER_PAGE_LIMIT = "under_page_limit"
SKIP_CRDOWNLOAD = "chrome_partial_download"


def count_pdf_pages(path: Path) -> int | None:
    """Return PDF page count via PyMuPDF, or None if it can't be determined
    (non-PDF, corrupt file, etc.).

    Used to pre-filter expensive-to-parse long PDFs. Returning None on
    failure makes the caller policy-neutral: the candidate still goes into
    the pool with a real parse, where its true failure mode (or success)
    surfaces with a useful error message.
    """
    if path.suffix.lower() != ".pdf":
        return None
    try:
        import fitz  # pymupdf

        with fitz.open(str(path)) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Candidate:
    doc: dict
    archive_file: Path
    root_doc_id: str  # full doc_id of the parent email (archive_gen slices to 16)


@dataclass
class RepairOutcome:
    doc_id: str
    updates: dict | None = None  # fields to merge into the doc row on rewrite
    error: str | None = None


@dataclass
class RepairComponents:
    """Thin DI wrapper so tests can inject fakes.

    Each field mirrors a real pipeline piece. The real `build_components`
    factory wires them to production implementations; tests supply fakes.
    """

    parse_attachment: Callable[[Path], Any]  # async (Path) -> markdown str
    chunk_text: Callable[..., list]  # DocumentChunker.chunk_text
    build_embedding_text: Callable[[str, str], str]
    generate_context: Callable[..., Any]  # async (doc_obj, preview) -> str
    embed_chunks: Callable[[list], Any]  # async (chunks) -> chunks
    update_attachment_markdown: Callable[..., str | None]


def find_candidates(
    output_dir: Path,
    max_pages: int | None = None,
    min_pages: int | None = None,
) -> tuple[list[Candidate], list[tuple[dict, str]]]:
    """Scan documents.jsonl + chunks.jsonl, return (repairable, skipped).

    ``max_pages`` / ``min_pages`` gate PDF candidates by page count (inclusive
    bounds). Non-PDFs bypass both filters. Unreadable PDFs (count returns
    None) also bypass — we'd rather surface the true parse error than hide
    it behind an opaque "couldn't count pages" skip.
    """
    docs_path = output_dir / "documents.jsonl"
    archive_root = output_dir / "archive"

    all_docs: list[dict] = []
    root_doc_id_by_uuid: dict[str, str] = {}
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            all_docs.append(d)
            if d.get("depth") == 0 and d.get("doc_id"):
                root_doc_id_by_uuid[d["id"]] = d["doc_id"]

    docs_with_chunks: set[str] = set()
    chunks_path = output_dir / "chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = c.get("document_id")
                if did:
                    docs_with_chunks.add(did)

    candidates: list[Candidate] = []
    skipped: list[tuple[dict, str]] = []
    for d in all_docs:
        if d.get("status") != "failed":
            continue
        err = d.get("error_message") or ""
        if "cost_optimizer" not in err:
            continue
        fp = d.get("file_path") or ""
        if "_extracted" in fp or "processed" in fp.lower():
            skipped.append((d, SKIP_ZIP_MEMBER))
            continue
        root_uuid = d.get("root_id") or ""
        root_doc_id = root_doc_id_by_uuid.get(root_uuid, "")
        if not root_doc_id:
            skipped.append((d, SKIP_NO_ROOT))
            continue
        rid16 = root_doc_id[:16]
        last_name = fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        archive_file = archive_root / rid16 / "attachments" / last_name
        if not archive_file.exists():
            skipped.append((d, SKIP_MISSING_FILE))
            continue
        if d["id"] in docs_with_chunks:
            skipped.append((d, SKIP_ALREADY_HAS_CHUNKS))
            continue
        # Chrome partial-download temp files: not valid PDF bytes, every
        # LlamaParse call returns a BadZipFile / parse error. Permanent skip.
        if archive_file.suffix.lower() == ".crdownload":
            skipped.append((d, SKIP_CRDOWNLOAD))
            continue
        if max_pages is not None or min_pages is not None:
            pages = count_pdf_pages(archive_file)
            if pages is not None:
                if max_pages is not None and pages > max_pages:
                    skipped.append((d, SKIP_OVER_PAGE_LIMIT))
                    continue
                if min_pages is not None and pages < min_pages:
                    skipped.append((d, SKIP_UNDER_PAGE_LIMIT))
                    continue
        candidates.append(
            Candidate(doc=d, archive_file=archive_file, root_doc_id=root_doc_id)
        )
    return candidates, skipped


async def repair_one(
    candidate: Candidate,
    components: RepairComponents,
    output_dir: Path,
) -> RepairOutcome:
    """Repair one attachment doc. Raises nothing; returns RepairOutcome."""
    from mtss.ingest.helpers import enrich_chunks_with_document_metadata
    from mtss.models.serializers import chunk_to_dict, dict_to_document

    doc_id = candidate.doc["id"]
    try:
        doc_obj = dict_to_document(candidate.doc)
        text = await components.parse_attachment(candidate.archive_file)
        if not text or not text.strip():
            return RepairOutcome(doc_id=doc_id, error="llamaparse_returned_empty")

        chunks = components.chunk_text(
            text=text,
            document_id=doc_obj.id,
            source_file=str(candidate.archive_file),
            is_markdown=True,
        )
        if not chunks:
            return RepairOutcome(doc_id=doc_id, error="chunker_returned_empty")

        enrich_chunks_with_document_metadata(chunks, doc_obj)

        context = await components.generate_context(doc_obj, text[:4000])
        for chunk in chunks:
            chunk.context_summary = context
            chunk.embedding_text = components.build_embedding_text(
                context, chunk.content
            )

        await components.embed_chunks(chunks)

        # Append chunks atomically (per-line). Do this before the archive .md
        # regenerate so a crash leaves us in a state where idempotency skip
        # will notice the chunks on retry (avoiding double-charge).
        chunks_path = output_dir / "chunks.jsonl"
        with chunks_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n"
                )
            f.flush()
            os.fsync(f.fileno())

        # Regenerate archive .md preview. Guarded — failure here is non-fatal;
        # the chunks are the authoritative repair. Rebuild via `ingest-update`
        # if this step is later noticed missing.
        md_path: str | None = None
        try:
            md_path = components.update_attachment_markdown(
                doc_id=candidate.root_doc_id,
                filename=candidate.doc["file_name"],
                content_type=candidate.doc.get("attachment_content_type")
                or "application/octet-stream",
                size_bytes=candidate.doc.get("attachment_size_bytes") or 0,
                parsed_content=text,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"archive .md regenerate failed for {candidate.doc['file_name']}: {e}"
            )

        updates: dict = {
            "status": "completed",
            "error_message": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if md_path:
            updates["archive_browse_uri"] = f"/archive/{md_path}"
            updates["archive_download_uri"] = (
                f"/archive/{md_path.removesuffix('.md')}"
            )
        return RepairOutcome(doc_id=doc_id, updates=updates)
    except Exception as e:  # noqa: BLE001
        return RepairOutcome(doc_id=doc_id, error=f"{type(e).__name__}: {e}")


def find_finalize_targets(output_dir: Path) -> list[dict]:
    """Docs that have chunks on disk but whose doc row still says failed.

    Happens when a mid-run kill cuts off the atomic docs.jsonl rewrite — the
    chunks/archive side has already been flushed per-doc, so the fix is pure
    metadata (flip status, derive archive URIs from .md presence).
    """
    docs_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    docs_with_chunks: set[str] = set()
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = c.get("document_id")
                if did:
                    docs_with_chunks.add(did)

    targets: list[dict] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") != "failed":
                continue
            if "cost_optimizer" not in (d.get("error_message") or ""):
                continue
            if d["id"] in docs_with_chunks:
                targets.append(d)
    return targets


def build_finalize_updates(
    output_dir: Path,
    targets: list[dict],
    root_doc_id_by_uuid: dict[str, str],
) -> dict[str, dict]:
    """Compute the metadata patch for each finalize target.

    Archive URIs are set only when the corresponding .md exists on disk.
    The on-disk name uses the same ``_sanitize_storage_key`` rules the
    archive writer applies (spaces → underscores, brackets stripped, etc.),
    so we must match against that — matching the raw ``file_name`` stem
    misses every attachment that had space/punctuation in its name.
    """
    from mtss.ingest.archive_generator import _sanitize_storage_key

    archive_root = output_dir / "archive"
    now_iso = datetime.now(timezone.utc).isoformat()
    updates: dict[str, dict] = {}
    for d in targets:
        patch: dict = {
            "status": "completed",
            "error_message": None,
            "updated_at": now_iso,
        }
        root_uuid = d.get("root_id") or ""
        root_doc_id = root_doc_id_by_uuid.get(root_uuid, "")
        if root_doc_id:
            rid16 = root_doc_id[:16]
            att_dir = archive_root / rid16 / "attachments"
            safe_name = _sanitize_storage_key(d.get("file_name", ""))
            md_name = f"{safe_name}.md"
            md_path = att_dir / md_name
            if md_path.exists():
                rel = f"{rid16}/attachments/{md_name}"
                patch["archive_browse_uri"] = f"/archive/{rel}"
                patch["archive_download_uri"] = f"/archive/{rid16}/attachments/{safe_name}"
        updates[d["id"]] = patch
    return updates


def rewrite_documents_jsonl(
    output_dir: Path, updates_by_id: dict[str, dict]
) -> int:
    """Atomic rewrite merging per-doc updates. Returns rows updated."""
    docs_path = output_dir / "documents.jsonl"
    tmp_path = docs_path.with_suffix(docs_path.suffix + ".tmp")
    count = 0
    with docs_path.open("r", encoding="utf-8") as src, tmp_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            d = json.loads(line)
            patch = updates_by_id.get(d.get("id"))
            if patch:
                d.update(patch)
                count += 1
            dst.write(json.dumps(d, ensure_ascii=False) + "\n")
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp_path, docs_path)
    return count


def build_components(output_dir: Path) -> RepairComponents:
    """Wire the real pipeline pieces — lazy imports so tests can avoid them.

    ArchiveGenerator must be pointed at the local-filesystem bucket under
    ``<output_dir>/archive``; its default ``ArchiveStorage()`` is the Supabase
    backend, which doesn't see local attachment files and causes the
    pre-upload ``file_exists`` check in ``update_attachment_markdown`` to
    fail silently ("Original file not found, skipping .md creation").
    """
    from mtss.ingest.archive_generator import ArchiveGenerator
    from mtss.parsers.chunker import ContextGenerator, DocumentChunker
    from mtss.parsers.llamaparse_parser import LlamaParseParser
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.storage.local_client import LocalBucketStorage

    parser = LlamaParseParser()
    chunker = DocumentChunker()
    context_gen = ContextGenerator()
    embedder = EmbeddingGenerator()
    archive_gen = ArchiveGenerator(
        storage=LocalBucketStorage(output_dir / "archive")
    )

    return RepairComponents(
        parse_attachment=parser.parse,
        chunk_text=chunker.chunk_text,
        build_embedding_text=context_gen.build_embedding_text,
        generate_context=context_gen.generate_context,
        embed_chunks=embedder.embed_chunks,
        update_attachment_markdown=archive_gen.update_attachment_markdown,
    )


async def _run(
    output_dir: Path,
    dry_run: bool,
    concurrency: int | None,
    limit: int | None,
    max_pages: int | None,
    min_pages: int | None,
    finalize: bool,
) -> int:
    if finalize:
        return _run_finalize(output_dir, dry_run)
    if concurrency is None:
        from mtss.config import get_settings

        concurrency = get_settings().max_concurrent_llamaparse
    candidates, skipped = find_candidates(
        output_dir, max_pages=max_pages, min_pages=min_pages
    )
    print(f"candidates: {len(candidates)}   skipped: {len(skipped)}")
    skip_counts: dict[str, int] = {}
    for _, reason in skipped:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
    for reason, n in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
        print(f"  skipped [{reason}]: {n}")

    if limit is not None:
        candidates = candidates[:limit]
        print(f"--limit applied: processing first {len(candidates)}")

    if not candidates or dry_run:
        print()
        print("sample candidates (first 5):")
        for c in candidates[:5]:
            print(
                f"  {c.doc['id'][:8]}..  {c.doc['file_name']}  -> "
                f"{c.archive_file.relative_to(output_dir)}"
            )
        if dry_run:
            print("dry-run: no changes written.")
        return 0

    print(f"concurrency: {concurrency}")
    components = build_components(output_dir)
    sem = asyncio.Semaphore(concurrency)
    updates_by_id: dict[str, dict] = {}
    successes = 0
    errors: list[tuple[str, str]] = []

    async def _run_one(c: Candidate) -> None:
        nonlocal successes
        async with sem:
            outcome = await repair_one(c, components, output_dir)
        if outcome.updates:
            updates_by_id[outcome.doc_id] = outcome.updates
            successes += 1
            print(f"  ok  {c.doc['file_name']}")
        else:
            errors.append((c.doc["file_name"], outcome.error or "unknown"))
            print(f"  ERR  {c.doc['file_name']}  — {outcome.error}")

    await asyncio.gather(*(_run_one(c) for c in candidates))

    rewritten = 0
    if updates_by_id:
        rewritten = rewrite_documents_jsonl(output_dir, updates_by_id)
    print()
    print(f"repaired: {successes} / {len(candidates)}   "
          f"rewrote {rewritten} doc rows")
    if errors:
        print(f"errors: {len(errors)}")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    return 0 if not errors else 1


def _run_finalize(output_dir: Path, dry_run: bool) -> int:
    """Flip doc rows whose chunks already exist on disk — no re-parse.

    Used to recover from a mid-run kill that wrote chunks per-doc but never
    reached the atomic docs.jsonl rewrite at the end of the batch.
    """
    # Build root lookup
    root_doc_id_by_uuid: dict[str, str] = {}
    with (output_dir / "documents.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("depth") == 0 and d.get("doc_id"):
                root_doc_id_by_uuid[d["id"]] = d["doc_id"]

    from mtss.ingest.archive_generator import _sanitize_storage_key

    targets = find_finalize_targets(output_dir)
    print(f"finalize targets: {len(targets)}")
    with_md = 0
    without_md = 0
    archive_root = output_dir / "archive"
    for d in targets:
        root_doc_id = root_doc_id_by_uuid.get(d.get("root_id") or "", "")
        rid16 = root_doc_id[:16] if root_doc_id else ""
        found = False
        if rid16:
            safe_name = _sanitize_storage_key(d.get("file_name", ""))
            if (archive_root / rid16 / "attachments" / f"{safe_name}.md").exists():
                found = True
        if found:
            with_md += 1
        else:
            without_md += 1
    print(f"  with archive .md:    {with_md}")
    print(f"  without archive .md: {without_md}")

    if dry_run or not targets:
        for d in targets[:5]:
            print(f"  sample: {d['id'][:8]}..  {d['file_name']}")
        if dry_run:
            print("dry-run: no changes written.")
        return 0

    updates = build_finalize_updates(output_dir, targets, root_doc_id_by_uuid)
    rewritten = rewrite_documents_jsonl(output_dir, updates)
    print(f"rewrote {rewritten} doc rows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Ingest output dir (default: data/output)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan only")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Parallel repair workers. Default: MAX_CONCURRENT_LLAMAPARSE from "
            "settings (same budget the ingest pipeline uses). Pass a lower "
            "number to throttle further; higher than the LlamaParse setting "
            "won't buy more parse throughput because the parser enforces its "
            "own semaphore internally."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N candidates (for pilot runs)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Skip PDFs with more than N pages. Pair with a later "
            "`--min-pages N+1` run to repair the deferred long PDFs on a "
            "separate token budget."
        ),
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=None,
        help=(
            "Skip PDFs with fewer than N pages. Use to target only the "
            "long-tail deferred by an earlier --max-pages run."
        ),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Don't parse anything — flip status=completed on docs that "
            "already have chunks on disk. Used to finish a run that was "
            "killed after chunks were persisted but before the atomic "
            "docs.jsonl rewrite at the end."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return asyncio.run(
        _run(
            args.output_dir,
            args.dry_run,
            args.concurrency,
            args.limit,
            args.max_pages,
            args.min_pages,
            args.finalize,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
