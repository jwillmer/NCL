"""Attachment processing for the ingest pipeline.

Handles individual attachment processing and ZIP extraction,
extracted from cli.py for better modularity.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .._io import read_bytes_async
from ..config import get_settings
from ..ingest.archive_generator import _sanitize_storage_key
from ..ingest.helpers import (
    apply_filter_metadata,
    enrich_chunks_with_document_metadata,
    get_format_name,
    noop_verbose,
    sanitize_error_message,
)
from ..models.document import ProcessingStatus

from .processing_trail import (
    STEP_CONTEXT,
    STEP_DECIDER,
    STEP_PARSE,
    STEP_SUMMARY,
    STEP_VISION,
)

if TYPE_CHECKING:
    from ..models.chunk import Chunk
    from ..models.document import Document
    from ..storage.unsupported_file_logger import UnsupportedFileLogger
    from .components import IngestComponents
    from .helpers import IssueTracker
    from .processing_trail import ProcessingTrail


logger = logging.getLogger(__name__)

_noop_verbose = noop_verbose  # backward compat alias


def _write_attachment_archive_md(
    archive_generator,
    attach_doc: "Document",
    email_doc_id: str,
    parsed_content: str,
    *,
    filename: str,
    content_type: str,
    size_bytes: int,
) -> str | None:
    """Write the attachment's .md preview to the archive and stamp browse/download URIs.

    Shared by non-ZIP (`_process_non_zip_attachment`) and ZIP-member
    (`_process_zip_member`) paths so the URI pattern can't drift — a class
    of bug CLAUDE.md used to call out explicitly for this file.

    Callers MUST check ``archive_generator``, ``parsed_content``, and
    ``email_doc_id`` truthiness before invoking; the helper assumes all three.
    """
    md_path = archive_generator.update_attachment_markdown(
        doc_id=email_doc_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        parsed_content=parsed_content,
    )
    if md_path:
        # md_path is already sanitized by _sanitize_storage_key — don't re-quote.
        attach_doc.archive_browse_uri = f"/archive/{md_path}"
        attach_doc.archive_download_uri = f"/archive/{md_path.removesuffix('.md')}"
    return md_path


def _apply_vessel_metadata_to_chunks(
    chunks: list,
    vessel_ids: list[str],
    vessel_types: list[str],
    vessel_classes: list[str],
) -> None:
    """Stamp vessel fields onto each chunk's metadata dict.

    Thin wrapper over ``apply_filter_metadata`` that handles the per-chunk
    fan-out + ``metadata=None`` initialization. No-op when all three vessel
    lists are empty.
    """
    if not (vessel_ids or vessel_types or vessel_classes):
        return
    for chunk in chunks:
        if chunk.metadata is None:
            chunk.metadata = {}
        apply_filter_metadata(
            chunk.metadata,
            vessel_ids=vessel_ids,
            vessel_types=vessel_types,
            vessel_classes=vessel_classes,
        )


def _extract_content_from_cached_markdown(cached_md: str) -> str | None:
    """Extract content section from cached attachment markdown.

    The cached markdown format has a header followed by "## Content" section.
    This extracts just the content portion for re-chunking.

    Args:
        cached_md: Full markdown content from cache.

    Returns:
        Extracted content section, or None if not found.
    """
    content_marker = "## Content\n"
    idx = cached_md.find(content_marker)
    if idx == -1:
        return None
    content = cached_md[idx + len(content_marker):].strip()
    return content if content else None


async def _decide_and_build_chunks(
    *,
    document: "Document",
    parsed_content: str,
    components: "IngestComponents",
    file_path: Path,
    filename: str,
    source_eml_path: str,
    vprint: Callable[[str, str | None], None] | None,
    file_ctx: str | None,
    trail: "ProcessingTrail | None" = None,
    trail_key: str | None = None,
) -> list["Chunk"]:
    """Run the embedding-mode decider and build chunks via the shared
    dispatcher. Used by both the non-ZIP and ZIP-member paths so they can't
    drift. When no context_generator is wired, force FULL (the decider's
    SUMMARY mode needs the LLM to synthesize the summary chunk)."""
    from ..models.document import EmbeddingMode
    from ..parsers.chunker import build_chunks_for_mode

    if components.context_generator is None:
        document.embedding_mode = EmbeddingMode.FULL
        return await build_chunks_for_mode(
            mode=EmbeddingMode.FULL,
            document=document,
            markdown=parsed_content,
            chunker=components.attachment_processor.chunker,
            context_generator=None,
            source_file=str(file_path),
        )

    from ..ingest.embedding_decider import decide_embedding_mode

    decision = await decide_embedding_mode(parsed_content, document)
    document.embedding_mode = decision.mode
    if trail is not None and trail_key is not None:
        # Triage model is only actually consulted in the medium-confidence band —
        # stamp the model only for triage-sourced reasons, otherwise leave it
        # None so consumers know the decision was deterministic.
        triage_reasons = {"triage_prose", "triage_dense", "triage_noise", "triage_failed"}
        triage_model: str | None = None
        if decision.reason in triage_reasons:
            ts = get_settings()
            triage_model = (
                ts.embedding_triage_llm_model or ts.get_model(ts.context_llm_model)
            )
        trail.stamp_attachment(
            trail_key,
            STEP_DECIDER,
            model=triage_model,
            mode=decision.mode.value,
            reason=decision.reason,
        )
    if vprint and file_ctx is not None:
        vprint(
            f"  -> embedding_mode={decision.mode.value} ({decision.reason})",
            file_ctx,
        )
    try:
        components.db.log_ingest_event(
            document_id=document.id,
            event_type="embedding_mode_decided",
            severity="info",
            message=f"{decision.mode.value}: {decision.reason}",
            file_path=str(file_path),
            file_name=filename,
            source_eml_path=source_eml_path,
        )
    except Exception:
        logger.debug("log_ingest_event(embedding_mode_decided) failed", exc_info=True)

    attach_chunks = await build_chunks_for_mode(
        mode=decision.mode,
        document=document,
        markdown=parsed_content,
        chunker=components.attachment_processor.chunker,
        context_generator=components.context_generator,
        source_file=str(file_path),
    )
    if trail is not None and trail_key is not None:
        from ..models.document import EmbeddingMode

        ctx_model = components.context_generator.model_name
        if decision.mode == EmbeddingMode.SUMMARY:
            # SUMMARY mode has no separate context step — the one LLM call is
            # the summary itself.
            trail.stamp_attachment(trail_key, STEP_SUMMARY, model=ctx_model)
        elif decision.mode == EmbeddingMode.FULL:
            trail.stamp_attachment(trail_key, STEP_CONTEXT, model=ctx_model)
        # METADATA_ONLY makes no LLM call — no stamp.
    if vprint and file_ctx is not None:
        vprint(
            f"  -> {len(attach_chunks)} chunks ({decision.mode.value})",
            file_ctx,
        )
    return attach_chunks


async def _archive_only_attachment(
    *,
    attachment,
    file_path: Path,
    email_doc: "Document",
    source_eml_path: str,
    components: "IngestComponents",
    skip_reason: str,
    archive_file_result,
    collect_docs: list["Document"] | None,
) -> None:
    """Register a PDF we deliberately chose not to parse.

    The email-level archive step already uploaded the original file — this
    helper only adds the doc record (status=FAILED, error_message carrying
    the pdf_too_large reason) so a follow-up `mtss mark-failed` + `mtss ingest
    --retry-failed` (or `mtss re-embed --doc-id <uuid> --mode summary`) can
    discover these docs by their error marker and process them later on a
    dedicated token budget. No stub .md is written; absence of the .md is the
    on-disk signal that the PDF hasn't been parsed yet.
    """
    attach_doc = components.hierarchy_manager.build_attachment_document(
        parent_doc=email_doc,
        attachment_path=file_path,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        original_filename=attachment.filename,
        archive_file_result=archive_file_result,
    )
    attach_doc.status = ProcessingStatus.FAILED
    attach_doc.error_message = skip_reason

    try:
        components.db.log_ingest_event(
            document_id=attach_doc.id,
            event_type="pdf_parse_skipped",
            severity="info",
            message=skip_reason,
            file_path=str(file_path),
            file_name=attachment.filename,
            source_eml_path=source_eml_path,
        )
    except Exception:  # noqa: BLE001
        pass

    if collect_docs is not None:
        collect_docs.append(attach_doc)


def _apply_oversized_pdf_peek(
    *,
    result,
    attach_doc_id,
    file_path: Path,
    file_name: str,
    source_eml_path: str,
    components: "IngestComponents",
    trail: "ProcessingTrail | None",
    trail_key: str | None,
) -> str:
    """Handle the oversized-PDF branch: stamp trail + emit event + return
    the preview markdown. Shared between the top-level and ZIP-member paths
    so they can't drift — see the sync invariant called out in CLAUDE.md."""
    if trail is not None and trail_key is not None:
        trail.stamp_attachment(trail_key, STEP_PARSE, parser="oversized_pdf_peek")
    components.db.log_ingest_event(
        document_id=attach_doc_id,
        event_type="oversized_pdf_peek",
        severity="info",
        message=f"peek used in lieu of full parse ({result.total_pages} pages)",
        file_path=str(file_path),
        file_name=file_name,
        source_eml_path=source_eml_path,
    )
    return result.preview_markdown


async def process_attachment(
    attachment,
    email_doc: "Document",
    source_eml_path: str,
    file_ctx: str,
    components: "IngestComponents",
    unsupported_logger: "UnsupportedFileLogger",
    archive_file_result=None,
    vessel_ids: list[str] | None = None,
    vessel_types: list[str] | None = None,
    vessel_classes: list[str] | None = None,
    force_reparse: bool = False,
    lenient: bool = False,
    on_verbose: Callable[[str, str | None], None] | None = None,
    issue_tracker: "IssueTracker | None" = None,
    email_context_summary: str | None = None,
    collect_docs: list["Document"] | None = None,
    on_member_complete: Callable[[], None] | None = None,
    trail: "ProcessingTrail | None" = None,
) -> list["Chunk"]:
    """Process a single attachment and return its chunks.

    Uses preprocessor for routing decisions.
    Also updates the archive with .md files for parsed attachments.
    Checks for cached parsed content before calling LlamaParse (unless force_reparse=True).
    """
    vprint = on_verbose or _noop_verbose
    chunks: list["Chunk"] = []
    vessel_ids = vessel_ids or []
    vessel_types = vessel_types or []
    vessel_classes = vessel_classes or []
    file_path = Path(attachment.saved_path)
    # Progress accounting: non-ZIP attachments tick once on function exit,
    # regardless of outcome (skip / success / error). ZIP branch ticks per
    # extracted member internally and sets this flag to suppress the outer
    # single tick.
    _suppress_tick = False

    try:
        # Format size for display
        size_kb = attachment.size_bytes / 1024 if attachment.size_bytes else 0
        size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
        format_name = get_format_name(attachment.content_type or "unknown")

        # Preprocess to get routing decision (classify_images=True for email-level attachments)
        result = await components.attachment_processor.preprocess(
            file_path, attachment.content_type, classify_images=True
        )

        # Determine processor type for display
        if result.is_zip:
            processor = "ZIP"
        elif result.is_image:
            processor = "Vision"
        elif result.parser_name:
            processor = result.parser_name.capitalize()
        else:
            processor = "unsupported"

        vprint(f"Attachment: {attachment.filename} ({format_name}, {size_str}) [{processor}]", file_ctx)

        # Handle based on preprocess result
        if not result.should_process:
            reason = result.skip_reason or "unsupported_format"

            # pdf_too_large: archive the original + stub .md but skip parsing.
            # The file has already been uploaded by the email-level archive step;
            # here we just register a doc record + stub md so browse/validate
            # see why it wasn't parsed and don't flag it as orphaned.
            if reason.startswith("pdf_too_large"):
                vprint(f"  -> Archived only (parse skipped): {reason}", file_ctx)
                await _archive_only_attachment(
                    attachment=attachment,
                    file_path=file_path,
                    email_doc=email_doc,
                    source_eml_path=source_eml_path,
                    components=components,
                    skip_reason=reason,
                    archive_file_result=archive_file_result,
                    collect_docs=collect_docs,
                )
                return chunks

            if "non_content" in reason.lower() or result.is_image:
                vprint("  -> Skipped: non-content image (logo/banner/signature)", file_ctx)
                reason = "classified_as_non_content"
            else:
                vprint(f"  -> Skipped: {reason}", file_ctx)
            await unsupported_logger.log_unsupported_file(
                file_path=file_path,
                reason=reason,
                source_eml_path=source_eml_path,
                parent_document_id=email_doc.id,
            )
            return chunks

        # Handle ZIP files
        if result.is_zip:
            _suppress_tick = True
            chunks.extend(
                await process_zip_attachment(
                    attachment=attachment,
                    email_doc=email_doc,
                    source_eml_path=source_eml_path,
                    file_ctx=file_ctx,
                    components=components,
                    unsupported_logger=unsupported_logger,
                    vessel_ids=vessel_ids,
                    vessel_types=vessel_types,
                    vessel_classes=vessel_classes,
                    force_reparse=force_reparse,
                    lenient=lenient,
                    on_verbose=on_verbose,
                    issue_tracker=issue_tracker,
                    email_context_summary=email_context_summary,
                    collect_docs=collect_docs,
                    on_member_complete=on_member_complete,
                    trail=trail,
                    zip_filename=attachment.filename,
                )
            )
            return chunks

        # ── Non-ZIP main body ──────────────────────────────────────────
        return await _process_non_zip_attachment(
            attachment=attachment,
            file_path=file_path,
            email_doc=email_doc,
            source_eml_path=source_eml_path,
            file_ctx=file_ctx,
            components=components,
            unsupported_logger=unsupported_logger,
            archive_file_result=archive_file_result,
            vessel_ids=vessel_ids,
            vessel_types=vessel_types,
            vessel_classes=vessel_classes,
            force_reparse=force_reparse,
            on_verbose=on_verbose,
            issue_tracker=issue_tracker,
            collect_docs=collect_docs,
            result=result,
            chunks=chunks,
            vprint=vprint,
            trail=trail,
        )
    finally:
        if on_member_complete and not _suppress_tick:
            try:
                on_member_complete()
            except Exception:  # noqa: BLE001
                pass


async def _process_non_zip_attachment(
    *,
    attachment,
    file_path: Path,
    email_doc: "Document",
    source_eml_path: str,
    file_ctx: str,
    components: "IngestComponents",
    unsupported_logger: "UnsupportedFileLogger",
    archive_file_result,
    vessel_ids: list[str],
    vessel_types: list[str],
    vessel_classes: list[str],
    force_reparse: bool,
    on_verbose: Callable[[str, str | None], None] | None,
    issue_tracker: "IssueTracker | None",
    collect_docs: list["Document"] | None,
    result,
    chunks: list["Chunk"],
    vprint: Callable[[str, str | None], None],
    trail: "ProcessingTrail | None" = None,
) -> list["Chunk"]:
    """Main body of process_attachment for non-ZIP attachments.

    Extracted so that process_attachment's outer try/finally can wrap the
    entire non-ZIP code path (for a single progress tick on exit) without
    requiring a single-try-with-body-inside rewrite.
    """
    # Build attachment document (deferred DB insert - caller persists atomically)
    attach_doc = components.hierarchy_manager.build_attachment_document(
        parent_doc=email_doc,
        attachment_path=file_path,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        original_filename=attachment.filename,
        archive_file_result=archive_file_result,
    )
    trail_key = attachment.filename
    if trail is not None:
        trail.register_attachment_document(attach_doc.id, trail_key)

    try:
        parsed_content: str | None = None

        if result.is_image and result.image_description:
            # Image was already classified and described during preprocessing
            chunk = components.attachment_processor.create_image_chunk(
                file_path, attach_doc.id, result.image_description, "meaningful"
            )
            chunks.append(chunk)
            parsed_content = result.image_description
            if trail is not None:
                trail.stamp_attachment(
                    trail_key,
                    STEP_VISION,
                    model=components.attachment_processor.image_processor.model_name,
                )
            vprint("  -> 1 chunk created (image described)", file_ctx)
            attach_doc.status = ProcessingStatus.COMPLETED
        elif result.is_image:
            # Image needs description (preprocessing didn't provide one)
            attach_chunks = await components.attachment_processor.process_document_image(
                file_path, attach_doc.id
            )
            chunks.extend(attach_chunks)
            if attach_chunks:
                parsed_content = attach_chunks[0].content
                if trail is not None:
                    trail.stamp_attachment(
                        trail_key,
                        STEP_VISION,
                        model=components.attachment_processor.image_processor.model_name,
                    )
            vprint(f"  -> {len(attach_chunks)} chunks created (image described)", file_ctx)
            attach_doc.status = ProcessingStatus.COMPLETED
        else:
            # Document - check for cached parsed content before calling LlamaParse
            cached_content: str | None = None
            if (
                not force_reparse
                and components.archive_generator
                and email_doc.doc_id
                and result.parser_name  # Only for documents that use parsers
            ):
                folder_id = email_doc.doc_id[:16]
                safe_filename = _sanitize_storage_key(attachment.filename)
                cached_md_path = f"{folder_id}/attachments/{safe_filename}.md"
                try:
                    if components.archive_generator.storage.file_exists(cached_md_path):
                        cached_md = components.archive_generator.storage.download_text(cached_md_path)
                        cached_content = _extract_content_from_cached_markdown(cached_md)
                        if cached_content:
                            vprint(f"  -> Using cached content ({len(cached_content)} chars)", file_ctx)
                except Exception as e:
                    vprint(f"  -> Cache check failed: {e}", file_ctx)
                    cached_content = None

            if cached_content:
                parsed_content = cached_content
                if trail is not None:
                    trail.stamp_attachment(trail_key, STEP_PARSE, parser="cached_md")
                vprint(
                    f"  -> cached content ({len(cached_content)} chars)", file_ctx
                )
            elif result.oversized_pdf and result.preview_markdown:
                # Oversized PDF: preprocessor already extracted a cheap local
                # peek. Feed that to the decider instead of running the full
                # cloud parser — SUMMARY/METADATA_ONLY is the right outcome
                # for multi-hundred-page sensor dumps.
                parsed_content = _apply_oversized_pdf_peek(
                    result=result,
                    attach_doc_id=attach_doc.id,
                    file_path=file_path,
                    file_name=attachment.filename,
                    source_eml_path=source_eml_path,
                    components=components,
                    trail=trail,
                    trail_key=trail_key,
                )
                vprint(
                    f"  -> oversized PDF peek ({result.total_pages} pages, "
                    f"{len(result.preview_markdown)} chars preview)",
                    file_ctx,
                )
            else:
                parsed_content, _parser_name, _parser_model = (
                    await components.attachment_processor.parse_to_text(
                        file_path, attachment.content_type
                    )
                )
                if trail is not None:
                    trail.stamp_attachment(
                        trail_key,
                        STEP_PARSE,
                        model=_parser_model,
                        parser=_parser_name,
                    )

                if not parsed_content:
                    vprint("  -> 0 chunks (document has no extractable text)", file_ctx)
                    components.db.log_ingest_event(
                        document_id=attach_doc.id,
                        event_type="no_body_chunks",
                        severity="info",
                        message="Attachment produced 0 chunks (no extractable text)",
                        file_path=str(file_path),
                        file_name=attachment.filename,
                        source_eml_path=source_eml_path,
                    )

            attach_chunks = []
            if parsed_content:
                attach_chunks = await _decide_and_build_chunks(
                    document=attach_doc,
                    parsed_content=parsed_content,
                    components=components,
                    file_path=file_path,
                    filename=attachment.filename,
                    source_eml_path=source_eml_path,
                    vprint=vprint,
                    file_ctx=file_ctx,
                    trail=trail,
                    trail_key=trail_key,
                )

            chunks.extend(attach_chunks)
            attach_doc.status = ProcessingStatus.COMPLETED

        # Update archive with .md file for this attachment
        vprint(f"  -> MD check: archive_gen={components.archive_generator is not None}, parsed_content={len(parsed_content) if parsed_content else 0} chars, doc_id={email_doc.doc_id[:16] if email_doc.doc_id else None}", file_ctx)
        if components.archive_generator and parsed_content and email_doc.doc_id:
            md_path = _write_attachment_archive_md(
                components.archive_generator,
                attach_doc,
                email_doc.doc_id,
                parsed_content,
                filename=attachment.filename,
                content_type=attachment.content_type or "application/octet-stream",
                size_bytes=attachment.size_bytes or 0,
            )
            if md_path:
                vprint(f"  -> Archive updated: {md_path}", file_ctx)
            else:
                vprint("  -> update_attachment_markdown returned None", file_ctx)
        else:
            vprint("  -> Skipping .md creation", file_ctx)

    except Exception as e:
        if issue_tracker:
            await issue_tracker.track_async(file_ctx, attachment.filename, str(e))
        attach_doc.status = ProcessingStatus.FAILED
        attach_doc.error_message = sanitize_error_message(str(e))
        await unsupported_logger.log_unsupported_file(
            file_path=file_path,
            reason="extraction_failed",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )

    # Collect document for atomic persist by caller
    if collect_docs is not None:
        collect_docs.append(attach_doc)

    # Enrich chunks with document citation metadata (source_id, source_title, archive URIs)
    enrich_chunks_with_document_metadata(chunks, attach_doc)
    _apply_vessel_metadata_to_chunks(chunks, vessel_ids, vessel_types, vessel_classes)

    return chunks


async def process_zip_attachment(
    attachment,
    email_doc: "Document",
    source_eml_path: str,
    file_ctx: str,
    components: "IngestComponents",
    unsupported_logger: "UnsupportedFileLogger",
    vessel_ids: list[str] | None = None,
    vessel_types: list[str] | None = None,
    vessel_classes: list[str] | None = None,
    force_reparse: bool = False,
    lenient: bool = False,
    on_verbose: Callable[[str, str | None], None] | None = None,
    issue_tracker: "IssueTracker | None" = None,
    email_context_summary: str | None = None,
    collect_docs: list["Document"] | None = None,
    on_member_complete: Callable[[], None] | None = None,
    trail: "ProcessingTrail | None" = None,
    zip_filename: str | None = None,
) -> list["Chunk"]:
    """Extract and process files from a ZIP attachment.

    Note: ZIP contents don't use the archive cache since they don't have
    pre-existing archive paths. They are always parsed fresh.
    """
    vprint = on_verbose or _noop_verbose
    chunks: list["Chunk"] = []
    vessel_ids = vessel_ids or []
    vessel_types = vessel_types or []
    vessel_classes = vessel_classes or []
    member_count = 0

    try:
        extracted_files = components.attachment_processor.extract_zip(
            Path(attachment.saved_path), lenient=lenient
        )
        member_count = len(extracted_files)

        settings = get_settings()
        sem = asyncio.Semaphore(max(1, settings.zip_member_concurrency))

        async def _run_one(extracted_path: Path, extracted_content_type: str):
            async with sem:
                try:
                    return await _process_zip_member(
                        extracted_path=extracted_path,
                        extracted_content_type=extracted_content_type,
                        attachment=attachment,
                        email_doc=email_doc,
                        source_eml_path=source_eml_path,
                        file_ctx=file_ctx,
                        components=components,
                        unsupported_logger=unsupported_logger,
                        vessel_ids=vessel_ids,
                        vessel_types=vessel_types,
                        vessel_classes=vessel_classes,
                        vprint=vprint,
                        issue_tracker=issue_tracker,
                        trail=trail,
                        zip_filename=zip_filename or attachment.filename,
                    )
                finally:
                    if on_member_complete:
                        try:
                            on_member_complete()
                        except Exception:  # noqa: BLE001
                            pass

        results = await asyncio.gather(
            *(_run_one(ep, ct) for ep, ct in extracted_files)
        )
        for attach_doc, attach_chunks in results:
            if attach_chunks:
                chunks.extend(attach_chunks)
            if attach_doc is not None and collect_docs is not None:
                collect_docs.append(attach_doc)

    except Exception as e:
        if issue_tracker:
            await issue_tracker.track_async(file_ctx, attachment.filename, f"ZIP extraction failed: {e}")
        await unsupported_logger.log_unsupported_file(
            file_path=Path(attachment.saved_path),
            reason="corrupted",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )
        # ZIP failed before/during extraction: the caller pre-sized progress
        # by expected member count. Catch up remaining ticks so the bar
        # doesn't stall.
        if on_member_complete:
            expected = _count_zip_members(Path(attachment.saved_path))
            for _ in range(max(0, expected - member_count)):
                try:
                    on_member_complete()
                except Exception:  # noqa: BLE001
                    pass

    return chunks


def _count_zip_members(zip_path: Path) -> int:
    """Best-effort count of top-level ZIP members for progress pre-sizing.

    Nested ZIPs are not recursed — the outer ZIP's top-level entries are
    close enough for a progress bar. Returns 0 on any read error (callers
    treat a 0-member ZIP as a single tick).
    """
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as zf:
            return sum(1 for m in zf.infolist() if not m.is_dir())
    except Exception:
        return 0


async def _process_zip_member(
    extracted_path: Path,
    extracted_content_type: str,
    attachment,
    email_doc: "Document",
    source_eml_path: str,
    file_ctx: str,
    components: "IngestComponents",
    unsupported_logger: "UnsupportedFileLogger",
    vessel_ids: list[str],
    vessel_types: list[str],
    vessel_classes: list[str],
    vprint: Callable[[str, str | None], None],
    issue_tracker: "IssueTracker | None",
    trail: "ProcessingTrail | None" = None,
    zip_filename: str | None = None,
) -> tuple["Document | None", list["Chunk"]]:
    """Process a single ZIP member.

    Returns (attach_doc, chunks). ``attach_doc`` is None when the member was
    preprocess-rejected (unsupported / classified as non-content). Callers
    run multiple invocations concurrently bounded by a semaphore, so this
    helper must not mutate shared state beyond the returned values. The only
    external side effects it performs are append-only logs (unsupported
    logger, ingest events) and per-path archive writes — both are safe
    against concurrent distinct-path callers.
    """
    result = await components.attachment_processor.preprocess(
        extracted_path, extracted_content_type, classify_images=False
    )

    if not result.should_process:
        await unsupported_logger.log_unsupported_file(
            file_path=extracted_path,
            reason=result.skip_reason or "unsupported_format",
            source_eml_path=source_eml_path,
            source_zip_path=attachment.saved_path,
            parent_document_id=email_doc.id,
        )
        return None, []

    attach_doc = components.hierarchy_manager.build_attachment_document(
        parent_doc=email_doc,
        attachment_path=extracted_path,
        content_type=extracted_content_type,
        size_bytes=extracted_path.stat().st_size,
        original_filename=extracted_path.name,
    )
    # ZIP members use "<zip_filename>/<member>" as trail key so identical
    # basenames across different ZIPs (or between top-level + nested) stay
    # distinct.
    trail_key = (
        f"{zip_filename}/{extracted_path.name}"
        if zip_filename
        else extracted_path.name
    )
    if trail is not None:
        trail.register_attachment_document(attach_doc.id, trail_key)

    attach_chunks: list["Chunk"] = []
    try:
        parsed_content = None
        if result.is_image:
            attach_chunks = await components.attachment_processor.process_document_image(
                extracted_path, attach_doc.id
            )
            if attach_chunks:
                parsed_content = attach_chunks[0].content
                if trail is not None:
                    trail.stamp_attachment(
                        trail_key,
                        STEP_VISION,
                        model=components.attachment_processor.image_processor.model_name,
                    )
        elif result.oversized_pdf and result.preview_markdown:
            # Oversized PDF inside a ZIP: same treatment as at the top level —
            # skip the full parser and feed the local peek to the decider.
            parsed_content = _apply_oversized_pdf_peek(
                result=result,
                attach_doc_id=attach_doc.id,
                file_path=extracted_path,
                file_name=extracted_path.name,
                source_eml_path=source_eml_path,
                components=components,
                trail=trail,
                trail_key=trail_key,
            )
        else:
            parsed_content, _zip_parser, _zip_model = (
                await components.attachment_processor.parse_to_text(
                    extracted_path, extracted_content_type
                )
            )
            if trail is not None:
                trail.stamp_attachment(
                    trail_key,
                    STEP_PARSE,
                    model=_zip_model,
                    parser=_zip_parser,
                )
            if not parsed_content:
                components.db.log_ingest_event(
                    document_id=attach_doc.id,
                    event_type="no_body_chunks",
                    severity="info",
                    message="ZIP-extracted attachment produced 0 chunks (no extractable text)",
                    file_path=str(extracted_path),
                    file_name=extracted_path.name,
                    source_eml_path=source_eml_path,
                )

        # Decider runs for any branch that produced parsed_content (normal
        # parse OR the oversized-PDF peek). Image path handles its own chunks.
        if not result.is_image and parsed_content:
            attach_chunks = await _decide_and_build_chunks(
                document=attach_doc,
                parsed_content=parsed_content,
                components=components,
                file_path=extracted_path,
                filename=extracted_path.name,
                source_eml_path=source_eml_path,
                vprint=None,
                file_ctx=None,
                trail=trail,
                trail_key=trail_key,
            )

        enrich_chunks_with_document_metadata(attach_chunks, attach_doc)
        _apply_vessel_metadata_to_chunks(
            attach_chunks, vessel_ids, vessel_types, vessel_classes
        )
        attach_doc.status = ProcessingStatus.COMPLETED

        # Update archive with .md file for this extracted file.
        # Must first upload the extracted original into the email's archive
        # folder — update_attachment_markdown refuses to write a preview
        # without its backing file. The email-level archive generator only
        # uploads top-level attachments (the ZIP itself), so ZIP members
        # need their own upload here.
        if components.archive_generator and parsed_content and email_doc.doc_id:
            folder_id = email_doc.doc_id[:16]
            safe_member_name = _sanitize_storage_key(extracted_path.name)
            original_archive_path = f"{folder_id}/attachments/{safe_member_name}"
            try:
                # Read + upload off the event loop so N concurrent ZIP-member
                # tasks (under Semaphore) actually overlap instead of serialising
                # behind sync file I/O on the loop thread.
                payload = await read_bytes_async(extracted_path)
                await asyncio.to_thread(
                    components.archive_generator.storage.upload_file,
                    original_archive_path,
                    payload,
                    extracted_content_type,
                )
            except Exception as e:
                vprint(f"  -> Failed to upload ZIP member original {safe_member_name}: {e}", file_ctx)

            _write_attachment_archive_md(
                components.archive_generator,
                attach_doc,
                email_doc.doc_id,
                parsed_content,
                filename=extracted_path.name,
                content_type=extracted_content_type,
                size_bytes=extracted_path.stat().st_size,
            )
    except Exception as e:
        if issue_tracker:
            await issue_tracker.track_async(file_ctx, f"{attachment.filename}/{extracted_path.name}", str(e))
        attach_doc.status = ProcessingStatus.FAILED
        attach_doc.error_message = sanitize_error_message(str(e))
        await unsupported_logger.log_unsupported_file(
            file_path=extracted_path,
            reason="extraction_failed",
            source_eml_path=source_eml_path,
            source_zip_path=attachment.saved_path,
            parent_document_id=email_doc.id,
        )

    return attach_doc, attach_chunks
