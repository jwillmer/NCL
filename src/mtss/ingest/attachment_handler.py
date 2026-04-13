"""Attachment processing for the ingest pipeline.

Handles individual attachment processing and ZIP extraction,
extracted from cli.py for better modularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..ingest.archive_generator import _sanitize_storage_key
from ..ingest.helpers import (
    apply_fallback_context,
    enrich_chunks_with_document_metadata,
    get_format_name,
    prepend_date_prefix,
)
from ..models.document import ProcessingStatus

if TYPE_CHECKING:
    from ..models.chunk import Chunk
    from ..models.document import Document
    from ..storage.unsupported_file_logger import UnsupportedFileLogger
    from .components import IngestComponents
    from .helpers import IssueTracker


def _noop_verbose(msg: str, file_ctx: str | None = None) -> None:
    """Default no-op verbose callback."""


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
        # Log skipped file
        reason = result.skip_reason or "unsupported_format"
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
            )
        )
        return chunks

    # Create attachment document for non-ZIP files
    attach_doc = await components.hierarchy_manager.create_attachment_document(
        parent_doc=email_doc,
        attachment_path=file_path,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        original_filename=attachment.filename,
        archive_file_result=archive_file_result,
    )

    try:
        parsed_content: str | None = None

        if result.is_image and result.image_description:
            # Image was already classified and described during preprocessing
            chunk = components.attachment_processor.create_image_chunk(
                file_path, attach_doc.id, result.image_description, "meaningful"
            )
            chunks.append(chunk)
            parsed_content = result.image_description
            vprint("  -> 1 chunk created (image described)", file_ctx)
            await components.db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
        elif result.is_image:
            # Image needs description (preprocessing didn't provide one)
            attach_chunks = await components.attachment_processor.process_document_image(
                file_path, attach_doc.id
            )
            chunks.extend(attach_chunks)
            if attach_chunks:
                parsed_content = attach_chunks[0].content
            vprint(f"  -> {len(attach_chunks)} chunks created (image described)", file_ctx)
            await components.db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)
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
                # Use cached content - create chunks directly
                attach_chunks = components.attachment_processor.chunker.chunk_text(
                    text=cached_content,
                    document_id=attach_doc.id,
                    source_file=str(file_path),
                    is_markdown=True,
                )
                parsed_content = cached_content
                vprint(f"  -> {len(attach_chunks)} chunks created (from cache)", file_ctx)
            else:
                # Parse with LlamaParse (or other parser)
                attach_chunks = await components.attachment_processor.process_attachment(
                    file_path, attach_doc.id, attachment.content_type
                )
                if attach_chunks:
                    # Combine all chunks for the .md file
                    parsed_content = "\n\n".join(c.content for c in attach_chunks if c.content)
                    vprint(f"  -> {len(attach_chunks)} chunks created", file_ctx)
                else:
                    vprint("  -> 0 chunks (document has no extractable text)", file_ctx)

            # Generate context summary for better search relevance
            attach_context_used = None
            if attach_chunks and components.context_generator and parsed_content:
                try:
                    attach_context = await components.context_generator.generate_context(
                        attach_doc, parsed_content[:4000]
                    )
                    if attach_context:
                        vprint(f"  -> Context generated: {len(attach_context)} chars", file_ctx)
                        attach_context_used = attach_context
                        # Apply context summary to ALL chunks
                        for chunk in attach_chunks:
                            chunk.context_summary = attach_context
                            chunk.embedding_text = components.context_generator.build_embedding_text(
                                attach_context, chunk.content
                            )
                except Exception as e:
                    vprint(f"  -> Context generation failed: {e}", file_ctx)

            # Inherit email context when no attachment-specific context was generated
            if attach_chunks and not attach_context_used and email_context_summary:
                vprint(f"  -> Inheriting email context ({len(email_context_summary)} chars)", file_ctx)
                apply_fallback_context(attach_chunks, email_context_summary, components.context_generator)

            prepend_date_prefix(attach_chunks, email_doc)

            chunks.extend(attach_chunks)
            await components.db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)

        # Update archive with .md file for this attachment
        vprint(f"  -> MD check: archive_gen={components.archive_generator is not None}, parsed_content={len(parsed_content) if parsed_content else 0} chars, doc_id={email_doc.doc_id[:16] if email_doc.doc_id else None}", file_ctx)
        if components.archive_generator and parsed_content and email_doc.doc_id:
            md_path = components.archive_generator.update_attachment_markdown(
                doc_id=email_doc.doc_id,
                filename=attachment.filename,
                content_type=attachment.content_type or "application/octet-stream",
                size_bytes=attachment.size_bytes or 0,
                parsed_content=parsed_content,
            )
            if md_path:
                vprint(f"  -> Archive updated: {md_path}", file_ctx)
                # Update document and chunks with archive URIs
                # md_path is already sanitized by _sanitize_storage_key, don't quote again
                browse_uri = f"/archive/{md_path}"
                download_path = md_path.removesuffix('.md')
                download_uri = f"/archive/{download_path}"
                attach_doc.archive_browse_uri = browse_uri
                attach_doc.archive_download_uri = download_uri
                await components.db.update_document_archive_uris(
                    attach_doc.id, browse_uri, download_uri
                )
            else:
                vprint("  -> update_attachment_markdown returned None", file_ctx)
        else:
            vprint("  -> Skipping .md creation", file_ctx)

    except Exception as e:
        if issue_tracker:
            await issue_tracker.track_async(file_ctx, attachment.filename, str(e))
        await components.db.update_document_status(attach_doc.id, ProcessingStatus.FAILED, str(e))
        await unsupported_logger.log_unsupported_file(
            file_path=file_path,
            reason="extraction_failed",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )

    # Enrich chunks with document citation metadata (source_id, source_title, archive URIs)
    enrich_chunks_with_document_metadata(chunks, attach_doc)

    # Add vessel metadata to chunk metadata for filtering
    if vessel_ids or vessel_types or vessel_classes:
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            if vessel_ids:
                chunk.metadata["vessel_ids"] = vessel_ids
            if vessel_types:
                chunk.metadata["vessel_types"] = vessel_types
            if vessel_classes:
                chunk.metadata["vessel_classes"] = vessel_classes

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
) -> list["Chunk"]:
    """Extract and process files from a ZIP attachment.

    Note: ZIP contents don't use the archive cache since they don't have
    pre-existing archive paths. They are always parsed fresh.
    """
    vprint = on_verbose or _noop_verbose  # noqa: F841 - kept for future verbose output
    chunks: list["Chunk"] = []
    vessel_ids = vessel_ids or []
    vessel_types = vessel_types or []
    vessel_classes = vessel_classes or []

    try:
        extracted_files = components.attachment_processor.extract_zip(
            Path(attachment.saved_path), lenient=lenient
        )
        for extracted_path, extracted_content_type in extracted_files:
            # Preprocess extracted file (classify_images=False - trust ZIP contents)
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
                continue

            # Create document for each extracted file
            attach_doc = await components.hierarchy_manager.create_attachment_document(
                parent_doc=email_doc,
                attachment_path=extracted_path,
                content_type=extracted_content_type,
                size_bytes=extracted_path.stat().st_size,
                original_filename=extracted_path.name,
            )

            # Process extracted file based on preprocess result
            try:
                parsed_content = None
                if result.is_image:
                    # Images from ZIPs - describe without classification
                    attach_chunks = await components.attachment_processor.process_document_image(
                        extracted_path, attach_doc.id
                    )
                    if attach_chunks:
                        parsed_content = attach_chunks[0].content
                else:
                    # Documents - use parser registry
                    attach_chunks = await components.attachment_processor.process_attachment(
                        extracted_path, attach_doc.id, extracted_content_type
                    )
                    if attach_chunks:
                        parsed_content = "\n\n".join(c.content for c in attach_chunks if c.content)
                # Enrich chunks with document citation metadata
                enrich_chunks_with_document_metadata(attach_chunks, attach_doc)
                # Add vessel metadata to chunk metadata for filtering
                if vessel_ids or vessel_types or vessel_classes:
                    for chunk in attach_chunks:
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        if vessel_ids:
                            chunk.metadata["vessel_ids"] = vessel_ids
                        if vessel_types:
                            chunk.metadata["vessel_types"] = vessel_types
                        if vessel_classes:
                            chunk.metadata["vessel_classes"] = vessel_classes
                # Inherit email context and add date prefix for ZIP-extracted chunks
                apply_fallback_context(attach_chunks, email_context_summary, components.context_generator)
                prepend_date_prefix(attach_chunks, email_doc)
                chunks.extend(attach_chunks)
                await components.db.update_document_status(attach_doc.id, ProcessingStatus.COMPLETED)

                # Update archive with .md file for this extracted file
                if components.archive_generator and parsed_content and email_doc.doc_id:
                    components.archive_generator.update_attachment_markdown(
                        doc_id=email_doc.doc_id,
                        filename=extracted_path.name,
                        content_type=extracted_content_type,
                        size_bytes=extracted_path.stat().st_size,
                        parsed_content=parsed_content,
                    )
            except Exception as e:
                if issue_tracker:
                    await issue_tracker.track_async(file_ctx, f"{attachment.filename}/{extracted_path.name}", str(e))
                await components.db.update_document_status(
                    attach_doc.id, ProcessingStatus.FAILED, str(e)
                )
                await unsupported_logger.log_unsupported_file(
                    file_path=extracted_path,
                    reason="extraction_failed",
                    source_eml_path=source_eml_path,
                    source_zip_path=attachment.saved_path,
                    parent_document_id=email_doc.id,
                )

    except Exception as e:
        if issue_tracker:
            await issue_tracker.track_async(file_ctx, attachment.filename, f"ZIP extraction failed: {e}")
        await unsupported_logger.log_unsupported_file(
            file_path=Path(attachment.saved_path),
            reason="corrupted",
            source_eml_path=source_eml_path,
            parent_document_id=email_doc.id,
        )

    return chunks
