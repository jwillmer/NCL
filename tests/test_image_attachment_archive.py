"""Regression test: image attachments must not UnboundLocalError on archive write.

History: ``_parser_name_for_archive`` / ``_parser_model_for_archive`` were
declared only inside the ``else:`` (document) branch of the non-ZIP
attachment handler. The archive write at the tail of the try block sits
outside the if/else, so image attachments that hit the vision path failed
with ``cannot access local variable '_parser_name_for_archive' where it is
not associated with a value`` — silently FAILing 106 image docs in
production. Keep this test lean (the fix is a single declaration hoist) so
the regression shape stays obvious.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mtss.ingest.attachment_handler import process_attachment
from mtss.models.chunk import Chunk
from mtss.models.document import EmbeddingMode, ParsedAttachment, ProcessingStatus


def _wire_components(*, image_description: str | None):
    """Components wired to force the image branches of process_attachment."""
    components = MagicMock()

    preprocess = MagicMock(
        should_process=True,
        is_image=True,
        is_zip=False,
        skip_reason=None,
        parser_name="image",
        image_description=image_description,
        oversized_pdf=False,
        preview_markdown=None,
        total_pages=None,
        peek_pages=None,
        content_type="image/jpeg",
    )
    components.attachment_processor = MagicMock()
    components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)
    components.attachment_processor.image_processor = MagicMock()
    components.attachment_processor.image_processor.model_name = "gemini-vision"
    components.attachment_processor.create_image_chunk = MagicMock(
        side_effect=lambda path, doc_id, desc, classification: Chunk(
            document_id=doc_id,
            content=desc,
            chunk_index=0,
            heading_path=["Image"],
        )
    )
    components.attachment_processor.process_document_image = AsyncMock(
        return_value=[
            Chunk(
                document_id="11111111-1111-1111-1111-111111111111",
                content="described image",
                chunk_index=0,
                heading_path=["Image"],
            )
        ]
    )

    components.hierarchy_manager = MagicMock()
    components.archive_generator = MagicMock()
    components.archive_generator.storage = MagicMock()
    components.archive_generator.storage.file_exists = MagicMock(return_value=False)
    components.archive_generator.update_attachment_markdown = MagicMock(return_value="archive.md")
    components.context_generator = None
    components.db = MagicMock()
    components.db.log_ingest_event = MagicMock()

    return components


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("preclassified", [True, False])
async def test_image_attachment_writes_archive_without_unboundlocal(
    tmp_path, sample_document, sample_attachment_document, preclassified
):
    """Both image branches — preclassified and post-preprocess — must reach
    the archive write. Previously this path raised UnboundLocalError."""
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0JFIF")  # minimal JPEG header

    attachment = ParsedAttachment(
        filename="photo.jpg",
        content_type="image/jpeg",
        size_bytes=image_path.stat().st_size,
        saved_path=str(image_path),
    )

    # doc_id is required for the archive write branch to trigger.
    sample_document.doc_id = "0123456789abcdef"

    components = _wire_components(
        image_description="A technical diagram." if preclassified else None,
    )
    components.hierarchy_manager.build_attachment_document = MagicMock(
        return_value=sample_attachment_document
    )

    unsupported_logger = MagicMock()
    unsupported_logger.log_unsupported_file = AsyncMock()

    chunks = await process_attachment(
        attachment=attachment,
        email_doc=sample_document,
        source_eml_path="test.eml",
        file_ctx="test.eml",
        components=components,
        unsupported_logger=unsupported_logger,
    )

    # Image path must have succeeded — not gotten redirected to the FAILED
    # extraction_failed branch that triggered when UnboundLocalError fired.
    unsupported_logger.log_unsupported_file.assert_not_called()
    assert sample_attachment_document.status == ProcessingStatus.COMPLETED
    assert len(chunks) >= 1
    # Image branches bypass the embedding decider, so the handler must stamp
    # the mode itself. Without this, image docs land in SQLite with
    # embedding_mode = NULL and downstream filters (re-embed, validate)
    # silently skip them.
    assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY


@pytest.mark.asyncio
@pytest.mark.unit
async def test_zip_image_member_stamps_embedding_mode(
    tmp_path, sample_document, sample_attachment_document
):
    """ZIP-extracted images go through the same image branch as top-level
    images and must end up with embedding_mode set. The non-ZIP and ZIP
    image branches drift historically — pin both."""
    from mtss.ingest.attachment_handler import process_zip_attachment

    image_path = tmp_path / "extracted" / "photo.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\xff\xd8\xff\xe0JFIF")

    zip_path = tmp_path / "bundle.zip"
    zip_path.write_bytes(b"PK\x03\x04fake-zip")
    attachment = ParsedAttachment(
        filename="bundle.zip",
        content_type="application/zip",
        size_bytes=zip_path.stat().st_size,
        saved_path=str(zip_path),
    )

    components = MagicMock()
    components.attachment_processor = MagicMock()
    components.attachment_processor.extract_zip = MagicMock(
        return_value=[(image_path, "image/jpeg")]
    )
    preprocess = MagicMock(
        should_process=True,
        is_image=True,
        skip_reason=None,
        oversized_pdf=False,
        preview_markdown=None,
    )
    components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)
    components.attachment_processor.image_processor = MagicMock()
    components.attachment_processor.image_processor.model_name = "gemini-vision"
    components.attachment_processor.process_document_image = AsyncMock(
        return_value=[]  # Even with zero chunks the mode must still be stamped.
    )
    components.hierarchy_manager = MagicMock()
    components.hierarchy_manager.build_attachment_document = MagicMock(
        return_value=sample_attachment_document
    )
    components.archive_generator = None
    components.context_generator = None
    components.db = MagicMock()
    components.db.log_ingest_event = MagicMock()

    sample_document.doc_id = "0123456789abcdef"

    unsupported_logger = MagicMock()
    unsupported_logger.log_unsupported_file = AsyncMock()

    collected: list = []
    await process_zip_attachment(
        attachment=attachment,
        email_doc=sample_document,
        source_eml_path="test.eml",
        file_ctx="test.eml",
        components=components,
        unsupported_logger=unsupported_logger,
        collect_docs=collected,
    )

    assert collected == [sample_attachment_document]
    assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY


@pytest.mark.asyncio
@pytest.mark.unit
async def test_empty_parse_attachment_stamps_embedding_mode(
    tmp_path, sample_document, sample_attachment_document
):
    """Document attachments whose parser returns empty content (e.g.
    image-only PDFs that local PyMuPDF can't read, .url shortcut files)
    must still receive embedding_mode = METADATA_ONLY.

    Regression: production DB had 16 such COMPLETED-with-0-chunks rows
    landing with embedding_mode = NULL. The empty-parse branch logged
    ``no_body_chunks`` and marked the doc COMPLETED but skipped the
    embedding decider (gated on ``if parsed_content:``), so the mode
    was never stamped — same shape as the image-branch bug.
    """
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake")

    attachment = ParsedAttachment(
        filename="scanned.pdf",
        content_type="application/pdf",
        size_bytes=pdf_path.stat().st_size,
        saved_path=str(pdf_path),
    )

    components = MagicMock()
    preprocess = MagicMock(
        should_process=True,
        is_image=False,
        is_zip=False,
        skip_reason=None,
        parser_name="local_pdf",
        image_description=None,
        oversized_pdf=False,
        preview_markdown=None,
        total_pages=None,
        peek_pages=None,
        content_type="application/pdf",
    )
    components.attachment_processor = MagicMock()
    components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)
    # Parser opened the file but extracted no text (image-only PDF).
    components.attachment_processor.parse_to_text = AsyncMock(
        return_value=("", "local_pdf", None)
    )

    components.hierarchy_manager = MagicMock()
    components.hierarchy_manager.build_attachment_document = MagicMock(
        return_value=sample_attachment_document
    )
    components.archive_generator = None
    components.context_generator = None
    components.db = MagicMock()
    components.db.log_ingest_event = MagicMock()

    sample_document.doc_id = "0123456789abcdef"

    unsupported_logger = MagicMock()
    unsupported_logger.log_unsupported_file = AsyncMock()

    chunks = await process_attachment(
        attachment=attachment,
        email_doc=sample_document,
        source_eml_path="test.eml",
        file_ctx="test.eml",
        components=components,
        unsupported_logger=unsupported_logger,
    )

    assert chunks == []
    assert sample_attachment_document.status == ProcessingStatus.COMPLETED
    assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY


@pytest.mark.asyncio
@pytest.mark.unit
async def test_empty_parse_zip_member_stamps_embedding_mode(
    tmp_path, sample_document, sample_attachment_document
):
    """ZIP-member parity for the empty-parse fix. Both top-level and
    ZIP-member empty-parse paths must stamp METADATA_ONLY."""
    from mtss.ingest.attachment_handler import process_zip_attachment

    extracted = tmp_path / "extracted" / "scanned.pdf"
    extracted.parent.mkdir(parents=True, exist_ok=True)
    extracted.write_bytes(b"%PDF-1.4\nfake")

    zip_path = tmp_path / "bundle.zip"
    zip_path.write_bytes(b"PK\x03\x04fake-zip")
    attachment = ParsedAttachment(
        filename="bundle.zip",
        content_type="application/zip",
        size_bytes=zip_path.stat().st_size,
        saved_path=str(zip_path),
    )

    components = MagicMock()
    components.attachment_processor = MagicMock()
    components.attachment_processor.extract_zip = MagicMock(
        return_value=[(extracted, "application/pdf")]
    )
    preprocess = MagicMock(
        should_process=True,
        is_image=False,
        skip_reason=None,
        oversized_pdf=False,
        preview_markdown=None,
    )
    components.attachment_processor.preprocess = AsyncMock(return_value=preprocess)
    components.attachment_processor.parse_to_text = AsyncMock(
        return_value=("", "local_pdf", None)
    )
    components.hierarchy_manager = MagicMock()
    components.hierarchy_manager.build_attachment_document = MagicMock(
        return_value=sample_attachment_document
    )
    components.archive_generator = None
    components.context_generator = None
    components.db = MagicMock()
    components.db.log_ingest_event = MagicMock()

    sample_document.doc_id = "0123456789abcdef"

    unsupported_logger = MagicMock()
    unsupported_logger.log_unsupported_file = AsyncMock()

    collected: list = []
    await process_zip_attachment(
        attachment=attachment,
        email_doc=sample_document,
        source_eml_path="test.eml",
        file_ctx="test.eml",
        components=components,
        unsupported_logger=unsupported_logger,
        collect_docs=collected,
    )

    assert collected == [sample_attachment_document]
    assert sample_attachment_document.status == ProcessingStatus.COMPLETED
    assert sample_attachment_document.embedding_mode == EmbeddingMode.METADATA_ONLY
