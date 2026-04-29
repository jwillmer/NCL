"""Tests for the CitationProcessor class."""

from unittest.mock import MagicMock, patch

import pytest

from mtss.models.chunk import (
    CitationValidationResult,
    RetrievalResult,
    ValidatedCitation,
)
from mtss.rag.citation_processor import CitationProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str = "aabbccddeeff",
    text: str = "Sample chunk text.",
    score: float = 0.9,
    doc_id: str = "doc001",
    source_id: str = "src001",
    source_title: str | None = "Inspection Report",
    section_path: list[str] | None = None,
    page_number: int | None = None,
    line_from: int | None = None,
    line_to: int | None = None,
    archive_browse_uri: str | None = None,
    archive_download_uri: str | None = None,
    image_uri: str | None = None,
    email_subject: str | None = None,
    email_date: str | None = None,
    email_initiator: str | None = None,
    **kwargs,
) -> RetrievalResult:
    """Helper to build a RetrievalResult with sensible defaults."""
    return RetrievalResult(
        text=text,
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_id=source_id,
        source_title=source_title,
        section_path=section_path or [],
        page_number=page_number,
        line_from=line_from,
        line_to=line_to,
        archive_browse_uri=archive_browse_uri,
        archive_download_uri=archive_download_uri,
        image_uri=image_uri,
        email_subject=email_subject,
        email_date=email_date,
        email_initiator=email_initiator,
        **kwargs,
    )


@pytest.fixture
def processor():
    """CitationProcessor with mocked ArchiveStorage."""
    with patch("mtss.rag.citation_processor.ArchiveStorage"):
        proc = CitationProcessor()
        # Default: archive verification returns True
        proc.storage.file_exists = MagicMock(return_value=True)
        return proc


# ---------------------------------------------------------------------------
# 1. test_build_citation_header — full metadata
# ---------------------------------------------------------------------------


def test_build_citation_header(processor: CitationProcessor):
    """Header includes CITE, title, email fields, page, and image URI."""
    result = _make_result(
        chunk_id="aabbccddeeff",
        source_title="MV Nordic Star Survey",
        email_subject="RE: Hull inspection",
        email_date="2024-06-15",
        email_initiator="alice@example.com",
        page_number=3,
        image_uri="/images/hull_photo.png",
    )

    header = processor._build_citation_header(result)

    assert header.startswith("[CITE:aabbccddeeff")
    assert 'title:"MV Nordic Star Survey"' in header
    assert 'subject:"RE: Hull inspection"' in header
    assert "date:2024-06-15" in header
    assert "from:alice@example.com" in header
    assert "page:3" in header
    assert "img:/images/hull_photo.png" in header
    # Enclosed in square brackets with pipe separators
    assert header.startswith("[")
    assert header.endswith("]")
    assert " | " in header


# ---------------------------------------------------------------------------
# 2. test_build_citation_header_minimal — only chunk_id
# ---------------------------------------------------------------------------


def test_build_citation_header_minimal(processor: CitationProcessor):
    """Header with no optional metadata contains only CITE field."""
    result = _make_result(
        chunk_id="112233445566",
        source_title=None,
        email_subject=None,
        email_date=None,
        email_initiator=None,
        page_number=None,
        image_uri=None,
    )

    header = processor._build_citation_header(result)

    assert header == "[CITE:112233445566]"
    # No pipe separators when there is only one field
    assert " | " not in header


# ---------------------------------------------------------------------------
# 3. test_build_context_from_results
# ---------------------------------------------------------------------------


def test_build_context_from_results(processor: CitationProcessor):
    """Context joins multiple results with --- separators and includes text."""
    results = [
        _make_result(chunk_id="aabbccddeeff", text="First chunk."),
        _make_result(chunk_id="112233445566", text="Second chunk."),
    ]

    context = processor.build_context(results)

    # Each block has header + newline + text
    assert "First chunk." in context
    assert "Second chunk." in context
    assert "CITE:aabbccddeeff" in context
    assert "CITE:112233445566" in context
    # Separated by horizontal rule
    assert "\n\n---\n\n" in context
    # Two blocks exactly
    assert context.count("---") == 1


# ---------------------------------------------------------------------------
# 3b. test_build_context_hybrid_*  (Phase-1 measurement candidate)
# ---------------------------------------------------------------------------


def test_build_context_hybrid_summary_chunk_skips_full_text(processor: CitationProcessor):
    """Summary-mode chunk emits only the [Context: ...] block; full text dropped."""
    sentinel_full_text = "RAW SENSOR DUMP " * 50  # would dominate the context
    summary_chunk = _make_result(
        chunk_id="aabbccddeeff",
        text=sentinel_full_text,
        embedding_mode="summary",
        # context_summary is the only kwarg not on _make_result's allowlist
        # so we pass it via **kwargs (RetrievalResult accepts it).
        context_summary="One-line LLM summary of 24 hours of engine sensors.",
    )

    context = processor.build_context_hybrid([summary_chunk])

    assert "[Context: One-line LLM summary of 24 hours of engine sensors.]" in context
    # Full-text MUST NOT leak into hybrid output for summary chunks.
    assert "RAW SENSOR DUMP" not in context
    # Header is still present so citations still resolve.
    assert "CITE:aabbccddeeff" in context


def test_build_context_hybrid_full_chunk_matches_build_context(processor: CitationProcessor):
    """For embedding_mode=='full', hybrid output equals the legacy builder output."""
    chunk = _make_result(
        chunk_id="112233445566",
        text="Detailed prose paragraph about hull inspection findings.",
        embedding_mode="full",
        context_summary="Hull inspection 2025-04-01.",
    )

    legacy = processor.build_context([chunk])
    hybrid = processor.build_context_hybrid([chunk])

    assert legacy == hybrid


def test_build_context_hybrid_metadata_only_chunk_matches_build_context(
    processor: CitationProcessor,
):
    """metadata_only chunks are tiny — leave them alone (lower-risk default)."""
    chunk = _make_result(
        chunk_id="ddeeff001122",
        text="filename: empty_attachment.pdf",
        embedding_mode="metadata_only",
        context_summary=None,
    )

    legacy = processor.build_context([chunk])
    hybrid = processor.build_context_hybrid([chunk])

    assert legacy == hybrid


def test_build_context_hybrid_summary_without_context_summary_falls_back(
    processor: CitationProcessor,
):
    """If embedding_mode=='summary' but context_summary is missing, retain full text.

    Defensive: skipping full text when there's no summary would leave the
    LLM with literally nothing for that chunk.
    """
    chunk = _make_result(
        chunk_id="334455667788",
        text="The only available text for this chunk.",
        embedding_mode="summary",
        context_summary=None,
    )

    hybrid = processor.build_context_hybrid([chunk])

    assert "The only available text for this chunk." in hybrid


def test_build_context_hybrid_unknown_mode_matches_build_context(
    processor: CitationProcessor,
):
    """embedding_mode=None (legacy/un-hydrated) behaves like full mode."""
    chunk = _make_result(
        chunk_id="445566778899",
        text="Body text.",
        embedding_mode=None,
        context_summary="Existing context summary.",
    )

    legacy = processor.build_context([chunk])
    hybrid = processor.build_context_hybrid([chunk])

    assert legacy == hybrid


def test_build_context_hybrid_mixed_results_savings_realised(
    processor: CitationProcessor,
):
    """Mixed batch: only summary-mode chunks are trimmed; others untouched."""
    full_chunk = _make_result(
        chunk_id="aaaaaaaaaaaa",
        text="Full prose chunk content.",
        embedding_mode="full",
    )
    summary_chunk = _make_result(
        chunk_id="bbbbbbbbbbbb",
        text="LARGE TABULAR DUMP " * 100,
        embedding_mode="summary",
        context_summary="Tabular log summary.",
    )

    hybrid = processor.build_context_hybrid([full_chunk, summary_chunk])

    assert "Full prose chunk content." in hybrid
    assert "[Context: Tabular log summary.]" in hybrid
    assert "LARGE TABULAR DUMP" not in hybrid
    # Both citation headers still present — both chunks remain citeable.
    assert "CITE:aaaaaaaaaaaa" in hybrid
    assert "CITE:bbbbbbbbbbbb" in hybrid


# ---------------------------------------------------------------------------
# 4. test_get_citation_map
# ---------------------------------------------------------------------------


def test_get_citation_map(processor: CitationProcessor):
    """Map keys are chunk_ids, values are the corresponding results."""
    r1 = _make_result(chunk_id="aabbccddeeff")
    r2 = _make_result(chunk_id="112233445566")

    cmap = processor.get_citation_map([r1, r2])

    assert set(cmap.keys()) == {"aabbccddeeff", "112233445566"}
    assert cmap["aabbccddeeff"] is r1
    assert cmap["112233445566"] is r2


# ---------------------------------------------------------------------------
# 5. test_validate_citations_valid
# ---------------------------------------------------------------------------


def test_validate_citations_valid(processor: CitationProcessor):
    """All cited chunk_ids exist in the map — no invalid citations."""
    r1 = _make_result(
        chunk_id="aabbccddeeff",
        source_title="Report A",
        page_number=2,
        line_from=10,
        line_to=20,
        archive_browse_uri="docs/report_a.md",
        archive_download_uri="docs/report_a.pdf",
    )
    r2 = _make_result(
        chunk_id="112233445566",
        source_title="Report B",
        archive_browse_uri="docs/report_b.md",
    )
    cmap = processor.get_citation_map([r1, r2])

    response = "Engine needs repair [C:aabbccddeeff]. Budget approved [C:112233445566]."
    result = processor.process_response(response, cmap)

    assert isinstance(result, CitationValidationResult)
    assert len(result.citations) == 2
    assert result.invalid_citations == []
    assert result.needs_retry is False
    # Chunk ids preserved
    cited_ids = {c.chunk_id for c in result.citations}
    assert cited_ids == {"aabbccddeeff", "112233445566"}
    # Lines tuple populated for r1
    c1 = next(c for c in result.citations if c.chunk_id == "aabbccddeeff")
    assert c1.lines == (10, 20)
    assert c1.page == 2


# ---------------------------------------------------------------------------
# 6. test_validate_citations_invalid
# ---------------------------------------------------------------------------


def test_validate_citations_invalid(processor: CitationProcessor):
    """Unknown chunk_ids are collected in invalid_citations and stripped."""
    r1 = _make_result(chunk_id="aabbccddeeff")
    cmap = processor.get_citation_map([r1])

    # "ffffffffffff" is valid hex and 12 chars but not in map
    response = "Fact A [C:aabbccddeeff]. Fact B [C:ffffffffffff]."
    result = processor.process_response(response, cmap)

    assert "ffffffffffff" in result.invalid_citations
    assert len(result.citations) == 1
    # Invalid marker removed from cleaned response
    assert "[C:ffffffffffff]" not in result.response
    # Valid marker still present
    assert "[C:aabbccddeeff]" in result.response


# ---------------------------------------------------------------------------
# 7. test_validate_citations_triggers_retry
# ---------------------------------------------------------------------------


def test_validate_citations_triggers_retry(processor: CitationProcessor):
    """>50% invalid citations triggers needs_retry=True."""
    r1 = _make_result(chunk_id="aabbccddeeff")
    cmap = processor.get_citation_map([r1])

    # 1 valid + 2 invalid = 3 total, 2/3 > 0.5
    response = (
        "A [C:aabbccddeeff]. "
        "B [C:ffffffffffff]. "
        "C [C:eeeeeeeeeeee]."
    )
    result = processor.process_response(response, cmap)

    assert result.needs_retry is True
    assert len(result.invalid_citations) == 2


# ---------------------------------------------------------------------------
# 8. test_replace_citation_markers
# ---------------------------------------------------------------------------


def test_replace_citation_markers(processor: CitationProcessor):
    """[C:chunk_id] markers replaced with <cite> HTML tags."""
    citations = [
        ValidatedCitation(
            index=1,
            chunk_id="aabbccddeeff",
            source_title="Hull Report",
            page=5,
            lines=(10, 20),
            archive_download_uri="/archive/hull.pdf",
            archive_verified=True,
        ),
    ]

    text = "The hull was inspected [C:aabbccddeeff] and passed."
    replaced = processor.replace_citation_markers(text, citations)

    assert "[C:aabbccddeeff]" not in replaced
    assert "<cite" in replaced
    assert 'id="aabbccddeeff"' in replaced
    assert 'title="Hull Report"' in replaced
    assert 'page="5"' in replaced
    assert 'lines="10-20"' in replaced
    assert 'download="hull.pdf"' in replaced
    assert ">1</cite>" in replaced


# ---------------------------------------------------------------------------
# 10. test_process_response_no_citations
# ---------------------------------------------------------------------------


def test_process_response_no_citations(processor: CitationProcessor):
    """Response without any citation markers returns empty lists."""
    cmap = {"aabbccddeeff": _make_result()}

    response = "No citations here."
    result = processor.process_response(response, cmap)

    assert result.citations == []
    assert result.invalid_citations == []
    assert result.missing_archives == []
    assert result.needs_retry is False
    assert result.response == "No citations here."
