"""Tests for the RAG query engine (search_only and retrieval)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtss.models.chunk import RetrievalResult
from mtss.rag.query_engine import RAGQueryEngine
from mtss.rag.retriever import _convert_to_retrieval_results

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536


def _make_db_row(
    *,
    content="Chunk text",
    similarity=0.9,
    file_path="/emails/test.eml",
    document_type="email",
    email_subject="Test Subject",
    email_initiator="alice@example.com",
    email_participants=None,
    email_date=None,
    heading_path=None,
    root_file_path=None,
    chunk_id="abc123def456",
    doc_id="doc-001",
    source_id="src-001",
    source_title="Test Email",
    section_path=None,
    page_number=None,
    line_from=None,
    line_to=None,
    archive_browse_uri=None,
    archive_download_uri=None,
    context_summary=None,
):
    """Build a dict that looks like a row returned by search_similar_chunks."""
    return {
        "content": content,
        "similarity": similarity,
        "file_path": file_path,
        "document_type": document_type,
        "email_subject": email_subject,
        "email_initiator": email_initiator,
        "email_participants": email_participants or ["alice@example.com", "bob@example.com"],
        "email_date": email_date,
        "heading_path": heading_path or ["Body"],
        "root_file_path": root_file_path,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_id": source_id,
        "source_title": source_title,
        "section_path": section_path,
        "page_number": page_number,
        "line_from": line_from,
        "line_to": line_to,
        "archive_browse_uri": archive_browse_uri,
        "archive_download_uri": archive_download_uri,
        "context_summary": context_summary,
    }


def _make_db_rows(n=3):
    """Return *n* distinct database rows."""
    return [
        _make_db_row(
            content=f"Chunk text {i}",
            similarity=0.9 - i * 0.05,
            chunk_id=f"chunk{i:08x}",
            file_path=f"/emails/email{i}.eml",
            email_subject=f"Subject {i}",
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.get_model = MagicMock(side_effect=lambda x: x)
    settings.rag_llm_model = "gpt-5-mini"
    settings.rerank_enabled = True
    settings.rerank_model = "cohere/rerank-v3.5"
    settings.rerank_top_n = 3
    settings.hybrid_search_enabled = True
    return settings


@pytest.fixture
def patches(mock_settings):
    """Context-manager that patches all external dependencies at once."""

    class Mocks:
        pass

    m = Mocks()

    with (
        patch("mtss.rag.query_engine.SupabaseClient") as mock_db_cls,
        patch("mtss.rag.query_engine.EmbeddingGenerator") as mock_emb_cls,
        patch("mtss.rag.query_engine.Reranker") as mock_reranker_cls,
    ):
        # SupabaseClient instance
        m.db = mock_db_cls.return_value
        m.db.search_similar_chunks = AsyncMock(return_value=_make_db_rows(3))
        m.db.close = AsyncMock()

        # EmbeddingGenerator instance
        m.embeddings = mock_emb_cls.return_value
        m.embeddings.generate_embedding = AsyncMock(return_value=FAKE_EMBEDDING)

        # Reranker instance (enabled with top_n=3 by default)
        m.reranker = mock_reranker_cls.return_value
        m.reranker.enabled = True
        m.reranker.top_n = 3
        m.reranker.rerank_results = AsyncMock(
            side_effect=lambda query, results, top_n=None: results[: top_n or 3]
        )

        m.settings = mock_settings
        yield m


# ---------------------------------------------------------------------------
# test_search_only_returns_retrieval_results
# ---------------------------------------------------------------------------


async def test_search_only_returns_retrieval_results(patches):
    """search_only() returns a list of RetrievalResult objects."""
    engine = RAGQueryEngine()
    results = await engine.search_only("test query")

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].text == "Chunk text 0"
    assert results[0].chunk_id == "chunk00000000"


# ---------------------------------------------------------------------------
# test_search_only_with_metadata_filter
# ---------------------------------------------------------------------------


async def test_search_only_with_metadata_filter(patches):
    """search_only() passes metadata_filter through to the DB client."""
    engine = RAGQueryEngine()
    custom_filter = {"vessel_ids": ["v-123"]}
    await engine.search_only("test", metadata_filter=custom_filter)

    call_kwargs = patches.db.search_similar_chunks.call_args.kwargs
    assert call_kwargs["metadata_filter"] == {"vessel_ids": ["v-123"]}


# ---------------------------------------------------------------------------
# test_convert_to_retrieval_results
# ---------------------------------------------------------------------------


async def test_convert_to_retrieval_results(patches):
    """_convert_to_retrieval_results() maps DB dicts to RetrievalResult objects."""
    rows = [
        _make_db_row(
            content="Image description",
            document_type="attachment_image",
            archive_download_uri="/archive/img.png",
            chunk_id="imgchunk0001",
        ),
        _make_db_row(
            content="Normal text",
            document_type="email",
            archive_download_uri=None,
            chunk_id="txtchunk0001",
        ),
    ]

    results = _convert_to_retrieval_results(rows)

    assert len(results) == 2

    img = results[0]
    assert isinstance(img, RetrievalResult)
    assert img.image_uri == "/archive/img.png"
    assert img.chunk_id == "imgchunk0001"
    assert img.text == "Image description"

    txt = results[1]
    assert txt.image_uri is None
    assert txt.chunk_id == "txtchunk0001"


# ---------------------------------------------------------------------------
# test_retrieval_result_serialization
# ---------------------------------------------------------------------------


def test_retrieval_result_serialization():
    """RetrievalResult.to_dict() and .from_dict() round-trip correctly."""
    original = RetrievalResult(
        text="Test content",
        score=0.85,
        chunk_id="abc123def456",
        doc_id="doc-001",
        source_id="src-001",
        source_title="Test Email",
        section_path=["Body"],
        page_number=3,
        line_from=10,
        line_to=20,
        archive_browse_uri="/archive/test.md",
        archive_download_uri="/archive/test.eml",
        email_subject="Subject",
        email_date="2024-01-15",
    )

    data = original.to_dict()
    restored = RetrievalResult.from_dict(data)

    assert restored.text == original.text
    assert restored.score == original.score
    assert restored.chunk_id == original.chunk_id
    assert restored.page_number == original.page_number
    assert restored.email_subject == original.email_subject
    assert restored.section_path == original.section_path


# ---------------------------------------------------------------------------
# test_context_summary_in_retrieval_result
# ---------------------------------------------------------------------------


async def test_context_summary_in_retrieval_result(patches):
    """context_summary should be captured from DB row into RetrievalResult."""
    rows = [
        _make_db_row(
            content="Engine failure at 14:30 UTC.",
            context_summary="Email about VLCC engine incident.",
            chunk_id="ctx_chunk_01",
        ),
    ]
    results = _convert_to_retrieval_results(rows)

    assert len(results) == 1
    assert results[0].context_summary == "Email about VLCC engine incident."


async def test_context_summary_none_when_missing(patches):
    """context_summary should be None when not in DB row."""
    rows = [_make_db_row(chunk_id="no_ctx")]
    results = _convert_to_retrieval_results(rows)

    assert results[0].context_summary is None


# ---------------------------------------------------------------------------
# test_context_summary_serialization
# ---------------------------------------------------------------------------


def test_context_summary_serialization():
    """context_summary should round-trip through to_dict/from_dict."""
    original = RetrievalResult(
        text="Test",
        score=0.9,
        chunk_id="abc",
        doc_id="doc",
        source_id="src",
        source_title="Title",
        section_path=[],
        context_summary="Document context here.",
    )
    data = original.to_dict()
    assert data["context_summary"] == "Document context here."

    restored = RetrievalResult.from_dict(data)
    assert restored.context_summary == "Document context here."


# ---------------------------------------------------------------------------
# test_hybrid_search_passes_query_text
# ---------------------------------------------------------------------------


async def test_hybrid_search_passes_query_text(patches):
    """retrieve() should pass query_text to search_similar_chunks when hybrid enabled."""
    engine = RAGQueryEngine()
    await engine.search_only("cargo damage")

    call_kwargs = patches.db.search_similar_chunks.call_args.kwargs
    assert call_kwargs.get("query_text") == "cargo damage"
