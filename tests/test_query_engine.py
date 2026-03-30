"""Characterization tests for the RAG query engine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtss.models.chunk import (
    CitationValidationResult,
    EnhancedRAGResponse,
    RetrievalResult,
    ValidatedCitation,
)
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
    settings.rag_llm_model = "gpt-4o"
    settings.chunk_display_max_chars = 2000
    settings.rerank_enabled = True
    settings.rerank_model = "cohere/rerank-english-v3.0"
    settings.rerank_top_n = 3
    return settings


@pytest.fixture
def patches(mock_settings):
    """Context-manager that patches all external dependencies at once."""

    class Mocks:
        pass

    m = Mocks()

    with (
        patch("mtss.rag.query_engine.get_settings", return_value=mock_settings),
        patch("mtss.rag.query_engine.SupabaseClient") as mock_db_cls,
        patch("mtss.rag.query_engine.EmbeddingGenerator") as mock_emb_cls,
        patch("mtss.rag.query_engine.Reranker") as mock_reranker_cls,
        patch("mtss.rag.query_engine.CitationProcessor") as mock_citation_cls,
        patch("mtss.rag.query_engine.get_langfuse_metadata", return_value={}),
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
        # By default, rerank_results returns its input sliced to top_n
        m.reranker.rerank_results = MagicMock(
            side_effect=lambda query, results, top_n=None: results[: top_n or 3]
        )

        # CitationProcessor instance
        m.citation_proc = mock_citation_cls.return_value
        m.citation_proc.get_citation_map = MagicMock(return_value={"chunk00000000": MagicMock()})
        m.citation_proc.build_context = MagicMock(return_value="built-context")
        m.citation_proc.get_system_prompt = MagicMock(return_value="system-prompt")
        m.citation_proc.process_response = MagicMock(
            return_value=CitationValidationResult(
                response="Answer with citations",
                citations=[
                    ValidatedCitation(index=1, chunk_id="chunk00000000", source_title="Test Email"),
                ],
                invalid_citations=[],
                needs_retry=False,
            )
        )
        m.citation_proc.replace_citation_markers = MagicMock(return_value="Final answer [1]")
        m.citation_proc.format_sources_section = MagicMock(return_value="\n\nSources:\n[1] Test Email")
        m.citation_proc.build_retry_hint = MagicMock(return_value="\nRetry hint")

        m.settings = mock_settings
        yield m


# ---------------------------------------------------------------------------
# 1. test_query_returns_response
# ---------------------------------------------------------------------------


async def test_query_returns_response(patches):
    """query() returns an EnhancedRAGResponse with answer and citations."""
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="The answer is 42."))]
        )
        engine = RAGQueryEngine()
        response = await engine.query("What is the answer?")

    assert isinstance(response, EnhancedRAGResponse)
    assert response.query == "What is the answer?"
    assert "Final answer" in response.answer
    assert len(response.citations) > 0


# ---------------------------------------------------------------------------
# 2. test_query_no_results
# ---------------------------------------------------------------------------


async def test_query_no_results(patches):
    """query() returns a canned 'no information' answer when DB returns nothing."""
    patches.db.search_similar_chunks = AsyncMock(return_value=[])

    engine = RAGQueryEngine()
    response = await engine.query("Something obscure?")

    assert "couldn't find any relevant information" in response.answer
    assert response.citations == []
    assert response.query == "Something obscure?"


# ---------------------------------------------------------------------------
# 3. test_query_with_reranking
# ---------------------------------------------------------------------------


async def test_query_with_reranking(patches):
    """Reranker is called when enabled and result count exceeds effective_top_n."""
    patches.db.search_similar_chunks = AsyncMock(return_value=_make_db_rows(5))

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Reranked answer."))]
        )
        engine = RAGQueryEngine()
        await engine.query("budget?", use_rerank=True)

    patches.reranker.rerank_results.assert_called_once()


# ---------------------------------------------------------------------------
# 4. test_query_skips_reranking_when_disabled
# ---------------------------------------------------------------------------


async def test_query_skips_reranking_when_disabled(patches):
    """Reranker is bypassed when use_rerank=False."""
    patches.db.search_similar_chunks = AsyncMock(return_value=_make_db_rows(5))

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="No rerank."))]
        )
        engine = RAGQueryEngine()
        await engine.query("budget?", use_rerank=False)

    patches.reranker.rerank_results.assert_not_called()


# ---------------------------------------------------------------------------
# 5. test_query_with_citations_returns_enhanced_response
# ---------------------------------------------------------------------------


async def test_query_with_citations_returns_enhanced_response(patches):
    """query_with_citations() returns an EnhancedRAGResponse with citations."""
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Answer [C:chunk00000000]."))]
        )
        engine = RAGQueryEngine()
        response = await engine.query_with_citations("What happened?")

    assert isinstance(response, EnhancedRAGResponse)
    assert response.query == "What happened?"
    assert "Final answer" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "chunk00000000"
    assert response.retry_count == 0


# ---------------------------------------------------------------------------
# 6. test_query_with_citations_no_results
# ---------------------------------------------------------------------------


async def test_query_with_citations_no_results(patches):
    """query_with_citations() returns canned answer when DB is empty."""
    patches.db.search_similar_chunks = AsyncMock(return_value=[])

    engine = RAGQueryEngine()
    response = await engine.query_with_citations("Anything?")

    assert isinstance(response, EnhancedRAGResponse)
    assert "couldn't find any relevant information" in response.answer
    assert response.citations == []


# ---------------------------------------------------------------------------
# 7. test_search_only_returns_retrieval_results
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
# 8. test_search_only_with_metadata_filter
# ---------------------------------------------------------------------------


async def test_search_only_with_metadata_filter(patches):
    """search_only() passes metadata_filter through to the DB client."""
    engine = RAGQueryEngine()
    custom_filter = {"vessel_ids": ["v-123"]}
    await engine.search_only("test", metadata_filter=custom_filter)

    call_kwargs = patches.db.search_similar_chunks.call_args.kwargs
    assert call_kwargs["metadata_filter"] == {"vessel_ids": ["v-123"]}


# ---------------------------------------------------------------------------
# 9. test_context_header_format
# ---------------------------------------------------------------------------


async def test_context_header_format(patches):
    """_build_context_header() produces the expected bracket-delimited format."""
    engine = RAGQueryEngine()

    result = RetrievalResult(
        text="Some text",
        score=0.9,
        chunk_id="abc123def456",
        doc_id="doc-001",
        source_id="src-001",
        source_title="Budget Review",
        section_path=["Overview", "Q2"],
        file_path="/emails/budget.eml",
        email_subject="Budget Review",
        email_participants=["alice@example.com", "bob@example.com", "carol@example.com", "dave@example.com"],
    )

    header = engine._build_context_header(result)

    assert header.startswith("[Source: ")
    assert header.endswith("]")
    assert "Email: Budget Review" in header
    assert "Participants: alice@example.com, bob@example.com, carol@example.com (+1 more)" in header
    assert "File: /emails/budget.eml" in header
    assert "Section: Overview > Q2" in header


# ---------------------------------------------------------------------------
# 10. test_generate_answer_prompt_structure
# ---------------------------------------------------------------------------


async def test_generate_answer_prompt_structure(patches):
    """_generate_answer_with_citations() calls litellm.completion with system + user messages."""
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="LLM says hello."))]
        )
        engine = RAGQueryEngine()
        result = await engine._generate_answer_with_citations("my question", "my context")

    assert result == "LLM says hello."

    call_kwargs = mock_completion.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "my question" in messages[1]["content"]
    assert "my context" in messages[1]["content"]


# ---------------------------------------------------------------------------
# 11. test_convert_to_retrieval_results
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
# 12. test_retrieval_result_serialization
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
