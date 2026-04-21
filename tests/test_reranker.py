"""Tests for the Reranker class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtss.models.chunk import RetrievalResult
from mtss.rag.reranker import Reranker


@pytest.fixture
def sample_results():
    """Create sample retrieval results for testing."""
    return [
        RetrievalResult(
            text="The project deadline has been moved to next Friday.",
            score=0.85,
            chunk_id="aaa111bbb222",
            doc_id="doc001",
            source_id="email1.eml",
            source_title="Project Update",
            section_path=["Email Body"],
            file_path="/path/to/email1.eml",
            document_type="email",
            email_subject="Project Update",
            email_initiator="alice@example.com",
            email_participants=["alice@example.com", "bob@example.com"],
            email_date="2024-01-15",
        ),
        RetrievalResult(
            text="We discussed the budget allocation for Q2.",
            score=0.82,
            chunk_id="bbb222ccc333",
            doc_id="doc002",
            source_id="email2.eml",
            source_title="Meeting Notes",
            section_path=["Email Body"],
            file_path="/path/to/email2.eml",
            document_type="email",
            email_subject="Meeting Notes",
            email_initiator="bob@example.com",
            email_participants=["bob@example.com", "carol@example.com"],
            email_date="2024-01-14",
        ),
        RetrievalResult(
            text="Budget breakdown: Marketing $50k, Engineering $100k.",
            score=0.78,
            chunk_id="ccc333ddd444",
            doc_id="doc003",
            source_id="attachment.pdf",
            source_title="Budget Report",
            section_path=["Budget Overview"],
            file_path="/path/to/attachment.pdf",
            document_type="attachment_pdf",
            email_subject="Project Update",
            email_initiator="alice@example.com",
            email_participants=["alice@example.com", "bob@example.com"],
            email_date="2024-01-15",
        ),
        RetrievalResult(
            text="Action items from last week have been completed.",
            score=0.75,
            chunk_id="ddd444eee555",
            doc_id="doc004",
            source_id="email3.eml",
            source_title="Weekly Standup",
            section_path=["Email Body"],
            file_path="/path/to/email3.eml",
            document_type="email",
            email_subject="Weekly Standup",
            email_initiator="carol@example.com",
            email_participants=["carol@example.com", "dave@example.com"],
            email_date="2024-01-13",
        ),
        RetrievalResult(
            text="Please submit your timesheets by EOD.",
            score=0.70,
            chunk_id="eee555fff666",
            doc_id="doc005",
            source_id="email4.eml",
            source_title="Reminder",
            section_path=["Email Body"],
            file_path="/path/to/email4.eml",
            document_type="email",
            email_subject="Reminder",
            email_initiator="dave@example.com",
            email_participants=["dave@example.com"],
            email_date="2024-01-12",
        ),
    ]


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.rerank_enabled = True
    settings.rerank_model = "cohere/rerank-4-fast"
    settings.rerank_top_n = 3
    settings.rerank_score_floor = 0.2
    settings.openrouter_api_key = "test-openrouter-key"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    return settings


def _mock_async_client(results_data):
    """Create a mock httpx.AsyncClient that returns rerank results."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": results_data}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.fixture
def mock_rerank_response_data():
    """Mock response data from OpenRouter rerank API."""
    # Simulate reranker reordering: originally [0,1,2,3,4], reranked to [2,0,1]
    return [
        {"index": 2, "relevance_score": 0.95},  # attachment.pdf - highest
        {"index": 0, "relevance_score": 0.88},  # email1.eml - second
        {"index": 1, "relevance_score": 0.72},  # email2.eml - third
    ]


class TestRerankerInit:
    """Tests for Reranker initialization."""

    def test_init_with_enabled_reranking(self, mock_settings):
        """Test initialization with reranking enabled."""
        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()

            assert reranker.enabled is True
            assert reranker.model == "cohere/rerank-4-fast"
            assert reranker.top_n == 3

    def test_init_with_disabled_reranking(self, mock_settings):
        """Test initialization with reranking disabled."""
        mock_settings.rerank_enabled = False

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()

            assert reranker.enabled is False

    def test_init_stores_api_credentials(self, mock_settings):
        """Test that Reranker stores OpenRouter API credentials from settings."""
        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            assert reranker._api_key == "test-openrouter-key"
            assert reranker._base_url == "https://openrouter.ai/api/v1"


class TestRerankerRerankResults:
    """Tests for Reranker.rerank_results method."""

    @pytest.mark.asyncio
    async def test_rerank_results_success(
        self, mock_settings, sample_results, mock_rerank_response_data
    ):
        """Test successful reranking of results."""
        mock_client = _mock_async_client(mock_rerank_response_data)

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                results = await reranker.rerank_results(
                    query="What is the project budget?",
                    results=sample_results,
                )

                assert len(results) == 3
                assert results[0].file_path == "/path/to/attachment.pdf"
                assert results[1].file_path == "/path/to/email1.eml"
                assert results[2].file_path == "/path/to/email2.eml"
                assert results[0].rerank_score == 0.95
                assert results[1].rerank_score == 0.88
                assert results[2].rerank_score == 0.72

    @pytest.mark.asyncio
    async def test_rerank_results_with_custom_top_n(
        self, mock_settings, sample_results, mock_rerank_response_data
    ):
        """Test reranking with custom top_n parameter."""
        mock_client = _mock_async_client(mock_rerank_response_data[:2])

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                results = await reranker.rerank_results(
                    query="What is the project budget?",
                    results=sample_results,
                    top_n=2,
                )

                assert len(results) == 2

    @pytest.mark.asyncio
    async def test_rerank_results_disabled(self, mock_settings, sample_results):
        """Test that disabled reranker returns truncated original results."""
        mock_settings.rerank_enabled = False

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = await reranker.rerank_results(
                query="What is the project budget?",
                results=sample_results,
            )

            assert len(results) == 3
            assert results[0].file_path == "/path/to/email1.eml"
            assert results[1].file_path == "/path/to/email2.eml"
            assert results[2].file_path == "/path/to/attachment.pdf"
            assert results[0].rerank_score is None

    @pytest.mark.asyncio
    async def test_rerank_results_empty_sources(self, mock_settings):
        """Test reranking with empty source list."""
        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = await reranker.rerank_results(
                query="What is the project budget?",
                results=[],
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_rerank_results_fewer_than_top_n(self, mock_settings, sample_results):
        """Test reranking when sources fewer than top_n."""
        mock_settings.rerank_top_n = 10

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = await reranker.rerank_results(
                query="What is the project budget?",
                results=sample_results[:2],
            )

            assert len(results) == 2
            assert results[0].rerank_score is None

    @pytest.mark.asyncio
    async def test_rerank_results_extracts_text(
        self, mock_settings, sample_results, mock_rerank_response_data
    ):
        """Test that reranker extracts enriched text for documents."""
        mock_client = _mock_async_client(mock_rerank_response_data)

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                await reranker.rerank_results(
                    query="What is the project budget?",
                    results=sample_results,
                )

                # Verify the async client's post was called with enriched documents
                call_kwargs = mock_client.post.call_args.kwargs
                json_body = call_kwargs["json"]
                assert json_body["query"] == "What is the project budget?"
                assert len(json_body["documents"]) == 5
                assert json_body["documents"][0].startswith("Project Update\n")
                assert "The project deadline has been moved to next Friday." in json_body["documents"][0]
                assert "Project Update | Budget Report" in json_body["documents"][2]

    @pytest.mark.asyncio
    async def test_rerank_score_floor(self, mock_settings, sample_results):
        """Test that low-scoring results are filtered out."""
        results_data = [
            {"index": 0, "relevance_score": 0.05},
            {"index": 1, "relevance_score": 0.45},
            {"index": 2, "relevance_score": 0.10},
        ]
        mock_client = _mock_async_client(results_data)

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                results = await reranker.rerank_results(
                    query="test", results=sample_results,
                )
                assert len(results) == 1
                assert results[0].rerank_score == 0.45

    @pytest.mark.asyncio
    async def test_rerank_score_floor_all_below_keeps_one(self, mock_settings, sample_results):
        """Test that at least one result is kept even when all scores are below floor."""
        results_data = [
            {"index": 0, "relevance_score": 0.05},
            {"index": 1, "relevance_score": 0.10},
            {"index": 2, "relevance_score": 0.15},
        ]
        mock_client = _mock_async_client(results_data)

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                results = await reranker.rerank_results(
                    query="test", results=sample_results,
                )
                assert len(results) == 1
                assert results[0].rerank_score == 0.05


class TestRerankerIntegration:
    """Integration-style tests for Reranker."""

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(
        self, mock_settings, sample_results, mock_rerank_response_data
    ):
        """Test that reranking preserves all result metadata."""
        mock_client = _mock_async_client(mock_rerank_response_data)

        with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
            with patch("mtss.rag.reranker.httpx.AsyncClient", return_value=mock_client):
                reranker = Reranker()
                results = await reranker.rerank_results(
                    query="What is the project budget?",
                    results=sample_results,
                )

                top_result = results[0]
                assert top_result.document_type == "attachment_pdf"
                assert top_result.email_subject == "Project Update"
                assert top_result.email_initiator == "alice@example.com"
                assert "alice@example.com" in top_result.email_participants
                assert top_result.email_date == "2024-01-15"
                assert top_result.section_path == ["Budget Overview"]
                assert top_result.score == 0.78
                assert top_result.rerank_score == 0.95
