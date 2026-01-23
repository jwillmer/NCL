"""Tests for the Reranker class."""

from unittest.mock import MagicMock, patch

import pytest

from mtss.models.chunk import SourceReference
from mtss.processing.reranker import Reranker


@pytest.fixture
def sample_sources():
    """Create sample source references for testing."""
    return [
        SourceReference(
            file_path="/path/to/email1.eml",
            document_type="email",
            email_subject="Project Update",
            email_initiator="alice@example.com",
            email_participants=["alice@example.com", "bob@example.com"],
            email_date="2024-01-15",
            chunk_content="The project deadline has been moved to next Friday.",
            similarity_score=0.85,
            heading_path=["Email Body"],
        ),
        SourceReference(
            file_path="/path/to/email2.eml",
            document_type="email",
            email_subject="Meeting Notes",
            email_initiator="bob@example.com",
            email_participants=["bob@example.com", "carol@example.com"],
            email_date="2024-01-14",
            chunk_content="We discussed the budget allocation for Q2.",
            similarity_score=0.82,
            heading_path=["Email Body"],
        ),
        SourceReference(
            file_path="/path/to/attachment.pdf",
            document_type="attachment_pdf",
            email_subject="Project Update",
            email_initiator="alice@example.com",
            email_participants=["alice@example.com", "bob@example.com"],
            email_date="2024-01-15",
            chunk_content="Budget breakdown: Marketing $50k, Engineering $100k.",
            similarity_score=0.78,
            heading_path=["Budget Overview"],
        ),
        SourceReference(
            file_path="/path/to/email3.eml",
            document_type="email",
            email_subject="Weekly Standup",
            email_initiator="carol@example.com",
            email_participants=["carol@example.com", "dave@example.com"],
            email_date="2024-01-13",
            chunk_content="Action items from last week have been completed.",
            similarity_score=0.75,
            heading_path=["Email Body"],
        ),
        SourceReference(
            file_path="/path/to/email4.eml",
            document_type="email",
            email_subject="Reminder",
            email_initiator="dave@example.com",
            email_participants=["dave@example.com"],
            email_date="2024-01-12",
            chunk_content="Please submit your timesheets by EOD.",
            similarity_score=0.70,
            heading_path=["Email Body"],
        ),
    ]


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.rerank_enabled = True
    settings.rerank_model = "cohere/rerank-english-v3.0"
    settings.rerank_top_n = 3
    settings.cohere_api_key = "test-api-key"
    return settings


@pytest.fixture
def mock_rerank_response():
    """Mock response from LiteLLM rerank."""
    response = MagicMock()
    # Simulate reranker reordering: originally [0,1,2,3,4], reranked to [2,0,1]
    result1 = MagicMock()
    result1.index = 2  # attachment.pdf - highest rerank score
    result1.relevance_score = 0.95

    result2 = MagicMock()
    result2.index = 0  # email1.eml - second highest
    result2.relevance_score = 0.88

    result3 = MagicMock()
    result3.index = 1  # email2.eml - third highest
    result3.relevance_score = 0.72

    response.results = [result1, result2, result3]
    return response


class TestRerankerInit:
    """Tests for Reranker initialization."""

    def test_init_with_enabled_reranking(self, mock_settings):
        """Test initialization with reranking enabled."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch.dict("os.environ", {}, clear=False):
                reranker = Reranker()

                assert reranker.enabled is True
                assert reranker.model == "cohere/rerank-english-v3.0"
                assert reranker.top_n == 3

    def test_init_with_disabled_reranking(self, mock_settings):
        """Test initialization with reranking disabled."""
        mock_settings.rerank_enabled = False

        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()

            assert reranker.enabled is False

    def test_init_does_not_set_cohere_api_key(self, mock_settings):
        """Test that Reranker no longer sets API key (done at module init)."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch.dict("os.environ", {}, clear=False) as env:
                # Clear any existing key
                env.pop("COHERE_API_KEY", None)
                Reranker()
                # Reranker should NOT set the key - it's done in mtss.__init__
                assert "COHERE_API_KEY" not in env or env.get("COHERE_API_KEY") != "test-api-key"

    def test_init_without_cohere_api_key(self, mock_settings):
        """Test initialization without Cohere API key."""
        mock_settings.cohere_api_key = None

        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch.dict("os.environ", {"COHERE_API_KEY": ""}, clear=False):
                # Should not raise an error
                reranker = Reranker()
                assert reranker.enabled is True


class TestRerankerRerankResults:
    """Tests for Reranker.rerank_results method."""

    def test_rerank_results_success(
        self, mock_settings, sample_sources, mock_rerank_response
    ):
        """Test successful reranking of results."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch(
                "mtss.processing.reranker.rerank", return_value=mock_rerank_response
            ):
                reranker = Reranker()
                results = reranker.rerank_results(
                    query="What is the project budget?",
                    sources=sample_sources,
                )

                # Should return top_n results (3)
                assert len(results) == 3

                # Results should be reordered by rerank score
                assert results[0].file_path == "/path/to/attachment.pdf"
                assert results[1].file_path == "/path/to/email1.eml"
                assert results[2].file_path == "/path/to/email2.eml"

                # Rerank scores should be populated
                assert results[0].rerank_score == 0.95
                assert results[1].rerank_score == 0.88
                assert results[2].rerank_score == 0.72

    def test_rerank_results_with_custom_top_n(
        self, mock_settings, sample_sources, mock_rerank_response
    ):
        """Test reranking with custom top_n parameter."""
        # Modify mock response to return 2 results
        mock_rerank_response.results = mock_rerank_response.results[:2]

        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch(
                "mtss.processing.reranker.rerank", return_value=mock_rerank_response
            ):
                reranker = Reranker()
                results = reranker.rerank_results(
                    query="What is the project budget?",
                    sources=sample_sources,
                    top_n=2,
                )

                assert len(results) == 2

    def test_rerank_results_disabled(self, mock_settings, sample_sources):
        """Test that disabled reranker returns truncated original results."""
        mock_settings.rerank_enabled = False

        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = reranker.rerank_results(
                query="What is the project budget?",
                sources=sample_sources,
            )

            # Should return first top_n results without reranking
            assert len(results) == 3
            assert results[0].file_path == "/path/to/email1.eml"
            assert results[1].file_path == "/path/to/email2.eml"
            assert results[2].file_path == "/path/to/attachment.pdf"

            # Rerank scores should not be set
            assert results[0].rerank_score is None

    def test_rerank_results_empty_sources(self, mock_settings):
        """Test reranking with empty source list."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = reranker.rerank_results(
                query="What is the project budget?",
                sources=[],
            )

            assert results == []

    def test_rerank_results_fewer_than_top_n(self, mock_settings, sample_sources):
        """Test reranking when sources fewer than top_n."""
        mock_settings.rerank_top_n = 10  # More than available sources

        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            reranker = Reranker()
            results = reranker.rerank_results(
                query="What is the project budget?",
                sources=sample_sources[:2],  # Only 2 sources
            )

            # Should return all sources without calling rerank API
            assert len(results) == 2
            assert results[0].rerank_score is None

    def test_rerank_results_extracts_chunk_content(
        self, mock_settings, sample_sources, mock_rerank_response
    ):
        """Test that reranker extracts chunk_content for documents."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch(
                "mtss.processing.reranker.rerank", return_value=mock_rerank_response
            ) as mock_rerank_fn:
                reranker = Reranker()
                reranker.rerank_results(
                    query="What is the project budget?",
                    sources=sample_sources,
                )

                # Verify rerank was called with correct documents
                call_kwargs = mock_rerank_fn.call_args.kwargs
                assert call_kwargs["query"] == "What is the project budget?"
                assert len(call_kwargs["documents"]) == 5
                assert (
                    call_kwargs["documents"][0]
                    == "The project deadline has been moved to next Friday."
                )


class TestRerankerIntegration:
    """Integration-style tests for Reranker."""

    def test_rerank_preserves_source_metadata(
        self, mock_settings, sample_sources, mock_rerank_response
    ):
        """Test that reranking preserves all source metadata."""
        with patch("mtss.processing.reranker.get_settings", return_value=mock_settings):
            with patch(
                "mtss.processing.reranker.rerank", return_value=mock_rerank_response
            ):
                reranker = Reranker()
                results = reranker.rerank_results(
                    query="What is the project budget?",
                    sources=sample_sources,
                )

                # Check that the top result (attachment.pdf) has all metadata
                top_result = results[0]
                assert top_result.document_type == "attachment_pdf"
                assert top_result.email_subject == "Project Update"
                assert top_result.email_initiator == "alice@example.com"
                assert "alice@example.com" in top_result.email_participants
                assert top_result.email_date == "2024-01-15"
                assert top_result.heading_path == ["Budget Overview"]
                assert top_result.similarity_score == 0.78  # Original score preserved
                assert top_result.rerank_score == 0.95  # New rerank score added
