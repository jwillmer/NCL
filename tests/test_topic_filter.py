"""Tests for topic-based pre-filtering."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from mtss.models.topic import ExtractedTopic, Topic, TopicSummary
from mtss.rag.topic_filter import (
    MatchedTopic,
    TopicFilter,
    TopicFilterResult,
    TopicMessages,
)


class TestTopicMessages:
    """Tests for user-facing message formatting."""

    def test_format_topics_single(self):
        """Should format single topic."""
        result = TopicMessages._format_topics(["Cargo Damage"])
        assert result == "'Cargo Damage'"

    def test_format_topics_two(self):
        """Should format two topics with 'and'."""
        result = TopicMessages._format_topics(["Cargo Damage", "Engine Issues"])
        assert result == "'Cargo Damage' and 'Engine Issues'"

    def test_format_topics_three(self):
        """Should format three topics with comma and 'and'."""
        result = TopicMessages._format_topics(["A", "B", "C"])
        assert result == "'A', 'B', and 'C'"

    def test_no_topic_match_single(self):
        """Should generate message for single unmatched topic."""
        msg = TopicMessages.no_topic_match(["Cargo Damage"])
        assert "Cargo Damage" in msg
        assert "this category isn't" in msg

    def test_no_topic_match_multiple(self):
        """Should generate message for multiple unmatched topics."""
        msg = TopicMessages.no_topic_match(["Cargo Damage", "Engine Issues"])
        assert "Cargo Damage" in msg
        assert "Engine Issues" in msg
        assert "these categories aren't" in msg

    def test_partial_topic_match(self):
        """Should generate message when some topics match."""
        msg = TopicMessages.partial_topic_match(
            matched=["Cargo Damage"],
            unmatched=["Unknown Topic"],
            count=5,
        )
        assert "Found 5 documents" in msg
        assert "Cargo Damage" in msg
        assert "Unknown Topic" in msg

    def test_empty_topics_with_suggestions(self):
        """Should generate message with related topic suggestions."""
        msg = TopicMessages.empty_topics(
            topic_names=["Cargo Damage"],
            suggested=["Hull Damage", "Ballast Systems"],
        )
        assert "Cargo Damage" in msg
        assert "don't have any documents" in msg
        assert "Hull Damage" in msg
        assert "Ballast Systems" in msg

    def test_no_vessel_match(self):
        """Should generate message when vessel filter excludes all results."""
        msg = TopicMessages.no_vessel_match(
            topic_names=["Cargo Damage"],
            topic_count=10,
            vessel_desc="vessel type VLCC",
        )
        assert "10 records" in msg
        assert "Cargo Damage" in msg
        assert "vessel type VLCC" in msg

    def test_topic_context(self):
        """Should generate context message for searching."""
        msg = TopicMessages.topic_context(["Engine Issues"], 25)
        assert "25 documents" in msg
        assert "Engine Issues" in msg


class TestTopicFilterResult:
    """Tests for TopicFilterResult dataclass."""

    def test_matched_topic_ids(self):
        """Should return list of matched topic IDs."""
        id1, id2 = uuid4(), uuid4()
        result = TopicFilterResult(
            matched_topics=[
                MatchedTopic(id=id1, name="a", display_name="A", chunk_count=1, filtered_count=1),
                MatchedTopic(id=id2, name="b", display_name="B", chunk_count=2, filtered_count=2),
            ]
        )
        assert result.matched_topic_ids == [id1, id2]

    def test_matched_topic_names(self):
        """Should return list of display names."""
        result = TopicFilterResult(
            matched_topics=[
                MatchedTopic(id=uuid4(), name="a", display_name="A Topic", chunk_count=1, filtered_count=1),
                MatchedTopic(id=uuid4(), name="b", display_name="B Topic", chunk_count=2, filtered_count=2),
            ]
        )
        assert result.matched_topic_names == ["A Topic", "B Topic"]


class TestTopicFilter:
    """Tests for TopicFilter."""

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock topic extractor."""
        extractor = AsyncMock()
        extractor.extract_topics_from_query = AsyncMock(return_value=[])
        return extractor

    @pytest.fixture
    def mock_matcher(self):
        """Create a mock topic matcher."""
        matcher = AsyncMock()
        matcher.find_topics_by_names = AsyncMock(return_value=[])
        matcher.find_related_topics = AsyncMock(return_value=[])
        return matcher

    @pytest.fixture
    def mock_db(self):
        """Create a mock database client."""
        db = AsyncMock()
        db.get_chunks_count_for_topic = AsyncMock(return_value=0)
        return db

    @pytest.fixture
    def topic_filter(self, mock_extractor, mock_matcher, mock_db):
        """Create a TopicFilter instance."""
        return TopicFilter(mock_extractor, mock_matcher, mock_db)

    @pytest.mark.asyncio
    async def test_no_topics_extracted(self, topic_filter):
        """Should return empty result for general queries."""
        result = await topic_filter.analyze_query("Hello there")

        assert result.detected_topics == []
        assert result.matched_topics == []
        assert result.should_skip_rag is False

    @pytest.mark.asyncio
    async def test_topic_extracted_but_no_match(
        self, topic_filter, mock_extractor, mock_matcher
    ):
        """Should return broad search message when topic not found."""
        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage")
        ]
        mock_matcher.find_topics_by_names.return_value = [("Cargo Damage", None)]

        result = await topic_filter.analyze_query("Cargo damage incidents?")

        assert result.detected_topics == ["Cargo Damage"]
        assert result.unmatched_topics == ["Cargo Damage"]
        assert result.should_skip_rag is False
        assert "Cargo Damage" in result.message

    @pytest.mark.asyncio
    async def test_topic_matched_with_results(
        self, topic_filter, mock_extractor, mock_matcher, mock_db
    ):
        """Should proceed when matched topic has results."""
        topic_id = uuid4()
        topic = Topic(
            id=topic_id,
            name="cargo damage",
            display_name="Cargo Damage",
            chunk_count=10,
        )
        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage")
        ]
        mock_matcher.find_topics_by_names.return_value = [("Cargo Damage", topic)]
        mock_db.get_chunks_count_for_topic.return_value = 10

        result = await topic_filter.analyze_query("Cargo damage incidents?")

        assert result.detected_topics == ["Cargo Damage"]
        assert len(result.matched_topics) == 1
        assert result.matched_topics[0].id == topic_id
        assert result.should_skip_rag is False
        assert result.total_chunk_count == 10

    @pytest.mark.asyncio
    async def test_topic_matched_but_empty(
        self, topic_filter, mock_extractor, mock_matcher, mock_db
    ):
        """Should skip RAG when matched topic has no documents."""
        topic_id = uuid4()
        topic = Topic(
            id=topic_id,
            name="cargo damage",
            display_name="Cargo Damage",
            chunk_count=0,
        )
        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage")
        ]
        mock_matcher.find_topics_by_names.return_value = [("Cargo Damage", topic)]
        mock_db.get_chunks_count_for_topic.return_value = 0

        result = await topic_filter.analyze_query("Cargo damage incidents?")

        assert result.should_skip_rag is True
        assert "don't have any documents" in result.message

    @pytest.mark.asyncio
    async def test_multi_topic_any_has_results(
        self, topic_filter, mock_extractor, mock_matcher, mock_db
    ):
        """Should proceed when ANY matched topic has results."""
        topic1 = Topic(id=uuid4(), name="cargo damage", display_name="Cargo Damage")
        topic2 = Topic(id=uuid4(), name="engine issues", display_name="Engine Issues")

        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage"),
            ExtractedTopic(name="Engine Issues"),
        ]
        mock_matcher.find_topics_by_names.return_value = [
            ("Cargo Damage", topic1),
            ("Engine Issues", topic2),
        ]
        # First topic has 0, second has 5
        mock_db.get_chunks_count_for_topic.side_effect = [0, 5, 0, 5]

        result = await topic_filter.analyze_query("Cargo damage and engine problems?")

        assert len(result.detected_topics) == 2
        assert len(result.matched_topics) == 2
        assert result.should_skip_rag is False  # Because topic2 has results
        assert result.total_chunk_count == 5

    @pytest.mark.asyncio
    async def test_multi_topic_all_empty(
        self, topic_filter, mock_extractor, mock_matcher, mock_db
    ):
        """Should skip RAG when ALL matched topics are empty."""
        topic1 = Topic(id=uuid4(), name="cargo damage", display_name="Cargo Damage")
        topic2 = Topic(id=uuid4(), name="engine issues", display_name="Engine Issues")

        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage"),
            ExtractedTopic(name="Engine Issues"),
        ]
        mock_matcher.find_topics_by_names.return_value = [
            ("Cargo Damage", topic1),
            ("Engine Issues", topic2),
        ]
        mock_db.get_chunks_count_for_topic.return_value = 0

        result = await topic_filter.analyze_query("Cargo damage and engine problems?")

        assert result.should_skip_rag is True

    @pytest.mark.asyncio
    async def test_vessel_filter_no_match(
        self, topic_filter, mock_extractor, mock_matcher, mock_db
    ):
        """Should skip RAG when vessel filter excludes all results."""
        topic = Topic(id=uuid4(), name="cargo damage", display_name="Cargo Damage")

        mock_extractor.extract_topics_from_query.return_value = [
            ExtractedTopic(name="Cargo Damage")
        ]
        mock_matcher.find_topics_by_names.return_value = [("Cargo Damage", topic)]
        # Total count is 10, but filtered count is 0
        mock_db.get_chunks_count_for_topic.side_effect = [10, 0]

        result = await topic_filter.analyze_query(
            "Cargo damage incidents?",
            vessel_filter={"vessel_types": ["VLCC"]},
        )

        assert result.should_skip_rag is True
        assert result.total_chunk_count == 10
        assert result.filtered_chunk_count == 0
        assert "VLCC" in result.message

    @pytest.mark.asyncio
    async def test_extraction_error_falls_back_to_broad(
        self, topic_filter, mock_extractor
    ):
        """Should fall back to broad search on extraction error."""
        mock_extractor.extract_topics_from_query.side_effect = Exception("API Error")

        result = await topic_filter.analyze_query("Cargo damage?")

        assert result.detected_topics == []
        assert result.should_skip_rag is False

    def test_describe_vessel_filter_id(self, topic_filter):
        """Should describe vessel ID filter."""
        desc = topic_filter._describe_vessel_filter({"vessel_ids": ["uuid"]})
        assert "selected vessel" in desc

    def test_describe_vessel_filter_type(self, topic_filter):
        """Should describe vessel type filter."""
        desc = topic_filter._describe_vessel_filter({"vessel_types": ["VLCC"]})
        assert "VLCC" in desc

    def test_describe_vessel_filter_class(self, topic_filter):
        """Should describe vessel class filter."""
        desc = topic_filter._describe_vessel_filter({"vessel_classes": ["Canopus Class"]})
        assert "Canopus Class" in desc
