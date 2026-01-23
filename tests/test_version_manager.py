"""Tests for version management and document decision logic.

Tests for VersionManager class that decides whether to insert, update, skip, or reprocess
documents during ingestion based on existing records and version numbers.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest


class TestVersionManager:
    """Tests for VersionManager class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock SupabaseClient."""
        db = MagicMock()
        db.get_document_by_doc_id = AsyncMock(return_value=None)
        db.get_document_by_source_id = AsyncMock(return_value=None)
        db.get_documents_below_version = AsyncMock(return_value=[])
        db.count_documents_below_version = AsyncMock(return_value=0)
        return db

    @pytest.fixture
    def version_manager(self, mock_db, comprehensive_mock_settings):
        """Create a VersionManager with mocked dependencies."""
        with patch(
            "mtss.processing.version_manager.get_settings",
            return_value=comprehensive_mock_settings,
        ):
            from mtss.processing.version_manager import VersionManager

            return VersionManager(db=mock_db)

    @pytest.fixture
    def existing_document(self, sample_document_id):
        """Create a mock existing document."""
        doc = MagicMock()
        doc.id = sample_document_id
        doc.source_id = "test/email.eml"
        doc.doc_id = "abc123def456"
        doc.ingest_version = 1
        return doc

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_insert_when_no_existing_document(self, version_manager, mock_db):
        """Should return insert action when document doesn't exist."""
        mock_db.get_document_by_doc_id.return_value = None
        mock_db.get_document_by_source_id.return_value = None

        result = await version_manager.check_document(
            source_id="new/email.eml",
            file_hash="abc123",
        )

        assert result.action == "insert"
        assert result.reason == "New document"
        assert result.existing_doc_id is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_skip_when_doc_id_exists_and_completed(
        self, version_manager, mock_db, existing_document
    ):
        """Should return skip action when same content already processed with current version."""
        existing_document.ingest_version = 1  # Same as current version
        mock_db.get_document_by_doc_id.return_value = existing_document

        result = await version_manager.check_document(
            source_id="test/email.eml",
            file_hash="content_hash_123",
        )

        assert result.action == "skip"
        assert "Already processed" in result.reason
        assert result.existing_doc_id == existing_document.id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_reprocess_when_ingest_version_outdated(
        self, version_manager, mock_db, existing_document, comprehensive_mock_settings
    ):
        """Should return reprocess action when document has older ingest version."""
        # Set existing document to older version
        existing_document.ingest_version = 0
        comprehensive_mock_settings.current_ingest_version = 1
        mock_db.get_document_by_doc_id.return_value = existing_document

        result = await version_manager.check_document(
            source_id="test/email.eml",
            file_hash="content_hash_123",
        )

        assert result.action == "reprocess"
        assert "upgraded" in result.reason.lower()
        assert result.existing_doc_id == existing_document.id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_update_when_source_id_exists_with_different_content(
        self, version_manager, mock_db, existing_document
    ):
        """Should return update action when same source has different content."""
        # No exact match (different hash = different doc_id)
        mock_db.get_document_by_doc_id.return_value = None
        # But source_id exists with old content
        mock_db.get_document_by_source_id.return_value = existing_document

        result = await version_manager.check_document(
            source_id="test/email.eml",
            file_hash="new_hash_different",
        )

        assert result.action == "update"
        assert "Content changed" in result.reason
        assert result.existing_doc_id == existing_document.id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_reprocess_candidates(
        self, version_manager, mock_db, existing_document
    ):
        """Should return documents below current version."""
        mock_db.get_documents_below_version.return_value = [existing_document]

        result = await version_manager.get_reprocess_candidates(target_version=2)

        mock_db.get_documents_below_version.assert_called_once_with(2, 100)
        assert len(result) == 1
        assert result[0] == existing_document

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_reprocess_candidates_uses_current_version_by_default(
        self, version_manager, mock_db, comprehensive_mock_settings
    ):
        """Should use current_ingest_version when target not specified."""
        mock_db.get_documents_below_version.return_value = []

        await version_manager.get_reprocess_candidates()

        mock_db.get_documents_below_version.assert_called_once_with(
            comprehensive_mock_settings.current_ingest_version, 100
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_reprocess_candidates_with_limit(self, version_manager, mock_db):
        """Should respect the limit parameter."""
        mock_db.get_documents_below_version.return_value = []

        await version_manager.get_reprocess_candidates(limit=50)

        mock_db.get_documents_below_version.assert_called_once()
        call_args = mock_db.get_documents_below_version.call_args
        assert call_args[0][1] == 50

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_count_reprocess_candidates(self, version_manager, mock_db):
        """Should return count of documents below target version."""
        mock_db.count_documents_below_version.return_value = 42

        result = await version_manager.count_reprocess_candidates(target_version=2)

        mock_db.count_documents_below_version.assert_called_once_with(2)
        assert result == 42


class TestIngestDecision:
    """Tests for IngestDecision dataclass."""

    @pytest.mark.unit
    def test_ingest_decision_insert(self):
        """Insert decision should have correct fields."""
        from mtss.processing.version_manager import IngestDecision

        decision = IngestDecision(
            action="insert",
            reason="New document",
        )

        assert decision.action == "insert"
        assert decision.reason == "New document"
        assert decision.existing_doc_id is None

    @pytest.mark.unit
    def test_ingest_decision_with_existing_doc(self):
        """Decision with existing doc should store the ID."""
        from mtss.processing.version_manager import IngestDecision

        doc_id = uuid4()
        decision = IngestDecision(
            action="skip",
            reason="Already processed",
            existing_doc_id=doc_id,
        )

        assert decision.action == "skip"
        assert decision.existing_doc_id == doc_id

    @pytest.mark.unit
    def test_ingest_decision_valid_actions(self):
        """All valid action values should be accepted."""
        from mtss.processing.version_manager import IngestDecision

        for action in ["insert", "update", "skip", "reprocess"]:
            decision = IngestDecision(action=action, reason="test")
            assert decision.action == action
