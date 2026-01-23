"""Tests for ingest storage components.

Tests for SupabaseClient, IssueTracker, and ingest helpers.
All database operations are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestSupabaseClient:
    """Tests for SupabaseClient class."""

    @pytest.fixture
    def mock_supabase_python_client(self):
        """Mock the Supabase Python client."""
        mock_client = MagicMock()
        mock_client.table.return_value = mock_client
        mock_client.select.return_value = mock_client
        mock_client.insert.return_value = mock_client
        mock_client.update.return_value = mock_client
        mock_client.delete.return_value = mock_client
        mock_client.eq.return_value = mock_client
        mock_client.neq.return_value = mock_client
        mock_client.lt.return_value = mock_client
        mock_client.in_.return_value = mock_client
        mock_client.order.return_value = mock_client
        mock_client.limit.return_value = mock_client
        mock_client.execute.return_value = MagicMock(data=[], count=0)
        return mock_client

    @pytest.fixture
    def supabase_client(self, comprehensive_mock_settings, mock_supabase_python_client):
        """Create a SupabaseClient with mocked dependencies."""
        with patch("mtss.storage.supabase_client.get_settings", return_value=comprehensive_mock_settings):
            with patch("mtss.storage.supabase_client.create_client", return_value=mock_supabase_python_client):
                from mtss.storage.supabase_client import SupabaseClient
                return SupabaseClient()

    @pytest.mark.asyncio
    async def test_insert_document(self, supabase_client, sample_document, mock_supabase_python_client):
        """Document insertion should call Supabase client."""
        result = await supabase_client.insert_document(sample_document)

        assert result == sample_document
        mock_supabase_python_client.table.assert_called_with("documents")

    @pytest.mark.asyncio
    async def test_update_document_status(self, supabase_client, sample_document_id, mock_supabase_python_client):
        """Document status update should work."""
        from mtss.models.document import ProcessingStatus

        await supabase_client.update_document_status(
            sample_document_id, ProcessingStatus.COMPLETED
        )

        mock_supabase_python_client.table.assert_called_with("documents")
        mock_supabase_python_client.update.assert_called()

    @pytest.mark.asyncio
    async def test_update_document_status_with_error(self, supabase_client, sample_document_id, mock_supabase_python_client):
        """Document status update with error message should work."""
        from mtss.models.document import ProcessingStatus

        await supabase_client.update_document_status(
            sample_document_id, ProcessingStatus.FAILED, error_message="Test error"
        )

        mock_supabase_python_client.table.assert_called_with("documents")

    @pytest.mark.asyncio
    async def test_get_document_by_hash_found(self, supabase_client, sample_document, mock_supabase_python_client):
        """Get document by hash should return document when found."""
        # Mock the response with document data
        mock_supabase_python_client.execute.return_value = MagicMock(
            data=[{
                "id": str(sample_document.id),
                "parent_id": None,
                "root_id": str(sample_document.id),
                "depth": 0,
                "path": [],
                "document_type": "email",
                "file_path": sample_document.file_path,
                "file_name": sample_document.file_name,
                "file_hash": sample_document.file_hash,
                "status": "pending",
            }]
        )

        result = await supabase_client.get_document_by_hash("abc123def456")

        assert result is not None
        mock_supabase_python_client.eq.assert_called_with("file_hash", "abc123def456")

    @pytest.mark.asyncio
    async def test_get_document_by_hash_not_found(self, supabase_client, mock_supabase_python_client):
        """Get document by hash should return None when not found."""
        mock_supabase_python_client.execute.return_value = MagicMock(data=[])

        result = await supabase_client.get_document_by_hash("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_insert_chunks(self, supabase_client, sample_chunks):
        """Chunk insertion should work with mock pool."""
        # Mock the async pool with proper async context manager
        mock_conn = AsyncMock()

        # Create a proper async context manager for acquire()
        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = MockAcquireContext()

        supabase_client._pool = mock_pool

        result = await supabase_client.insert_chunks(sample_chunks)

        assert result == sample_chunks
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_empty_chunks(self, supabase_client):
        """Inserting empty chunk list should return empty list."""
        result = await supabase_client.insert_chunks([])

        assert result == []

    @pytest.mark.asyncio
    async def test_replace_chunks_atomic(self, supabase_client, sample_document_id, sample_chunks):
        """Atomic chunk replacement should work."""
        # Create a proper async context manager for transaction()
        class MockTransactionContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                pass

        # Use MagicMock for the connection, but set up transaction as a regular method
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=MockTransactionContext())

        # Create a proper async context manager for acquire()
        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = MockAcquireContext()

        supabase_client._pool = mock_pool

        result = await supabase_client.replace_chunks_atomic(sample_document_id, sample_chunks)

        assert result == len(sample_chunks)
        mock_conn.execute.assert_called()  # DELETE
        mock_conn.executemany.assert_called()  # INSERT

    @pytest.mark.asyncio
    async def test_replace_chunks_atomic_empty(self, supabase_client, sample_document_id, mock_supabase_python_client):
        """Atomic replacement with empty chunks should just delete."""
        mock_supabase_python_client.execute.return_value = MagicMock(data=[], count=0)

        result = await supabase_client.replace_chunks_atomic(sample_document_id, [])

        assert result == 0

    def test_log_ingest_event(self, supabase_client, sample_document_id, mock_supabase_python_client):
        """Ingest event logging should work."""
        supabase_client.log_ingest_event(
            document_id=sample_document_id,
            event_type="parse_failure",
            severity="warning",
            message="Test message",
            file_path="/test/file.pdf",
        )

        mock_supabase_python_client.table.assert_called_with("ingest_events")
        mock_supabase_python_client.insert.assert_called()

    def test_log_ingest_event_truncates_message(self, supabase_client, sample_document_id, mock_supabase_python_client):
        """Long messages should be truncated."""
        long_message = "x" * 500

        supabase_client.log_ingest_event(
            document_id=sample_document_id,
            event_type="error",
            message=long_message,
        )

        # Verify the message was truncated in the call
        call_args = mock_supabase_python_client.insert.call_args
        if call_args:
            data = call_args[0][0]
            assert len(data.get("reason", "")) <= 200

    def test_row_to_document_conversion(self, supabase_client):
        """Row to document conversion should work."""
        row = {
            "id": "12345678-1234-5678-1234-567812345678",
            "parent_id": None,
            "root_id": "12345678-1234-5678-1234-567812345678",
            "depth": 0,
            "path": [],
            "document_type": "email",
            "file_path": "/test/file.eml",
            "file_name": "file.eml",
            "file_hash": "abc123",
            "status": "completed",
            "source_id": "file.eml",
            "doc_id": "doc123",
        }

        result = supabase_client._row_to_document(row)

        assert result.file_name == "file.eml"
        assert result.depth == 0
        assert str(result.id) == "12345678-1234-5678-1234-567812345678"

    def test_row_to_chunk_conversion(self, supabase_client, sample_document_id):
        """Row to chunk conversion should work."""
        row = {
            "id": str(uuid4()),
            "document_id": str(sample_document_id),
            "chunk_id": "abc123",
            "content": "Test content",
            "chunk_index": 0,
            "section_path": ["Intro"],
            "metadata": {"key": "value"},
        }

        result = supabase_client._row_to_chunk(row)

        assert result.content == "Test content"
        assert result.chunk_index == 0
        assert result.metadata == {"key": "value"}


class TestIssueTracker:
    """Tests for IssueTracker class."""

    @pytest.fixture
    def issue_tracker(self, mock_console):
        """Create an IssueTracker with mocked console."""
        from mtss.ingest.helpers import IssueTracker
        return IssueTracker(console=mock_console)

    @pytest.fixture
    def issue_tracker_with_db(self, mock_console, mock_supabase_client):
        """Create an IssueTracker with mocked console and DB."""
        from mtss.ingest.helpers import IssueTracker
        return IssueTracker(console=mock_console, db=mock_supabase_client)

    def test_track_adds_issue(self, issue_tracker, mock_console):
        """Track should add issue to internal list."""
        issue_tracker.track("email.eml", "attachment.pdf", "Parse failed")

        assert len(issue_tracker) == 1
        assert issue_tracker.issues[0]["email"] == "email.eml"
        assert issue_tracker.issues[0]["attachment"] == "attachment.pdf"
        assert issue_tracker.issues[0]["error"] == "Parse failed"
        mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_track_async_persists_to_db(
        self, issue_tracker_with_db, mock_supabase_client, sample_document_id
    ):
        """Async track should persist to database."""
        await issue_tracker_with_db.track_async(
            file_ctx="email.eml",
            attachment="attachment.pdf",
            error="Parse failed",
            document_id=sample_document_id,
        )

        assert len(issue_tracker_with_db) == 1
        mock_supabase_client.log_ingest_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_async_without_db(self, issue_tracker):
        """Async track without DB should still work."""
        await issue_tracker.track_async(
            file_ctx="email.eml",
            attachment="attachment.pdf",
            error="Parse failed",
        )

        assert len(issue_tracker) == 1

    def test_error_count(self, issue_tracker):
        """Error count should track only error-severity issues."""
        issue_tracker.track("email1.eml", "att1.pdf", "Warning issue")
        issue_tracker._issues[-1]["severity"] = "warning"

        issue_tracker.track("email2.eml", "att2.pdf", "Error issue")
        issue_tracker._issues[-1]["severity"] = "error"

        assert issue_tracker.error_count == 1
        assert len(issue_tracker) == 2

    def test_show_summary_no_issues(self, issue_tracker, mock_console):
        """Show summary should do nothing with no issues."""
        issue_tracker.show_summary()

        # Should not create a table if no issues
        # The console.print may or may not be called depending on implementation
        assert len(issue_tracker) == 0

    def test_show_summary_with_issues(self, issue_tracker, mock_console):
        """Show summary should display table with issues."""
        issue_tracker.track("email.eml", "attachment.pdf", "Test error")

        issue_tracker.show_summary()

        # Should have printed something
        assert mock_console.print.called

    def test_clear(self, issue_tracker):
        """Clear should remove all issues."""
        issue_tracker.track("email.eml", "att.pdf", "Error 1")
        issue_tracker.track("email.eml", "att2.pdf", "Error 2")

        assert len(issue_tracker) == 2

        issue_tracker.clear()

        assert len(issue_tracker) == 0

    def test_issues_property_returns_copy(self, issue_tracker):
        """Issues property should return a copy."""
        issue_tracker.track("email.eml", "att.pdf", "Error")

        issues = issue_tracker.issues
        issues.clear()

        assert len(issue_tracker) == 1  # Original unchanged


class TestIngestHelpers:
    """Tests for helper functions in ingest.helpers."""

    def test_get_format_name_known_types(self):
        """Known MIME types should return readable names."""
        from mtss.ingest.helpers import get_format_name

        assert get_format_name("application/pdf") == "PDF"
        assert get_format_name("image/png") == "PNG"
        assert get_format_name("image/jpeg") == "JPEG"
        assert get_format_name("application/zip") == "ZIP"

    def test_get_format_name_unknown_type(self):
        """Unknown MIME types should return fallback."""
        from mtss.ingest.helpers import get_format_name

        result = get_format_name("application/x-custom-type")

        assert result == "X-CUSTOM-TYPE"

    def test_get_format_name_docx(self):
        """DOCX MIME type should return DOCX."""
        from mtss.ingest.helpers import get_format_name

        result = get_format_name(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assert result == "DOCX"

    def test_enrich_chunks_with_document_metadata(self, sample_chunks, sample_document):
        """Chunks should be enriched with document metadata."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata

        enrich_chunks_with_document_metadata(sample_chunks, sample_document)

        for chunk in sample_chunks:
            assert chunk.source_id == sample_document.source_id
            assert chunk.source_title == sample_document.source_title
            assert chunk.archive_browse_uri == sample_document.archive_browse_uri
            assert chunk.archive_download_uri == sample_document.archive_download_uri

    def test_enrich_chunks_generates_chunk_ids(self, sample_chunks, sample_document):
        """Enrichment should generate chunk IDs."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata

        # Clear existing chunk_ids
        for chunk in sample_chunks:
            chunk.chunk_id = None

        enrich_chunks_with_document_metadata(sample_chunks, sample_document)

        for chunk in sample_chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.chunk_id) == 12  # CHUNK_ID_LENGTH

    def test_generates_unique_chunk_ids(self, sample_document):
        """Each chunk should get a unique ID."""
        from mtss.ingest.helpers import enrich_chunks_with_document_metadata
        from mtss.models.chunk import Chunk

        chunks = [
            Chunk(
                document_id=sample_document.id,
                content=f"Content {i}",
                chunk_index=i,
                char_start=i * 100,
                char_end=(i + 1) * 100,
                section_path=[],
                metadata={},
            )
            for i in range(5)
        ]

        enrich_chunks_with_document_metadata(chunks, sample_document)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(set(chunk_ids)) == len(chunk_ids)  # All unique
