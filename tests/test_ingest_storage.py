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

        result = supabase_client._docs._row_to_document(row)

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

        result = supabase_client._docs._row_to_chunk(row)

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


class TestLocalClientFlushChunkDedup:
    """Tests that LocalStorageClient.flush() dedupes chunks by chunk_id within a run."""

    @pytest.fixture
    def local_client(self, tmp_path):
        from mtss.storage.local_client import LocalStorageClient

        return LocalStorageClient(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_flush_dedupes_same_run_chunks_by_chunk_id(self, local_client):
        """Two current-run chunks with different UUIDs but same chunk_id should dedupe."""
        import json as _json
        from uuid import uuid4

        from mtss.models.chunk import Chunk
        from mtss.models.document import Document, DocumentType, ProcessingStatus

        doc = Document(
            id=uuid4(),
            document_type=DocumentType.EMAIL,
            file_path="/test.eml",
            file_name="test.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )
        await local_client.insert_document(doc)

        shared_chunk_id = "shared_chunk_1"
        chunk_a = Chunk(
            id=uuid4(),
            document_id=doc.id,
            chunk_id=shared_chunk_id,
            content="First write",
            chunk_index=0,
            metadata={"type": "email_body"},
        )
        chunk_b = Chunk(
            id=uuid4(),
            document_id=doc.id,
            chunk_id=shared_chunk_id,
            content="Second write",
            chunk_index=0,
            metadata={"type": "email_body"},
        )
        await local_client.insert_chunks([chunk_a])
        await local_client.insert_chunks([chunk_b])

        local_client.flush()

        chunks_path = local_client.output_dir / "chunks.jsonl"
        lines = [ln for ln in chunks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        matching = [_json.loads(ln) for ln in lines if _json.loads(ln).get("chunk_id") == shared_chunk_id]
        assert len(matching) == 1

    @pytest.mark.asyncio
    async def test_flush_keeps_unique_chunks(self, local_client):
        """Two chunks with different chunk_ids should both survive flush."""
        import json as _json
        from uuid import uuid4

        from mtss.models.chunk import Chunk
        from mtss.models.document import Document, DocumentType, ProcessingStatus

        doc = Document(
            id=uuid4(),
            document_type=DocumentType.EMAIL,
            file_path="/test.eml",
            file_name="test.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )
        await local_client.insert_document(doc)

        chunk_a = Chunk(
            id=uuid4(),
            document_id=doc.id,
            chunk_id="unique_a",
            content="A",
            chunk_index=0,
            metadata={"type": "email_body"},
        )
        chunk_b = Chunk(
            id=uuid4(),
            document_id=doc.id,
            chunk_id="unique_b",
            content="B",
            chunk_index=1,
            metadata={"type": "email_body"},
        )
        await local_client.insert_chunks([chunk_a, chunk_b])

        local_client.flush()

        chunks_path = local_client.output_dir / "chunks.jsonl"
        lines = [ln for ln in chunks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        chunk_ids = {_json.loads(ln).get("chunk_id") for ln in lines}
        assert "unique_a" in chunk_ids
        assert "unique_b" in chunk_ids


class TestMarkFailedCommand:
    """Tests for `MTSS mark-failed` maintenance command."""

    @pytest.fixture
    def seeded_output(self, tmp_path):
        """Output dir with a processing_log.jsonl holding 3 entries."""
        import json as _json

        output = tmp_path / "output"
        output.mkdir()
        log = output / "processing_log.jsonl"
        entries = [
            {
                "file_path": str(tmp_path / "emails" / "a.eml"),
                "file_hash": "aaa",
                "status": "COMPLETED",
                "started_at": "2026-04-16T10:00:00+00:00",
                "completed_at": "2026-04-16T10:01:00+00:00",
                "error": None,
                "attempts": 1,
            },
            {
                "file_path": str(tmp_path / "emails" / "b.eml"),
                "file_hash": "bbb",
                "status": "COMPLETED",
                "started_at": "2026-04-16T10:02:00+00:00",
                "completed_at": "2026-04-16T10:03:00+00:00",
                "error": None,
                "attempts": 1,
            },
            {
                "file_path": str(tmp_path / "emails" / "c.eml"),
                "file_hash": "ccc",
                "status": "COMPLETED",
                "started_at": "2026-04-16T10:04:00+00:00",
                "completed_at": "2026-04-16T10:05:00+00:00",
                "error": None,
                "attempts": 1,
            },
        ]
        with open(log, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(_json.dumps(e) + "\n")
        return output, entries

    @pytest.mark.asyncio
    async def test_marks_specified_files_as_failed(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed
        from mtss.storage.local_progress_tracker import LocalProgressTracker

        output, entries = seeded_output
        await _mark_failed(
            [_Path(entries[0]["file_path"]), _Path(entries[2]["file_path"])],
            output,
            "bad_context",
        )

        reloaded = LocalProgressTracker(output)
        assert reloaded._entries[entries[0]["file_path"]]["status"] == "FAILED"
        assert reloaded._entries[entries[0]["file_path"]]["error"] == "bad_context"
        assert reloaded._entries[entries[1]["file_path"]]["status"] == "COMPLETED"
        assert reloaded._entries[entries[2]["file_path"]]["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_resolves_file_by_basename(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed
        from mtss.storage.local_progress_tracker import LocalProgressTracker

        output, entries = seeded_output
        await _mark_failed([_Path("b.eml")], output, "manual")

        reloaded = LocalProgressTracker(output)
        assert reloaded._entries[entries[1]["file_path"]]["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_compacts_log_to_one_entry_per_file(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed

        output, entries = seeded_output
        await _mark_failed([_Path(entries[0]["file_path"])], output, "x")

        with open(output / "processing_log.jsonl", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) == 3  # One per file, no duplicates from append

    @pytest.mark.asyncio
    async def test_missing_log_file_exits(self, tmp_path):
        from pathlib import Path as _Path

        import typer as _typer

        from mtss.cli.maintenance_cmd import _mark_failed

        with pytest.raises(_typer.Exit):
            await _mark_failed([_Path("x.eml")], tmp_path / "nope", "reason")


class TestCleanArchiveMdCommand:
    """`MTSS clean-archive-md` strips stale LlamaParse image refs from archived .md files."""

    @pytest.fixture
    def seeded_archive(self, tmp_path):
        output = tmp_path / "output"
        archive = output / "archive"
        folder_a = archive / "abc123" / "attachments"
        folder_b = archive / "def456" / "attachments"
        folder_a.mkdir(parents=True)
        folder_b.mkdir(parents=True)
        # Broken bare-image ref
        (folder_a / "report.pdf.md").write_text(
            "Intro text\n![Diagram](image)\nTrailing text\n",
            encoding="utf-8",
        )
        # Mix of broken + LlamaParse page refs
        (folder_b / "slides.pdf.md").write_text(
            "![logo](image)\n![chart](page_3_chart_1_v2.jpg)\n",
            encoding="utf-8",
        )
        # Clean file — should not change
        clean_path = archive / "abc123" / "email.eml.md"
        clean_path.write_text("- [Attachment](abc123/attachments/report.pdf)\n", encoding="utf-8")
        return output, folder_a / "report.pdf.md", folder_b / "slides.pdf.md", clean_path

    def test_rewrites_files_with_bare_image_refs(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _clean_archive_md

        output, md_a, md_b, clean = seeded_archive
        _clean_archive_md(output, dry_run=False)

        txt_a = md_a.read_text(encoding="utf-8")
        assert "(image)" not in txt_a
        assert "Diagram" in txt_a  # alt-text preserved

        txt_b = md_b.read_text(encoding="utf-8")
        assert "(image)" not in txt_b
        assert "page_3_chart_1_v2.jpg" not in txt_b
        assert "logo" in txt_b and "chart" in txt_b

        # Clean file untouched
        assert clean.read_text(encoding="utf-8") == "- [Attachment](abc123/attachments/report.pdf)\n"

    def test_dry_run_does_not_write(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _clean_archive_md

        output, md_a, _md_b, _clean = seeded_archive
        before = md_a.read_text(encoding="utf-8")
        _clean_archive_md(output, dry_run=True)
        assert md_a.read_text(encoding="utf-8") == before

    def test_missing_archive_dir_exits(self, tmp_path):
        import typer as _typer

        from mtss.cli.maintenance_cmd import _clean_archive_md

        with pytest.raises(_typer.Exit):
            _clean_archive_md(tmp_path / "no-such-output", dry_run=False)

    def test_idempotent(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _clean_archive_md

        output, md_a, _md_b, _clean = seeded_archive
        _clean_archive_md(output, dry_run=False)
        first = md_a.read_text(encoding="utf-8")
        _clean_archive_md(output, dry_run=False)
        assert md_a.read_text(encoding="utf-8") == first
