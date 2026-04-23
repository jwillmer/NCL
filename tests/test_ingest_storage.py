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

    @pytest.mark.asyncio
    async def test_insert_chunks_strips_nul_bytes_in_text(
        self, supabase_client, sample_chunks
    ):
        """Postgres TEXT rejects \\x00 as invalid UTF-8; SQLite ingest may have
        smuggled NUL bytes through (e.g. from a malformed PDF extract). The
        asyncpg bind must strip them so the import doesn't abort on a single
        poisoned chunk."""
        sample_chunks[0].content = "before\x00after"
        sample_chunks[1].context_summary = "ctx\x00nul"
        sample_chunks[2].section_path = ["head\x00er", "clean"]
        sample_chunks[3].metadata = {"k": "v\x00alue", "nested": ["a\x00b"]}

        mock_conn = AsyncMock()

        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = MockAcquireContext()
        supabase_client._pool = mock_pool

        await supabase_client.insert_chunks(sample_chunks)

        records = mock_conn.executemany.call_args.args[1]
        # Records layout: (id, chunk_id, document_id, content, chunk_index,
        # context_summary, embedding_text, section_path, ...
        # ... , embedding, metadata)
        all_text_fields = []
        for rec in records:
            all_text_fields.extend(
                [
                    rec[1],   # chunk_id
                    rec[3],   # content
                    rec[5],   # context_summary
                    rec[6] or "",  # embedding_text
                    *(rec[7] or []),  # section_path
                    rec[19],  # metadata (jsonb string)
                ]
            )
        assert all("\x00" not in f for f in all_text_fields if f is not None)
        # Non-NUL content is preserved verbatim.
        assert records[0][3] == "beforeafter"
        assert records[1][5] == "ctxnul"
        assert records[2][7] == ["header", "clean"]
        # metadata jsonb stays valid JSON, NUL stripped.
        import json as _json
        md = _json.loads(records[3][19])
        assert md == {"k": "value", "nested": ["ab"]}

    @pytest.mark.asyncio
    async def test_persist_ingest_result_strips_nul_bytes_in_document_fields(
        self, supabase_client, sample_document
    ):
        """Document TEXT fields (subject, participants, source_title, ...)
        must also have NUL stripped before binding."""
        sample_document.source_title = "Subject\x00 with nul"
        sample_document.file_name = "file\x00.eml"
        sample_document.email_metadata.subject = "S\x00ubj"
        sample_document.email_metadata.participants = ["a\x00@b", "c@d"]

        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.executemany = AsyncMock()

        class MockTransactionContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                pass

        mock_conn.transaction = MagicMock(return_value=MockTransactionContext())

        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = MockAcquireContext()
        supabase_client._pool = mock_pool

        await supabase_client.persist_ingest_result(
            email_doc=sample_document,
            attachment_docs=[],
            chunks=[],
        )

        # _insert_document_pg binds positional args; find the INSERT call.
        insert_call = next(
            c for c in mock_conn.execute.call_args_list
            if "INSERT INTO documents" in c.args[0]
        )
        bound = insert_call.args[1:]
        str_args = [a for a in bound if isinstance(a, str)]
        list_args = [a for a in bound if isinstance(a, list)]
        assert all("\x00" not in s for s in str_args)
        for lst in list_args:
            assert all("\x00" not in x for x in lst if isinstance(x, str))
        # Non-NUL portion preserved.
        assert "Subject with nul" in str_args
        assert "file.eml" in str_args
        assert "Subj" in str_args

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


class TestMarkFailedCommand:
    """Tests for `MTSS mark-failed` maintenance command."""

    @pytest.fixture
    def seeded_output(self, tmp_path):
        """Output dir with an ingest.db whose processing_log holds 3 entries."""
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output = tmp_path / "output"
        output.mkdir()
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
        tracker = SqliteProgressTracker(output)
        for e in entries:
            tracker._conn.execute(
                """
                INSERT INTO processing_log(file_path, file_hash, status, started_at,
                                           completed_at, attempts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    e["file_path"], e["file_hash"], e["status"],
                    e["started_at"], e["completed_at"], e["attempts"],
                ),
            )
        tracker.close()
        return output, entries

    @pytest.mark.asyncio
    async def test_marks_specified_files_as_failed(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output, entries = seeded_output
        await _mark_failed(
            [_Path(entries[0]["file_path"]), _Path(entries[2]["file_path"])],
            output,
            "bad_context",
        )

        reloaded = SqliteProgressTracker(output)
        by_path = {e["file_path"]: e for e in reloaded.iter_entries()}
        reloaded.close()
        assert by_path[entries[0]["file_path"]]["status"] == "FAILED"
        assert by_path[entries[0]["file_path"]]["error"] == "bad_context"
        assert by_path[entries[1]["file_path"]]["status"] == "COMPLETED"
        assert by_path[entries[2]["file_path"]]["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_resolves_file_by_basename(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output, entries = seeded_output
        await _mark_failed([_Path("b.eml")], output, "manual")

        reloaded = SqliteProgressTracker(output)
        by_path = {e["file_path"]: e for e in reloaded.iter_entries()}
        reloaded.close()
        assert by_path[entries[1]["file_path"]]["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_get_failed_files_returns_all_failed_regardless_of_attempts(self, tmp_path):
        """`--retry-failed` is an explicit user command and must retry *every*
        FAILED entry regardless of how many prior attempts accumulated.

        Regression: an earlier implementation filtered ``attempts < 3``, which
        made ``ingest --retry-failed`` silently say "Retrying 0 failed files"
        once every FAILED entry had exhausted its auto-retry budget. The
        attempts counter stays on the entry for visibility, but does not
        gate retry eligibility.
        """
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output = tmp_path / "output"
        output.mkdir()
        entries = [
            # Fresh failure.
            (str(tmp_path / "emails" / "fresh.eml"), "f1", "FAILED", "x", 1),
            # Stuck at the old cap — previously invisible to retry when attempts-gate
            # still applied. Must surface regardless of attempts today.
            (str(tmp_path / "emails" / "stuck.eml"), "f2", "FAILED", "Timed out", 3),
            (str(tmp_path / "emails" / "very_stuck.eml"), "f3", "FAILED", "x", 99),
            # COMPLETED — must NOT appear.
            (str(tmp_path / "emails" / "ok.eml"), "f4", "COMPLETED", None, 1),
        ]
        tracker = SqliteProgressTracker(output)
        for fp, fh, status, err, attempts in entries:
            tracker._conn.execute(
                "INSERT INTO processing_log("
                "file_path, file_hash, status, error, attempts"
                ") VALUES (?, ?, ?, ?, ?)",
                (fp, fh, status, err, attempts),
            )
        result = sorted(p.name for p in await tracker.get_failed_files())
        tracker.close()
        assert result == ["fresh.eml", "stuck.eml", "very_stuck.eml"]

    @pytest.mark.asyncio
    async def test_compacts_log_to_one_entry_per_file(self, seeded_output):
        from pathlib import Path as _Path

        from mtss.cli.maintenance_cmd import _mark_failed
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output, entries = seeded_output
        await _mark_failed([_Path(entries[0]["file_path"])], output, "x")

        # processing_log has PK(file_path) so duplicates are structurally
        # impossible — three seeded entries, still three rows after mark-failed.
        tracker = SqliteProgressTracker(output)
        rows = tracker.iter_entries()
        tracker.close()
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_missing_log_file_exits(self, tmp_path):
        from pathlib import Path as _Path

        import typer as _typer

        from mtss.cli.maintenance_cmd import _mark_failed

        with pytest.raises(_typer.Exit):
            await _mark_failed([_Path("x.eml")], tmp_path / "nope", "reason")

    def test_compact_is_noop_safe(self, tmp_path):
        """``SqliteProgressTracker.compact()`` is just a WAL checkpoint — a
        failing checkpoint must not corrupt the existing rows. With WAL the
        canonical file is never truncated in place, so the class of
        rewrite-during-crash bug the old JSONL ``compact`` had is structurally
        gone. This test locks that contract in: the row survives compact().
        """
        from mtss.storage.sqlite_progress_tracker import SqliteProgressTracker

        output = tmp_path / "output"
        output.mkdir()

        tracker = SqliteProgressTracker(output)
        tracker._conn.execute(
            "INSERT INTO processing_log(file_path, file_hash, status) "
            "VALUES ('/a.eml', 'a', 'COMPLETED')"
        )
        tracker.compact()
        rows = tracker.iter_entries()
        tracker.close()

        assert len(rows) == 1
        assert rows[0]["file_path"] == "/a.eml"


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


class TestHealArchiveCommand:
    """`mtss heal-archive` uploads local files missing from Supabase Storage."""

    @pytest.fixture
    def seeded_archive(self, tmp_path):
        output = tmp_path / "output"
        folder = output / "archive" / "abc123" / "attachments"
        folder.mkdir(parents=True)
        (folder / "present.md").write_text("already uploaded\n", encoding="utf-8")
        (folder / "missing.md").write_text("needs upload\n", encoding="utf-8")
        (folder / "missing.xlsx").write_bytes(b"\x50\x4b\x03\x04")  # zip magic
        return output

    def _storage_stub(self, remote_contents: dict[str, list[str]]):
        from unittest.mock import MagicMock

        from mtss.storage.archive_storage import ArchiveStorageError

        storage = MagicMock()
        storage.bucket_name = "archive-test"
        uploaded: list[tuple[str, int, str]] = []

        def _list_folder(folder: str):
            if folder in remote_contents:
                return [{"name": n} for n in remote_contents[folder]]
            raise ArchiveStorageError(f"missing folder: {folder}")

        def _upload_file(path, content, content_type):
            uploaded.append((path, len(content), content_type))
            return path

        storage.list_folder.side_effect = _list_folder
        storage.upload_file.side_effect = _upload_file
        return storage, uploaded

    def test_dry_run_lists_missing_but_does_not_upload(self, seeded_archive):
        from unittest.mock import patch

        from mtss.cli.maintenance_cmd import _heal_archive

        storage, uploaded = self._storage_stub(
            {"abc123/attachments": ["present.md"]}
        )
        with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
            _heal_archive(seeded_archive, dry_run=True, limit=0)

        assert uploaded == []
        storage.list_folder.assert_called_once_with("abc123/attachments")

    def test_apply_uploads_only_missing_files(self, seeded_archive):
        from unittest.mock import patch

        from mtss.cli.maintenance_cmd import _heal_archive

        storage, uploaded = self._storage_stub(
            {"abc123/attachments": ["present.md"]}
        )
        with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
            _heal_archive(seeded_archive, dry_run=False, limit=0)

        uploaded_keys = sorted(k for k, _, _ in uploaded)
        assert uploaded_keys == [
            "abc123/attachments/missing.md",
            "abc123/attachments/missing.xlsx",
        ]
        # Markdown got text content-type, xlsx fell through to octet-stream.
        md_upload = next(u for u in uploaded if u[0].endswith(".md"))
        assert "markdown" in md_upload[2] or "text" in md_upload[2]

    def test_apply_respects_limit(self, seeded_archive):
        from unittest.mock import patch

        from mtss.cli.maintenance_cmd import _heal_archive

        storage, uploaded = self._storage_stub(
            {"abc123/attachments": ["present.md"]}
        )
        with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
            _heal_archive(seeded_archive, dry_run=False, limit=1)

        assert len(uploaded) == 1

    def test_missing_folder_uploads_everything(self, seeded_archive):
        """Folder absent remotely → every local file in it is queued."""
        from unittest.mock import patch

        from mtss.cli.maintenance_cmd import _heal_archive

        storage, uploaded = self._storage_stub({})  # empty bucket
        with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
            _heal_archive(seeded_archive, dry_run=False, limit=0)

        uploaded_keys = sorted(k for k, _, _ in uploaded)
        assert uploaded_keys == [
            "abc123/attachments/missing.md",
            "abc123/attachments/missing.xlsx",
            "abc123/attachments/present.md",
        ]

    def test_no_archive_dir_exits(self, tmp_path):
        import typer as _typer

        from mtss.cli.maintenance_cmd import _heal_archive

        with pytest.raises(_typer.Exit):
            _heal_archive(tmp_path / "nope", dry_run=True, limit=0)

    def test_local_filenames_are_sanitized_before_comparison(self, tmp_path):
        """Regression: `mtss import` uploads under the ``async_upload``
        sanitizer (`%` → `_`, keeps spaces). `heal-archive` must apply the
        same sanitizer to local names before diffing against the bucket,
        or every file containing `%` looks "missing" even though it's
        already uploaded under the sanitized key.
        """
        from unittest.mock import patch

        from mtss.cli.maintenance_cmd import _heal_archive

        output = tmp_path / "output"
        folder = output / "archive" / "abc123" / "attachments"
        folder.mkdir(parents=True)
        # Local disk name still has `%` (ingest-time sanitizer preserves it).
        (folder / "report_10%.xlsx").write_bytes(b"x")

        # Bucket has the import-time sanitized version (`%` → `_`).
        storage, uploaded = self._storage_stub(
            {"abc123/attachments": ["report_10_.xlsx"]}
        )
        with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
            _heal_archive(output, dry_run=False, limit=0)

        # No upload — the file is already present under its sanitized key.
        assert uploaded == []


class TestDeleteDocumentForReprocessClearsIndexes:
    """Regression: prior-loaded docs must be evicted from the in-memory
    dedup cache when reprocessed, otherwise insert_document early-returns
    on re-insert and the replacement email row is never written.

    Real-world symptom: retry-failed run for an email marked the email
    COMPLETED in processing_log but the email's row was missing from
    ``documents``, leaving its attachment children orphaned with
    root_id pointing to a non-existent UUID. FK CASCADE makes this a
    structural impossibility now, but the cache-eviction on
    ``delete_document_for_reprocess`` is still the belt-and-braces check.
    """

    @pytest.mark.asyncio
    async def test_reprocess_after_restart_writes_new_email_row(self, tmp_path):
        from uuid import uuid4

        from mtss.models.document import Document, DocumentType, ProcessingStatus
        from mtss.storage.sqlite_client import SqliteStorageClient

        output_dir = tmp_path / "output"

        # --- Prior run: write an email doc, close ---
        client_a = SqliteStorageClient(output_dir=output_dir)
        old_id = uuid4()
        old_email = Document(
            id=old_id,
            doc_id="deadbeefcafebabe",
            source_id="same.eml",
            file_hash="hash_v1",
            document_type=DocumentType.EMAIL,
            file_path="/same.eml",
            file_name="same.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )
        await client_a.insert_document(old_email)
        client_a._conn.close()

        # --- Fresh process: load prior data, then reprocess same email ---
        client_b = SqliteStorageClient(output_dir=output_dir)
        # __post_init__ rehydrates caches from the DB
        existing = await client_b.get_document_by_doc_id("deadbeefcafebabe")
        assert existing is not None, "precondition: prior doc must be discoverable by doc_id"

        client_b.delete_document_for_reprocess(existing.id)
        # After cleanup, cache must not resurrect the stale doc
        assert await client_b.get_document_by_doc_id("deadbeefcafebabe") is None, \
            "delete_document_for_reprocess must evict prior doc from doc_id cache"

        # Insert NEW email (same doc_id, different uuid)
        new_id = uuid4()
        new_email = Document(
            id=new_id,
            doc_id="deadbeefcafebabe",
            source_id="same.eml",
            file_hash="hash_v1",
            document_type=DocumentType.EMAIL,
            file_path="/same.eml",
            file_name="same.eml",
            depth=0,
            status=ProcessingStatus.COMPLETED,
        )
        stored = await client_b.insert_document(new_email)
        assert stored.id == new_id, \
            "insert_document returned stale prior doc instead of writing the new one"

        # documents table should hold exactly one row with this doc_id and the new UUID
        rows = list(client_b._conn.execute(
            "SELECT id FROM documents WHERE doc_id = ?", ("deadbeefcafebabe",)
        ))
        client_b._conn.close()
        assert len(rows) == 1
        assert rows[0]["id"] == str(new_id)


class TestFixTableMdCommand:
    """`MTSS fix-table-md` rewrites legacy broken-pipe tables in archived .md
    files as valid GFM.

    Backstory: ``LocalXlsxParser`` and ``LocalCsvParser`` historically emitted
    rows with `" | ".join(cells)` — no leading/trailing ``|`` and no delimiter
    row after the header. Markdown viewers collapse those blocks into a
    soft-wrapped paragraph. This maintenance command patches the output in
    place without re-parsing the source attachment.
    """

    @pytest.fixture
    def seeded_archive(self, tmp_path):
        output = tmp_path / "output"
        archive = output / "archive"
        folder_a = archive / "aaa" / "attachments"
        folder_b = archive / "bbb" / "attachments"
        folder_a.mkdir(parents=True)
        folder_b.mkdir(parents=True)

        # Broken xlsx.md — `## Sheet` header followed by pipe-joined rows,
        # no delimiter, no leading/trailing `|`. Modelled on the real
        # Weekly_Onboard_LO_Analysis_Log_22.xlsx.md output.
        broken_xlsx = (
            "# Weekly Log.xlsx\n"
            "\n"
            "**Type:** application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\n"
            "\n"
            "---\n"
            "\n"
            "## Content\n"
            "\n"
            "## Sheet1\n"
            "\n"
            "Date | Engine | Notes\n"
            "2022-01-28 | ME | Added 500 lt\n"
            "2022-02-04 | AE1 | -\n"
            "2022-02-11 | AE2 | OK\n"
            "\n"
            "Other prose content here.\n"
        )
        (folder_a / "weekly.xlsx.md").write_text(broken_xlsx, encoding="utf-8")

        # Broken csv.md — no sheet header, just rows.
        broken_csv = (
            "# export.csv\n"
            "\n"
            "---\n"
            "\n"
            "## Content\n"
            "\n"
            "name | value\n"
            "alpha | 1\n"
            "beta | 2\n"
        )
        (folder_b / "export.csv.md").write_text(broken_csv, encoding="utf-8")

        # Already-valid file — must not change.
        good_path = folder_a / "good.xlsx.md"
        good_path.write_text(
            "## Sheet1\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
            encoding="utf-8",
        )

        # Non-xlsx/csv .md — must be skipped entirely.
        untouched = archive / "aaa" / "email.eml.md"
        untouched.write_text(
            "Body of an email with a | pipe | in prose.\n",
            encoding="utf-8",
        )
        return output, folder_a / "weekly.xlsx.md", folder_b / "export.csv.md", good_path, untouched

    def test_apply_rewrites_broken_table_as_valid_gfm(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _fix_table_md

        output, xlsx_md, csv_md, good, untouched = seeded_archive
        _fix_table_md(output, dry_run=False)

        txt = xlsx_md.read_text(encoding="utf-8")
        lines = txt.splitlines()
        # Locate the rewritten table — first line after "## Sheet1" blank line.
        assert "| Date | Engine | Notes |" in lines
        # Delimiter row follows immediately.
        header_idx = lines.index("| Date | Engine | Notes |")
        assert lines[header_idx + 1] == "|---|---|---|"
        # Original data preserved row-for-row.
        assert "| 2022-01-28 | ME | Added 500 lt |" in lines
        assert "| 2022-02-04 | AE1 | - |" in lines
        assert "| 2022-02-11 | AE2 | OK |" in lines
        # Surrounding content preserved.
        assert "# Weekly Log.xlsx" in txt
        assert "Other prose content here." in txt

        csv_txt = csv_md.read_text(encoding="utf-8")
        csv_lines = csv_txt.splitlines()
        assert "| name | value |" in csv_lines
        header_idx_csv = csv_lines.index("| name | value |")
        assert csv_lines[header_idx_csv + 1] == "|---|---|"
        assert "| alpha | 1 |" in csv_lines
        assert "| beta | 2 |" in csv_lines

    def test_dry_run_does_not_write(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _fix_table_md

        output, xlsx_md, csv_md, _good, _untouched = seeded_archive
        before_xlsx = xlsx_md.read_text(encoding="utf-8")
        before_csv = csv_md.read_text(encoding="utf-8")

        _fix_table_md(output, dry_run=True)

        assert xlsx_md.read_text(encoding="utf-8") == before_xlsx
        assert csv_md.read_text(encoding="utf-8") == before_csv

    def test_idempotent(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _fix_table_md

        output, xlsx_md, _csv_md, _good, _untouched = seeded_archive
        _fix_table_md(output, dry_run=False)
        first = xlsx_md.read_text(encoding="utf-8")
        _fix_table_md(output, dry_run=False)
        assert xlsx_md.read_text(encoding="utf-8") == first

    def test_already_valid_tables_untouched(self, seeded_archive):
        from mtss.cli.maintenance_cmd import _fix_table_md

        output, _xlsx_md, _csv_md, good, _untouched = seeded_archive
        before = good.read_text(encoding="utf-8")
        _fix_table_md(output, dry_run=False)
        assert good.read_text(encoding="utf-8") == before

    def test_non_table_md_files_ignored(self, seeded_archive):
        """Command only walks *.xlsx.md and *.csv.md — the email .md with a
        pipe in prose must remain untouched."""
        from mtss.cli.maintenance_cmd import _fix_table_md

        output, _xlsx_md, _csv_md, _good, untouched = seeded_archive
        before = untouched.read_text(encoding="utf-8")
        _fix_table_md(output, dry_run=False)
        assert untouched.read_text(encoding="utf-8") == before

    def test_missing_archive_dir_exits(self, tmp_path):
        import typer as _typer

        from mtss.cli.maintenance_cmd import _fix_table_md

        with pytest.raises(_typer.Exit):
            _fix_table_md(tmp_path / "no-such-output", dry_run=False)

    def test_block_widens_to_max_row_width(self, tmp_path):
        """Legacy rows have no escaping, so a cell-internal ``|`` is
        indistinguishable from a separator. The fixer splits greedily and
        widens the block to the max row width — the header is padded with
        empty cells so the delimiter row still matches GFM column-count
        rules. Data is preserved; no truncation or silent merge.
        """
        from mtss.cli.maintenance_cmd import _fix_table_md

        output = tmp_path / "output"
        folder = output / "archive" / "zzz" / "attachments"
        folder.mkdir(parents=True)
        md = folder / "mix.csv.md"
        md.write_text(
            "## Content\n"
            "\n"
            "name | note\n"
            "alpha | one|two\n",
            encoding="utf-8",
        )
        _fix_table_md(output, dry_run=False)
        text = md.read_text(encoding="utf-8")
        # Delimiter has 3 columns (widest row wins).
        assert "|---|---|---|" in text
        # Every original token still present; data row is valid GFM.
        assert "| alpha | one | two |" in text
        assert "| name | note |  |" in text

    def test_real_world_broken_xlsx_sample_becomes_gfm(self, tmp_path):
        """Representative subset of the real
        ``Weekly_Onboard_LO_Analysis_Log_22.xlsx.md`` that sits in
        production archive. Reproduces the exact structure — a sheet
        header whose first two rows both have leading empty cells
        rendered as ``" | value"`` — and asserts the post-fix table
        is valid GFM and parses as a table.
        """
        from mtss.cli.maintenance_cmd import _fix_table_md

        output = tmp_path / "output"
        folder = output / "archive" / "0b0a97618b16d6fe9d2bde33bde37e1f" / "attachments"
        folder.mkdir(parents=True)
        md = folder / "Weekly_Onboard_LO_Analysis_Log_22.xlsx.md"
        sample = (
            "# Weekly Onboard  LO Analysis Log (22).xlsx\n"
            "\n"
            "**Type:** application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\n"
            "**Size:** 28.8 KB\n"
            "**Extracted:** 2026-04-19 23:50:13\n"
            "\n"
            "---\n"
            "\n"
            "## Content\n"
            "\n"
            "## Sheet1\n"
            "\n"
            " | MAIN ENGINE |  |  | A/E #1 |  |  | STERN TUBE | \n"
            "Date | TBN | Water % | Viscosity | TBN | Water % | Viscosity | Water %\n"
            "2022-01-28 00:00:00 | 12 | <0.02 | OK | 27 | <0.02 | OK | <0.02\n"
            "2022-02-04 00:00:00 | 13 | <0.02 | OK | 25 | <0.02 | OK | <0.02\n"
            "2022-02-11 00:00:00 | 13 | <0.02 | OK | 25 | <0.02 | OK | <0.02\n"
        )
        md.write_text(sample, encoding="utf-8")

        # Dry-run first.
        before = md.read_text(encoding="utf-8")
        _fix_table_md(output, dry_run=True)
        assert md.read_text(encoding="utf-8") == before, "dry-run must not write"

        # Apply.
        _fix_table_md(output, dry_run=False)
        after = md.read_text(encoding="utf-8")
        assert after != before
        # The first table row is now valid GFM (leading + trailing pipe)
        # and contains the header cell from the source.
        assert "| MAIN ENGINE |" in after
        for line in after.splitlines():
            if "MAIN ENGINE" in line:
                assert line.startswith("| ") and line.endswith("|"), (
                    f"header row not valid GFM: {line!r}"
                )
                break
        # A delimiter row follows the first data row (exact column count: 8).
        assert "|---|---|---|---|---|---|---|---|" in after
        # Original data preserved — dates still present.
        assert "2022-01-28 00:00:00" in after
        assert "2022-02-11 00:00:00" in after
        # Every non-blank line in the table block starts with `|` now.
        in_table = False
        for line in after.splitlines():
            if line.startswith("|---"):
                in_table = True
                continue
            if in_table and not line.strip():
                break
            if in_table:
                assert line.startswith("| "), f"broken row in table: {line!r}"

        # Idempotency.
        _fix_table_md(output, dry_run=False)
        assert md.read_text(encoding="utf-8") == after
