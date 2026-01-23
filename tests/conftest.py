"""Pytest configuration and fixtures for MTSS tests."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_eml_content():
    """Sample EML file content for testing."""
    return b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <test123@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset="utf-8"

This is the plain text body of the test email.

--boundary123
Content-Type: application/pdf; name="test.pdf"
Content-Disposition: attachment; filename="test.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRv
YmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAw
IG9iago8PC9UeXBlL1BhZ2UvTWVkaWFCb3hbMCAwIDYxMiA3OTJdL1BhcmVudCAyIDAgUj4+CmVu
ZG9iagp4cmVmCjAgNAowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMTUgMDAwMDAgbiAKMDAw
MDAwMDA2NiAwMDAwMCBuIAowMDAwMDAwMTIzIDAwMDAwIG4gCnRyYWlsZXIKPDwvU2l6ZSA0L1Jv
b3QgMSAwIFI+PgpzdGFydHhyZWYKMjAwCiUlRU9GCg==

--boundary123--
"""


@pytest.fixture
def sample_eml_file(temp_dir, sample_eml_content):
    """Create a sample EML file for testing."""
    eml_path = temp_dir / "test_email.eml"
    eml_path.write_bytes(sample_eml_content)
    return eml_path


@pytest.fixture
def simple_eml_content():
    """Simple EML file content without attachments."""
    return b"""From: sender@example.com
To: recipient@example.com
Subject: Simple Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Content-Type: text/plain; charset="utf-8"

This is a simple test email without any attachments.
It has multiple lines.
"""


@pytest.fixture
def simple_eml_file(temp_dir, simple_eml_content):
    """Create a simple EML file without attachments."""
    eml_path = temp_dir / "simple_email.eml"
    eml_path.write_bytes(simple_eml_content)
    return eml_path


@pytest.fixture
def real_eml_file():
    """Path to the real test email with multiple attachments (PDF, ZIP, PNG)."""
    return Path(__file__).parent / "fixtures" / "test_email.eml"


@pytest.fixture
def mock_settings():
    """Mock settings for testing without requiring environment variables."""
    settings = MagicMock()
    settings.enable_ocr = False
    settings.enable_picture_description = False
    settings.embedding_model = "text-embedding-3-small"
    settings.chunk_size_tokens = 512
    settings.data_source_dir = Path("./data/emails")
    settings.attachments_dir = Path("./data/attachments")
    return settings


# ==================== New Fixtures for Ingest Tests ====================


@pytest.fixture
def comprehensive_mock_settings():
    """Extended settings with all config values needed for ingest testing."""
    settings = MagicMock()
    # Basic settings
    settings.enable_ocr = False
    settings.enable_picture_description = False
    settings.embedding_model = "text-embedding-3-small"
    settings.embedding_dimensions = 1536
    settings.embedding_max_tokens = 8191
    settings.max_concurrent_embeddings = 5
    settings.embedding_batch_size = 100
    # Chunking settings
    settings.chunk_size_tokens = 512
    settings.chunk_overlap_tokens = 50
    # Context generation settings
    settings.context_llm_model = "gpt-4o-mini"
    settings.context_llm_fallback = "gpt-4o"
    settings.get_model = MagicMock(side_effect=lambda x: x)
    # Directory settings
    settings.data_source_dir = Path("./data/emails")
    settings.eml_source_dir = Path("./data/emails")
    settings.attachments_dir = Path("./data/attachments")
    # ZIP settings
    settings.zip_max_depth = 3
    settings.zip_max_files = 100
    settings.zip_max_total_size_mb = 100
    # Database settings
    settings.supabase_url = "https://test.supabase.co"
    settings.supabase_key = "test-key"
    settings.supabase_db_url = "postgresql://test:test@localhost:5432/test"
    # Archive settings
    settings.archive_base_url = "https://archive.example.com"
    settings.current_ingest_version = 1
    return settings


@pytest.fixture
def sample_document_id():
    """Fixed UUID for testing."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_document(sample_document_id):
    """Sample Document model for testing."""
    from mtss.models.document import (
        Document,
        DocumentType,
        EmailMetadata,
        ProcessingStatus,
    )

    return Document(
        id=sample_document_id,
        parent_id=None,
        root_id=sample_document_id,
        depth=0,
        path=[str(sample_document_id)],
        document_type=DocumentType.EMAIL,
        file_path="/data/emails/test.eml",
        file_name="test.eml",
        file_hash="abc123def456",
        source_id="test.eml",
        doc_id="doc123abc",
        content_version=1,
        ingest_version=1,
        source_title="Test Email Subject",
        archive_browse_uri="/archive/doc123abc/test.eml.md",
        archive_download_uri="/archive/doc123abc/test.eml",
        email_metadata=EmailMetadata(
            subject="Test Email Subject",
            participants=["sender@example.com", "recipient@example.com"],
            initiator="sender@example.com",
            date_start=datetime(2024, 1, 1, 12, 0, 0),
            message_count=1,
        ),
        status=ProcessingStatus.PENDING,
    )


@pytest.fixture
def sample_attachment_document(sample_document_id):
    """Sample attachment Document model for testing."""
    from mtss.models.document import (
        AttachmentMetadata,
        Document,
        DocumentType,
        ProcessingStatus,
    )

    attachment_id = UUID("87654321-4321-8765-4321-876543218765")
    return Document(
        id=attachment_id,
        parent_id=sample_document_id,
        root_id=sample_document_id,
        depth=1,
        path=[str(sample_document_id), str(attachment_id)],
        document_type=DocumentType.ATTACHMENT_PDF,
        file_path="/data/attachments/report.pdf",
        file_name="report.pdf",
        file_hash="pdf123hash",
        source_id="test.eml/report.pdf",
        doc_id="pdfdoc123",
        content_version=1,
        ingest_version=1,
        source_title="report.pdf",
        archive_browse_uri="/archive/doc123abc/attachments/report.pdf.md",
        archive_download_uri="/archive/doc123abc/attachments/report.pdf",
        attachment_metadata=AttachmentMetadata(
            content_type="application/pdf",
            size_bytes=12345,
            original_filename="report.pdf",
        ),
        status=ProcessingStatus.PENDING,
    )


@pytest.fixture
def sample_chunk(sample_document_id):
    """Sample Chunk model for testing."""
    from mtss.models.chunk import Chunk

    return Chunk(
        id=uuid4(),
        document_id=sample_document_id,
        chunk_id="abc123def456",
        content="This is sample chunk content for testing purposes.",
        chunk_index=0,
        context_summary="This is an email about testing.",
        embedding_text="This is an email about testing.\n\nThis is sample chunk content for testing purposes.",
        section_path=["Introduction", "Overview"],
        section_title="Overview",
        source_title="Test Email Subject",
        source_id="test.eml",
        line_from=1,
        line_to=5,
        char_start=0,
        char_end=50,
        archive_browse_uri="/archive/doc123abc/test.eml.md",
        archive_download_uri="/archive/doc123abc/test.eml",
        embedding=[0.1] * 1536,
        metadata={"source_file": "/data/emails/test.eml"},
    )


@pytest.fixture
def sample_chunks(sample_document_id):
    """Multiple sample chunks for batch testing."""
    from mtss.models.chunk import Chunk

    return [
        Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id=f"chunk{i:03d}",
            content=f"Sample chunk content number {i}.",
            chunk_index=i,
            section_path=[],
            metadata={},
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client with both sync and async methods."""
    client = MagicMock()
    # Async methods
    client.insert_document = AsyncMock()
    client.update_document_status = AsyncMock()
    client.get_document_by_hash = AsyncMock(return_value=None)
    client.get_document_by_id = AsyncMock()
    client.get_document_ancestry = AsyncMock(return_value=[])
    client.get_document_children = AsyncMock(return_value=[])
    client.insert_chunks = AsyncMock()
    client.replace_chunks_atomic = AsyncMock()
    client.delete_chunks_by_document = AsyncMock(return_value=0)
    client.get_chunks_by_document = AsyncMock(return_value=[])
    client.get_pool = AsyncMock()
    client.close = AsyncMock()
    # Ingest-update specific async methods
    client.get_all_root_source_ids = AsyncMock(return_value={})
    client.get_document_by_source_id = AsyncMock(return_value=None)
    client.delete_orphaned_documents = AsyncMock(return_value=0)
    client.update_document_archive_browse_uri = AsyncMock()
    client.update_chunk_context = AsyncMock()
    # Sync methods
    client.log_ingest_event = MagicMock()
    client.get_chunk_by_id = MagicMock(return_value=None)
    client.delete_document_for_reprocess = MagicMock()
    return client


@pytest.fixture
def mock_embedding_response():
    """Mock LiteLLM embedding response for single embedding."""
    response = MagicMock()
    response.data = [{"embedding": [0.1] * 1536}]
    return response


@pytest.fixture
def mock_batch_embedding_response():
    """Mock LiteLLM embedding response for batch embeddings."""
    response = MagicMock()
    response.data = [{"embedding": [0.1 * (i + 1)] * 1536} for i in range(5)]
    return response


@pytest.fixture
def mock_llm_completion_response():
    """Mock LiteLLM completion response for context generation."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="This is an email from sender@example.com dated 2024-01-01 about testing procedures."
            )
        )
    ]
    return response


@pytest.fixture
def sample_email_metadata():
    """Sample EmailMetadata for testing."""
    from mtss.models.document import EmailMetadata

    return EmailMetadata(
        subject="RE: Project Update",
        participants=["alice@example.com", "bob@example.com", "charlie@example.com"],
        initiator="alice@example.com",
        date_start=datetime(2024, 1, 15, 10, 30, 0),
        date_end=datetime(2024, 1, 15, 14, 45, 0),
        message_count=3,
    )


@pytest.fixture
def sample_parsed_email(sample_email_metadata):
    """Sample ParsedEmail for testing."""
    from mtss.models.document import ParsedEmail

    return ParsedEmail(
        metadata=sample_email_metadata,
        full_text="""From: alice@example.com
Subject: RE: Project Update

Hi team,

Please find attached the latest project status report.

Best regards,
Alice""",
        attachments=[],
    )


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for chunking tests."""
    return """# Introduction

This is the introduction section of the document.

## Overview

This section provides an overview of the project.

### Key Points

- Point one about the project
- Point two about implementation
- Point three about timeline

## Technical Details

The technical implementation involves several components.

### Architecture

The system uses a microservices architecture.

### Database

PostgreSQL with pgvector extension for similarity search.
"""


@pytest.fixture
def sample_email_body():
    """Sample email body content for testing."""
    return """From: John Smith <john@example.com>
To: Jane Doe <jane@example.com>
Date: January 15, 2024

Hi Jane,

I wanted to follow up on our discussion about the vessel maintenance schedule.
The MV Nordic Star requires immediate attention for the following items:

1. Engine oil change
2. Hull inspection
3. Navigation equipment calibration

Please coordinate with the port authority for scheduling.

Best regards,
John

---
Previous message:

From: Jane Doe <jane@example.com>
To: John Smith <john@example.com>

John,

Can you provide the maintenance requirements for the fleet?

Thanks,
Jane
"""


@pytest.fixture
def mock_console():
    """Mock Rich console for IssueTracker tests."""
    console = MagicMock()
    console.print = MagicMock()
    return console


# ==================== Ingest-Update Test Fixtures ====================


@pytest.fixture
def mock_archive_storage():
    """Mock ArchiveStorage for bucket operations."""
    storage = MagicMock()
    storage.file_exists = MagicMock(return_value=False)
    storage.download_file = MagicMock(return_value=b"# Markdown content")
    storage.upload_file = MagicMock()
    storage.delete_folder = MagicMock()
    return storage


@pytest.fixture
def mock_ingest_components(mock_supabase_client, mock_archive_storage):
    """Mock IngestComponents dataclass."""
    components = MagicMock()
    components.db = mock_supabase_client
    components.archive_storage = mock_archive_storage
    components.eml_parser = MagicMock()
    components.chunker = MagicMock()
    components.context_generator = MagicMock()
    components.embeddings = MagicMock()
    components.archive_generator = MagicMock()
    return components


@pytest.fixture
def sample_document_missing_archive(sample_document):
    """Document with no archive_browse_uri."""
    doc = sample_document.model_copy()
    doc.archive_browse_uri = None
    doc.archive_download_uri = None
    return doc


@pytest.fixture
def sample_chunks_missing_lines(sample_document_id):
    """Chunks with NULL line_from/line_to."""
    from mtss.models.chunk import Chunk

    return [
        Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id=f"chunk{i:03d}",
            content=f"Sample content {i}",
            chunk_index=i,
            line_from=None,
            line_to=None,
            section_path=[],
            metadata={},
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_chunks_missing_context(sample_document_id):
    """Chunks with NULL context_summary."""
    from mtss.models.chunk import Chunk

    return [
        Chunk(
            id=uuid4(),
            document_id=sample_document_id,
            chunk_id=f"chunk{i:03d}",
            content=f"Sample content {i}",
            chunk_index=i,
            line_from=1,
            line_to=10,
            context_summary=None,
            section_path=[],
            metadata={},
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_image_document(sample_document_id):
    """Sample image attachment Document for testing."""
    from mtss.models.document import (
        AttachmentMetadata,
        Document,
        DocumentType,
        ProcessingStatus,
    )

    image_id = UUID("11111111-2222-3333-4444-555555555555")
    return Document(
        id=image_id,
        parent_id=sample_document_id,
        root_id=sample_document_id,
        depth=1,
        path=[str(sample_document_id), str(image_id)],
        document_type=DocumentType.ATTACHMENT_IMAGE,
        file_path="/data/attachments/image.png",
        file_name="image.png",
        file_hash="img123hash",
        source_id="test.eml/image.png",
        doc_id="imgdoc123",
        content_version=1,
        ingest_version=1,
        source_title="image.png",
        archive_browse_uri=None,  # Images typically don't have browse URIs
        archive_download_uri="/archive/doc123abc/attachments/image.png",
        attachment_metadata=AttachmentMetadata(
            content_type="image/png",
            size_bytes=54321,
            original_filename="image.png",
        ),
        status=ProcessingStatus.PENDING,
    )


# ==================== Local Storage Test Fixtures ====================


@pytest.fixture
def local_ingest_output(tmp_path):
    """Create local storage for capturing ingest output.

    This fixture provides a way to inspect what ingest would write
    to Supabase without actually connecting to the database.

    Example usage:
        def test_ingest(local_ingest_output):
            # Run ingest with local_ingest_output.db as the db client
            # Check output:
            docs = local_ingest_output.read_documents_jsonl()
            chunks = local_ingest_output.read_chunks_jsonl()
    """
    from tests.local_storage import LocalIngestOutput

    return LocalIngestOutput.create(tmp_path)


@pytest.fixture
def local_db_client(tmp_path):
    """Create a local database client for testing."""
    from tests.local_storage import LocalStorageClient

    return LocalStorageClient(tmp_path / "database")


@pytest.fixture
def local_bucket_storage(tmp_path):
    """Create a local bucket storage for testing."""
    from tests.local_storage import LocalBucketStorage

    return LocalBucketStorage(tmp_path / "bucket")
