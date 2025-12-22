"""Pytest configuration and fixtures for NCL tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

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
