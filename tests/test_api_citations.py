"""Tests for the /api/citations/{chunk_id} endpoint.

Covers the two recent behavior changes:
- download URL is signed at response time so the UI can open it in a new
  tab without a Bearer header;
- archive content-fetch errors surface via logger.exception instead of
  silently swallowing to ``content: None``.
"""

import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from mtss.models.chunk import Chunk
from mtss.storage.archive_storage import ArchiveStorageError


def _build_chunk(
    *,
    archive_browse_uri: str | None = "/archive/abc1234567890/attachments/file.xlsx.md",
    archive_download_uri: str | None = "/archive/abc1234567890/attachments/file.xlsx",
) -> Chunk:
    return Chunk(
        document_id=uuid4(),
        chunk_id="aabbccddee12",
        content="body",
        chunk_index=0,
        source_title="file.xlsx",
        line_from=1,
        line_to=10,
        archive_browse_uri=archive_browse_uri,
        archive_download_uri=archive_download_uri,
    )


def _storage_factory(mock_storage: MagicMock):
    def _factory(*args, **kwargs):  # noqa: ARG001 - match ArchiveStorage() signature
        return mock_storage

    return _factory


@pytest.mark.asyncio
async def test_citation_returns_signed_download_url_and_content(app, client, auth_headers):
    """Happy path: content downloaded, signed URL minted, both in response."""
    chunk = _build_chunk()
    supabase = app.state.mock_supabase_rest_client
    with patch(
        "mtss.api.main.SupabaseClient.get_chunk_by_id", return_value=chunk
    ), patch("mtss.api.main.SupabaseClient", autospec=False) as MockClient:
        MockClient.return_value.get_chunk_by_id = MagicMock(return_value=chunk)

        storage = MagicMock()
        storage.download_file = MagicMock(return_value=b"# markdown")
        storage.create_signed_url = MagicMock(return_value="https://cdn.example/x?token=abc")

        with patch("mtss.api.main.ArchiveStorage", side_effect=_storage_factory(storage)):
            response = await client.get(
                f"/api/citations/{chunk.chunk_id}", headers=auth_headers
            )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] == "# markdown"
    assert body["archive_download_signed_url"] == "https://cdn.example/x?token=abc"
    # Path passed to storage is bucket-relative (no leading /archive/)
    storage.download_file.assert_called_once_with("abc1234567890/attachments/file.xlsx.md")
    storage.create_signed_url.assert_called_once_with(
        "abc1234567890/attachments/file.xlsx", expires_in=300
    )
    # `supabase` fixture is kept so the `app` teardown has a reference
    del supabase


@pytest.mark.asyncio
async def test_citation_content_error_is_logged_not_swallowed(
    app, client, auth_headers, caplog
):
    """Download failures must reach the logger via logger.exception.

    Regression for the pre-fix behavior where a generic try/except mapped
    every error to ``logger.warning`` + ``content=None``, hiding real
    misconfiguration (wrong bucket, expired creds, missing object).
    """
    chunk = _build_chunk()
    storage = MagicMock()
    storage.download_file = MagicMock(
        side_effect=ArchiveStorageError("simulated bucket miss")
    )
    storage.create_signed_url = MagicMock(return_value="https://cdn.example/x?token=abc")

    with patch(
        "mtss.api.main.SupabaseClient.get_chunk_by_id", return_value=chunk
    ), patch("mtss.api.main.SupabaseClient") as MockClient:
        MockClient.return_value.get_chunk_by_id = MagicMock(return_value=chunk)
        with patch("mtss.api.main.ArchiveStorage", side_effect=_storage_factory(storage)):
            with caplog.at_level(logging.ERROR, logger="mtss.api.main"):
                response = await client.get(
                    f"/api/citations/{chunk.chunk_id}", headers=auth_headers
                )

    assert response.status_code == 200, response.text
    body = response.json()
    # Content absent; the rest of the response still renders.
    assert body["content"] is None
    assert body["source_title"] == "file.xlsx"
    # Signed URL still produced — download and signing are independent paths.
    assert body["archive_download_signed_url"] == "https://cdn.example/x?token=abc"
    # logger.exception attaches exc_info — pin that to prevent a future
    # "logger.warning" regression.
    archive_records = [
        r for r in caplog.records
        if "Archive .md not found" in r.getMessage()
    ]
    assert archive_records, f"expected archive error log; got {[r.getMessage() for r in caplog.records]}"
    assert archive_records[0].exc_info is not None
    del app  # silence unused-fixture lint; app is required for client wiring


@pytest.mark.asyncio
async def test_citation_signed_url_failure_does_not_break_response(
    app, client, auth_headers, caplog
):
    """If signing fails, content still flows and signed URL is null."""
    chunk = _build_chunk()
    storage = MagicMock()
    storage.download_file = MagicMock(return_value=b"# body")
    storage.create_signed_url = MagicMock(
        side_effect=ArchiveStorageError("no such object")
    )

    with patch(
        "mtss.api.main.SupabaseClient.get_chunk_by_id", return_value=chunk
    ), patch("mtss.api.main.SupabaseClient") as MockClient:
        MockClient.return_value.get_chunk_by_id = MagicMock(return_value=chunk)
        with patch("mtss.api.main.ArchiveStorage", side_effect=_storage_factory(storage)):
            with caplog.at_level(logging.ERROR, logger="mtss.api.main"):
                response = await client.get(
                    f"/api/citations/{chunk.chunk_id}", headers=auth_headers
                )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] == "# body"
    assert body["archive_download_signed_url"] is None
    del app


@pytest.mark.asyncio
async def test_citation_without_archive_uris_omits_signed_url(
    app, client, auth_headers
):
    """Chunks with no archive URIs (e.g. synthesised) get neither content nor a signed URL."""
    chunk = _build_chunk(archive_browse_uri=None, archive_download_uri=None)
    storage = MagicMock()
    storage.download_file = MagicMock()
    storage.create_signed_url = MagicMock()

    with patch(
        "mtss.api.main.SupabaseClient.get_chunk_by_id", return_value=chunk
    ), patch("mtss.api.main.SupabaseClient") as MockClient:
        MockClient.return_value.get_chunk_by_id = MagicMock(return_value=chunk)
        with patch("mtss.api.main.ArchiveStorage", side_effect=_storage_factory(storage)):
            response = await client.get(
                f"/api/citations/{chunk.chunk_id}", headers=auth_headers
            )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] is None
    assert body["archive_download_uri"] is None
    assert body["archive_download_signed_url"] is None
    storage.download_file.assert_not_called()
    storage.create_signed_url.assert_not_called()
    del app
