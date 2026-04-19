"""Tests for persisting ProcessingTrail to archive metadata.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtss.ingest.archive_generator import ArchiveGenerator
from mtss.ingest.processing_trail import ProcessingTrail
from mtss.storage.local_client import LocalBucketStorage
from mtss.utils import compute_folder_id


@pytest.fixture
def archive_setup(tmp_path: Path):
    """Build an ArchiveGenerator + a pre-existing metadata.json."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    storage = LocalBucketStorage(bucket_dir=archive_dir)

    doc_id = "abcdef1234567890" + "00" * 8
    folder_id = compute_folder_id(doc_id)
    seed_metadata = {
        "doc_id": doc_id,
        "folder_id": folder_id,
        "subject": "Test subject",
        "participants": ["a@x.com"],
        "attachments": {},
    }
    storage.upload_text(
        f"{folder_id}/metadata.json",
        json.dumps(seed_metadata, indent=2),
        "application/json",
    )

    gen = ArchiveGenerator(ingest_root=tmp_path / "source", storage=storage)
    return gen, storage, doc_id, folder_id


def _read_metadata(storage: LocalBucketStorage, folder_id: str) -> dict:
    return json.loads(storage.download_text(f"{folder_id}/metadata.json"))


def test_finalize_adds_processing_key_to_metadata(archive_setup):
    gen, storage, doc_id, folder_id = archive_setup

    trail = ProcessingTrail()
    trail.stamp_email("parse", parser="eml_local")
    trail.stamp_email("context", model="openrouter/openai/gpt-5-nano")
    trail.stamp_attachment(
        "spec.pdf", "parse", model="google/gemini-2.5-flash", parser="gemini_pdf"
    )

    ok = gen.finalize_metadata_processing(doc_id, trail.to_json())
    assert ok is True

    data = _read_metadata(storage, folder_id)
    # Seed fields must still be intact
    assert data["subject"] == "Test subject"
    # New processing key
    assert "processing" in data
    assert "parse" in data["processing"]["email"]
    assert data["processing"]["attachments"]["spec.pdf"]["parse"]["parser"] == "gemini_pdf"


def test_finalize_overwrites_prior_processing(archive_setup):
    gen, storage, doc_id, folder_id = archive_setup

    first = ProcessingTrail()
    first.stamp_email("embed", model="text-embedding-3-small", chunk_count=5)
    gen.finalize_metadata_processing(doc_id, first.to_json())

    second = ProcessingTrail()
    second.stamp_email("embed", model="text-embedding-3-large", chunk_count=7)
    gen.finalize_metadata_processing(doc_id, second.to_json())

    data = _read_metadata(storage, folder_id)
    assert data["processing"]["email"]["embed"]["model"] == "text-embedding-3-large"
    assert data["processing"]["email"]["embed"]["chunk_count"] == 7


def test_finalize_returns_false_when_metadata_missing(archive_setup):
    gen, storage, _doc_id, folder_id = archive_setup
    missing_doc_id = "1111111122222222" + "00" * 8

    trail = ProcessingTrail()
    trail.stamp_email("parse", parser="eml_local")
    ok = gen.finalize_metadata_processing(missing_doc_id, trail.to_json())

    assert ok is False


def test_finalize_never_raises_on_bad_json(tmp_path: Path):
    """Corrupt metadata.json → finalize returns False, doesn't break ingest."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    storage = LocalBucketStorage(bucket_dir=archive_dir)
    doc_id = "f" * 16 + "0" * 16
    folder_id = compute_folder_id(doc_id)

    # Write malformed JSON
    storage.upload_text(f"{folder_id}/metadata.json", "{not valid json", "application/json")

    gen = ArchiveGenerator(ingest_root=tmp_path / "source", storage=storage)
    trail = ProcessingTrail()
    trail.stamp_email("parse", parser="eml_local")

    ok = gen.finalize_metadata_processing(doc_id, trail.to_json())
    assert ok is False  # Swallows the error, doesn't raise
