"""Unit tests for ProcessingTrail."""

from __future__ import annotations

import re
from uuid import uuid4

import pytest

from mtss.ingest.processing_trail import ProcessingTrail


ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_stamp_email_records_ran_at_and_model():
    t = ProcessingTrail()
    t.stamp_email("parse", model=None, parser="eml_local")
    out = t.to_json()

    assert "parse" in out["email"]
    entry = out["email"]["parse"]
    assert ISO_Z.match(entry["ran_at"])
    assert entry["parser"] == "eml_local"
    assert "model" not in entry  # None model is omitted


def test_stamp_email_with_model_keeps_model_field():
    t = ProcessingTrail()
    t.stamp_email("context", model="openrouter/openai/gpt-5-nano")
    out = t.to_json()

    assert out["email"]["context"]["model"] == "openrouter/openai/gpt-5-nano"


def test_stamp_attachment_keyed_by_filename():
    t = ProcessingTrail()
    t.stamp_attachment("spec.pdf", "parse", model="gemini-2.5-flash", pages=12)
    t.stamp_attachment("img.png", "vision", model="openrouter/google/gemini-2.5-flash")

    out = t.to_json()
    assert set(out["attachments"].keys()) == {"spec.pdf", "img.png"}
    assert out["attachments"]["spec.pdf"]["parse"]["pages"] == 12
    assert out["attachments"]["img.png"]["vision"]["model"] == (
        "openrouter/google/gemini-2.5-flash"
    )


def test_stamp_is_current_state_only_last_write_wins():
    t = ProcessingTrail()
    t.stamp_email("context", model="modelA")
    t.stamp_email("context", model="modelB", fallback_used=True)

    entry = t.to_json()["email"]["context"]
    assert entry["model"] == "modelB"
    assert entry["fallback_used"] is True


def test_stamp_attachment_different_steps_coexist():
    t = ProcessingTrail()
    t.stamp_attachment("spec.pdf", "parse", model="gemini-2.5-flash")
    t.stamp_attachment("spec.pdf", "context", model="gpt-5-nano")
    t.stamp_attachment("spec.pdf", "embed", model="text-embedding-3-large", chunk_count=8)

    spec = t.to_json()["attachments"]["spec.pdf"]
    assert set(spec.keys()) == {"parse", "context", "embed"}
    assert spec["embed"]["chunk_count"] == 8


def test_stamp_by_document_id_routes_to_email():
    t = ProcessingTrail()
    email_doc_id = str(uuid4())
    t.register_email_document(email_doc_id)

    t.stamp_by_document_id(
        email_doc_id, "embed", model="text-embedding-3-large", chunk_count=3
    )

    out = t.to_json()
    assert out["email"]["embed"]["chunk_count"] == 3
    assert out["attachments"] == {}


def test_stamp_by_document_id_routes_to_attachment():
    t = ProcessingTrail()
    att_doc_id = str(uuid4())
    t.register_attachment_document(att_doc_id, "spec.pdf")

    t.stamp_by_document_id(
        att_doc_id, "embed", model="text-embedding-3-large", chunk_count=8
    )

    out = t.to_json()
    assert out["attachments"]["spec.pdf"]["embed"]["chunk_count"] == 8
    assert out["email"] == {}


def test_stamp_by_document_id_unknown_is_noop():
    """Unknown document_ids silently drop — lets callers fan out over
    chunk.document_id without pre-filtering."""
    t = ProcessingTrail()
    t.stamp_by_document_id(str(uuid4()), "embed", model="text-embedding-3-large")

    assert t.to_json() == {"email": {}, "attachments": {}}


def test_to_json_empty_trail_has_both_keys():
    t = ProcessingTrail()
    out = t.to_json()
    assert out == {"email": {}, "attachments": {}}


def test_details_none_values_are_dropped():
    t = ProcessingTrail()
    t.stamp_email("decider", model=None, mode="full", reason=None)

    entry = t.to_json()["email"]["decider"]
    assert entry["mode"] == "full"
    assert "reason" not in entry
    assert "model" not in entry


def test_accepts_uuid_objects_for_document_id():
    t = ProcessingTrail()
    doc_uuid = uuid4()
    t.register_attachment_document(doc_uuid, "spec.pdf")

    t.stamp_by_document_id(doc_uuid, "embed", model="text-embedding-3-large")

    assert "embed" in t.to_json()["attachments"]["spec.pdf"]
