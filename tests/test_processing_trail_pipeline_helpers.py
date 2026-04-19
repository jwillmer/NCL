"""Tests for the pipeline/attachment_handler helpers that populate the trail.

These cover the two small pure helpers added alongside trail wiring:
- `_stamp_embed_by_document` (pipeline)
- `_resolve_parser_model` (attachment_handler)
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from mtss.ingest.attachment_handler import _resolve_parser_model
from mtss.ingest.pipeline import _stamp_embed_by_document
from mtss.ingest.processing_trail import ProcessingTrail


def test_stamp_embed_groups_chunks_by_document_id():
    trail = ProcessingTrail()
    email_uuid = uuid4()
    att_a_uuid = uuid4()
    att_b_uuid = uuid4()
    trail.register_email_document(email_uuid)
    trail.register_attachment_document(att_a_uuid, "a.pdf")
    trail.register_attachment_document(att_b_uuid, "b.pdf")

    chunks = [
        SimpleNamespace(document_id=email_uuid),
        SimpleNamespace(document_id=email_uuid),
        SimpleNamespace(document_id=att_a_uuid),
        SimpleNamespace(document_id=att_b_uuid),
        SimpleNamespace(document_id=att_b_uuid),
        SimpleNamespace(document_id=att_b_uuid),
    ]

    _stamp_embed_by_document(trail, chunks, embedding_model="text-embedding-3-large")

    data = trail.to_json()
    assert data["email"]["embed"]["chunk_count"] == 2
    assert data["email"]["embed"]["model"] == "text-embedding-3-large"
    assert data["attachments"]["a.pdf"]["embed"]["chunk_count"] == 1
    assert data["attachments"]["b.pdf"]["embed"]["chunk_count"] == 3


def test_stamp_embed_skips_unregistered_documents():
    """Embedding stamps for unknown document_ids silently drop — callers
    fan out over chunks without pre-filtering."""
    trail = ProcessingTrail()
    unknown = uuid4()
    _stamp_embed_by_document(
        trail,
        [SimpleNamespace(document_id=unknown)],
        embedding_model="text-embedding-3-large",
    )
    assert trail.to_json() == {"email": {}, "attachments": {}}


def test_resolve_parser_model_local_parser_returns_none():
    assert _resolve_parser_model("local_pdf") is None
    assert _resolve_parser_model("eml_local") is None
    assert _resolve_parser_model(None) is None
    assert _resolve_parser_model("") is None


def test_resolve_parser_model_gemini_returns_configured_model(monkeypatch):
    from mtss import config as cfg_mod

    cfg_mod.get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setenv("GEMINI_PDF_MODEL", "openrouter/google/gemini-2.5-flash")
    try:
        result = _resolve_parser_model("gemini_pdf")
        # Settings may surface the raw env value (expected) or a pre-resolved
        # alias. Just assert the shape — the model field is present and
        # references Gemini.
        assert result is not None
        assert "gemini" in result.lower()
    finally:
        cfg_mod.get_settings.cache_clear()  # type: ignore[attr-defined]


def test_resolve_parser_model_llamaparse_returns_marker():
    """llama-cloud 2.x doesn't expose a user-facing model name — we record the
    tier instead so the trail still distinguishes LLM-parsed from local-parsed."""
    assert _resolve_parser_model("llamaparse") == "llamaparse:agentic"
