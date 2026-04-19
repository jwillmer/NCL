"""Tests for the pipeline/attachment_handler helpers that populate the trail.

Covers:
- `_stamp_embed_by_document` (pipeline)
- `BaseParser.model_name` contract — deterministic parsers return ``None``;
  LLM-backed parsers expose the provider/model string stamped onto the trail.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

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


def test_base_parser_model_name_defaults_to_none():
    """Deterministic parsers inherit ``None`` — the trail records parser name only."""
    from mtss.parsers.base import BaseParser

    class _Dummy(BaseParser):
        name = "dummy"

        async def parse(self, file_path):  # pragma: no cover - not exercised
            return ""

    assert _Dummy().model_name is None


def test_gemini_parser_model_name_returns_configured_model(monkeypatch):
    from mtss import config as cfg_mod
    from mtss.parsers.gemini_pdf_parser import GeminiPDFParser

    cfg_mod.get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setenv("GEMINI_PDF_MODEL", "openrouter/google/gemini-2.5-flash")
    try:
        result = GeminiPDFParser().model_name
        assert result is not None
        assert "gemini" in result.lower()
    finally:
        cfg_mod.get_settings.cache_clear()  # type: ignore[attr-defined]


def test_llamaparse_parser_model_name_returns_tier_marker():
    """llama-cloud 2.x exposes no user-facing model name — we record the tier
    so the trail still distinguishes LLM-parsed from local-parsed."""
    from mtss.parsers.llamaparse_parser import LlamaParseParser

    assert LlamaParseParser().model_name == "llamaparse:agentic"
