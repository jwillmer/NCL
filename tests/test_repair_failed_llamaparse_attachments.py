"""Tests for scripts/repair_failed_llamaparse_attachments.py.

These lock the repair script's contract against the already-ingested data it
will touch — classification rules, idempotency, atomic rewrite, and the
end-to-end flow for a happy-path candidate. All tests run fully in-memory
with ``tmp_path`` and fake components; no network, no real LlamaParse API.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

_REPAIR_PATH = (
    Path(__file__).parent.parent / "scripts" / "repair_failed_llamaparse_attachments.py"
)


@pytest.fixture(scope="module")
def repair_mod():
    """Load the script as a module (it lives under scripts/, not src/)."""
    spec = importlib.util.spec_from_file_location("repair_llp", _REPAIR_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repair_llp"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _failed_attachment_doc(
    *,
    id_: str,
    root_id: str,
    file_name: str,
    file_path_on_disk: str,
    error: str = "LlamaParse failed: AsyncParsingResource.parse() got an unexpected keyword argument 'cost_optimizer'",
) -> dict:
    return {
        "id": id_,
        "source_id": f"src:{file_name}",
        "doc_id": "a" * 32,
        "ingest_version": 5,
        "document_type": "attachment_pdf",
        "file_path": file_path_on_disk,
        "file_name": file_name,
        "file_hash": "h",
        "parent_id": root_id,
        "root_id": root_id,
        "depth": 1,
        "source_title": file_name,
        "archive_path": None,
        "archive_browse_uri": None,
        "archive_download_uri": None,
        "status": "failed",
        "error_message": error,
        "attachment_content_type": "application/pdf",
        "attachment_size_bytes": 1024,
    }


def _root_email_doc(id_: str, doc_id: str) -> dict:
    return {
        "id": id_,
        "source_id": f"src:{id_}.eml",
        "doc_id": doc_id,
        "ingest_version": 5,
        "document_type": "email",
        "file_path": f"data/emails/{id_}.eml",
        "file_name": f"{id_}.eml",
        "file_hash": "he",
        "parent_id": None,
        "root_id": id_,
        "depth": 0,
        "source_title": id_,
        "archive_path": doc_id[:16],
        "archive_browse_uri": None,
        "archive_download_uri": None,
        "status": "completed",
        "error_message": None,
    }


# ---------------------------------------------------------------------------
# find_candidates
# ---------------------------------------------------------------------------


class TestFindCandidates:
    """Classification rules: what goes into the straightforward bucket, what
    gets skipped, and why."""

    def test_classifies_each_bucket(self, repair_mod, tmp_path):
        output = tmp_path / "output"
        archive = output / "archive"
        root_uuid = str(uuid4())
        root_doc_id = "b" * 32  # → folder "bbbbbbbbbbbbbbbb"
        rid16 = root_doc_id[:16]

        # 1. Straightforward: archive file present, failed with cost_optimizer
        straight_id = str(uuid4())
        straight_file = archive / rid16 / "attachments" / "good.pdf"
        straight_file.parent.mkdir(parents=True)
        straight_file.write_bytes(b"%PDF-1.4\n")

        # 2. ZIP-member: file_path references _extracted temp path
        zip_id = str(uuid4())

        # 3. Missing file: file_path points to archive but file isn't there
        missing_id = str(uuid4())

        # 4. Already has chunks: skip (idempotent)
        already_id = str(uuid4())
        already_file = archive / rid16 / "attachments" / "already.pdf"
        already_file.write_bytes(b"%PDF-1.4\n")

        # 5. Failed but NOT cost_optimizer error — not our concern
        other_err_id = str(uuid4())

        docs = [
            _root_email_doc(root_uuid, root_doc_id),
            _failed_attachment_doc(
                id_=straight_id, root_id=root_uuid, file_name="good.pdf",
                file_path_on_disk=f"{rid16}/attachments/good.pdf",
            ),
            _failed_attachment_doc(
                id_=zip_id, root_id=root_uuid, file_name="inside.pdf",
                file_path_on_disk="data\\processed\\attachments\\x\\thing_extracted\\inside.pdf",
            ),
            _failed_attachment_doc(
                id_=missing_id, root_id=root_uuid, file_name="gone.pdf",
                file_path_on_disk=f"{rid16}/attachments/gone.pdf",
            ),
            _failed_attachment_doc(
                id_=already_id, root_id=root_uuid, file_name="already.pdf",
                file_path_on_disk=f"{rid16}/attachments/already.pdf",
            ),
            _failed_attachment_doc(
                id_=other_err_id, root_id=root_uuid, file_name="unrelated.pdf",
                file_path_on_disk=f"{rid16}/attachments/unrelated.pdf",
                error="Invalid object in /Pages",
            ),
        ]
        _write_jsonl(output / "documents.jsonl", docs)
        _write_jsonl(
            output / "chunks.jsonl",
            [{"document_id": already_id, "content": "existing"}],
        )

        candidates, skipped = repair_mod.find_candidates(output)

        assert [c.doc["id"] for c in candidates] == [straight_id]

        reasons = {d["id"]: r for d, r in skipped}
        assert reasons[zip_id] == repair_mod.SKIP_ZIP_MEMBER
        assert reasons[missing_id] == repair_mod.SKIP_MISSING_FILE
        assert reasons[already_id] == repair_mod.SKIP_ALREADY_HAS_CHUNKS
        assert other_err_id not in reasons, (
            "docs with non-cost_optimizer errors must not appear at all — "
            "they're out of this script's remit"
        )

    def test_page_filters_gate_pdf_candidates(self, repair_mod, tmp_path):
        """--max-pages drops large PDFs; --min-pages drops small ones.
        Non-PDF attachments always bypass the filter. .crdownload partial-
        download files are permanently skipped regardless of bounds.
        """
        output = tmp_path / "output"
        archive = output / "archive"
        root_uuid = str(uuid4())
        root_doc_id = "f" * 32
        rid16 = root_doc_id[:16]
        att = archive / rid16 / "attachments"
        att.mkdir(parents=True)

        # Use a tiny builtin PDF writer to make 1-page and 100-page PDFs.
        # PyMuPDF can create fixtures cheaply; skip if not importable (it's
        # already a project dep, so this always works in CI).
        import fitz

        small_pdf = att / "small.pdf"
        big_pdf = att / "big.pdf"
        docx_file = att / "office.docx"
        crdl_file = att / "partial.pdf.crdownload"

        doc_small = fitz.open()
        doc_small.new_page()
        doc_small.save(str(small_pdf))
        doc_small.close()

        doc_big = fitz.open()
        for _ in range(100):
            doc_big.new_page()
        doc_big.save(str(big_pdf))
        doc_big.close()

        docx_file.write_bytes(b"PK\x03\x04fake-docx")
        crdl_file.write_bytes(b"incomplete-download-bytes")

        small_id = str(uuid4())
        big_id = str(uuid4())
        docx_id = str(uuid4())
        crdl_id = str(uuid4())

        docs = [
            _root_email_doc(root_uuid, root_doc_id),
            _failed_attachment_doc(
                id_=small_id, root_id=root_uuid, file_name="small.pdf",
                file_path_on_disk=f"{rid16}/attachments/small.pdf",
            ),
            _failed_attachment_doc(
                id_=big_id, root_id=root_uuid, file_name="big.pdf",
                file_path_on_disk=f"{rid16}/attachments/big.pdf",
            ),
            _failed_attachment_doc(
                id_=docx_id, root_id=root_uuid, file_name="office.docx",
                file_path_on_disk=f"{rid16}/attachments/office.docx",
            ),
            _failed_attachment_doc(
                id_=crdl_id, root_id=root_uuid, file_name="partial.pdf.crdownload",
                file_path_on_disk=f"{rid16}/attachments/partial.pdf.crdownload",
            ),
        ]
        _write_jsonl(output / "documents.jsonl", docs)

        # --max-pages 40: big.pdf dropped; small.pdf + docx kept; crdl skipped
        candidates, skipped = repair_mod.find_candidates(output, max_pages=40)
        kept_ids = {c.doc["id"] for c in candidates}
        reasons = {d["id"]: r for d, r in skipped}
        assert kept_ids == {small_id, docx_id}
        assert reasons[big_id] == repair_mod.SKIP_OVER_PAGE_LIMIT
        assert reasons[crdl_id] == repair_mod.SKIP_CRDOWNLOAD

        # --min-pages 50: only big.pdf passes; small.pdf dropped; docx passes
        # (non-PDF, bypasses page filter); crdl still skipped.
        candidates, skipped = repair_mod.find_candidates(output, min_pages=50)
        kept_ids = {c.doc["id"] for c in candidates}
        reasons = {d["id"]: r for d, r in skipped}
        assert kept_ids == {big_id, docx_id}
        assert reasons[small_id] == repair_mod.SKIP_UNDER_PAGE_LIMIT
        assert reasons[crdl_id] == repair_mod.SKIP_CRDOWNLOAD

    def test_ignores_completed_docs(self, repair_mod, tmp_path):
        output = tmp_path / "output"
        root_uuid = str(uuid4())
        completed = _failed_attachment_doc(
            id_=str(uuid4()), root_id=root_uuid, file_name="ok.pdf",
            file_path_on_disk="x",
        )
        completed["status"] = "completed"
        completed["error_message"] = None
        _write_jsonl(
            output / "documents.jsonl",
            [_root_email_doc(root_uuid, "c" * 32), completed],
        )
        candidates, skipped = repair_mod.find_candidates(output)
        assert candidates == []
        assert skipped == []


# ---------------------------------------------------------------------------
# repair_one
# ---------------------------------------------------------------------------


class TestRepairOne:
    """End-to-end happy path with fake components. Asserts the pipeline
    sequence (parse → chunk → enrich → context → embed → persist) runs and
    writes valid chunk rows + returns the doc-row patch."""

    @pytest.mark.asyncio
    async def test_happy_path_writes_chunks_and_returns_updates(
        self, repair_mod, tmp_path
    ):
        output = tmp_path / "output"
        archive = output / "archive"
        root_uuid = str(uuid4())
        root_doc_id = "d" * 32
        rid16 = root_doc_id[:16]
        pdf = archive / rid16 / "attachments" / "report.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")

        doc_uuid = str(uuid4())
        docs = [
            _root_email_doc(root_uuid, root_doc_id),
            _failed_attachment_doc(
                id_=doc_uuid, root_id=root_uuid, file_name="report.pdf",
                file_path_on_disk=f"{rid16}/attachments/report.pdf",
            ),
        ]
        _write_jsonl(output / "documents.jsonl", docs)

        fake_chunk_a = _fake_chunk(content="alpha", doc_id=doc_uuid)
        fake_chunk_b = _fake_chunk(content="beta", doc_id=doc_uuid)

        def _chunk_text(*, text, document_id, source_file, is_markdown=True,
                        metadata=None):
            assert is_markdown is True
            assert text.startswith("# Report"), \
                "parser output must flow unmodified into chunker"
            fake_chunk_a.document_id = document_id
            fake_chunk_b.document_id = document_id
            return [fake_chunk_a, fake_chunk_b]

        embed_calls: list = []

        async def _embed_chunks(chunks):
            for i, c in enumerate(chunks):
                c.embedding = [float(i), float(i) + 0.5]
            embed_calls.append(list(chunks))
            return chunks

        components = repair_mod.RepairComponents(
            parse_attachment=AsyncMock(return_value="# Report\n\nContent body."),
            chunk_text=_chunk_text,
            build_embedding_text=lambda ctx, content: f"ctx:{ctx}|c:{content}",
            generate_context=AsyncMock(return_value="doc-ctx"),
            embed_chunks=_embed_chunks,
            update_attachment_markdown=MagicMock(
                return_value=f"{rid16}/attachments/report.pdf.md"
            ),
        )

        candidate = repair_mod.Candidate(
            doc=docs[1], archive_file=pdf, root_doc_id=root_doc_id
        )
        outcome = await repair_mod.repair_one(candidate, components, output)

        assert outcome.error is None
        assert outcome.updates["status"] == "completed"
        assert outcome.updates["error_message"] is None
        assert outcome.updates["archive_browse_uri"] == (
            f"/archive/{rid16}/attachments/report.pdf.md"
        )
        assert outcome.updates["archive_download_uri"] == (
            f"/archive/{rid16}/attachments/report.pdf"
        )

        # Chunks were appended
        chunks_written = [
            json.loads(line)
            for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(chunks_written) == 2
        assert chunks_written[0]["document_id"] == doc_uuid
        assert chunks_written[0]["context_summary"] == "doc-ctx"
        assert chunks_written[0]["embedding_text"].startswith("ctx:doc-ctx|")
        assert chunks_written[0]["embedding"] == [0.0, 0.5]
        # Citation enrichment via enrich_chunks_with_document_metadata
        assert chunks_written[0]["source_id"] == "src:report.pdf"
        # archive URIs on chunk come from the doc object (which we update on
        # the doc row, not the chunk) — enrich runs BEFORE the doc.update, so
        # these are the pre-repair URIs (None). That's fine; retriever
        # falls through to doc.archive_browse_uri anyway.

        # embed_chunks was called
        assert len(embed_calls) == 1
        # archive_gen was called
        components.update_attachment_markdown.assert_called_once()
        kwargs = components.update_attachment_markdown.call_args.kwargs
        assert kwargs["doc_id"] == root_doc_id
        assert kwargs["filename"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_parser_empty_returns_error_outcome(self, repair_mod, tmp_path):
        """Zero-content extraction is an error outcome, not a chunk write.
        The doc row must not get flipped to ``completed`` in that case."""
        output = tmp_path / "output"
        (output / "archive").mkdir(parents=True)
        pdf = output / "archive" / "f.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        doc_uuid = str(uuid4())
        doc = _failed_attachment_doc(
            id_=doc_uuid, root_id=str(uuid4()), file_name="f.pdf",
            file_path_on_disk="f.pdf",
        )

        components = repair_mod.RepairComponents(
            parse_attachment=AsyncMock(return_value=""),
            chunk_text=MagicMock(),
            build_embedding_text=MagicMock(),
            generate_context=AsyncMock(),
            embed_chunks=AsyncMock(),
            update_attachment_markdown=MagicMock(),
        )
        candidate = repair_mod.Candidate(
            doc=doc, archive_file=pdf, root_doc_id="e" * 32
        )
        outcome = await repair_mod.repair_one(candidate, components, output)
        assert outcome.updates is None
        assert outcome.error == "llamaparse_returned_empty"
        assert not (output / "chunks.jsonl").exists()
        components.chunk_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_parser_exception_returns_error_outcome(
        self, repair_mod, tmp_path
    ):
        output = tmp_path / "output"
        pdf = tmp_path / "f.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        doc_uuid = str(uuid4())
        doc = _failed_attachment_doc(
            id_=doc_uuid, root_id=str(uuid4()), file_name="f.pdf",
            file_path_on_disk="f.pdf",
        )
        components = repair_mod.RepairComponents(
            parse_attachment=AsyncMock(side_effect=ValueError("boom")),
            chunk_text=MagicMock(),
            build_embedding_text=MagicMock(),
            generate_context=AsyncMock(),
            embed_chunks=AsyncMock(),
            update_attachment_markdown=MagicMock(),
        )
        candidate = repair_mod.Candidate(
            doc=doc, archive_file=pdf, root_doc_id="f" * 32
        )
        outcome = await repair_mod.repair_one(candidate, components, output)
        assert outcome.updates is None
        assert "boom" in (outcome.error or "")


# ---------------------------------------------------------------------------
# rewrite_documents_jsonl
# ---------------------------------------------------------------------------


class TestRewriteDocumentsJsonl:
    def test_patches_only_target_rows(self, repair_mod, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        docs = [
            {"id": "a", "status": "failed", "error_message": "x", "file_name": "a"},
            {"id": "b", "status": "completed", "error_message": None, "file_name": "b"},
            {"id": "c", "status": "failed", "error_message": "x", "file_name": "c"},
        ]
        _write_jsonl(output / "documents.jsonl", docs)

        updates = {
            "a": {"status": "completed", "error_message": None, "updated_at": "2026-01-01"},
            "c": {"status": "completed", "error_message": None, "updated_at": "2026-01-02"},
        }
        count = repair_mod.rewrite_documents_jsonl(output, updates)
        assert count == 2

        after = [
            json.loads(l)
            for l in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert {d["id"]: d["status"] for d in after} == {
            "a": "completed", "b": "completed", "c": "completed"
        }
        # b untouched
        assert after[1]["file_name"] == "b"
        # temp file cleaned up
        assert not (output / "documents.jsonl.tmp").exists()


# ---------------------------------------------------------------------------
# Finalize mode
# ---------------------------------------------------------------------------


class TestFinalizeMode:
    """`--finalize` flips doc rows whose chunks already exist on disk.

    Regression scenario: a long repair run was killed after chunks got
    persisted per-doc but before the atomic documents.jsonl rewrite at the
    end. Those docs still read as failed; finalize cleans that up without
    re-parsing.
    """

    def test_detects_targets_and_patches_status_plus_archive_uri(
        self, repair_mod, tmp_path
    ):
        output = tmp_path / "output"
        archive = output / "archive"
        root_uuid = str(uuid4())
        root_doc_id = "g" * 32
        rid16 = root_doc_id[:16]
        att_dir = archive / rid16 / "attachments"
        att_dir.mkdir(parents=True)

        # A: chunks + .md on disk → finalize with archive URIs
        #    file_name has space+period (realistic — email attachment titles
        #    regularly do), so the .md on disk is the sanitized form.
        a_id = str(uuid4())
        a_raw = "A11. IAPP_2.pdf"  # raw doc file_name
        a_safe = "A11._IAPP_2.pdf"  # what _sanitize_storage_key produces
        (att_dir / a_safe).write_bytes(b"%PDF-1.4")
        (att_dir / f"{a_safe}.md").write_text("# a\n\nbody", encoding="utf-8")

        # B: chunks on disk, no .md → finalize without archive URIs
        b_id = str(uuid4())
        (att_dir / "b.pdf").write_bytes(b"%PDF-1.4")

        # C: failed but no chunks → NOT a target (normal repair path)
        c_id = str(uuid4())

        # D: already completed → never a target
        d_id = str(uuid4())

        docs = [
            _root_email_doc(root_uuid, root_doc_id),
            _failed_attachment_doc(
                id_=a_id, root_id=root_uuid, file_name=a_raw,
                file_path_on_disk=f"{rid16}/attachments/{a_safe}",
            ),
            _failed_attachment_doc(
                id_=b_id, root_id=root_uuid, file_name="b.pdf",
                file_path_on_disk=f"{rid16}/attachments/b.pdf",
            ),
            _failed_attachment_doc(
                id_=c_id, root_id=root_uuid, file_name="c.pdf",
                file_path_on_disk=f"{rid16}/attachments/c.pdf",
            ),
            {**_failed_attachment_doc(
                id_=d_id, root_id=root_uuid, file_name="d.pdf",
                file_path_on_disk=f"{rid16}/attachments/d.pdf",
            ), "status": "completed", "error_message": None},
        ]
        _write_jsonl(output / "documents.jsonl", docs)
        _write_jsonl(
            output / "chunks.jsonl",
            [
                {"document_id": a_id, "content": "x"},
                {"document_id": b_id, "content": "y"},
                {"document_id": d_id, "content": "z"},  # completed already
            ],
        )

        targets = repair_mod.find_finalize_targets(output)
        target_ids = {t["id"] for t in targets}
        assert target_ids == {a_id, b_id}, (
            "Only docs with chunks AND status=failed+cost_optimizer qualify; "
            "failed-but-chunkless (normal repair path) and already-completed "
            "must be excluded."
        )

        root_map = {root_uuid: root_doc_id}
        patches = repair_mod.build_finalize_updates(output, targets, root_map)

        assert patches[a_id]["status"] == "completed"
        assert patches[a_id]["error_message"] is None
        assert patches[a_id]["archive_browse_uri"] == (
            f"/archive/{rid16}/attachments/{a_safe}.md"
        )
        assert patches[a_id]["archive_download_uri"] == (
            f"/archive/{rid16}/attachments/{a_safe}"
        )

        assert patches[b_id]["status"] == "completed"
        assert patches[b_id]["error_message"] is None
        assert "archive_browse_uri" not in patches[b_id], (
            "docs without an on-disk .md must not get a broken URI — "
            "validate ingest would flag it."
        )


# ---------------------------------------------------------------------------
# build_components wiring
# ---------------------------------------------------------------------------


class TestBuildComponentsStorageWiring:
    """The ArchiveGenerator returned by ``build_components`` must be pointed at
    the *local* archive bucket, not the default Supabase-backed ArchiveStorage.

    Regression: the default ``ArchiveGenerator()`` uses ``ArchiveStorage``
    which calls ``file_exists`` against Supabase — it returns False for local
    files and the pre-upload guard in ``update_attachment_markdown`` silently
    skips ``.md`` regeneration, leaving repaired docs with null archive URIs
    and no browsable previews.
    """

    def test_build_components_uses_local_bucket_storage(self, repair_mod, tmp_path):
        from mtss.storage.local_client import LocalBucketStorage

        output = tmp_path / "output"
        (output / "archive").mkdir(parents=True)

        components = repair_mod.build_components(output)

        # Walk the closure back to the ArchiveGenerator to inspect its storage
        # (update_attachment_markdown is a bound method).
        archive_gen = components.update_attachment_markdown.__self__
        assert isinstance(archive_gen.storage, LocalBucketStorage), (
            f"ArchiveGenerator.storage must be LocalBucketStorage for repair; "
            f"got {type(archive_gen.storage).__name__}"
        )

    def test_regenerates_md_when_original_is_on_local_disk(
        self, repair_mod, tmp_path
    ):
        """End-to-end wiring check: given a real attachment file under the
        local archive tree, ``update_attachment_markdown`` writes the .md
        file and returns its relative path."""
        output = tmp_path / "output"
        folder_id = "0123456789abcdef"
        att_dir = output / "archive" / folder_id / "attachments"
        att_dir.mkdir(parents=True)
        pdf = att_dir / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        components = repair_mod.build_components(output)

        md_path = components.update_attachment_markdown(
            doc_id=folder_id + "x" * 16,  # archive_gen slices to 16
            filename="sample.pdf",
            content_type="application/pdf",
            size_bytes=pdf.stat().st_size,
            parsed_content="# Title\n\nExtracted body.",
        )
        assert md_path == f"{folder_id}/attachments/sample.pdf.md"
        md_on_disk = output / "archive" / md_path
        assert md_on_disk.exists()
        assert "Extracted body." in md_on_disk.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_chunk(*, content: str, doc_id: str):
    """Produce a minimal Chunk-like object that ``chunk_to_dict`` can read."""
    from mtss.models.chunk import Chunk

    return Chunk(
        document_id=doc_id,  # overwritten by chunk_text
        content=content,
        chunk_index=0,
    )
