"""Crash-safety tests for atomic writes + fsync in canonical JSONL output.

Covers ``src/mtss/_io.py`` primitives and the refactored
``LocalStorageClient.flush()``. A crashed flush must NEVER leave a
half-written canonical file — that data costs hours and real money to
regenerate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from mtss._io import atomic_write_text, fsync_append_line


# --------------------------------------------------------------------------
# _io primitives
# --------------------------------------------------------------------------


def test_atomic_write_uses_tmp_then_rename(tmp_path: Path):
    """atomic_write_text writes to {path}.tmp and then os.replace's it."""
    target = tmp_path / "documents.jsonl"
    tmp_target = tmp_path / "documents.jsonl.tmp"

    real_replace = os.replace
    observed: dict = {}

    def spy_replace(src, dst):
        # Capture tmp state at the moment of rename.
        observed["tmp_exists_at_replace"] = Path(src).exists()
        observed["src"] = str(src)
        observed["dst"] = str(dst)
        return real_replace(src, dst)

    with patch("mtss._io.os.replace", side_effect=spy_replace) as mock_replace:
        atomic_write_text(target, "hello\nworld\n")

    assert mock_replace.call_count == 1
    assert observed["tmp_exists_at_replace"] is True
    assert observed["src"] == str(tmp_target)
    assert observed["dst"] == str(target)
    # After replace the tmp is gone and the target has the content.
    assert not tmp_target.exists()
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"


def test_atomic_write_cleans_tmp_on_failure(tmp_path: Path):
    """If the write raises mid-write, the tmp file is cleaned up and the target is untouched."""
    target = tmp_path / "documents.jsonl"
    tmp_target = tmp_path / "documents.jsonl.tmp"

    # Seed an existing "canonical" file we must not clobber.
    target.write_text("ORIGINAL\n", encoding="utf-8")

    class BoomIO:
        def __init__(self, real_open_fn, path, mode, encoding):
            self._f = real_open_fn(path, mode, encoding=encoding)

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *a):
            return self._f.__exit__(*a)

        def write(self, data):
            raise OSError("simulated disk failure")

        def flush(self):
            self._f.flush()

        def fileno(self):
            return self._f.fileno()

    real_open = open

    def boom_open(path, mode="r", *args, **kwargs):
        # Only blow up on the tmp write.
        if str(path).endswith(".tmp") and "w" in mode:
            return BoomIO(real_open, path, mode, kwargs.get("encoding"))
        return real_open(path, mode, *args, **kwargs)

    with patch("mtss._io.open", side_effect=boom_open):
        with pytest.raises(OSError, match="simulated disk failure"):
            atomic_write_text(target, "REPLACEMENT\n")

    # No tmp leak, original preserved.
    assert not tmp_target.exists()
    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_atomic_write_fsyncs_before_rename(tmp_path: Path):
    """fsync must be called before os.replace so data is durable on disk pre-rename."""
    target = tmp_path / "topics.jsonl"

    call_order: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        call_order.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        call_order.append("replace")
        return real_replace(src, dst)

    with patch("mtss._io.os.fsync", side_effect=spy_fsync), \
         patch("mtss._io.os.replace", side_effect=spy_replace):
        atomic_write_text(target, "t\n")

    assert "fsync" in call_order, "expected os.fsync to be called"
    assert "replace" in call_order, "expected os.replace to be called"
    assert call_order.index("fsync") < call_order.index("replace"), (
        f"fsync must precede replace, got order: {call_order}"
    )


def test_fsync_append_line_fsyncs(tmp_path: Path):
    """fsync_append_line fsyncs after the append write."""
    target = tmp_path / "processing_log.jsonl"

    call_order: list[str] = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        call_order.append("fsync")
        return real_fsync(fd)

    with patch("mtss._io.os.fsync", side_effect=spy_fsync) as mock_fsync:
        fsync_append_line(target, '{"a": 1}')
        fsync_append_line(target, '{"b": 2}')

    assert mock_fsync.call_count == 2
    assert call_order == ["fsync", "fsync"]
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n{"b": 2}\n'


# --------------------------------------------------------------------------
# flush() crash simulation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_survives_simulated_crash(local_db_client):
    """A crash between documents.jsonl and chunks.jsonl writes must not corrupt either file.

    After a successful first flush we have canonical docs + chunks on disk. We
    then simulate an OSError raised while flush() is writing chunks.jsonl
    during a second flush. Assert:
    - documents.jsonl reflects the new documents (first atomic write succeeded)
    - chunks.jsonl retains prior-run content (second write aborted before replace)
    - no .tmp files leak in the output directory
    """
    from mtss.models.chunk import Chunk
    from mtss.models.document import Document, DocumentType, ProcessingStatus

    # --- First pass: seed canonical docs + chunks ----------------------
    doc_v1 = Document(
        id=uuid4(),
        document_type=DocumentType.EMAIL,
        file_path="/v1.eml",
        file_name="v1.eml",
        depth=0,
        status=ProcessingStatus.COMPLETED,
    )
    await local_db_client.insert_document(doc_v1)
    chunk_v1 = Chunk(
        id=uuid4(),
        document_id=doc_v1.id,
        chunk_id="chunk_v1",
        content="original content",
        chunk_index=0,
        metadata={"type": "email_body"},
    )
    await local_db_client.insert_chunks([chunk_v1])
    local_db_client.flush()

    docs_path = local_db_client.output_dir / "documents.jsonl"
    chunks_path = local_db_client.output_dir / "chunks.jsonl"

    docs_after_first = docs_path.read_text(encoding="utf-8")
    chunks_after_first = chunks_path.read_text(encoding="utf-8")
    assert "v1.eml" in docs_after_first
    assert "chunk_v1" in chunks_after_first

    # --- Second pass: stage a new doc + chunk, then simulate crash -----
    doc_v2 = Document(
        id=uuid4(),
        document_type=DocumentType.EMAIL,
        file_path="/v2.eml",
        file_name="v2.eml",
        depth=0,
        status=ProcessingStatus.COMPLETED,
    )
    await local_db_client.insert_document(doc_v2)
    chunk_v2 = Chunk(
        id=uuid4(),
        document_id=doc_v2.id,
        chunk_id="chunk_v2",
        content="new content",
        chunk_index=0,
        metadata={"type": "email_body"},
    )
    await local_db_client.insert_chunks([chunk_v2])

    # Snapshot disk state right before the crashing flush. insert_chunks/
    # insert_document append via _append_jsonl, so the file has grown — but
    # those appends are durable, and flush() should either atomically replace
    # the file with the new consolidated content, or (on crash) leave the
    # pre-flush on-disk content completely intact.
    docs_pre_flush = docs_path.read_text(encoding="utf-8")
    chunks_pre_flush = chunks_path.read_text(encoding="utf-8")

    # Patch atomic_write_text as used inside local_client so the chunks.jsonl
    # write blows up AFTER documents.jsonl was already atomically replaced.
    from mtss.storage import local_client as lc_module

    real_atomic = lc_module.atomic_write_text

    def flaky_atomic(path, content, *, encoding="utf-8"):
        if path.name == "chunks.jsonl":
            raise OSError("simulated disk failure during chunks write")
        return real_atomic(path, content, encoding=encoding)

    with patch.object(lc_module, "atomic_write_text", side_effect=flaky_atomic):
        with pytest.raises(OSError, match="simulated disk failure"):
            local_db_client.flush()

    # documents.jsonl was the first atomic write — it should reflect the flushed
    # (consolidated, deduped) content including both v1 and v2.
    docs_after_crash = docs_path.read_text(encoding="utf-8")
    assert "v1.eml" in docs_after_crash
    assert "v2.eml" in docs_after_crash, (
        "documents.jsonl atomic write should have succeeded before the crash"
    )

    # chunks.jsonl write aborted before os.replace — on-disk content is EXACTLY
    # whatever was there immediately before flush() started (the appended lines
    # from insert_chunks, fsync'd). No half-written truncation from a
    # plain open('w') crash.
    chunks_after_crash = chunks_path.read_text(encoding="utf-8")
    assert chunks_after_crash == chunks_pre_flush, (
        "chunks.jsonl on-disk content must be bit-identical to its pre-flush "
        "state when the atomic rewrite aborts"
    )
    # Prior content is still there (we did not lose chunk_v1).
    assert "chunk_v1" in chunks_after_crash

    # No .tmp leaks anywhere in the output dir.
    leaked = list(local_db_client.output_dir.glob("*.tmp"))
    assert leaked == [], f"unexpected .tmp leak after crash: {leaked}"

    # Sanity: files on disk are still valid JSONL (no truncated half-lines).
    for line in docs_after_crash.splitlines():
        if line.strip():
            json.loads(line)
    for line in chunks_after_crash.splitlines():
        if line.strip():
            json.loads(line)

    # Silence unused-variable lint: we assert via the pre_flush snapshots.
    _ = docs_pre_flush
