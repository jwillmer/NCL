"""Per-check tests for the 22 extracted ingest validation helpers.

Each `_check_<descriptor>` function in `mtss.cli.validate_cmd` encapsulates one
of the 22 numbered integrity checks the `mtss validate ingest` CLI runs. This
file exercises each in isolation with minimal in-memory fixtures, asserting on
the exact `(issues, warnings)` strings — those strings are part of the public
CLI contract.

The sibling `tests/test_sanitize_migration.py::TestValidateNewChecks` suite
remains the semantic regression gate for checks that had prior targeted
coverage; these tests extend the coverage to every extracted function.
"""

from __future__ import annotations

from collections import Counter
from uuid import uuid4

import pytest

from mtss.cli.validate_cmd import (
    _check_archive_uris,
    _check_broken_archive_uris,
    _check_broken_markdown_links,
    _check_chunk_positions,
    _check_context_summary,
    _check_docs_without_chunks,
    _check_document_types,
    _check_duplicate_file_hashes,
    _check_duplicate_ids,
    _check_duplicate_uuids,
    _check_email_metadata,
    _check_embedding_completeness,
    _check_embedding_mode_coverage,
    _check_embedding_mode_inheritance,
    _check_embedding_vector_sanity,
    _check_empty_content,
    _check_encoded_filenames,
    _check_encoded_uris,
    _check_failed_documents,
    _check_orphan_archive_folders,
    _check_orphan_attachments,
    _check_orphan_chunks,
    _check_outdated_ingest_version,
    _check_remote_archive_uris,
    _check_processing_log,
    _check_residual_image_refs,
    _check_schema_parity,
    _check_single_chunk_modes,
    _check_sqlite_integrity,
    _check_stale_processing_entries,
    _check_stale_topic_refs,
    _check_thread_root_consistency,
    _check_topic_count_accuracy,
    _check_topic_health,
    _check_trailing_dot_filenames,
    _check_unknown_vessel_mentions,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_doc(
    *,
    uuid=None,
    doc_id="abc0000000000000",
    doc_type="email",
    file_name="test.eml",
    depth=0,
    status="completed",
    root_id=None,
    archive_path=None,
    browse_uri=None,
    download_uri=None,
    source_id=None,
    email_participants=None,
    email_date_start=None,
    email_date_end=None,
):
    uuid = uuid or str(uuid4())
    return {
        "id": uuid,
        "doc_id": doc_id,
        "document_type": doc_type,
        "file_name": file_name,
        "file_path": f"/emails/{file_name}",
        "depth": depth,
        "status": status,
        "root_id": root_id or uuid,
        "archive_path": archive_path if archive_path is not None else doc_id[:16],
        "archive_browse_uri": browse_uri,
        "archive_download_uri": download_uri,
        "source_id": source_id or file_name,
        "email_participants": email_participants if email_participants is not None else ["a@b.com"],
        "email_date_start": email_date_start,
        "email_date_end": email_date_end,
    }


def _make_chunk(
    doc_uuid,
    *,
    chunk_id=None,
    content="hello world",
    embedding=(0.1, 0.2),
    context_summary="context",
    embedding_text="embedding text",
    topic_ids=None,
    char_start=0,
    char_end=10,
):
    return {
        "id": str(uuid4()),
        "document_id": doc_uuid,
        "chunk_id": chunk_id or f"chunk_{uuid4().hex[:8]}",
        "content": content,
        "chunk_index": 0,
        "embedding": list(embedding) if embedding else None,
        "context_summary": context_summary,
        "embedding_text": embedding_text,
        "metadata": {"topic_ids": topic_ids} if topic_ids else {},
        "char_start": char_start,
        "char_end": char_end,
    }


def _make_topic(tid, *, chunk_count=0, document_count=0, embedding=(0.1, 0.2)):
    return {
        "id": tid,
        "name": f"topic-{tid}",
        "chunk_count": chunk_count,
        "document_count": document_count,
        "embedding": list(embedding) if embedding else None,
    }


# ---------------------------------------------------------------------------
# Check 1: duplicate UUIDs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_duplicate_uuids_flags_dupe():
    shared_id = str(uuid4())
    docs = [_make_doc(uuid=shared_id), _make_doc(uuid=shared_id)]
    # Only the last wins in the dict
    doc_by_uuid = {d["id"]: d for d in docs}
    issues, warnings = _check_duplicate_uuids(docs, doc_by_uuid)
    assert warnings == []
    assert len(issues) == 1
    assert "Duplicate document UUIDs" in issues[0]
    assert "2 rows but 1 unique" in issues[0]


@pytest.mark.unit
def test_check_duplicate_uuids_clean():
    docs = [_make_doc(uuid=str(uuid4())) for _ in range(3)]
    doc_by_uuid = {d["id"]: d for d in docs}
    assert _check_duplicate_uuids(docs, doc_by_uuid) == ([], [])


# ---------------------------------------------------------------------------
# Check 2: processing log
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_processing_log_flags_failed_entries():
    proc_by_file = {
        "a.eml": {"file_path": "a.eml", "status": "FAILED", "error": "boom"},
        "b.eml": {"file_path": "b.eml", "status": "COMPLETED"},
    }
    issues, warnings = _check_processing_log(proc_by_file)
    assert warnings == []
    assert issues[0] == "1 files not COMPLETED in processing log"
    assert any("a.eml" in i and "status=FAILED" in i for i in issues[1:])


@pytest.mark.unit
def test_check_processing_log_all_completed():
    proc_by_file = {
        "a.eml": {"file_path": "a.eml", "status": "COMPLETED"},
        "b.eml": {"file_path": "b.eml", "status": "COMPLETED"},
    }
    assert _check_processing_log(proc_by_file) == ([], [])


# ---------------------------------------------------------------------------
# Check 3: document type breakdown (informational only)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_document_types_is_informational():
    docs = [_make_doc(doc_type="email"), _make_doc(doc_type="attachment_document")]
    assert _check_document_types(docs) == ([], [])


# ---------------------------------------------------------------------------
# Check 4: orphan chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_orphan_chunks_flags_unresolvable_document_id():
    doc = _make_doc()
    doc_by_uuid = {doc["id"]: doc}
    good_chunk = _make_chunk(doc["id"])
    bad_chunk = _make_chunk("nonexistent-uuid")
    issues, warnings = _check_orphan_chunks([good_chunk, bad_chunk], doc_by_uuid)
    assert warnings == []
    assert issues == ["1 chunks reference non-existent documents"]


@pytest.mark.unit
def test_check_orphan_chunks_all_resolved():
    doc = _make_doc()
    doc_by_uuid = {doc["id"]: doc}
    chunks = [_make_chunk(doc["id"]) for _ in range(3)]
    assert _check_orphan_chunks(chunks, doc_by_uuid) == ([], [])


# ---------------------------------------------------------------------------
# Check 5: embedding completeness + dimensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_embedding_completeness_flags_missing_and_mixed_dims():
    doc = _make_doc()
    chunks = [
        _make_chunk(doc["id"], embedding=(0.1, 0.2)),
        _make_chunk(doc["id"], embedding=None),
        _make_chunk(doc["id"], embedding=(0.1, 0.2, 0.3)),
    ]
    issues, warnings = _check_embedding_completeness(chunks)
    assert warnings == []
    assert any("1/3 chunks missing embeddings" in i for i in issues)
    assert any("Inconsistent embedding dimensions" in i for i in issues)


@pytest.mark.unit
def test_check_embedding_completeness_clean():
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], embedding=(0.1, 0.2)) for _ in range(3)]
    assert _check_embedding_completeness(chunks) == ([], [])


# ---------------------------------------------------------------------------
# Check 6: empty content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_empty_content_flags_blank_strings():
    doc = _make_doc()
    chunks = [
        _make_chunk(doc["id"], content="good"),
        _make_chunk(doc["id"], content=""),
        _make_chunk(doc["id"], content="   \n"),
    ]
    issues, warnings = _check_empty_content(chunks)
    assert warnings == []
    assert issues == ["2 chunks have empty content"]


@pytest.mark.unit
def test_check_empty_content_all_non_empty():
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], content="hi") for _ in range(2)]
    assert _check_empty_content(chunks) == ([], [])


# ---------------------------------------------------------------------------
# Check 7: context_summary / embedding_text
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_context_summary_flags_missing_context_groups_by_email():
    email = _make_doc(doc_type="email", source_id="thread.eml")
    # attachment rolled up under the email root
    attachment = _make_doc(
        uuid=str(uuid4()),
        doc_type="attachment_document",
        source_id="thread.eml/report.pdf",
        root_id=email["id"],
    )
    docs = [email, attachment]
    doc_by_uuid = {d["id"]: d for d in docs}
    chunks = [
        _make_chunk(email["id"], context_summary="", embedding_text="ok"),
        _make_chunk(attachment["id"], context_summary="ok", embedding_text=""),
    ]
    issues, warnings = _check_context_summary(chunks, docs, doc_by_uuid)
    assert issues == []
    # Warning header + one detail line (both chunks roll up to "thread.eml")
    assert len(warnings) == 2
    assert "2/2 text chunks missing context_summary/embedding_text" in warnings[0]
    assert "thread.eml" in warnings[1]


@pytest.mark.unit
def test_check_context_summary_skips_image_docs():
    image = _make_doc(doc_type="attachment_image")
    docs = [image]
    doc_by_uuid = {d["id"]: d for d in docs}
    # Image chunk missing context — should not flag
    chunks = [_make_chunk(image["id"], context_summary="", embedding_text="")]
    assert _check_context_summary(chunks, docs, doc_by_uuid) == ([], [])


# ---------------------------------------------------------------------------
# Check 8: docs without chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_docs_without_chunks_flags_only_unexpected():
    kept = _make_doc(doc_type="attachment_document")
    img = _make_doc(doc_type="attachment_image")
    failed = _make_doc(doc_type="attachment_document", status="failed")
    filtered = _make_doc(doc_type="email")
    docs = [kept, img, failed, filtered]
    chunk_doc_ids = Counter()
    filtered_uuids = {filtered["id"]}
    issues, warnings = _check_docs_without_chunks(docs, chunk_doc_ids, filtered_uuids)
    assert warnings == []
    assert issues[0] == "1 document(s) have no chunks (not explained by events)"
    # Only the kept doc is reported in the detail line
    assert any(kept["file_path"][:70] in i for i in issues[1:])


@pytest.mark.unit
def test_check_docs_without_chunks_all_have_chunks():
    doc = _make_doc(doc_type="attachment_document")
    chunk_doc_ids = Counter({doc["id"]: 3})
    assert _check_docs_without_chunks([doc], chunk_doc_ids, set()) == ([], [])


# ---------------------------------------------------------------------------
# Check 9: orphan attachments (root_id dangles)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_orphan_attachments_flags_dangling_root_id():
    email = _make_doc(doc_type="email")
    email_uuids = {email["id"]}
    orphan = _make_doc(
        uuid=str(uuid4()),
        doc_type="attachment_document",
        root_id="does-not-exist",
    )
    _, warnings = _check_orphan_attachments([email, orphan], email_uuids)
    assert warnings == ["1 attachments have root_id not matching any email"]


@pytest.mark.unit
def test_check_orphan_attachments_clean():
    email = _make_doc(doc_type="email")
    att = _make_doc(
        uuid=str(uuid4()),
        doc_type="attachment_document",
        root_id=email["id"],
    )
    assert _check_orphan_attachments([email, att], {email["id"]}) == ([], [])


# ---------------------------------------------------------------------------
# Check 10: failed documents
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_failed_documents_flags_failed():
    failed_doc = _make_doc(status="failed")
    failed_doc["error_message"] = "boom"
    _, warnings = _check_failed_documents(
        [failed_doc, _make_doc()], events=[], verbose=False
    )
    assert warnings == [
        "1 document(s) have status='failed' in the documents table "
        "(no matching extraction_failed event)"
    ]


@pytest.mark.unit
def test_check_failed_documents_verbose_includes_error_detail():
    failed_doc = _make_doc(status="failed")
    failed_doc["error_message"] = "parser crashed"
    _, warnings = _check_failed_documents([failed_doc], events=[], verbose=True)
    assert len(warnings) == 2
    assert "parser crashed" in warnings[1]


@pytest.mark.unit
def test_check_failed_documents_clean():
    assert _check_failed_documents([_make_doc()], events=[], verbose=False) == ([], [])


@pytest.mark.unit
def test_check_failed_documents_suppressed_by_matching_event():
    """A failed doc with a matching extraction_failed event is expected — no warning.

    Reproduces the live case: LlamaParse raised "produced no content" on a PDF,
    which both (a) marked the attach_doc as failed-in-place and (b) logged an
    extraction_failed event. The validator must cross-reference them instead
    of double-reporting.
    """
    root_uuid = str(uuid4())
    failed_doc = _make_doc(
        status="failed",
        doc_type="attachment_pdf",
        root_id=root_uuid,
        source_id="86547629_ese5irrf.r44.eml/5415da512a101_as_a4(61)a3(8).pdf",
    )
    matching_event = {
        "event_type": "extraction_failed",
        "parent_document_id": root_uuid,
        "file_name": "5415DA512A101_AS_A4(61)A3(8).pdf",
    }

    _, warnings = _check_failed_documents(
        [failed_doc], events=[matching_event], verbose=False
    )

    assert warnings == []


@pytest.mark.unit
def test_check_failed_documents_event_for_different_email_does_not_suppress():
    """Event for a different email must not mask this doc's failure."""
    failed_doc = _make_doc(
        status="failed",
        doc_type="attachment_pdf",
        root_id=str(uuid4()),
        source_id="real.eml/real.pdf",
    )
    unrelated_event = {
        "event_type": "extraction_failed",
        "parent_document_id": str(uuid4()),  # different email
        "file_name": "real.pdf",
    }

    _, warnings = _check_failed_documents(
        [failed_doc], events=[unrelated_event], verbose=False
    )

    assert len(warnings) == 1
    assert "status='failed'" in warnings[0]


# ---------------------------------------------------------------------------
# Check 11: trailing-dot filenames
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_trailing_dot_filenames_flags():
    doc = _make_doc(source_id="broken_name.")
    _, warnings = _check_trailing_dot_filenames([doc], verbose=False)
    assert warnings == [
        "1 document(s) have source_id ending in '.' (potential parsing issues)"
    ]


@pytest.mark.unit
def test_check_trailing_dot_filenames_clean():
    doc = _make_doc(source_id="ok.eml")
    assert _check_trailing_dot_filenames([doc], verbose=False) == ([], [])


# ---------------------------------------------------------------------------
# Check 12: topic health
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_topic_health_flags_missing_embeddings_and_zero_counts():
    topics = [
        _make_topic("t1", document_count=5),
        _make_topic("t2", document_count=0, embedding=None),
    ]
    issues, warnings = _check_topic_health(topics)
    assert issues == []
    assert any("1/2 topics missing embeddings" in w for w in warnings)
    assert any("1/2 topics have document_count=0" in w for w in warnings)


@pytest.mark.unit
def test_check_topic_health_clean():
    topics = [_make_topic("t1", document_count=3)]
    assert _check_topic_health(topics) == ([], [])


# ---------------------------------------------------------------------------
# Check 13: archive URIs / folders
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_archive_uris_flags_missing_uris_and_missing_folder():
    # Email with all three URI fields None — flagged as missing URIs
    missing = _make_doc(doc_type="email", browse_uri=None, download_uri=None)
    missing["archive_path"] = None
    # Email with URIs set but the folder doesn't actually exist
    no_folder = _make_doc(
        doc_type="email",
        browse_uri="/archive/xyz/email.md",
        download_uri="/archive/xyz/email.eml",
        archive_path="xyz",
    )
    docs = [missing, no_folder]
    email_uuids = {d["id"] for d in docs}
    # A different folder on disk, so no_folder is flagged
    issues, warnings = _check_archive_uris(docs, email_uuids, {"other"}, verbose=False)
    assert any("2/2 emails missing archive URIs" in i or "1/2 emails missing archive URIs" in i
               for i in issues)
    assert any("emails have no archive folder on disk" in w for w in warnings)


@pytest.mark.unit
def test_check_archive_uris_clean():
    email = _make_doc(
        doc_type="email",
        browse_uri="/archive/abc/email.md",
        download_uri="/archive/abc/email.eml",
        archive_path="abc",
    )
    # archive_folders on disk contains the matching folder id
    assert _check_archive_uris([email], {email["id"]}, {"abc"}, verbose=False) == ([], [])


# ---------------------------------------------------------------------------
# Check 14: stale topic references
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_stale_topic_refs_flags_unknown_ids():
    topics = [_make_topic("t1")]
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], topic_ids=["t1", "gone"])]
    issues, _ = _check_stale_topic_refs(chunks, topics)
    assert issues == [
        "1 chunk topic_ids reference 1 non-existent topics (stale from merges)"
    ]


@pytest.mark.unit
def test_check_stale_topic_refs_clean():
    topics = [_make_topic("t1")]
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], topic_ids=["t1"])]
    assert _check_stale_topic_refs(chunks, topics) == ([], [])


# ---------------------------------------------------------------------------
# Check 15: topic count accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_topic_count_accuracy_flags_mismatched_counts():
    # Topic claims chunk_count=5/document_count=5 but only 1 chunk ref exists
    topics = [_make_topic("t1", chunk_count=5, document_count=5)]
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], topic_ids=["t1"])]
    issues, _ = _check_topic_count_accuracy(chunks, topics)
    assert issues == [
        "1/1 topics have stale counts "
        "(JSONL counts don't match actual chunk references)"
    ]


@pytest.mark.unit
def test_check_topic_count_accuracy_clean():
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], topic_ids=["t1"])]
    topics = [_make_topic("t1", chunk_count=1, document_count=1)]
    assert _check_topic_count_accuracy(chunks, topics) == ([], [])


# ---------------------------------------------------------------------------
# Check 16: archive URIs point to missing files
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_broken_archive_uris_flags_missing_targets(tmp_path):
    archive_dir = tmp_path / "archive"
    (archive_dir / "abc").mkdir(parents=True)
    (archive_dir / "abc" / "email.md").write_text("ok")

    doc = _make_doc(
        browse_uri="/archive/abc/email.md",  # exists
        download_uri="/archive/abc/missing.eml",  # does not
    )
    issues, _ = _check_broken_archive_uris([doc], archive_dir)
    assert any("1 archive URIs point to missing files on disk" in i for i in issues)
    # Detail line includes file_name + relative path
    assert any("missing.eml" in i for i in issues)


@pytest.mark.unit
def test_check_broken_archive_uris_all_resolve(tmp_path):
    archive_dir = tmp_path / "archive"
    (archive_dir / "abc").mkdir(parents=True)
    (archive_dir / "abc" / "email.md").write_text("ok")
    (archive_dir / "abc" / "email.eml").write_text("ok")
    doc = _make_doc(
        browse_uri="/archive/abc/email.md",
        download_uri="/archive/abc/email.eml",
    )
    assert _check_broken_archive_uris([doc], archive_dir) == ([], [])


# ---------------------------------------------------------------------------
# Check 35: remote archive URIs resolve in Supabase Storage (opt-in)
# ---------------------------------------------------------------------------


def _remote_check_with_bucket(docs, chunks, bucket_contents, bucket_name="archive-test"):
    """Run check 35 with an ``ArchiveStorage`` stub whose bucket holds the
    given ``{folder: [filename, ...]}`` map. Missing folders raise
    ``ArchiveStorageError`` to mirror real storage behavior."""
    from unittest.mock import MagicMock, patch

    from mtss.storage.archive_storage import ArchiveStorageError

    storage = MagicMock()
    storage.bucket_name = bucket_name

    def _list_folder(folder):
        if folder in bucket_contents:
            return [{"name": name} for name in bucket_contents[folder]]
        raise ArchiveStorageError(f"missing: {folder}")

    storage.list_folder.side_effect = _list_folder

    # The check imports ArchiveStorage inside the function, so patch the
    # canonical class target rather than a module-level alias.
    with patch("mtss.storage.archive_storage.ArchiveStorage", return_value=storage):
        return _check_remote_archive_uris(docs, chunks, verbose=False)


@pytest.mark.unit
def test_check_remote_archive_uris_all_present():
    doc = _make_doc(
        browse_uri="/archive/abc/email.md",
        download_uri="/archive/abc/email.eml",
    )
    bucket = {"abc": ["email.md", "email.eml"]}
    issues, warnings = _remote_check_with_bucket([doc], [], bucket)
    assert issues == []
    assert warnings == []


@pytest.mark.unit
def test_check_remote_archive_uris_flags_missing_object():
    doc = _make_doc(
        browse_uri="/archive/abc/email.md",
        download_uri="/archive/abc/email.eml",
    )
    bucket = {"abc": ["email.md"]}  # email.eml absent
    issues, _ = _remote_check_with_bucket([doc], [], bucket)
    assert any("1 archive URIs point to missing objects" in i for i in issues)
    assert any("abc/email.eml" in i for i in issues)


@pytest.mark.unit
def test_check_remote_archive_uris_flags_missing_folder():
    doc = _make_doc(browse_uri="/archive/gone/email.md")
    bucket: dict[str, list[str]] = {}  # folder entirely absent
    issues, _ = _remote_check_with_bucket([doc], [], bucket)
    assert any("archive URIs point to missing objects" in i for i in issues)
    assert any("gone/email.md" in i for i in issues)


@pytest.mark.unit
def test_check_remote_archive_uris_no_uris_is_noop():
    doc = _make_doc()  # no browse/download URIs
    assert _check_remote_archive_uris([doc], [], verbose=False) == ([], [])


# ---------------------------------------------------------------------------
# Check 17: duplicate doc_ids + chunk_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_duplicate_ids_flags_doc_and_chunk_dupes():
    docs = [
        _make_doc(doc_id="dup"),
        _make_doc(doc_id="dup"),
        _make_doc(doc_id="unique"),
    ]
    doc = docs[0]
    chunks = [
        _make_chunk(doc["id"], chunk_id="c1"),
        _make_chunk(doc["id"], chunk_id="c1"),
    ]
    issues, _ = _check_duplicate_ids(docs, chunks, verbose=False)
    assert any("1 duplicate doc_ids" in i for i in issues)
    assert any("1 duplicate chunk_ids" in i for i in issues)


@pytest.mark.unit
def test_check_duplicate_ids_clean():
    docs = [_make_doc(doc_id=f"doc{i}") for i in range(3)]
    chunks = [_make_chunk(docs[0]["id"], chunk_id=f"c{i}") for i in range(3)]
    assert _check_duplicate_ids(docs, chunks, verbose=False) == ([], [])


# ---------------------------------------------------------------------------
# Check 18: encoded filenames on disk
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_encoded_filenames_flags_percent_escaped(tmp_path):
    archive = tmp_path / "archive" / "abc" / "attachments"
    archive.mkdir(parents=True)
    (archive / "file%20name.pdf").write_bytes(b"x")
    (archive / "clean.pdf").write_bytes(b"x")

    issues, _ = _check_encoded_filenames(tmp_path / "archive", verbose=False)
    assert issues == [
        "1 archive files have URL-encoded names (run migration script to fix)"
    ]


@pytest.mark.unit
def test_check_encoded_filenames_clean(tmp_path):
    archive = tmp_path / "archive" / "abc"
    archive.mkdir(parents=True)
    (archive / "clean.pdf").write_bytes(b"x")
    assert _check_encoded_filenames(tmp_path / "archive", verbose=False) == ([], [])


# ---------------------------------------------------------------------------
# Check 19: encoded URIs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_encoded_uris_flags_percent_in_uri():
    doc = _make_doc(browse_uri="/archive/abc/file%20name.pdf.md")
    issues, _ = _check_encoded_uris([doc])
    assert issues == [
        "1 archive URIs contain URL-encoding (run migration script to fix)"
    ]


@pytest.mark.unit
def test_check_encoded_uris_clean():
    doc = _make_doc(browse_uri="/archive/abc/clean_file.md")
    assert _check_encoded_uris([doc]) == ([], [])


# ---------------------------------------------------------------------------
# Check 20: broken markdown links
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_broken_markdown_links_flags_missing_target(tmp_path):
    archive_dir = tmp_path / "archive"
    folder = archive_dir / "abc123"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text(
        "# hi\n\n- [Report](abc123/attachments/missing.pdf)\n"
    )
    docs = [_make_doc(doc_id="abc123xxxxxxxxxx", depth=0, source_id="thread.eml",
                      archive_path="abc123")]
    _, warnings = _check_broken_markdown_links(docs, archive_dir)
    assert any("1 broken markdown links" in w for w in warnings)


@pytest.mark.unit
def test_check_broken_markdown_links_all_valid(tmp_path):
    archive_dir = tmp_path / "archive"
    folder = archive_dir / "abc"
    (folder / "attachments").mkdir(parents=True)
    (folder / "attachments" / "report.pdf").write_bytes(b"pdf")
    (folder / "email.md").write_text("- [Report](abc/attachments/report.pdf)\n")
    docs = [_make_doc(archive_path="abc")]
    assert _check_broken_markdown_links(docs, archive_dir) == ([], [])


@pytest.mark.unit
def test_check_broken_markdown_links_ignores_prose_matches(tmp_path):
    """Engineer-prose artifacts like "[PC-JB1](17&18)" and Outlook-flattened
    "[cid:...](sample)" accidentally match markdown syntax but aren't real
    links. The check must skip them; otherwise every maritime technical
    email with bracketed component labels reports as a broken link.

    Regression source: 2026-04-17 1000-email validate run surfaced 4 such
    cases from 4 distinct emails — none were actually broken, just prose
    literally present in the source ``text/plain`` body.
    """
    archive_dir = tmp_path / "archive"
    folder = archive_dir / "prose"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text(
        "Report from M/T Vessel\n\n"
        "The cables to the Kyma JB [PC-JB1](17&18) are connected.\n\n"
        "[cid:17336829136755e6e11dffa694867792@fleet.marantankers.com](sample)\n"
    )
    docs = [_make_doc(archive_path="prose")]
    assert _check_broken_markdown_links(docs, archive_dir) == ([], [])


@pytest.mark.unit
def test_check_broken_markdown_links_still_flags_image_form_bare_tokens(tmp_path):
    """Image-form broken targets (``![alt](doge)``) must NOT be filtered as
    prose — they should have been removed by ``strip_llamaparse_image_refs``.
    If one surfaces here, the strip regex needs another pattern, so keep
    reporting it so we notice.
    """
    archive_dir = tmp_path / "archive"
    folder = archive_dir / "img"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text("![Screen showing alarm](doge)\n")
    docs = [_make_doc(archive_path="img")]
    _, warnings = _check_broken_markdown_links(docs, archive_dir)
    assert any("1 broken markdown links" in w for w in warnings)


@pytest.mark.unit
def test_check_broken_markdown_links_still_flags_real_missing_attachment(tmp_path):
    """A link-form reference to a path that looks like a real file
    (has an extension or a path separator) must still be flagged. Only the
    prose-shape (no sep, no extension, no scheme) false positives get
    filtered."""
    archive_dir = tmp_path / "archive"
    folder = archive_dir / "real"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text(
        "- [Attachment](real/attachments/gone.pdf)\n"
        "- [Nested](sub/dir/also-missing)\n"
    )
    docs = [_make_doc(archive_path="real")]
    _, warnings = _check_broken_markdown_links(docs, archive_dir)
    # Both remain broken: one has an extension, the other has a path separator.
    assert any("2 broken markdown links" in w for w in warnings)


# ---------------------------------------------------------------------------
# Check 21: chunk positions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_chunk_positions_flags_invalid():
    doc = _make_doc()
    chunks = [
        _make_chunk(doc["id"], char_start=0, char_end=10),  # good
        _make_chunk(doc["id"], char_start=-1, char_end=-1),  # thread digest — skip
        _make_chunk(doc["id"], char_start=-5, char_end=3),  # bad (negative start)
        _make_chunk(doc["id"], char_start=50, char_end=10),  # bad (start > end)
    ]
    _, warnings = _check_chunk_positions(chunks)
    assert warnings == [
        "2 chunks have invalid char positions (negative or start > end)"
    ]


@pytest.mark.unit
def test_check_chunk_positions_clean():
    doc = _make_doc()
    chunks = [
        _make_chunk(doc["id"], char_start=0, char_end=10),
        _make_chunk(doc["id"], char_start=-1, char_end=-1),
    ]
    assert _check_chunk_positions(chunks) == ([], [])


# ---------------------------------------------------------------------------
# Check 22: email metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_email_metadata_flags_bad_dates_and_missing_participants():
    bad_date = _make_doc(
        doc_type="email",
        email_date_start="2025-06-01",
        email_date_end="2024-01-01",
    )
    no_parts = _make_doc(doc_type="email", email_participants=[])
    _, warnings = _check_email_metadata([bad_date, no_parts])
    assert "1 emails have date_start > date_end" in warnings
    assert "1 emails have no participants" in warnings


@pytest.mark.unit
def test_check_email_metadata_clean():
    ok = _make_doc(
        doc_type="email",
        email_date_start="2024-01-01",
        email_date_end="2024-01-02",
        email_participants=["a@b.com"],
    )
    # Non-email docs are ignored entirely
    attachment = _make_doc(doc_type="attachment_document", email_participants=None)
    assert _check_email_metadata([ok, attachment]) == ([], [])


# ---------------------------------------------------------------------------
# Extended checks (23+)
# ---------------------------------------------------------------------------


def _make_mini_db(tmp_path, extra_sql: str = "") -> "sqlite3.Connection":
    """Create a minimal schema-correct DB for extended-check tests.

    Uses the live PROCESSING_LOG_SCHEMA_SQL so processing_log stays
    aligned with the production schema. `extra_sql` lets tests inject
    rows or violations.
    """
    import sqlite3
    from mtss.storage.sqlite_client import _SCHEMA_SQL
    db = tmp_path / "ingest.db"
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    if extra_sql:
        conn.executescript(extra_sql)
    return conn


# --- Check 23: SQLite integrity ----------------------------------------------


@pytest.mark.unit
def test_check_sqlite_integrity_clean(tmp_path):
    conn = _make_mini_db(tmp_path)
    try:
        assert _check_sqlite_integrity(conn) == ([], [])
    finally:
        conn.close()


@pytest.mark.unit
def test_check_sqlite_integrity_flags_fk_violation(tmp_path):
    # Insert a chunk_topics row pointing at a non-existent chunk and topic
    # by temporarily disabling FK enforcement.
    conn = _make_mini_db(tmp_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO chunk_topics(chunk_id, topic_id) VALUES (?, ?)",
            ("ghost-chunk", "ghost-topic"),
        )
        conn.execute("PRAGMA foreign_keys=ON")
        issues, _ = _check_sqlite_integrity(conn)
        assert any("foreign-key violations" in i for i in issues)
    finally:
        conn.close()


# --- Check 24: schema parity --------------------------------------------------


@pytest.mark.unit
def test_check_schema_parity_clean(tmp_path):
    conn = _make_mini_db(tmp_path)
    try:
        assert _check_schema_parity(conn) == ([], [])
    finally:
        conn.close()


@pytest.mark.unit
def test_check_schema_parity_flags_missing_column(tmp_path):
    # Drop-and-recreate processing_log without the ingest_version column.
    conn = _make_mini_db(tmp_path)
    try:
        conn.executescript(
            "DROP TABLE processing_log;"
            "CREATE TABLE processing_log ("
            "  file_path TEXT PRIMARY KEY, file_hash TEXT NOT NULL,"
            "  status TEXT NOT NULL, started_at TEXT, completed_at TEXT,"
            "  duration_seconds REAL, attempts INTEGER DEFAULT 0, error TEXT"
            ");"
        )
        issues, _ = _check_schema_parity(conn)
        assert any("processing_log" in i and "ingest_version" in i for i in issues)
    finally:
        conn.close()


# --- Check 25: embedding_mode coverage ---------------------------------------


@pytest.mark.unit
def test_check_embedding_mode_coverage_flags_missing():
    docs = [
        {**_make_doc(doc_type="attachment_document"), "embedding_mode": "full"},
        {**_make_doc(doc_type="attachment_document"), "embedding_mode": None},
        {**_make_doc(doc_type="attachment_document", status="failed")},  # ignored
        {**_make_doc(doc_type="attachment_image")},  # ignored
    ]
    issues, _ = _check_embedding_mode_coverage(docs)
    assert any("1 document(s) missing embedding_mode" in i for i in issues)


@pytest.mark.unit
def test_check_embedding_mode_coverage_flags_invalid():
    docs = [
        {**_make_doc(doc_type="attachment_document"), "embedding_mode": "bogus"},
    ]
    issues, _ = _check_embedding_mode_coverage(docs)
    assert any("unknown embedding_mode values" in i for i in issues)


@pytest.mark.unit
def test_check_embedding_mode_coverage_clean():
    docs = [
        {**_make_doc(doc_type="email"), "embedding_mode": "full"},
        {**_make_doc(doc_type="attachment_document"), "embedding_mode": "summary"},
    ]
    assert _check_embedding_mode_coverage(docs) == ([], [])


# --- Check 26: embedding_mode inheritance ------------------------------------


@pytest.mark.unit
def test_check_embedding_mode_inheritance_flags_mismatch():
    doc = {**_make_doc(), "embedding_mode": "full"}
    by_uuid = {doc["id"]: doc}
    chunks = [
        {**_make_chunk(doc["id"]), "embedding_mode": "summary"},  # mismatch
        {**_make_chunk(doc["id"]), "embedding_mode": None},       # missing
        {**_make_chunk(doc["id"]), "embedding_mode": "full"},     # ok
    ]
    issues, warnings = _check_embedding_mode_inheritance(chunks, by_uuid)
    assert any("disagrees with parent document" in i for i in issues)
    assert any("missing embedding_mode inherited" in w for w in warnings)


@pytest.mark.unit
def test_check_embedding_mode_inheritance_clean():
    doc = {**_make_doc(), "embedding_mode": "full"}
    by_uuid = {doc["id"]: doc}
    chunks = [{**_make_chunk(doc["id"]), "embedding_mode": "full"} for _ in range(3)]
    assert _check_embedding_mode_inheritance(chunks, by_uuid) == ([], [])


# --- Check 27: single-chunk modes --------------------------------------------


@pytest.mark.unit
def test_check_single_chunk_modes_flags_multi_chunk_summary():
    doc = {**_make_doc(), "embedding_mode": "summary"}
    counts = Counter({doc["id"]: 3})
    issues, _ = _check_single_chunk_modes([doc], counts)
    assert any("do not have exactly 1 chunk" in i for i in issues)


@pytest.mark.unit
def test_check_single_chunk_modes_clean():
    summary_doc = {**_make_doc(), "embedding_mode": "summary"}
    meta_doc = {**_make_doc(), "embedding_mode": "metadata_only"}
    full_doc = {**_make_doc(), "embedding_mode": "full"}
    counts = Counter({
        summary_doc["id"]: 1, meta_doc["id"]: 1, full_doc["id"]: 42,
    })
    assert _check_single_chunk_modes(
        [summary_doc, meta_doc, full_doc], counts
    ) == ([], [])


# --- Check 28: orphan archive folders ----------------------------------------


@pytest.mark.unit
def test_check_orphan_archive_folders_flags_extras():
    from mtss.utils import compute_folder_id
    doc = _make_doc(doc_id="kept00000000000000000000000000aa")
    expected_folder = compute_folder_id(doc["doc_id"])
    on_disk = {expected_folder, "orphan_folder_1", "orphan_folder_2"}
    _, warnings = _check_orphan_archive_folders([doc], on_disk)
    assert any("2 archive folder(s) on disk with no matching email" in w for w in warnings)


@pytest.mark.unit
def test_check_orphan_archive_folders_clean():
    from mtss.utils import compute_folder_id
    doc = _make_doc(doc_id="kept00000000000000000000000000aa")
    expected = compute_folder_id(doc["doc_id"])
    assert _check_orphan_archive_folders([doc], {expected}) == ([], [])


# --- Check 29: residual image refs -------------------------------------------


@pytest.mark.unit
def test_check_residual_image_refs_flags_llamaparse_artifacts(tmp_path):
    archive = tmp_path / "archive"
    folder = archive / "abc"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text(
        "# Report\n"
        "![fig](page_1_image_2.jpg)\n"
        "<img src=\"foo.png\" alt=\"bar\"/>\n"
        "![gen](image_3.png)\n"
    )
    _, warnings = _check_residual_image_refs(archive)
    assert any("residual image ref" in w for w in warnings)


@pytest.mark.unit
def test_check_residual_image_refs_clean(tmp_path):
    archive = tmp_path / "archive"
    folder = archive / "ok"
    folder.mkdir(parents=True)
    (folder / "email.md").write_text("# Clean\n\nJust prose, no refs.\n")
    assert _check_residual_image_refs(archive) == ([], [])


# --- Check 30: duplicate file_hash ------------------------------------------


@pytest.mark.unit
def test_check_duplicate_file_hashes_flags_emails_with_same_hash():
    docs = [
        {**_make_doc(source_id="a.eml"), "file_hash": "h1"},
        {**_make_doc(source_id="b.eml"), "file_hash": "h1"},
        {**_make_doc(source_id="c.eml"), "file_hash": "h2"},
    ]
    _, warnings = _check_duplicate_file_hashes(docs)
    assert any("map to multiple email documents" in w for w in warnings)


@pytest.mark.unit
def test_check_duplicate_file_hashes_ignores_attachment_dupes():
    docs = [
        {**_make_doc(doc_type="attachment_document", source_id="x"), "file_hash": "same"},
        {**_make_doc(doc_type="attachment_document", source_id="y"), "file_hash": "same"},
    ]
    assert _check_duplicate_file_hashes(docs) == ([], [])


# --- Check 31: embedding vector sanity --------------------------------------


@pytest.mark.unit
def test_check_embedding_vector_sanity_flags_zero_and_nan():
    doc = _make_doc()
    chunks = [
        _make_chunk(doc["id"], embedding=(0.0, 0.0, 0.0)),
        _make_chunk(doc["id"], embedding=(float("nan"), 0.1)),
        _make_chunk(doc["id"], embedding=(float("inf"), 0.1)),
        _make_chunk(doc["id"], embedding=(0.1, 0.2)),  # good
    ]
    issues, _ = _check_embedding_vector_sanity(chunks)
    assert any("are all zeros" in i for i in issues)
    assert any("NaN or Inf" in i for i in issues)


@pytest.mark.unit
def test_check_embedding_vector_sanity_clean():
    doc = _make_doc()
    chunks = [_make_chunk(doc["id"], embedding=(0.1, 0.2)) for _ in range(3)]
    assert _check_embedding_vector_sanity(chunks) == ([], [])


# --- Check 32: outdated ingest_version --------------------------------------


@pytest.mark.unit
def test_check_outdated_ingest_version_flags_older_rows():
    docs = [
        {**_make_doc(), "ingest_version": 3},
        {**_make_doc(), "ingest_version": 5},
        {**_make_doc(), "ingest_version": 5},
    ]
    _, warnings = _check_outdated_ingest_version(docs, current_version=5)
    assert any("below current ingest_version" in w and "v3=1" in w for w in warnings)


@pytest.mark.unit
def test_check_outdated_ingest_version_clean():
    docs = [{**_make_doc(), "ingest_version": 5} for _ in range(3)]
    assert _check_outdated_ingest_version(docs, current_version=5) == ([], [])


# --- Check 33: thread-root consistency --------------------------------------


@pytest.mark.unit
def test_check_thread_root_consistency_flags_email_with_foreign_root():
    email = _make_doc(doc_type="email")
    email["root_id"] = "some-other-uuid"  # email should point to itself
    issues, _ = _check_thread_root_consistency([email], {email["id"]})
    assert any("root_id != id" in i for i in issues)


@pytest.mark.unit
def test_check_thread_root_consistency_flags_attachment_with_missing_root():
    email = _make_doc(doc_type="email")
    att = _make_doc(doc_type="attachment_document", root_id="ghost-uuid")
    email_uuids = {email["id"]}
    _, warnings = _check_thread_root_consistency([email, att], email_uuids)
    assert any("root_id pointing to a non-email" in w for w in warnings)


@pytest.mark.unit
def test_check_thread_root_consistency_clean():
    email = _make_doc(doc_type="email")
    att = _make_doc(doc_type="attachment_document", root_id=email["id"])
    assert _check_thread_root_consistency([email, att], {email["id"]}) == ([], [])


# --- Check 34: stale processing entries -------------------------------------


@pytest.mark.unit
def test_check_stale_processing_entries_flags_old_rows(tmp_path):
    from datetime import datetime, timedelta, timezone
    conn = _make_mini_db(tmp_path)
    try:
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO processing_log(file_path, file_hash, status, started_at) "
            "VALUES (?, ?, 'PROCESSING', ?)",
            ("stuck.eml", "h", old_ts),
        )
        _, warnings = _check_stale_processing_entries(conn)
        assert any("stuck in PROCESSING" in w for w in warnings)
    finally:
        conn.close()


@pytest.mark.unit
def test_check_stale_processing_entries_clean(tmp_path):
    from datetime import datetime, timezone
    conn = _make_mini_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO processing_log(file_path, file_hash, status, started_at) "
            "VALUES (?, ?, 'PROCESSING', ?)",
            ("fresh.eml", "h", datetime.now(timezone.utc).isoformat()),
        )
        assert _check_stale_processing_entries(conn) == ([], [])
    finally:
        conn.close()


# --- Check 36: unknown vessel mentions --------------------------------------


@pytest.mark.unit
def test_check_unknown_vessel_mentions_flags_typo():
    canonical = {"MARAN CANOPUS", "MARAN APOLLO"}
    email = _make_doc(doc_type="email")
    chunk = _make_chunk(
        email["id"],
        content="Update on MARAN CANNOPUS arrival; MARAN APOLLO already departed.",
    )
    issues, warnings = _check_unknown_vessel_mentions(
        [chunk], [email], canonical, {email["archive_path"]: email["source_id"]},
    )
    assert issues == []
    # The typo "MARAN CANNOPUS" should be surfaced; the valid "MARAN APOLLO" should not.
    joined = "\n".join(warnings)
    assert "MARAN CANNOPUS" in joined
    assert "MARAN APOLLO" not in joined


@pytest.mark.unit
def test_check_unknown_vessel_mentions_clean():
    canonical = {"MARAN CANOPUS"}
    email = _make_doc(doc_type="email")
    chunk = _make_chunk(email["id"], content="Routine status update for MARAN CANOPUS.")
    assert _check_unknown_vessel_mentions(
        [chunk], [email], canonical, {},
    ) == ([], [])


@pytest.mark.unit
def test_check_unknown_vessel_mentions_empty_register_warns_once():
    email = _make_doc(doc_type="email")
    chunk = _make_chunk(email["id"], content="MARAN CANOPUS sailing")
    issues, warnings = _check_unknown_vessel_mentions(
        [chunk], [email], set(), {},
    )
    # Empty register skips the body of the check entirely — just emit one
    # advisory warning so the caller sees why no findings were produced.
    assert issues == []
    assert len(warnings) == 1
    assert "vessel register is empty" in warnings[0]


@pytest.mark.unit
def test_check_unknown_vessel_mentions_attributes_to_source_email():
    canonical = {"MARAN APOLLO"}  # Non-empty so the check actually runs.
    email = _make_doc(doc_type="email", source_id="incident-2024-08.eml")
    chunk = _make_chunk(email["id"], content="M.T. NEW DISCOVERY visible on AIS")
    folder_to_email = {email["archive_path"]: email["source_id"]}
    _, warnings = _check_unknown_vessel_mentions(
        [chunk], [email], canonical, folder_to_email,
    )
    # First-seen attribution should point at the source email by name.
    joined = "\n".join(warnings)
    assert "incident-2024-08.eml" in joined
