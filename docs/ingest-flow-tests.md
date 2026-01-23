# Ingest Flow Test Documentation

This document describes each step in the ingest processing pipeline, explains what it does and why, and references the tests that validate each step.

## Table of Contents

1. [Ingest Flow Steps](#ingest-flow-steps)
2. [Update Flow Steps](#update-flow-steps)
3. [Test File Summary](#test-file-summary)
4. [Running Tests](#running-tests)

---

## Ingest Flow Steps

### Step 1: Parse Email

**What it does:** Extracts metadata, body text, and attachments from an EML file.

**Why:** This is the entry point for all email data into the system. The parser handles multi-part MIME messages, character encoding issues, and extracts individual messages from conversation threads.

**Key operations:**
- Extract subject, sender, recipients, dates from headers
- Decode body text (handling multiple charsets with fallbacks)
- Identify and save attachments to temp directory
- Parse conversation threads into individual messages

**Tested by:**
- `test_ingest_flow.py::TestIngestFlowIntegration::test_full_email_processing_flow`
- `test_eml_parser.py::TestEmlParser::*`

---

### Step 2: Parse Attachments

**What it does:** Saves attachments to a temporary folder for processing.

**Why:** Attachments need to be extracted to disk before they can be processed by document parsers (LlamaParse) or classified by the image processor.

**Key operations:**
- Decode base64/quoted-printable content
- Preserve original filename (sanitized for filesystem)
- Track content type and file size

**Tested by:**
- `test_ingest_flow.py::TestIngestFlowIntegration::test_attachment_processing_flow`
- `test_eml_parser.py::TestEmlParser::test_parse_email_with_attachments`

---

### Step 3: Compute File Hash

**What it does:** Computes SHA-256 hash of the file content.

**Why:** The hash enables content-addressable deduplication. If the same file is ingested twice, the hash will match and we can skip reprocessing.

**Key operations:**
- Stream file content through SHA-256
- Return 64-character hex digest

**Tested by:**
- `test_ingest_processing.py::TestHierarchyManager::test_compute_file_hash`
- `test_ingest_processing.py::TestHierarchyManager::test_compute_file_hash_deterministic`

---

### Step 4: Compute Source ID

**What it does:** Normalizes the file path to create a stable source identifier.

**Why:** Source IDs enable tracking documents across re-ingests when content changes. The normalized form (lowercase, forward slashes, relative to ingest root) ensures consistent identification regardless of platform or absolute path.

**Key operations:**
- Convert to relative path from ingest root
- Normalize separators to forward slashes
- Convert to lowercase

**Tested by:**
- `test_ingest_processing.py::TestUtils::test_normalize_source_id`
- `test_ingest_processing.py::TestUtils::test_normalize_source_id_lowercase`

---

### Step 5: Compute Doc ID

**What it does:** Computes a content-addressable document ID from source_id and file_hash.

**Why:** The doc_id uniquely identifies a specific version of a document. It combines "where" (source_id) with "what" (file_hash) to create a stable identifier for deduplication and version tracking.

**Key operations:**
- Hash source_id + file_hash together
- Truncate to 16 characters for readability

**Tested by:**
- `test_ingest_processing.py::TestUtils::test_compute_doc_id_deterministic`
- `test_ingest_processing.py::TestUtils::test_compute_doc_id_different_for_different_inputs`

---

### Step 6: Version Check

**What it does:** Decides what action to take based on existing documents in the database.

**Why:** Prevents duplicate processing and enables incremental updates. The version manager checks if the document exists and what action to take.

**Actions:**
- `insert`: New document, never seen before
- `update`: Same source, but content changed
- `skip`: Already processed with current ingest version
- `reprocess`: Processed with older ingest logic version

**Tested by:**
- `test_version_manager.py::TestVersionManager::test_skip_when_doc_id_exists_and_completed`
- `test_version_manager.py::TestVersionManager::test_reprocess_when_ingest_version_outdated`
- `test_version_manager.py::TestVersionManager::test_update_when_source_id_exists_with_different_content`
- `test_version_manager.py::TestVersionManager::test_insert_when_no_existing_document`

---

### Step 7: Create Email Document

**What it does:** Inserts a document record in the database hierarchy.

**Why:** The document record serves as the root of a document tree. Attachments become children of this document, enabling hierarchical queries and cascade deletions.

**Key operations:**
- Create Document model with all metadata
- Set depth=0, parent_id=None, root_id=self.id
- Insert into documents table

**Tested by:**
- `test_ingest_processing.py::TestHierarchyManager::test_create_email_document`
- `test_ingest_storage.py::TestSupabaseClient::test_insert_document`

---

### Step 8: Generate Archive

**What it does:** Uploads original files and markdown previews to Supabase Storage.

**Why:** The archive enables source citations with clickable links. Users can view markdown previews or download original files directly from the UI.

**Key operations:**
- Create folder structure: `{doc_id[:16]}/`
- Upload original EML file
- Generate email.eml.md with formatted metadata and messages
- Upload attachment originals and .md previews

**Tested by:**
- `test_archive_generator.py::TestArchiveGenerator::test_generates_email_archive_markdown`
- `test_archive_generator.py::TestArchiveGenerator::test_generates_attachment_archive_markdown`
- `test_archive_generator.py::TestArchiveGenerator::test_uploads_original_files_to_bucket`
- `test_archive_generator.py::TestArchiveGenerator::test_returns_correct_uris`

---

### Step 9: Vessel Matching

**What it does:** Finds vessel names mentioned in the email content.

**Why:** Enables filtering search results by vessel. Documents are tagged with vessel IDs during ingest so users can scope their queries to specific vessels.

**Key operations:**
- Search subject and body for vessel names/aliases
- Use word boundary matching to avoid false positives
- Return set of vessel UUIDs

**Tested by:**
- `test_vessel_matcher.py::TestVesselMatcher::test_finds_vessel_name_in_subject`
- `test_vessel_matcher.py::TestVesselMatcher::test_finds_vessel_name_in_body`
- `test_vessel_matcher.py::TestVesselMatcher::test_finds_vessel_alias`
- `test_vessel_matcher.py::TestVesselMatcher::test_no_match_returns_empty_list`

---

### Step 10: Generate Context

**What it does:** Uses LLM to generate a summary for embedding quality improvement.

**Why:** Contextual chunking significantly improves retrieval quality (35-67% improvement). The LLM-generated summary is prepended to each chunk's embedding text.

**Key operations:**
- Build prompt with document metadata and content preview
- Call LLM (gpt-4o-mini) for summary generation
- Fall back to metadata-based context on failure

**Tested by:**
- `test_ingest_processing.py::TestContextGenerator::test_generate_context_success`
- `test_ingest_processing.py::TestContextGenerator::test_generate_context_fallback_on_error`
- `test_ingest_processing.py::TestContextGenerator::test_build_embedding_text`

---

### Step 11: Chunk Email Body

**What it does:** Splits email body into chunks with line tracking.

**Why:** Large documents need to be split into chunks that fit within embedding model context windows. Line tracking enables precise citations.

**Key operations:**
- Use LangChain text splitter with markdown awareness
- Track line_from, line_to for each chunk
- Track char_start, char_end for chunk_id generation
- Preserve heading hierarchy in section_path

**Tested by:**
- `test_ingest_processing.py::TestDocumentChunker::test_chunk_short_text_single_chunk`
- `test_ingest_processing.py::TestDocumentChunker::test_chunk_long_text_multiple_chunks`
- `test_ingest_processing.py::TestDocumentChunker::test_chunk_tracks_line_numbers`
- `test_ingest_processing.py::TestDocumentChunker::test_chunk_tracks_character_positions`

---

### Step 12: Create Attachment Documents

**What it does:** Inserts child document records for each attachment.

**Why:** Attachments are separate documents in the hierarchy, enabling independent chunking, status tracking, and cascade deletion.

**Key operations:**
- Create Document with parent_id pointing to email document
- Set depth = parent.depth + 1
- Compute source_id as "email_source_id/attachment_filename"

**Tested by:**
- `test_ingest_processing.py::TestHierarchyManager::test_create_attachment_document`

---

### Step 13: Process Attachments

**What it does:** Parses PDFs, images, and ZIP archives into text.

**Why:** Attachment content needs to be extracted and made searchable. Different file types require different parsers.

**Key operations:**
- Route by content type (PDF, image, ZIP, Office docs)
- Use LlamaParse for documents
- Use Vision API for images (with classification)
- Extract and recurse for ZIP files

**Tested by:**
- `test_ingest_processing.py::TestAttachmentProcessor::test_classifies_pdf_as_document`
- `test_ingest_processing.py::TestAttachmentProcessor::test_classifies_image_as_meaningful_vs_decorative`
- `test_ingest_processing.py::TestAttachmentProcessor::test_extracts_zip_contents`
- `test_ingest_processing.py::TestAttachmentProcessor::test_handles_nested_zip`
- `test_attachment_processor.py::*`

---

### Step 14: Enrich Chunks

**What it does:** Adds document metadata and generates chunk_id for each chunk.

**Why:** Chunks need citation metadata for the UI to display source links. The chunk_id enables direct linking to specific content.

**Key operations:**
- Generate chunk_id from doc_id + char_start + char_end
- Copy source_id, source_title from document
- Copy archive_browse_uri, archive_download_uri

**Tested by:**
- `test_ingest_storage.py::TestIngestHelpers::test_enrich_chunks_sets_chunk_id`
- `test_ingest_storage.py::TestIngestHelpers::test_enrich_chunks_sets_source_metadata`

---

### Step 15: Generate Embeddings

**What it does:** Calls OpenAI embedding API to vectorize chunk content.

**Why:** Vector embeddings enable semantic similarity search. The embedding_text (context + content) is what gets embedded.

**Key operations:**
- Batch chunks for efficiency
- Truncate to max tokens if needed
- Handle rate limits with retry

**Tested by:**
- `test_ingest_processing.py::TestEmbeddingGenerator::test_generate_single_embedding`
- `test_ingest_processing.py::TestEmbeddingGenerator::test_generate_embeddings_batch`
- `test_ingest_processing.py::TestEmbeddingGenerator::test_embed_chunks`

---

### Step 16: Store Chunks

**What it does:** Batch inserts chunks with embeddings to the database.

**Why:** Bulk insert is more efficient than individual inserts. The chunks table stores content, embeddings, and citation metadata.

**Key operations:**
- Use asyncpg for bulk insert
- Include all citation fields (chunk_id, source_id, archive_uris, etc.)
- Store embedding as pgvector type

**Tested by:**
- `test_ingest_storage.py::TestSupabaseClient::test_insert_chunks`
- `test_ingest_storage.py::TestSupabaseClient::test_insert_chunks_empty_list`

---

### Step 17: Update Document Status

**What it does:** Marks the document as COMPLETED in the database.

**Why:** Status tracking enables progress monitoring, retry of failed documents, and filtering in queries.

**Key operations:**
- Update status to COMPLETED
- Set processed_at timestamp
- Optionally set error_message for FAILED status

**Tested by:**
- `test_ingest_storage.py::TestSupabaseClient::test_update_document_status`
- `test_ingest_storage.py::TestSupabaseClient::test_update_document_status_with_error`

---

### Step 18: Log Events

**What it does:** Records processing events (errors, warnings) for monitoring.

**Why:** Ingest events provide visibility into processing quality. The ingest_events table can be queried to find documents with issues.

**Key operations:**
- Insert event with type, severity, message
- Link to parent document for context
- Truncate long messages for security

**Tested by:**
- `test_ingest_storage.py::TestSupabaseClient::test_log_ingest_event`
- `test_ingest_storage.py::TestSupabaseClient::test_log_ingest_event_truncates_message`

---

## Update Flow Steps

The update flow (`MTSS ingest-update`) fixes documents that were ingested with missing data.

### Step 1: Find Orphans

**What it does:** Detects documents in the database whose source files no longer exist.

**Why:** Source files may be moved or deleted. Orphaned documents should be flagged or removed to maintain data integrity.

**Tested by:**
- `test_ingest_update_flow.py::TestFindOrphanedDocuments::test_finds_orphaned_documents`
- `test_ingest_update_flow.py::TestFindOrphanedDocuments::test_no_orphans_when_all_files_exist`

---

### Step 2: Scan Issues

**What it does:** Checks documents for missing archive_browse_uri, line numbers, or context.

**Why:** Earlier ingest versions may have skipped certain fields. Scanning identifies what needs to be fixed.

**Tested by:**
- `test_ingest_update_flow.py::TestScanIngestIssues::test_detects_missing_archive`
- `test_ingest_update_flow.py::TestScanIngestIssues::test_detects_missing_lines`
- `test_ingest_update_flow.py::TestScanIngestIssues::test_detects_missing_context`

---

### Step 3: Fix Archives

**What it does:** Regenerates or links existing archive files.

**Why:** Documents missing archive_browse_uri cannot provide source links in the UI.

**Tested by:**
- `test_ingest_update_flow.py::TestFixMissingArchives::test_regenerates_archive`
- `test_ingest_update_flow.py::TestFixMissingArchives::test_links_existing_archive`

---

### Step 4: Fix Line Numbers

**What it does:** Re-chunks content with line tracking enabled.

**Why:** Chunks missing line_from/line_to cannot provide precise line-level citations.

**Tested by:**
- `test_ingest_update_flow.py::TestFixMissingLines::test_rechunks_with_line_tracking`
- `test_ingest_update_flow.py::TestFixMissingLines::test_preserves_chunk_content`

---

### Step 5: Fix Context

**What it does:** Regenerates LLM context summaries for chunks.

**Why:** Chunks missing context_summary have lower retrieval quality.

**Tested by:**
- `test_ingest_update_flow.py::TestFixMissingContext::test_generates_context_summary`
- `test_ingest_update_flow.py::TestFixMissingContext::test_updates_embedding_text`

---

### Step 6: Fix Orchestration

**What it does:** Executes fixes in correct dependency order.

**Why:** Some fixes depend on others (e.g., archive must exist before line fix can update URIs).

**Tested by:**
- `test_ingest_update_flow.py::TestFixDocumentIssues::test_executes_fixes_in_order`
- `test_ingest_update_flow.py::TestFixDocumentIssues::test_skips_already_fixed_documents`

---

## Test File Summary

| File | Purpose | Markers |
|------|---------|---------|
| `test_eml_parser.py` | Email parsing | `@pytest.mark.unit` |
| `test_attachment_processor.py` | Attachment extraction | `@pytest.mark.unit` |
| `test_ingest_processing.py` | Chunker, embeddings, hierarchy, utils | `@pytest.mark.unit` |
| `test_ingest_storage.py` | Database client, helpers | `@pytest.mark.unit` |
| `test_ingest_flow.py` | Full ingest flow with mocks | `@pytest.mark.integration` |
| `test_ingest_update_flow.py` | Update flow with mocks | `@pytest.mark.integration` |
| `test_version_manager.py` | Version checking logic | `@pytest.mark.unit` |
| `test_archive_generator.py` | Archive generation | `@pytest.mark.unit` |
| `test_vessel_matcher.py` | Vessel name matching | `@pytest.mark.unit` |
| `test_ingest_consistency.py` | Ingest/update consistency | `@pytest.mark.integration` |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run ingest-specific tests
pytest tests/test_ingest*.py tests/test_version_manager.py tests/test_archive_generator.py tests/test_vessel_matcher.py -v

# Run with coverage
pytest tests/ --cov=mtss --cov-report=html

# Run only fast unit tests
pytest tests/ -m "unit" -v

# Run integration tests (slower)
pytest tests/ -m "integration" -v

# Verify consistency tests
pytest tests/test_ingest_consistency.py -v
```

---

## Coverage Gaps

The following areas have limited or no test coverage:

1. **Full end-to-end integration** - Requires real database/storage
2. **LlamaParse parsing** - Requires API key and credits
3. **Vision API classification** - Requires API key
4. **Rate limit handling** - Difficult to test reliably

These are typically tested manually or in staging environments.
