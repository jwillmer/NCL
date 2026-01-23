# Ingest Flow Documentation

This document provides detailed flowcharts of the ingest and ingest-update logic for the MTSS Email RAG Pipeline.

## Table of Contents

1. [Main Ingest Flow](#main-ingest-flow)
2. [Single Email Processing Flow](#single-email-processing-flow)
3. [Ingest-Update Flow](#ingest-update-flow)
4. [Data Integrity](#data-integrity)

---

## Main Ingest Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGEST COMMAND ENTRY                               │
│                         (cli.py:118 - ingest())                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                                            │
│    - Load settings and vessel registry                                       │
│    - Initialize SupabaseClient, ProgressTracker                             │
│    - Create IngestComponents (shared with ingest-update)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FILE CLASSIFICATION (classify_files_for_queues)                          │
│    - Peek at each EML file's attachments                                    │
│    - Fast lane: Emails with no attachments or images only                   │
│    - Slow lane: Emails with documents requiring LlamaParse                  │
│    - Worker split: 60% fast / 40% slow                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. FILE SELECTION STRATEGY                                                   │
│    ┌──────────────────┬──────────────────┬──────────────────┐               │
│    │ --resume (default)│ --retry-failed   │ --reprocess-out  │               │
│    │ Get pending files │ Reset stale files│ Get files with   │               │
│    │ not yet processed │ Retry failed     │ old ingest_ver   │               │
│    └──────────────────┴──────────────────┴──────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. CONCURRENT PROCESSING                                                     │
│    - Async workers process files from fast/slow queues                      │
│    - Each worker calls _process_single_email()                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Single Email Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _process_single_email() - cli.py:418                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: VERSION/DEDUPLICATION CHECK                                          │
│                                                                              │
│ - Compute file_hash (SHA-256) and source_id (normalized path)               │
│ - Compute doc_id = hash(source_id + file_hash)                              │
│                                                                              │
│ Decision Matrix (VersionManager.check_document):                            │
│ ┌──────────────────┬────────────────────────────────────────────┐           │
│ │ Action           │ Condition                                   │           │
│ ├──────────────────┼────────────────────────────────────────────┤           │
│ │ SKIP             │ doc_id exists with COMPLETED status        │           │
│ │ REPROCESS        │ doc_id exists with older ingest_version    │           │
│ │ UPDATE           │ source_id exists with different content    │           │
│ │ INSERT           │ No existing document found                 │           │
│ └──────────────────┴────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PARSE EMAIL                                                          │
│                                                                              │
│ - eml_parser.parse_file(eml_path) → ParsedEmail                             │
│ - Extracts: metadata, body (plain/HTML), attachments                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: GENERATE ARCHIVE                                                     │
│                                                                              │
│ - Creates folder: {doc_id[:16]}/                                            │
│ - Stores original attachments                                               │
│ - Stores .md files for parsed content                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CREATE EMAIL DOCUMENT                                                │
│                                                                              │
│ - hierarchy_manager.create_email_document()                                 │
│ - Stores: doc_id, source_id, file_hash, ingest_version, metadata            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: ENTITY EXTRACTION                                                    │
│                                                                              │
│ - vessel_matcher.find_vessels_in_email(subject, body)                       │
│ - Returns list of vessel_ids for filtering                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: GENERATE CONTEXT (LLM)                                               │
│                                                                              │
│ - context_generator.generate_context(email_doc, body_text[:4000])           │
│ - Creates semantic summary for better embedding quality                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: CHUNK EMAIL BODY                                                     │
│                                                                              │
│ - split_into_messages(body_text) → individual messages                      │
│ - For each message:                                                         │
│   - remove_boilerplate_from_message()                                       │
│   - Compute chunk_id from doc_id + char positions                           │
│   - Build embedding_text with context                                       │
│   - Create Chunk with vessel_ids in metadata                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: PROCESS ATTACHMENTS (for each attachment)                            │
│                                                                              │
│ 8a. Preprocess:                                                             │
│     - Check if ZIP → route to ZIP handler                                   │
│     - Check if image → classify content vs decorative                       │
│     - Otherwise → route to document parser                                  │
│                                                                              │
│ 8b. Create child document in hierarchy                                      │
│                                                                              │
│ 8c. Parse/Extract content:                                                  │
│     ┌────────────┬──────────────────────────────────────────┐               │
│     │ Type       │ Handler                                   │               │
│     ├────────────┼──────────────────────────────────────────┤               │
│     │ ZIP        │ extract_zip() → recursive processing     │               │
│     │ Image      │ Vision model describes → create chunk    │               │
│     │ Document   │ LlamaParse/TextParser → chunk            │               │
│     └────────────┴──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: GENERATE EMBEDDINGS                                                  │
│                                                                              │
│ - embeddings.embed_chunks(all_chunks)                                       │
│ - Batches to OpenAI API via LiteLLM                                         │
│ - Truncates to token limit if needed                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 10: STORE TO DATABASE                                                   │
│                                                                              │
│ - db.insert_chunks(chunks_with_embeddings)                                  │
│ - db.update_document_status(doc_id, COMPLETED)                              │
│ - tracker.mark_completed(eml_path)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 11: CLEANUP                                                             │
│                                                                              │
│ - Delete temporary attachment folder                                        │
│ - Safety check: only under managed directory                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ingest-Update Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INGEST-UPDATE COMMAND ENTRY                            │
│                     (cli.py:1893 - ingest_update())                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1a: SCAN FOR ISSUES (_scan_ingest_issues)                             │
│                                                                              │
│ For each document in DB:                                                    │
│   - Check if source .eml file exists                                        │
│   - Check if archive exists in storage bucket                               │
│   - Check if chunks have line numbers (line_from, line_to)                  │
│   - Check if chunks have context summaries                                  │
│                                                                              │
│ Output: List of IssueRecord with issues and cached chunks                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1b: FIND ORPHANED DOCUMENTS (_find_orphaned_documents)                │
│                                                                              │
│ - Get all root source_ids from DB                                           │
│ - Compare against existing .eml files                                       │
│ - Mark documents without source files as orphans                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2a: REMOVE ORPHANS                                                     │
│                                                                              │
│ - db.delete_orphaned_documents(orphan_ids)                                  │
│ - Cascades to child documents and chunks                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2b: FIX DOCUMENT ISSUES (_fix_document_issues)                        │
│                                                                              │
│ For each IssueRecord:                                                       │
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FIX MISSING ARCHIVES (_fix_missing_archives)                            │ │
│ │ - First check if archive exists in bucket but DB link is missing        │ │
│ │ - If found → update DB link only                                        │ │
│ │ - If missing → regenerate from parsed email                             │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FIX MISSING LINE NUMBERS (_fix_missing_lines)                           │ │
│ │ - Re-chunk from archive markdown                                        │ │
│ │ - Use replace_chunks_atomic() for atomic replacement                    │ │
│ │ - Regenerate embeddings                                                 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FIX MISSING CONTEXT (_fix_missing_context)                              │ │
│ │ - Regenerate LLM context summaries                                      │ │
│ │ - Update embedding_text with context                                    │ │
│ │ - Re-embed chunks                                                       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Integrity

The ingest command includes data integrity protections to ensure reliable data ingestion.

### Error Handling Behavior

By default, the ingest command fails documents when critical data loss is detected:

| Error Type | Default Behavior | Rationale |
|------------|------------------|-----------|
| Archive generation fails | Log warning, continue | Non-critical backup |
| Context generation fails | Log warning, continue | Degraded but functional |
| Attachment parse fails | Log error, **FAIL document** | Data loss |
| ZIP member fails | Log error, **FAIL document** | Data loss |

### Lenient Mode

Use `--lenient` flag for backward compatibility with softer error handling:

```bash
# Strict mode (default) - fail on data loss
uv run MTSS ingest --source ./data/emails

# Lenient mode - log errors but continue processing
uv run MTSS ingest --source ./data/emails --lenient
```

### Monitoring Ingest Quality

Processing events are logged to the `ingest_events` table for visibility:

```sql
-- View all events by type
SELECT event_type, severity, COUNT(*)
FROM ingest_events
GROUP BY event_type, severity;

-- Find documents with errors
SELECT parent_document_id, event_type, message
FROM ingest_events
WHERE severity = 'error'
ORDER BY discovered_at DESC;

-- Get events for a specific document
SELECT * FROM ingest_events
WHERE parent_document_id = 'your-doc-uuid';
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `unsupported_file` | File format not supported (existing behavior) |
| `encoding_fallback` | Character replacement used during decoding |
| `parse_failure` | Parser returned empty or error |
| `archive_failure` | Archive generation failed |
| `context_failure` | LLM context generation failed |
| `empty_content` | Content empty after processing |

### Retry Logic

LLM and embedding API calls include automatic retry with exponential backoff:

- **Max retries:** 3
- **Backoff schedule:** 1s, 2s, 4s
- **Errors retried:** Rate limits, timeouts, transient failures
