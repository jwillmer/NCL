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
│                         (cli/ingest_cmd.py:118 - ingest())                             │
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
│    - Slow lane: Emails with docs needing a heavy parser (Gemini/LlamaParse) │
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
│                    process_email() - ingest/pipeline.py:418                     │
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
│ STEP 6b: TOPIC EXTRACTION                                                    │
│                                                                              │
│ Uses archived markdown content (cleaner than raw email body):              │
│ - Banners, signatures, and HTML artifacts already removed                  │
│ - Falls back to raw body_text if archive not available                     │
│                                                                              │
│ Multi-source extraction (for long email threads):                           │
│ - Subject line: Usually contains core topic                                 │
│ - Original message: Bottom of thread = problem description (NOT solutions) │
│ - Context summary: Semantic-rich overview from Step 6                       │
│                                                                              │
│ topic_extractor.extract_topics(structured_input) → 1-5 topics per email    │
│ - For each extracted topic:                                                 │
│   - topic_matcher.get_or_create_topic() with semantic deduplication        │
│   - Dedup threshold: >=0.85 auto-merge, <0.85 create new                   │
│ - Store topic_ids in chunk.metadata for query-time filtering               │
│ - Topics enable early-return optimization in RAG queries                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: CHUNK EMAIL BODY                                                     │
│                                                                              │
│ - clean_email_body() → LLM boundary detection + regex boilerplate removal   │
│ - split_into_messages(cleaned_body) → individual messages                   │
│ - For each message:                                                         │
│   - remove_boilerplate_from_message()                                       │
│   - Compute chunk_id from doc_id + char positions                           │
│   - Build embedding_text with context                                       │
│   - Create Chunk with vessel_ids and topic_ids in metadata                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
          ┌─────────────────────┴─────────────────────┐
          │                                           │
          ▼                                           ▼
┌───────────────────────────┐  ┌──────────────────────────────────────────────┐
│ STEP 7b: THREAD DIGEST    │  │ STEP 8: PROCESS ATTACHMENTS (concurrent)    │
│ (runs in parallel)        │  │                                              │
│                           │  │ 8a. Route by type:                           │
│ For multi-message threads │  │     - ZIP → extract, recurse                 │
│ (2+ messages):            │  │     - Image → classify, describe if content  │
│ - Sanitize input          │  │     - PDF simple → PyMuPDF4LLM (free)        │
│ - LLM summarizes thread   │  │     - PDF complex → Gemini 2.5 Flash         │
│ - Creates one digest      │  │     - .docx/.xlsx/.csv/.html → local         │
│   chunk (type=            │  │     - .doc/.xls/.ppt → LlamaParse            │
│   thread_digest)          │  │                                              │
│ - Tagged with same        │  │ 8b. Create child document in hierarchy       │
│   vessel/topic metadata   │  │                                              │
│                           │  │ 8c. embedding_decider → full / summary /     │
│                           │  │     metadata_only (per-doc, stamped on doc + │
│                           │  │     every chunk)                             │
│                           │  │                                              │
│                           │  │ 8d. Chunk per chosen mode + chunk metadata   │
└─────────┬─────────────────┘  └──────────────────────┬───────────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: GENERATE EMBEDDINGS                                                  │
│                                                                              │
│ - embeddings.embed_chunks(all_chunks including digest)                      │
│ - Batches to OpenAI API via LiteLLM                                         │
│ - Truncates to token limit if needed                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 10: ATOMIC PERSIST                                                      │
│                                                                              │
│ - Single asyncpg transaction:                                               │
│   - Insert email document + attachment documents                            │
│   - Insert all chunks with embeddings                                       │
│   - Update topic counts                                                     │
│ - Rollback on any failure (no partial state)                                │
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
│                     (cli/ingest_cmd.py:1893 - ingest_update())                         │
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

## Input Sanitization

Sanitization is applied at multiple stages to prevent corrupted data from entering the pipeline.

### Stage 1: EML Parsing (`parsers/eml_parser.py`)

| What | How | Why |
|------|-----|-----|
| UTF-8 BOM | Strip `\xef\xbb\xbf` prefix | Some email clients add BOM that breaks multipart parsing |
| HTML body | Convert to plain text, preserve hyperlinks as markdown `[text](url)` | UI renders markdown |
| Encoding fallback | Try declared charset, then UTF-8, then latin-1 with `errors="replace"` | Never crash on bad encoding |

### Stage 2: Email Cleaning (`parsers/email_cleaner.py`)

| What | How | Why |
|------|-----|-----|
| Content boundaries | LLM detects first/last meaningful word anchors | Zero-modification extraction of relevant content |
| Boilerplate | Regex removes signatures, contact blocks, mailto syntax, CID image refs | Noise reduction for embeddings |
| Short messages | Filter messages <20 words after cleaning | Removes auto-replies, signature-only messages |

### Stage 3: Archive Storage Keys (`ingest/archive_generator.py`)

`_sanitize_storage_key()` applied to all filenames stored in archive:

| Character | Replacement | Why |
|-----------|-------------|-----|
| `[ ] ( ) ~` | `_` | Break markdown link syntax |
| Non-ASCII | NFKD transliteration to ASCII | Storage key compatibility |
| All-non-ASCII names | Hex-encoded fallback | Prevent empty keys |
| Spaces, `' , # &` | `_` or removed | URL-safe keys, no encoding needed |
| Multiple `_` | Collapsed to single `_` | Clean keys |

### Stage 4: Markdown Heading Escaping (`ingest/archive_generator.py`)

`_escape_markdown_heading()` applied to email subjects and filenames in generated `.md` files:

| Character | Escaped to | Why |
|-----------|------------|-----|
| `\` | `\\` | Prevent escape sequences |
| `[ ]` | `\[ \]` | Prevent link creation |
| `#` | `\#` | Prevent heading injection |
| `` ` `` | `` \` `` | Prevent code span injection |

### Stage 5: Topic Extraction (`processing/topics.py`)

`sanitize_input()` applied before LLM calls:
- Strips control characters (preserves `\n`, `\t`)
- Truncates to `max_length` (default 6000 chars)
- Mitigates prompt injection patterns

### Stage 6: ZIP Extraction (`parsers/attachment_processor.py`)

| Check | Action |
|-------|--------|
| `..` in path | Skip file (path traversal) |
| Absolute paths | Skip file |
| Windows drive letters | Skip file |
| Hidden files (`.`, `__MACOSX`) | Skip file |
| Compression ratio >100:1 | Skip file (ZIP bomb) |

---

## Validation (`mtss validate ingest`)

Post-ingest validation catches issues that slip through the pipeline. Run after every ingest.

### Checks (22 total)

| # | Check | Severity | What it catches |
|---|-------|----------|-----------------|
| 1 | Duplicate UUIDs | Error | Serialization bugs |
| 2 | Processing log status | Error | Files stuck in non-COMPLETED state |
| 3-5 | Chunk-document linkage | Error | Orphan chunks, missing embeddings |
| 6 | Empty chunk content | Error | Chunks with no text |
| 7 | Context/embedding_text presence | Warning | LLM failures during context generation |
| 8 | Documents without chunks | Error | Processing failures not caught |
| 9-10 | Hierarchy and status | Warning | Broken parent chains, failed docs in output |
| 11 | Trailing-dot filenames | Warning | Potential parsing issues |
| 12-15 | Topic health | Warning/Error | Missing embeddings, stale refs, inaccurate counts |
| 16 | Archive URI file existence | Error | URIs pointing to missing files on disk |
| 17 | Duplicate doc_ids/chunk_ids | Error | Content-addressable ID collisions |
| 18-19 | URL-encoded names/URIs | Error | Pre-migration state needing cleanup |
| 20 | Broken markdown internal links | Warning | Sanitization mismatches in archive .md files |
| 21 | Chunk position validity | Warning | `char_start > char_end` or negative positions |
| 22 | Email metadata consistency | Warning | `date_start > date_end`, missing participants |

### Running Validation

```bash
# Validate local ingest output
mtss validate ingest [--output-dir PATH] [--verbose]

# Compare local output against Supabase (after import)
mtss validate import [--output-dir PATH] [--verbose]
```

---

## Test Coverage

### Regression Tests (`tests/test_archive_uris.py`)

Covers 7 categories of archive URI bugs:
1. No double `/archive/` prefix in URIs
2. Sanitized filename matching for `archive_file_result`
3. Chunk URI propagation from document
4. Citation processor strips `/archive/` prefix
5. No double URL-encoding (`%2520`)
6. Frontend `stripArchivePrefix` defense-in-depth
7. **Download links and [View] links use sanitized filenames** (prevents broken markdown)

### Archive Generator Tests (`tests/test_archive_generator.py`)

- Markdown generation, attachment processing, URI construction
- `_sanitize_storage_key`: brackets, parens, tilde, non-ASCII, hex fallback

### Pipeline Tests (`tests/test_ingest_*.py`)

- Full email processing flow (parse → chunk → embed)
- Ingest consistency (deterministic output)
- Ingest update/repair flow
- Import roundtrip verification

### Parser Tests

- EML parsing: UTF-8 BOM, attachments, metadata, HTML-to-text with link preservation
- Attachment processor: ZIP extraction, image classification, routing
- Image filter: size/dimension/filename pattern filtering

---

## Data Integrity

The ingest command includes data integrity protections to ensure reliable data ingestion.

### Crash Safety

| Mechanism | Location | Protection |
|-----------|----------|------------|
| Progress tracking with `fsync` | `local_progress_tracker.py` | No duplicate processing after power loss |
| Atomic persist | `persist_ingest_result()` | All documents + chunks + topic counts in one transaction |
| Managed cleanup | Attachment folder deletion only under managed directory | No accidental data loss |

### Error Handling Behavior

By default, the ingest command fails documents when critical data loss is detected:

| Error Type | Default Behavior | Rationale |
|------------|------------------|-----------|
| Archive generation fails | Log warning, continue | Non-critical backup |
| Context generation fails | Log warning, continue | Degraded but functional |
| Attachment parse fails | Log error, **FAIL document** | Data loss |
| ZIP member fails | Log error, **FAIL document** | Data loss |

### Lenient Mode

```bash
# Strict mode (default) - fail on data loss
mtss ingest --source ./data/emails

# Lenient mode - log errors but continue processing
mtss ingest --source ./data/emails --lenient
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `unsupported_file` | File format not supported |
| `encoding_fallback` | Character replacement used during decoding |
| `parse_failure` | Parser returned empty or error |
| `archive_failure` | Archive generation failed |
| `context_failure` | LLM context generation failed |
| `empty_content` | Content empty after processing |
| `classified_as_non_content` | Image filtered as decorative/tracking |
| `message_filtered` | Message too short after boilerplate removal |
| `no_body_chunks` | Email body produced 0 chunks |

### Retry Logic

LLM and embedding API calls include automatic retry with exponential backoff:

- **Max retries:** 3
- **Backoff schedule:** 1s, 2s, 4s
- **Errors retried:** Rate limits, timeouts, transient failures
