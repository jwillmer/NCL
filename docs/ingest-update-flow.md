# Ingest-Update Flow

This document provides detailed flowcharts for the `ingest-update` command, which validates and repairs ingested data.

## Table of Contents

1. [Command Overview](#command-overview)
2. [Issue Detection Flow](#issue-detection-flow)
3. [Orphan Detection Flow](#orphan-detection-flow)
4. [Fix Pipeline](#fix-pipeline)
5. [Dry-Run vs Execute](#dry-run-vs-execute)
6. [Atomic Operations](#atomic-operations)

---

## Command Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INGEST-UPDATE COMMAND ENTRY                           │
│                       (cli.py:1907 - ingest_update())                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SCAN                                                               │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1a. Find orphaned documents                                             │ │
│ │     _find_orphaned_documents() - cli.py:2084                            │ │
│ │     Compare DB source_ids against existing .eml files                   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1b. Scan for document issues                                            │ │
│ │     _scan_ingest_issues() - cli.py:2114                                 │ │
│ │     Check each document for: archive, lines, context                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: REPORT (--dry-run stops here)                                      │
│                                                                             │
│ Display summary:                                                            │
│   - Orphaned documents count                                                │
│   - Documents with archive issues                                           │
│   - Documents with missing line numbers                                     │
│   - Documents with missing context                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                            (if not --dry-run)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: FIX                                                                │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 3a. Remove orphaned documents                                           │ │
│ │     db.delete_orphaned_documents(orphan_ids)                            │ │
│ │     Cascades to children and chunks                                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 3b. Fix document issues (for each IssueRecord)                          │ │
│ │     _fix_document_issues() - cli.py:2258                                │ │
│ │     Executes fixes in dependency order                                  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Issue Detection Flow

The scan phase checks each document for four types of issues.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ISSUE DETECTION - _scan_ingest_issues()                  │
│                              (cli.py:2114)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                      For each .eml file in source_dir
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Lookup document in database                                         │
│                                                                             │
│ - Compute source_id from file path                                          │
│ - Query documents table for matching source_id                              │
│ - If not found → skip (not yet ingested)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Load document tree                                                  │
│                                                                             │
│ - Get root document (the email)                                             │
│ - Get all child documents (attachments)                                     │
│ - Prefetch chunks for performance                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Check each issue type                                               │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ CHECK: Missing Archive                                                  │ │
│ │ Condition: document.archive_browse_uri is NULL                          │ │
│ │ Applies to: Root document and child documents                           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ CHECK: Missing Line Numbers                                             │ │
│ │ Condition: Any chunk has line_from = NULL or line_to = NULL             │ │
│ │ Applies to: Root document and child documents (except images)           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ CHECK: Missing Context                                                  │ │
│ │ Condition: Any chunk has context_summary = NULL                         │ │
│ │ Applies to: Root document and child documents (except images)           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Record issues                                                       │
│                                                                             │
│ If any issues found → create IssueRecord with:                              │
│   - eml_path: Path to source file                                           │
│   - doc: Root document object                                               │
│   - child_docs: List of child documents                                     │
│   - issues: List of issue type strings                                      │
│   - cached_chunks: Prefetched chunks by document ID                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Issue Types:**

| Issue String | Document Type | Field Checked |
|--------------|---------------|---------------|
| `missing_archive` | Root email | `archive_browse_uri` |
| `missing_child_archive` | Attachment | `archive_browse_uri` |
| `missing_lines` | Root email | `chunk.line_from`, `chunk.line_to` |
| `missing_child_lines` | Attachment | `chunk.line_from`, `chunk.line_to` |
| `missing_context` | Root email | `chunk.context_summary` |
| `missing_child_context` | Attachment | `chunk.context_summary` |

---

## Orphan Detection Flow

Orphan detection runs before issue scanning to identify documents that should be removed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ORPHAN DETECTION - _find_orphaned_documents()              │
│                              (cli.py:2084)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Build set of existing source files                                  │
│                                                                             │
│ - Recursively find all .eml files in source_dir                             │
│ - Compute source_id for each (normalized relative path)                     │
│ - Store in set: existing_files                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Get all root documents from database                                │
│                                                                             │
│ - Query documents where depth = 0 (root documents only)                     │
│ - Returns dict: {source_id: doc_id}                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Find orphans                                                        │
│                                                                             │
│ For each (source_id, doc_id) in DB:                                         │
│   if source_id NOT IN existing_files:                                       │
│     → Add doc_id to orphan_ids list                                         │
│                                                                             │
│ Orphan = document in DB but source .eml file no longer exists               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Return orphan IDs for deletion                                      │
│                                                                             │
│ Returns: List[UUID] of root document IDs to delete                          │
│ Deletion will cascade to child documents and chunks                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fix Pipeline

Fixes execute in a specific order due to dependencies between issue types.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIX PIPELINE - _fix_document_issues()                    │
│                              (cli.py:2258)                                  │
│                                                                             │
│  CRITICAL: Fixes must run in this order due to dependencies                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: FIX ARCHIVES (must run first)                                       │
│         _fix_missing_archives() - cli.py:2296                               │
│                                                                             │
│ WHY FIRST: Chunks depend on archive .md content for re-chunking             │
│                                                                             │
│ Process:                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1. Check if archive exists in bucket but DB link is missing             │ │
│ │    └─ If found → Update DB link only (fast path)                        │ │
│ │                                                                         │ │
│ │ 2. If archive not in bucket:                                            │ │
│ │    └─ Parse email from source .eml file                                 │ │
│ │    └─ Generate archive using same function as regular ingest            │ │
│ │    └─ Upload to bucket                                                  │ │
│ │    └─ Update DB with browse URI                                         │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: FIX LINE NUMBERS (must run second)                                  │
│         _fix_missing_lines() - cli.py:2436                                  │
│                                                                             │
│ WHY SECOND: Context generation needs proper chunk structure                 │
│                                                                             │
│ Process:                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1. Read markdown content from archive                                   │ │
│ │                                                                         │ │
│ │ 2. Re-chunk using same chunker as regular ingest                        │ │
│ │    └─ Produces chunks with line_from, line_to                           │ │
│ │                                                                         │ │
│ │ 3. Generate embeddings for new chunks                                   │ │
│ │                                                                         │ │
│ │ 4. Atomic replace: delete old chunks + insert new in single transaction │ │
│ │    └─ Uses replace_chunks_atomic() for safety                           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ Note: Image attachments are skipped (single chunks without line tracking)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: FIX CONTEXT (runs last)                                             │
│         _fix_missing_context() - cli.py:2528                                │
│                                                                             │
│ WHY LAST: Needs proper chunks with correct structure                        │
│                                                                             │
│ Process:                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1. Get content from archive or chunk content                            │ │
│ │                                                                         │ │
│ │ 2. Generate context summary using LLM                                   │ │
│ │    └─ Uses same ContextGenerator as regular ingest                      │ │
│ │    └─ Includes retry logic for API failures                             │ │
│ │                                                                         │ │
│ │ 3. Update embedding_text with context prefix                            │ │
│ │                                                                         │ │
│ │ 4. Re-generate embeddings for affected chunks                           │ │
│ │                                                                         │ │
│ │ 5. Update chunks in database                                            │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ Note: Image attachments are skipped (use image descriptions, not context)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Fix Dependency Chain:**

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Archive    │────>│   Line Numbers   │────>│   Context    │
│              │     │                  │     │              │
│ Must exist   │     │ Needs archive    │     │ Needs proper │
│ for chunks   │     │ .md content      │     │ chunks       │
│ to be fixed  │     │ to re-chunk      │     │ to generate  │
└──────────────┘     └──────────────────┘     └──────────────┘
```

---

## Dry-Run vs Execute

The `--dry-run` flag controls whether fixes are applied.

```
                            ingest-update
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │ Phase 1: Scan Issues  │
                      │ (always runs)         │
                      └───────────────────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │ Phase 2: Report       │
                      │ (always runs)         │
                      └───────────────────────┘
                                  │
                                  ▼
                         ┌───────────────┐
                         │  --dry-run?   │
                         └───────────────┘
                          │           │
                     Yes  │           │  No
                          ▼           ▼
            ┌─────────────────┐  ┌─────────────────┐
            │  EXIT           │  │ Phase 3: Fix    │
            │                 │  │ Apply all fixes │
            │  Issues shown   │  │ Update database │
            │  No changes     │  │ Regenerate data │
            └─────────────────┘  └─────────────────┘
```

**Dry-run output example:**

```
Scanning: ./data/source
Found 150 .eml files
Scanning for issues... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

┌─────────────────────────────────────┐
│         Ingest-Update Summary       │
├─────────────────────────────────────┤
│ Orphaned documents:              3  │
│ Missing archives:               12  │
│ Missing line numbers:           45  │
│ Missing context:                23  │
├─────────────────────────────────────┤
│ Total documents with issues:    67  │
└─────────────────────────────────────┘

[DRY RUN] No changes made. Run without --dry-run to fix.
```

---

## Atomic Operations

The fix pipeline uses atomic operations to prevent data loss.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATOMIC CHUNK REPLACEMENT                                 │
│                                                                             │
│ Used by: _fix_missing_lines() when re-chunking documents                    │
│ Function: db.replace_chunks_atomic(doc_id, old_ids, new_chunks)             │
└─────────────────────────────────────────────────────────────────────────────┘

Normal update (dangerous):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   1. DELETE old chunks                                                      │
│   2. (if failure here) → DATA LOST! Old chunks gone, new not inserted       │
│   3. INSERT new chunks                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Atomic replace (safe):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   BEGIN TRANSACTION                                                         │
│     │                                                                       │
│     ├─ DELETE old chunks                                                    │
│     ├─ INSERT new chunks                                                    │
│     │                                                                       │
│     └─ If any step fails → ROLLBACK (old chunks preserved)                  │
│                                                                             │
│   COMMIT (only if all steps succeed)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Safety guarantees:**

| Operation | Failure Behavior | Data State |
|-----------|------------------|------------|
| Orphan deletion | Cascade delete fails → rollback | Orphan retained |
| Archive regeneration | Upload fails → no DB update | Old state preserved |
| Line number fix | Insert fails → rollback | Old chunks preserved |
| Context fix | LLM fails → skip with retry | Chunks unchanged |

**Retry Logic:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLM and Embedding API calls include automatic retry                         │
│                                                                             │
│ Max retries: 3                                                              │
│ Backoff: 1s → 2s → 4s                                                       │
│ Retried errors: Rate limits, timeouts, transient failures                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing

The ingest-update flow is covered by automated tests in `tests/test_ingest_update_flow.py`.

### Running Tests

```bash
# Run all ingest-update tests
pytest tests/test_ingest_update_flow.py -v

# Run with coverage
pytest tests/test_ingest_update_flow.py --cov=mtss.cli --cov-report=term-missing
```

### Test Coverage

| Component | Test Class |
|-----------|------------|
| Orphan detection | `TestFindOrphanedDocuments` |
| Issue scanning | `TestScanIngestIssues` |
| Archive fixes | `TestFixMissingArchives` |
| Line number fixes | `TestFixMissingLines` |
| Context fixes | `TestFixMissingContext` |
| Fix orchestration | `TestFixDocumentIssues` |
| Dry-run mode | `TestDryRunMode` |
| Security/errors | `TestSecurityAndEdgeCases` |

All tests run without external dependencies (Supabase, OpenAI, etc.) using mocks.
