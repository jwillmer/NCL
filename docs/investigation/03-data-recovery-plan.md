---
purpose: Recovery strategy for production data after Supabase Storage auto-clear
status: investigation-complete
date: 2026-04-13
---

# Data Recovery Plan

## 1. Current Data Inventory

### Local data (intact)

| Location | Contents | Size | Count |
|---|---|---|---|
| `data/emails/` | Raw EML files | ~6 GB | 6,289 |
| `data/processed/attachments/` | Extracted attachments per doc_id folder | 256 MB | 117 folders |
| `data/processed/estimate/` | Cached ingest estimate data | 432 MB | 243 entries |
| `data/reports/` | Ingest failure reports (JSON + CSV) | small | 21 pairs |

The `data/processed/attachments/` folders are keyed by `folder_id` (first 16 chars of `doc_id`), each containing extracted attachment files (PDFs, images, Office docs). These are the raw attachment files that were saved during EML parsing, not the parsed/converted markdown content. No `.md` cache files exist locally -- those lived only in Supabase Storage.

The `data/source/` directory (configured as `DATA_SOURCE_DIR`) does not exist. EML files are at `data/emails/`, so the `.env` variable `DATA_SOURCE_DIR` needs to be set to `./data/emails` for ingest to find them.

### Cloud data (status uncertain)

| Location | Contents | Status |
|---|---|---|
| Supabase PostgreSQL: `documents` | Document hierarchy (email + attachment records) | **Unknown -- needs probing** |
| Supabase PostgreSQL: `chunks` | Embedding vectors, content, citation metadata | **Unknown -- needs probing** |
| Supabase PostgreSQL: `processing_log` | File-level processing state (completed/failed/processing) | **Unknown -- needs probing** |
| Supabase PostgreSQL: `topics` | Extracted topics with embeddings | **Unknown -- needs probing** |
| Supabase PostgreSQL: `vessels` | Vessel registry | **Unknown -- needs probing** |
| Supabase PostgreSQL: `ingest_events` | Processing event log | **Unknown -- needs probing** |
| Supabase Storage: `archive` bucket | Browsable markdown + original files | **Auto-cleared (confirmed)** |

### What was in the archive bucket

For each processed email, the archive bucket contained a folder `{doc_id[:16]}/` with:

- `email.eml` -- original EML file (copy)
- `email.eml.md` -- markdown rendering of the email conversation
- `metadata.json` -- programmatic access metadata
- `attachments/{filename}` -- original attachment files
- `attachments/{filename}.md` -- parsed content as markdown (the LlamaParse cache)

This is the **only location** where parsed attachment `.md` files existed. The local `data/processed/attachments/` folders contain only the raw attachment binaries, not the parsed markdown.

## 2. Impact Assessment

### What is lost

1. **Archive bucket (confirmed lost):** All browsable markdown files, all cached LlamaParse output (`.md` files for attachments), metadata JSON files, and copies of original files in the bucket.

2. **LlamaParse cache (critical):** The attachment `.md` files in the archive bucket served as the LlamaParse parse cache. During re-ingest, the pipeline checks `{folder_id}/attachments/{filename}.md` in the bucket before calling LlamaParse (see `attachment_handler.py` lines 170-188). With the bucket empty, every document-type attachment (PDFs, Office files) will need to be re-parsed through LlamaParse, incurring cost and time.

### What can be recovered without re-processing

- **If DB is intact:** All document records, chunks (with embeddings), topics, vessels, processing_log -- the entire RAG search pipeline would work. Only archive links (`archive_browse_uri`, `archive_download_uri`) would be broken (pointing to non-existent bucket files). Search and retrieval would still function; only "View source" / "Download original" links in citations would 404.

- **If DB is partially intact:** Depends on what's left. The pipeline can detect and skip already-completed documents. Partial documents (status != COMPLETED) get cleaned up and retried.

### What requires re-processing

- The archive bucket must be fully rebuilt regardless of DB state.
- If DB is empty, everything must be re-ingested from scratch.
- LlamaParse must re-parse all document attachments (no local cache exists).

## 3. Pipeline Resume/Reprocess Capabilities

### How skip detection works (three layers)

1. **VersionManager (primary, `version_manager.py`):** Computes `doc_id = hash(source_id + file_hash)` and queries `documents` table by `doc_id`. If found with `ingest_version >= current_version`, returns `action="skip"`. If found with lower version, returns `action="reprocess"`. If not found but `source_id` matches an existing doc, returns `action="update"`. Otherwise returns `action="insert"`.

2. **Legacy hash check (fallback):** If no VersionManager, queries `documents` by `file_hash`. Skips if status is COMPLETED.

3. **Orphan safety check:** After both checks, queries by computed `target_doc_id`. If found and COMPLETED, skips. If found and not COMPLETED, cleans up for retry.

4. **ProgressTracker (`progress_tracker.py`):** Filters pending files by comparing file hashes against `processing_log` table entries with status COMPLETED or PROCESSING.

### Handling of partial/incomplete data

- Documents with `status != COMPLETED` are detected during skip checks and cleaned up via `delete_document_for_reprocess()`, which deletes the document (CASCADE deletes children and chunks) and ingest_events.
- The `get_pending_files()` method only considers files as processed if their hash appears in `processing_log` with status COMPLETED or PROCESSING.
- There is no built-in mechanism to detect "document exists in DB but archive bucket is missing." The `ingest-update` command checks for missing archives but only within `archive_browse_uri` fields.

### Reprocess capabilities

| Command | What it does |
|---|---|
| `MTSS ingest` | Default resume mode: skips files whose hash is in `processing_log` as completed/processing |
| `MTSS ingest --no-resume` | Processes all EML files but still skips via VersionManager doc_id lookup in `documents` table |
| `MTSS ingest --retry-failed` | Retries files with `status=failed` in `processing_log` (< 3 attempts) |
| `MTSS ingest --reprocess-outdated` | Finds files with `ingest_version < current_version` and reprocesses them |
| `MTSS ingest-update` | Scans for orphans, missing archives, missing line numbers, missing context, missing topics |
| `MTSS ingest-update --dry-run` | Shows issues without fixing |
| `MTSS reprocess` | Shows documents needing reprocess but **not yet implemented** (prints "use clean and re-ingest") |
| `MTSS clean` | Deletes ALL data (DB tables + local processed files + storage bucket) |
| `MTSS reset-failures <file>` | Deletes specific failed documents from DB for retry |
| `MTSS reset-stale` | Resets stuck "processing" entries to "failed" |

## 4. Recovery Strategy

### Decision tree

```
Is the Supabase PostgreSQL database intact?
|
+-- YES (all tables have data)
|   |
|   Is data complete? (documents.status all COMPLETED, chunks exist)
|   |
|   +-- YES -> Scenario A: Archive-Only Rebuild
|   +-- PARTIAL -> Scenario B: Incremental Recovery
|
+-- NO (empty or missing tables)
    |
    +-> Scenario C: Full Re-ingest
```

### Scenario A: Archive-Only Rebuild (DB intact, bucket empty)

The DB has all documents, chunks, embeddings, topics, and vessels. Only the archive bucket is missing.

**What works:** RAG search, query answering, topic filtering, vessel filtering -- all functional.

**What is broken:** Citation "View" and "Download" links return 404.

**Recovery steps:**

1. Verify DB state: `MTSS stats` to see processing counts.
2. Fix `DATA_SOURCE_DIR` in `.env` to point to `./data/emails`.
3. Run `MTSS ingest-update --dry-run` to see what issues exist (will detect missing archives).
4. Run `MTSS ingest-update` to regenerate archives.
   - For email archives: re-parses EML files and uploads `email.eml`, `email.eml.md`, `metadata.json` to bucket.
   - For attachment archives: uses existing chunk content from DB to reconstruct `.md` files, re-uploads original attachments from parsed email.
   - This does NOT re-run LlamaParse -- it uses existing chunk content stored in the DB.
5. Run `MTSS ingest-update` a second time to verify zero issues remain.

**Cost:** Negligible (no LLM/embedding calls; only Supabase Storage uploads).

**Risk:** Low. The repair logic in `repair.py` > `fix_missing_archives()` handles both cases:
- If attachment file is found in parsed email and on disk: uploads original + regenerates `.md` from DB chunks.
- If attachment file is NOT found in parsed email: deletes the entire email document for clean re-ingest.

### Scenario B: Incremental Recovery (DB partial, bucket empty)

Some documents are in the DB, others are missing. Some may have incomplete status.

**Recovery steps:**

1. Fix `DATA_SOURCE_DIR` in `.env` to point to `./data/emails`.
2. Run `MTSS stats` to see current state.
3. Run `MTSS reset-stale` to clear any stuck "processing" entries.
4. Run `MTSS ingest --retry-failed` to retry any failed files.
5. Run `MTSS ingest` (default resume mode) to process missing files.
   - ProgressTracker will identify files not in `processing_log` as pending.
   - VersionManager will verify against `documents` table.
   - Already-completed documents are skipped.
   - New/missing documents are fully processed (including LlamaParse for attachments).
6. Run `MTSS ingest-update` to fix missing archives on previously-completed documents.

**Cost:** LlamaParse charges for any new document attachments that need parsing. Embedding costs for new chunks. Context/topic LLM calls for new documents.

**Risk:** Medium. The pipeline handles partial data well for individual documents (clean up and retry), but if there are thousands of partially-ingested documents, the cleanup phase could be slow due to per-document DB queries.

### Scenario C: Full Re-ingest (DB empty)

Database has been wiped or is inaccessible. Start from scratch.

**Recovery steps:**

1. Fix `DATA_SOURCE_DIR` in `.env` to point to `./data/emails`.
2. Clean local processed files: delete `data/processed/attachments/` contents (stale from previous run).
3. Verify vessel registry: `MTSS entities load-vessels` to reload vessel CSV.
4. Run `MTSS estimate` to get cost estimate before committing.
5. Run `MTSS ingest --no-resume` to process all 6,289 EML files.
6. Monitor with `MTSS stats` periodically.
7. After completion, run `MTSS ingest --retry-failed` for any failures.
8. Run `MTSS ingest-update --dry-run` to check for any remaining issues.

**Cost:** Full LlamaParse cost for all document attachments. Previous estimate run had 243 entries, suggesting significant attachment count. Full embedding cost for all chunks. Full LLM cost for context summaries and topic extraction.

**Risk:** High cost (use `MTSS estimate` first). The `--lenient` flag is recommended to prevent a single attachment failure from blocking its parent email. Long runtime (previous ingest reports span Jan 15-26, 2026).

## 5. LlamaParse Cache Considerations

### Can we reuse local `data/processed/attachments/`?

**No, not directly.** The local attachment folders contain raw binary files (PDFs, images, Office docs), not parsed markdown. The parsed markdown was only stored in the Supabase Storage bucket as `.md` files. With the bucket cleared, these cache files are gone.

The local attachment files ARE useful during re-ingest because:
- The EML parser extracts attachments to `data/processed/attachments/{folder_id}/` during parsing.
- These files are the input to LlamaParse/vision processing.
- However, each ingest run re-extracts from the EML file anyway, so the cached extraction saves minimal time.
- After successful processing, the pipeline deletes the attachment folder (`pipeline.py` lines 418-424).

### LlamaParse cost mitigation

If budget is a concern for Scenario C:
- LlamaParse caches results on their side for 48 hours. If the previous run was recent, some results may be cached. However, the last ingest reports are from January 2026 (3 months ago), so this is not applicable.
- Process in batches using `--batch-size` to control throughput.
- Use `--lenient` to avoid failures blocking progress.

## 6. Processing Order and Dependencies

The pipeline processes each EML file atomically:

1. Parse EML -> extract body + attachments
2. Generate archive folder in bucket (upload EML, create structure)
3. Create email document in DB hierarchy
4. Match vessels in email content
5. Create body text chunks
6. Generate context summary (LLM call)
7. Extract topics (LLM call)
8. Process each attachment:
   a. Check for cached `.md` in bucket (will miss, since bucket is empty)
   b. Parse via LlamaParse / vision / text processor
   c. Create attachment document in DB (child of email doc)
   d. Create attachment chunks
   e. Upload `.md` to bucket
9. Generate email markdown with [View] links
10. Embed all chunks (email body + attachments)
11. Insert chunks to DB
12. Update topic counts
13. Mark email document as COMPLETED
14. Clean up local attachment folder

Attachment documents are children of their email document (via `parent_id` and `root_id`). Deleting a parent via `delete_document_for_reprocess` cascades to all children and their chunks.

## 7. Risks and Fallback Options

| Risk | Mitigation |
|---|---|
| DB is partially corrupted (some tables intact, others not) | Run `MTSS stats` first; if it errors, check individual table access via Supabase dashboard |
| LlamaParse API cost for full re-ingest | Run `MTSS estimate` first; consider processing in batches over multiple days |
| Interrupted re-ingest | Pipeline is resumable by default; use `MTSS ingest` to continue where it left off |
| `ingest-update` fails on some documents | It processes each document independently; failures are logged but don't block others |
| Attachment files referenced in DB but missing from bucket | `ingest-update` handles this: uploads original from parsed email + regenerates `.md` from chunk content |
| Attachment not found in parsed email during repair | `repair.py` deletes the entire email document for clean re-ingest (line 362) |
| DATA_SOURCE_DIR misconfigured | Currently set to `./data/source` which does not exist; must be changed to `./data/emails` |
| Topic counts become inconsistent after partial recovery | `ingest-update` can fix missing topics; for count accuracy, a full topic recount may be needed |
| Processing_log has stale "processing" entries | `MTSS reset-stale` clears entries older than 60 minutes |

### Fallback: Nuclear option

If recovery is too complex or data integrity is questionable:

1. `MTSS clean --yes` (deletes everything from DB, local processed files, and storage bucket)
2. `MTSS entities load-vessels` (reload vessel registry)
3. `MTSS ingest --lenient` (full re-ingest from EML files)

This is the safest path but most expensive in terms of LlamaParse and LLM costs.

## 8. Recommended First Steps

1. **Probe DB state:** Run `MTSS stats` to check if `processing_log` table is accessible and has data.
2. **Fix source dir:** Update `.env` to set `DATA_SOURCE_DIR=./data/emails`.
3. **Decide scenario:** Based on DB probe results, follow Scenario A, B, or C above.
4. **Estimate costs:** If Scenario B or C, run `MTSS estimate` to understand LlamaParse costs.
5. **Execute recovery:** Follow the step-by-step procedure for the chosen scenario.
