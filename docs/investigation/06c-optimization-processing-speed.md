---
title: "Optimization: Processing Speed"
category: investigation
status: proposal
date: 2026-04-13
scope: ingest pipeline
---

# Processing Speed Optimization Proposals

Analysis of the ingest pipeline for 6,289 EML files (6 GB) with attachments, identifying bottlenecks and proposing concrete speedups.

## Current Architecture Summary

### Processing flow per email (sequential)

```
1. Hash check + version check         (DB query, ~50ms)
2. Parse EML file                      (local I/O, ~10ms)
3. Generate archive (extract files)    (local I/O + Supabase upload, ~200-500ms)
4. Create email document in DB         (DB insert, ~50ms)
5. Vessel matching                     (local, ~1ms)
6. Context summary generation          (LLM API call, ~500-1500ms)
7. Topic extraction                    (LLM API call, ~500-1500ms)
8. Topic matching (per topic)          (Embedding API + DB, ~200ms each)
9. Split body into message chunks      (local, ~5ms)
10. For each attachment (sequential):
    a. Preprocess/classify             (Vision API for images, ~1000ms)
    b. Check cache for parsed content  (Supabase Storage, ~100ms)
    c. Parse with LlamaParse           (external API, ~5-30s per document)
    d. Generate context summary        (LLM API call, ~500-1500ms)
    e. Chunk text                      (local, ~5ms)
    f. Update archive markdown         (Supabase Storage upload, ~100ms)
11. Generate embeddings (all chunks)   (OpenAI API, ~200-500ms per batch)
12. Insert chunks to DB                (asyncpg batch, ~50-100ms)
13. Mark completed + cleanup           (DB update + local I/O, ~50ms)
```

### Concurrency model

- `MAX_CONCURRENT_FILES=5`: Worker pool with fast/slow lane queues (60/40 split)
- `MAX_CONCURRENT_EMBEDDINGS=5`: Not currently used as a semaphore (batches run sequentially within each email)
- `EMBEDDING_BATCH_SIZE=100`: Texts per OpenAI API call
- `BATCH_SIZE=10`: Not used in the hot path (legacy setting)
- asyncpg pool: `min_size=2, max_size=10`

### Bottleneck classification

| Category | Operations | Est. time per email |
|----------|-----------|-------------------|
| External API (slow) | LlamaParse parsing | 5-30s per document |
| External API (medium) | Vision classify/describe, LLM context, LLM topics | 0.5-1.5s each |
| External API (fast) | Embeddings, topic embedding match | 0.2-0.5s per batch |
| Network I/O | Supabase Storage upload, DB queries via REST | 50-200ms each |
| Local compute | EML parsing, chunking, vessel matching, hashing | <50ms total |

---

## Proposal 1: Parallelize attachments within a single email

### Current behavior

Attachments are processed sequentially in a `for` loop (pipeline.py line 342-376). An email with 5 PDF attachments waits 25-150 seconds serially.

### Proposed change

Use `asyncio.gather()` or a semaphore-bounded gather to process attachments concurrently within each email. Since attachments are independent (they share the parent `email_doc` read-only, and each creates its own `attach_doc`), they can safely run in parallel.

```
# Instead of:
for attachment in parsed_email.attachments:
    chunks = await process_attachment(...)

# Do:
sem = asyncio.Semaphore(3)  # limit per-email concurrency
async def process_one(att):
    async with sem:
        return await process_attachment(att, ...)
results = await asyncio.gather(*[process_one(a) for a in parsed_email.attachments])
```

### Analysis

- **Expected speedup:** 40-60% for emails with multiple document attachments (the most time-consuming category). A 5-attachment email goes from ~75s to ~25-30s.
- **Bottleneck addressed:** LlamaParse serialization within a single email.
- **Risk:** Low. Attachments are independent. The only shared state is `email_doc` (read-only at this point) and `components` (all thread-safe). LlamaParse API may rate-limit if too many parallel calls -- the semaphore mitigates this.
- **Complexity:** Low. ~15 lines changed in `pipeline.py`. Need to handle the attachment index for progress reporting.

---

## Proposal 2: Parallelize context and topic generation

### Current behavior

In `pipeline.py`, context generation (line 199-207) runs first, then topic extraction (line 213-276) runs after it. These are independent LLM calls that could overlap.

### Proposed change

Run context summary and topic extraction concurrently using `asyncio.gather()`:

```python
context_task = generate_context(...)
topic_task = extract_topics(...)
context_summary, extracted_topics = await asyncio.gather(context_task, topic_task)
```

**Complication:** The current code feeds `context_summary` into the topic extraction input (line 264: `topic_input_parts.append(f"Summary: {context_summary}")`). Two options:

- **Option A:** Drop context_summary from topic input. The subject + original message + body content already provide strong signal. The context summary adds marginal value to topic extraction.
- **Option B:** Keep sequential but pipeline with other work (start attachment processing while topics run).

### Analysis

- **Expected speedup:** 10-15% overall. Saves ~0.5-1.5s per email by overlapping two LLM calls.
- **Bottleneck addressed:** Sequential LLM API calls within email body processing.
- **Risk:** Low for Option A (topics may be very slightly less accurate without context summary in the prompt, but subject + body is the primary signal). Medium for Option B (more complex orchestration).
- **Complexity:** Low for Option A (~10 lines). Medium for Option B (requires restructuring the pipeline flow).

---

## Proposal 3: Pipeline body processing with attachment processing

### Current behavior

The pipeline is fully sequential: body chunks are created (steps 5-9), then attachments are processed (step 10), then ALL chunks are embedded together (step 11). This means attachment parsing cannot start until body processing (including two LLM calls) completes.

### Proposed change

Start attachment processing in parallel with context/topic generation, since attachments don't depend on context or topics (those get added to chunk metadata later):

```python
# Start attachment processing immediately after email parsing
attachment_task = asyncio.create_task(process_all_attachments(...))

# Meanwhile, do context generation and topic extraction
context_summary = await generate_context(...)
topic_ids = await extract_topics(...)

# Build body chunks with context/topics
body_chunks = build_body_chunks(context_summary, topic_ids)

# Wait for attachments, then enrich their chunks with vessel/topic metadata
attachment_chunks = await attachment_task
enrich_chunks(attachment_chunks, vessel_ids, topic_ids, context_summary)

# Embed and store all chunks
all_chunks = body_chunks + attachment_chunks
```

### Analysis

- **Expected speedup:** 20-30% for emails with attachments. LlamaParse parsing (5-30s) runs concurrently with LLM context+topics (~2-3s), so the LLM time is completely hidden.
- **Bottleneck addressed:** Sequential dependency between body LLM processing and attachment parsing.
- **Risk:** Medium. Requires restructuring pipeline.py significantly. Attachment context summaries are currently generated inside `attachment_handler.py` after parsing -- these would need to remain sequential within each attachment but can run in parallel with body processing. Topic/vessel metadata must be applied to attachment chunks after the gather.
- **Complexity:** Medium. ~50-80 lines of restructuring.

---

## Proposal 4: Increase MAX_CONCURRENT_FILES from 5 to 8-10

### Current behavior

Only 5 emails are processed concurrently. Given that the primary bottleneck is external API latency (not local CPU or memory), the machine is likely underutilized.

### Proposed change

Increase `MAX_CONCURRENT_FILES` to 8-10 for local machines with good network. The fast/slow lane split already prevents LlamaParse overload (60% fast workers handle image-only emails quickly).

### Analysis

- **Expected speedup:** 30-50% throughput increase for fast-lane emails (which are I/O-bound on LLM APIs). Slow-lane improvement depends on LlamaParse rate limits.
- **Bottleneck addressed:** Under-utilization of async I/O capacity.
- **Risk:** Medium. LlamaParse has rate limits (varies by plan). OpenAI embedding/LLM APIs have generous limits but can 429. The existing retry-with-backoff logic handles this. Memory usage increases linearly (~50-100MB per concurrent email for attachment data).
- **Complexity:** Trivial. Change one config value and test. Consider making it dynamic based on lane queue sizes.

### Recommendation

Test with 8 first, monitor LlamaParse 429 rates. The retry logic is already robust.

---

## Proposal 5: Progressive embedding -- embed body chunks while attachments parse

### Current behavior

All chunks (body + all attachments) are collected, then embedded in one batch at the end (pipeline.py line 398-401). This means embedding cannot start until the slowest attachment finishes parsing.

### Proposed change

Embed and store body chunks immediately after they are ready, without waiting for attachments. Then embed attachment chunks as each attachment completes:

```python
# Embed body chunks immediately
if body_chunks:
    body_chunks = await embeddings.embed_chunks(body_chunks)
    await db.insert_chunks(body_chunks)

# Process attachments, embed each as it completes
for attachment in attachments:
    att_chunks = await process_attachment(...)
    if att_chunks:
        att_chunks = await embeddings.embed_chunks(att_chunks)
        await db.insert_chunks(att_chunks)
```

Or with parallel attachments (Proposal 1), embed in a callback after each attachment.

### Analysis

- **Expected speedup:** 5-10% overall. Body chunks (usually 1-5) are small and embed quickly. The real value is reducing peak memory (no need to hold all chunks in memory) and enabling earlier searchability.
- **Bottleneck addressed:** Waiting for all attachments before starting any embedding.
- **Risk:** Medium. If the email fails mid-processing, some chunks are already in the DB but the document is not marked COMPLETED. The existing cleanup logic handles this (partial documents are cleaned up on retry via `delete_document_for_reprocess`). However, this increases DB write frequency.
- **Complexity:** Medium. Requires restructuring the chunk collection pattern and ensuring the document status flow still works.

---

## Proposal 6: Batch Supabase Storage uploads with async concurrency

### Current behavior

Archive generation uploads files to Supabase Storage one at a time. For an email with 5 attachments, that is 5+ sequential HTTP uploads (original files + markdown previews).

### Proposed change

Use `asyncio.gather()` for parallel Storage uploads. The archive generator currently uses synchronous Supabase client calls -- wrap them in `anyio.to_thread.run_sync()` or switch to async HTTP.

### Analysis

- **Expected speedup:** 5-10% for emails with many attachments. Each upload is ~100-200ms, so 10 uploads save ~1-1.5s.
- **Bottleneck addressed:** Sequential Supabase Storage uploads.
- **Risk:** Low. Uploads are independent. Supabase Storage can handle concurrent uploads.
- **Complexity:** Medium. The `ArchiveStorage` class uses synchronous `storage3` client. Would need async wrappers or a different client.

---

## Proposal 7: Cache and batch topic matching embeddings

### Current behavior

For each extracted topic (1-5 per email), `TopicMatcher.get_or_create_topic()` calls `embeddings.generate_embedding()` individually, then queries the DB for similar topics. That is 1-5 separate embedding API calls per email just for topics.

### Proposed change

Batch all topic name embeddings into a single API call:

```python
topic_names = [t.name for t in extracted_topics]
embeddings = await self.embeddings.generate_embeddings_batch(topic_names)
for topic, embedding in zip(extracted_topics, embeddings):
    topic_id = await match_or_create_with_embedding(topic, embedding)
```

Additionally, maintain an in-memory cache of all known topic embeddings (there will only be ~50-200 topics total) to skip the DB similarity query for already-known topics.

### Analysis

- **Expected speedup:** 3-5% overall. Saves ~200ms per email (from 5 x 200ms to 1 x 200ms for topic embeddings).
- **Bottleneck addressed:** Per-topic embedding API calls.
- **Risk:** Low. The topic matcher already has a name cache (`_name_cache`). Extending it with embeddings is straightforward.
- **Complexity:** Low. ~20 lines in `TopicMatcher`, add batch embedding support.

---

## Proposal 8: Local preprocessing before API calls

### Current behavior

Every image goes through the Vision API for classification (logo/banner/signature/meaningful). Even small images (< 5KB, likely icons) and images with dimensions that clearly indicate logos (e.g., 200x50px) get sent to the API.

### Proposed change

Add local heuristics before API calls:

1. **Size filter:** Images < 5KB are almost always icons/logos. Skip API call.
2. **Dimension filter:** Images with extreme aspect ratios (> 5:1) or very small dimensions (< 100x100) are likely logos/icons. Skip API call.
3. **Image hash deduplication:** The same logo appears in hundreds of emails. Hash the image content and cache classification results in memory or a local SQLite file.

The estimator already has some of this logic (`images_meaningful` vs `images_skipped` based on heuristics). Bring those heuristics into the live pipeline.

### Analysis

- **Expected speedup:** 5-15% for emails with images. Roughly 50-70% of email images are logos/banners based on typical maritime email patterns. At ~1s per Vision API call, skipping 3-4 images per email saves 3-4s.
- **Bottleneck addressed:** Unnecessary Vision API calls for obvious non-content images.
- **Risk:** Low. False positives (skipping a meaningful small image) are possible but unlikely for maritime incident reports. The existing estimator heuristics are already proven.
- **Complexity:** Low. ~30 lines in `preprocessor.py`. The `ImageProcessor` already has an `is_supported()` method; add a `should_skip_locally()` check before calling classify.

---

## Proposal 9: Optimize resumability check overhead

### Current behavior

For each file, `process_email()` performs up to 4 DB queries before starting:

1. `version_manager.check_document()` -- queries by `source_id` (line 90-91)
2. `get_document_by_id()` if existing_doc_id found (line 95)
3. `compute_doc_id()` + `get_document_by_doc_id()` safety check (line 136-137)
4. `tracker.mark_started()` -- upsert to processing_log (lines 120-146)

That is 3-4 synchronous Supabase REST API calls per file just for skip-checking.

### Proposed change

**Batch pre-fetch:** Before the worker loop starts, load all known `source_id -> doc_id` mappings and `processing_log` status into memory in one or two bulk queries. Then the per-file check becomes an in-memory dict lookup.

```python
# Before processing loop:
known_docs = await db.get_all_root_source_ids()  # Already exists!
processed_hashes = await tracker._get_processed_hashes()  # Already exists!

# Per file: O(1) dict lookup instead of 3 HTTP requests
```

The existing `get_pending_files()` already does this bulk approach for the initial file list, but `process_email()` redundantly checks again per file. This double-checking is a safety net but could be optimized by passing the pre-fetched data.

### Analysis

- **Expected speedup:** 2-5% overall. Saves ~150-300ms per file on skip-check overhead. For 6,289 files, that is 15-30 minutes saved.
- **Bottleneck addressed:** Redundant DB queries for resumability checking.
- **Risk:** Low. The pre-fetched data could become stale during a long ingest run (another process inserts documents), but this is a single-writer scenario. The safety check at line 136 (orphan cleanup) catches edge cases.
- **Complexity:** Low-Medium. Pass pre-fetched data to `process_email()` and skip the per-file queries when data is available. ~20-30 lines.

---

## Proposal 10: Use asyncpg for all DB operations (replace Supabase REST client)

### Current behavior

The codebase uses two database access patterns:
- **Supabase REST client** (synchronous, via `self.client.table(...).select/insert/update`): Used for documents, processing_log, ingest_events, topic queries
- **asyncpg pool** (async, via `pool.acquire()`): Used only for chunk inserts and vector search

The REST client adds ~50-100ms of HTTP overhead per call vs ~5-10ms for asyncpg direct queries. Many operations wrap sync REST calls in `anyio.to_thread.run_sync()`, adding thread context-switch overhead.

### Proposed change

Migrate all hot-path DB operations to asyncpg:
- `insert_document()` -- currently REST, called once per email + once per attachment
- `update_document_status()` -- called 2-3x per email
- `mark_started()` / `mark_completed()` -- processing_log upserts
- `get_document_by_hash()` / `get_document_by_doc_id()` -- skip checks

Keep REST client only for admin/maintenance operations where latency does not matter.

### Analysis

- **Expected speedup:** 5-10% overall. Each email makes ~5-10 DB calls on the hot path. Saving ~50ms per call = 250-500ms per email.
- **Bottleneck addressed:** HTTP overhead for database operations on the critical path.
- **Risk:** Medium. Requires writing raw SQL for operations currently handled by the Supabase SDK's query builder. More code to maintain. Need to ensure transaction safety.
- **Complexity:** High. Each repository method needs an asyncpg equivalent. ~200-300 lines of SQL. Could be done incrementally (migrate hottest paths first).

---

## Priority ranking

| Priority | Proposal | Speedup | Complexity | Risk |
|----------|----------|---------|------------|------|
| 1 | P1: Parallelize attachments within email | 40-60% | Low | Low |
| 2 | P4: Increase MAX_CONCURRENT_FILES to 8-10 | 30-50% | Trivial | Medium |
| 3 | P3: Pipeline body + attachment processing | 20-30% | Medium | Medium |
| 4 | P2: Parallelize context + topic generation | 10-15% | Low | Low |
| 5 | P8: Local image preprocessing heuristics | 5-15% | Low | Low |
| 6 | P9: Batch pre-fetch resumability data | 2-5% | Low-Med | Low |
| 7 | P7: Batch topic matching embeddings | 3-5% | Low | Low |
| 8 | P10: Migrate hot-path DB to asyncpg | 5-10% | High | Medium |
| 9 | P5: Progressive embedding | 5-10% | Medium | Medium |
| 10 | P6: Batch storage uploads | 5-10% | Medium | Low |

## Recommended implementation order

**Phase 1 (quick wins, 1-2 days):** P1 + P4 + P2
- Parallelize attachments, bump concurrency, overlap context/topics
- Expected combined speedup: 50-70%
- Low risk, low complexity

**Phase 2 (medium effort, 2-3 days):** P8 + P7 + P9
- Local image filtering, batch topic embeddings, pre-fetch skip data
- Expected additional speedup: 10-20%
- Low risk

**Phase 3 (larger refactors, 3-5 days):** P3 + P5
- Pipeline restructuring for overlapped body/attachment processing
- Expected additional speedup: 15-25%
- Medium risk, requires careful testing

**Phase 4 (long-term, 1-2 weeks):** P10 + P6
- Database layer migration, async storage uploads
- Expected additional speedup: 10-15%
- Higher complexity but improves overall architecture

## Estimated total processing time

### Current (conservative estimate)

- ~6,289 files at ~30s average per file (including skips) with 5 concurrent workers
- Effective throughput: ~5 files every 30s = 10 files/min
- Total: ~10.5 hours

### After Phase 1

- Average per-file drops to ~15-20s, concurrency rises to 8-10
- Effective throughput: ~25-35 files/min
- Total: ~3-4 hours

### After all phases

- Average per-file drops to ~10-15s, optimized concurrency
- Effective throughput: ~40-60 files/min
- Total: ~1.5-2.5 hours
