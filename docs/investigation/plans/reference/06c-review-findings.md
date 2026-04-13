---
purpose: Review findings for 06c (Processing Speed Optimization)
status: complete
date: 2026-04-13
source: 06c-optimization-processing-speed.md
---

# 06c Review Findings: Processing Speed Optimization

## Critical Finding: Bottleneck Profile Shift

The 06c proposals were written assuming the current architecture where LlamaParse is the
dominant time bottleneck (5-30s per document). With the approved optimization plan, the
bottleneck profile changes dramatically:

| Bottleneck | Current pipeline | After optimization |
|------------|------------------|--------------------|
| PDF parsing (simple) | LlamaParse: 5-30s | Local PyMuPDF4LLM: ~0.01s |
| PDF parsing (complex) | LlamaParse: 5-30s | GPT-4.1-mini batch: async, ~2-5s |
| DOCX/XLSX parsing | LlamaParse: 5-15s | Local python-docx/openpyxl: ~0.01s |
| Image classification | Vision API: ~1s | Local heuristic skip 52% + GPT-4.1-nano: ~0.5s |
| Context summary | LLM API: 0.5-1.5s | LLM API: 0.5-1.5s (unchanged) |
| Topic extraction | LLM API: 0.5-1.5s | LLM API: 0.5-1.5s (unchanged) |
| Topic embedding match | Embedding API: ~0.2s each | Local cosine sim: <1ms (local-only) |
| Embeddings | Embedding API: 0.2-0.5s/batch | Embedding API: 0.2-0.5s/batch (unchanged) |
| DB operations | Supabase REST: 50-200ms each | Local JSONL: <1ms (local-only) |

**New dominant bottleneck after optimization: LLM API calls (context + topics) at 1-3s
per email, and the Embedding API at 0.2-0.5s per batch. Everything else becomes near-instant
for local-only ingest.**

This means the speedup estimates in 06c are inflated. The actual per-email time drops from
~30s average to ~3-5s average just from the parser and storage changes alone, before any
concurrency optimization.

---

## Proposal-by-Proposal Review

### P1: Parallelize attachments within email -- APPROVED (reduced scope)

**Feasibility confirmed.** Code review of `pipeline.py` (lines 342-376) and
`attachment_handler.py` confirms:

- Attachments are processed in a sequential `for` loop
- Each attachment creates its own `attach_doc` via `hierarchy_manager.create_attachment_document()`
- The `email_doc` is read-only at this point (created at line 173, not modified after)
- `components` is shared but all methods called are stateless or handle their own state
- `HierarchyManager.create_attachment_document()` creates a new `Document` per call with
  unique IDs derived from `parent_doc.source_id + filename` -- no collision risk
- The only shared mutable state is `email_chunks` (the list being extended), but with
  `asyncio.gather()` each coroutine returns its own chunk list, concatenated after gather

**Revised speedup estimate:** With local parsers replacing LlamaParse, most attachments
process in <100ms (local PDF/DOCX/XLSX parsing). The remaining slow attachments are:
- Complex/scanned PDFs (still needs cloud API, ~2-5s)
- Images needing Vision API description (~0.5-1s)
- Attachment context summary generation (LLM API, ~0.5-1.5s)

For an email with 5 attachments where 2 need cloud parsing, parallelization saves
~2-4s instead of the claimed 45-50s. Still worthwhile but much less dramatic than 06c's
"40-60%" claim.

**Decision:** Implement but with lower priority than 06c suggests. The per-email
semaphore of 3 concurrent attachments is sensible to avoid API rate limits.

### P4: Increase MAX_CONCURRENT_FILES to 8-10 -- APPROVED

**Feasibility confirmed.** `config.py` line 134 defines `max_concurrent_files = 5`.
`ingest_cmd.py` uses this for worker pool sizing (line 259) with fast/slow lane split.

**No global mutable state concerns for local-only ingest:**
- `LocalStorageClient` uses append-only JSONL writes (serialized by asyncio event loop)
- In-memory indexes (`_documents_by_hash`, etc.) are dict operations -- single-threaded in asyncio
- `TopicMatcher._name_cache` is a dict (safe in single-threaded asyncio)
- File I/O for archive writes goes to unique paths per email (no conflicts)

**Revised speedup estimate:** With local parsing making most emails process in 3-5s, the
bottleneck becomes API rate limits (LLM, embeddings) rather than concurrency slots.
Increasing to 8-10 workers helps saturate API quotas. Realistic throughput improvement:
20-40% for the LLM-bound portion.

**Decision:** Implement. Change default to 8 in config. Trivial, no downside.

### P2: Parallelize context + topic generation -- DEFERRED (likely moot)

**Analysis:** The task description notes this may be moot if we consolidate context + topics
into a single LLM call. Code review confirms the dependency:

- `pipeline.py` line 199-207: context generation runs first
- `pipeline.py` line 262-264: context_summary is fed into topic extraction input
- Topic extraction uses subject + original message + context_summary

**If consolidated into one call:** Both outputs come from a single LLM request, so there
is nothing to parallelize. The single call saves ~0.5-1.5s per email (one fewer API
round-trip) and produces both context summary and topics together. This is strictly
better than P2.

**If NOT consolidated:** Parallelizing requires dropping context_summary from topic input
(Option A in 06c). The context summary adds marginal value to topic extraction since
subject + body already provide strong signal. But even then, the savings are only
~0.5-1.5s per email. Not worth the code complexity if consolidation is planned.

**Decision:** DEFER. Evaluate consolidation first. If consolidation is implemented, P2
becomes irrelevant. If consolidation is rejected for quality reasons, revisit P2 as
a simple Option A implementation.

### P3: Pipeline body + attachment overlap -- DEFERRED

**Analysis:** The proposal suggests starting attachment processing in parallel with
context/topic generation. Code review of the data flow:

- Attachment processing (`attachment_handler.py`) does NOT require `context_summary` or
  `topic_ids` during its core work (parsing, chunking, creating attachment documents)
- Context summary for attachments is generated independently inside `attachment_handler.py`
  (line 214-228) using the attachment's own parsed content
- `topic_ids` and `vessel_ids` are applied to attachment chunk metadata AFTER processing
  (lines 275-284 in `attachment_handler.py`)

So technically, attachments COULD start parsing while body context/topics generate. But:

1. With local parsers, attachment parsing is near-instant (~10ms for PDF/DOCX)
2. The only slow attachment operations are cloud API calls (complex PDFs, images)
3. Context/topic generation takes ~1-3s
4. The overlap window is small and the restructuring effort is medium (~50-80 lines)
5. P1 (parallel attachments) already captures most of the benefit

**Revised speedup estimate:** Maybe 1-2s per email in the best case (complex PDF
parsing overlapped with body LLM calls). Not worth the restructuring complexity given
that P1 already parallelizes the expensive attachment work.

**Decision:** DEFER. The restructuring adds complexity for diminishing returns once P1
and local parsers are in place. Revisit only if profiling shows a clear bottleneck.

### P7: Batch topic embeddings -- APPROVED (for Supabase mode) / IRRELEVANT (for local-only)

**Analysis:** `TopicMatcher.get_or_create_topic()` (topics.py line 320-369):

1. Checks `_name_cache` (in-memory dict) -- O(1)
2. Checks exact name match in DB -- `db.get_topic_by_name()`
3. Generates embedding via API -- `embeddings.generate_embedding(name)`
4. Searches DB for similar topics -- `db.find_similar_topics()`

For **local-only ingest**, steps 2 and 4 are in-memory operations (<1ms). Step 3 is
the only API call. Batching 3-5 topic embeddings into one call saves ~800ms per email
(4 x 200ms eliminated).

However, the `_name_cache` (line 315) means most topics after the first ~100 emails
will be cache hits (maritime reports have a finite topic vocabulary). So the embedding
API call only happens for genuinely new topics, which decreases rapidly.

**For local-only ingest:** The batching is worth ~800ms per email only during the
initial topic-building phase (first ~100-200 emails). After that, cache hits dominate.
Across 6,289 emails, the net saving is small (maybe 2-3 minutes total).

**Decision:** APPROVED as a nice-to-have. Implement when touching `TopicMatcher` for
other reasons. Low effort (~20 lines), low risk, marginal but free benefit.

### P5: Progressive embedding -- DEFERRED

**Analysis:** Currently all chunks are embedded in one batch at the end (pipeline.py
line 398-401). P5 proposes embedding body chunks immediately, then each attachment
as it completes.

**For local-only ingest:** Embedding is the one operation that MUST use an external API
(no local embedding model in the plan). With chunk size changing from 512 to 1024
tokens, most emails will have fewer chunks. The current batching approach is actually
more efficient for API calls (fewer round-trips with larger batches).

Progressive embedding makes sense for a system where you want early searchability.
For local-only ingest (writing JSONL files), there is no searchability until import.
The batch approach is strictly better for throughput.

**Decision:** DEFER. Irrelevant for local-only ingest. Reconsider when building the
production Supabase pipeline if early searchability matters.

### P8: Local image preprocessing -- ALREADY APPROVED (from 06a)

Confirmed in `optimization-plan.md` Phase 1a. The image pre-filter heuristic from
the estimator (`_is_meaningful_image()`) will be ported to the live pipeline. This
is documented in the optimization plan and will be implemented as part of Phase 1.

No additional action needed from 06c review.

### P9: Batch pre-fetch resumability -- IRRELEVANT for local-only

The proposal targets Supabase REST API round-trips for skip-checking (3-4 HTTP calls
per file at ~50-100ms each). For local-only ingest, these checks are in-memory dict
lookups (<1ms). No optimization needed.

**Decision:** SKIP for local-only. Already irrelevant.

### P10: asyncpg migration -- IRRELEVANT for local-only

No database in local-only mode. All operations are in-memory + JSONL writes.

**Decision:** SKIP for local-only. Already marked DEFERRED.

### P6: Batch storage uploads -- IRRELEVANT for local-only

Local file writes are <1ms. No Supabase Storage uploads.

**Decision:** SKIP for local-only. Already irrelevant.

---

## Revised Speed Estimates for Local-Only Ingest

### Per-email time breakdown (after all approved optimizations)

| Operation | Time | Notes |
|-----------|------|-------|
| Hash check + version check | <1ms | In-memory dict lookup |
| Parse EML | ~10ms | Local I/O |
| Generate archive | ~20ms | Local file writes |
| Create email document | <1ms | In-memory + JSONL append |
| Vessel matching | <1ms | Local |
| Context summary (LLM) | 500-1500ms | API call (dominant cost) |
| Topic extraction (LLM) | 500-1500ms | API call (or consolidated) |
| Topic matching | <1ms | In-memory cosine similarity |
| Body chunking | ~5ms | Local |
| Attachment parsing (simple) | ~10ms each | Local PyMuPDF4LLM / python-docx |
| Attachment parsing (complex) | 2-5s each | Cloud API (GPT-4.1-mini) |
| Image description | 0.5-1s each | Vision API (after pre-filter) |
| Attachment context (LLM) | 500-1500ms each | API call per attachment |
| Embedding generation | 200-500ms | API batch (all chunks) |
| Insert chunks | <1ms | JSONL append |
| **Typical email (no complex attachments)** | **~2-4s** | Dominated by LLM calls |
| **Email with 1 complex PDF** | **~5-8s** | PDF API + LLM calls |
| **Email with 5 complex PDFs (P1 parallel)** | **~6-10s** | Parallel parsing helps here |

### Total ingest time estimate

- 6,289 emails with 8 concurrent workers
- Average: ~4s per email (mix of simple and complex)
- Effective throughput: ~120 files/min
- **Estimated total: ~53 minutes**

Compare to 06c's original estimates:
- Current: ~10.5 hours
- After Phase 1: ~3-4 hours
- After all phases: ~1.5-2.5 hours

The local-only architecture with local parsers achieves better results than 06c's
"after all phases" estimate, because eliminating network I/O for storage and DB
operations is more impactful than any of the proposed concurrency optimizations.

---

## Summary of Decisions

| Proposal | Decision | Rationale |
|----------|----------|-----------|
| P1: Parallel attachments | APPROVED | Low effort, helps with remaining cloud-parsed attachments |
| P4: Concurrent files 8-10 | APPROVED | Trivial config change, helps saturate API quotas |
| P2: Parallel context+topics | DEFERRED | Likely moot if LLM calls are consolidated |
| P3: Body+attachment overlap | DEFERRED | Diminishing returns with local parsers + P1 |
| P7: Batch topic embeddings | APPROVED (low priority) | Nice-to-have, implement when touching TopicMatcher |
| P5: Progressive embedding | DEFERRED | Irrelevant for local-only JSONL output |
| P8: Local image preprocessing | ALREADY APPROVED | Part of optimization Phase 1 (from 06a) |
| P9: Batch pre-fetch | SKIP | Irrelevant for local-only (in-memory lookups) |
| P10: asyncpg migration | SKIP | Irrelevant for local-only (no database) |
| P6: Batch storage uploads | SKIP | Irrelevant for local-only (local file writes) |

## Integration into Implementation Plan

The three approved proposals (P1, P4, P7) should be implemented during the same phase
as the pipeline changes for local-only ingest, to avoid touching `pipeline.py` twice:

- **P4** (config change): Add to local-only-ingest-plan Step 8 (CLI changes) -- change
  default `max_concurrent_files` from 5 to 8
- **P1** (parallel attachments): Add as a new step in local-only-ingest-plan, after
  Step 7 (create_local_ingest_components), since it modifies `pipeline.py`
- **P7** (batch topic embeddings): Add as a nice-to-have step, implement when the
  topic system is being modified anyway
