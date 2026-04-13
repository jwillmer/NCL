---
purpose: Master tracking document synthesizing all MTSS ingest pipeline investigation findings
status: investigation-phase
date: 2026-04-13
documents: 10 investigation reports (01, 02, 03, 06a, 06b, 06c, 06d, 07a, 07b, 09)
---

# MTSS Ingest Pipeline Investigation -- Master Tracking

## Overview

- **Project:** MTSS (Maritime Technical Safety System) -- Email RAG pipeline for maritime incident reports
- **Scope:** Evaluate and improve the MTSS ingest pipeline before processing ~7 GB of new email data (6,289 EML files)
- **Date:** 2026-04-13
- **Status:** Investigation phase -- all research complete, awaiting decisions and implementation
- **Production scale:** ~50 GB ingest data, ~100 GB database (from project memory)

## Executive Summary

The MTSS ingest pipeline is architecturally sound but has critical bugs, configuration errors, and missing capabilities that must be addressed before the next full ingest run. The most urgent finding is a **reranker bug in the agent path** (`src/mtss/api/agent.py` line 460) that silently disables the Cohere cross-encoder reranker in production -- the very component documented to improve accuracy by 20-35%. Additionally, the **DATA_SOURCE_DIR** environment variable points to a nonexistent `./data/source` directory when the actual EML files reside at `./data/emails`, and the **Supabase Storage archive bucket has been auto-cleared**, meaning all LlamaParse-cached `.md` files are lost and must be regenerated.

A full ingest of 6,289 EML files is estimated to cost **$140-$225** depending on optimizations applied. LlamaParse document parsing ($75-$157) and Vision API image processing ($63-$84) account for ~95% of the cost. Two high-impact, low-effort optimizations -- tiered PDF parsing (routing simple PDFs to local parsers) and local image pre-filtering (reusing the estimator's heuristic) -- could reduce the first-run cost by **36-56%**, bringing the total down to $99-$143. Processing time is estimated at 3-10 hours depending on concurrency settings and parallelization improvements.

The retrieval/response quality investigation uncovered eight classes of improvement, with the most impactful being: missing date filtering in `match_chunks` (temporal queries return arbitrary results), no aggregation query path (count queries produce misleading answers), and insufficient result diversity for cross-vessel comparison queries. A phased implementation roadmap spanning 4 phases is proposed, with Phase 0 (critical bug fixes) as a prerequisite before any ingest or optimization work.

## Critical Findings

Items that must be fixed before any ingest run or production use:

- **Reranker bug (07a):** Agent path sets `top_k=rerank_top_n` (5), so only 5 candidates are retrieved and reranking is effectively skipped. File: `src/mtss/api/agent.py`, line 460. The intended behavior is retrieve 20, rerank to 5.
- **DATA_SOURCE_DIR mismatch (03):** `.env` variable `DATA_SOURCE_DIR` is set to `./data/source` which does not exist. EML files are at `./data/emails`. Ingest will find zero files until corrected.
- **LlamaParse cache lost (03):** Supabase Storage `archive` bucket was auto-cleared. All cached `.md` files for parsed attachments are gone. Every document attachment must be re-parsed through LlamaParse on the next ingest, incurring full parsing cost.
- **Missing date filtering (07b):** The `match_chunks` function has no date parameters. Temporal queries ("last 2 years", "since January") rely entirely on semantic similarity, producing unreliable results. Dates are stored in `documents.email_date_start` but never used as filter criteria.
- **Image pre-filter not applied in pipeline (06a):** The estimator has a working heuristic (`_is_meaningful_image()`) that filters ~73% of images as non-content (logos, banners, signatures), but **this heuristic is only used in the estimator, not in the actual ingest pipeline**. All ~20,783 images go through the Vision API for classification.
- **Estimator underestimates cost by ~24% (01):** The `mtss estimate` command misses context summary LLM calls, topic extraction calls, and undercounts image processing (classifies ALL images, not just meaningful ones). Reported estimate: ~$132; actual expected: ~$164.

## Task Tracker

| Task ID | Title | Status | Document | Key Finding | Decision Needed | Next Steps |
|---------|-------|--------|----------|-------------|-----------------|------------|
| T-01 | Cost estimation | Research complete | [01](01-cost-estimation.md) | Full ingest: $140-$200; estimator underestimates by ~24% ($132 vs $164) | Accept cost range? | Run `mtss estimate --source ./data/emails` for exact numbers |
| T-02 | Local storage design | Proposal | [02](02-local-storage-design.md) | JSONL + archive folder structure; 10 missing methods in LocalStorageClient | Implement before or after cloud ingest? | Phase 1: extend LocalStorageClient; Phase 2: progress tracker |
| T-03 | Data recovery plan | Investigation complete | [03](03-data-recovery-plan.md) | Archive bucket lost; DB status unknown; 3 recovery scenarios defined | Probe DB state first; choose Scenario A/B/C | Run `mtss stats` to probe DB; fix DATA_SOURCE_DIR |
| T-06a | Cost reduction | Proposal | [06a](06a-optimization-cost-reduction.md) | Tiered PDF parsing saves $63-$94 (28-42%); image pre-filter saves $19-$32 (8-14%) | Implement tiered parsing? | Extract `_is_meaningful_image()` to shared utility |
| T-06b | Retrieval quality | Proposal | [06b](06b-optimization-retrieval-quality.md) | 8 proposals; top 3: attachment context inheritance, chunking strategy, temporal context | Which proposals to prioritize? | Start with P6 (attachment context) -- low effort, high impact |
| T-06c | Processing speed | Proposal | [06c](06c-optimization-processing-speed.md) | Parallelize attachments = 40-60% speedup; bump concurrency to 8-10 = 30-50% | Accept Phase 1 quick wins? | Implement P1+P4+P2 in pipeline.py (~1-2 days) |
| T-07a | Search optimization | Investigation | [07a](07a-response-search-optimization.md) | CRITICAL: reranker silently disabled in agent path; 15 proposals across 5 priorities | Fix P0 bug immediately | Change agent.py line 460 to `top_k=20` |
| T-07b | Scenario analysis | Investigation | [07b](07b-response-scenario-analysis.md) | 6 query scenarios analyzed; date filtering and aggregation queries are biggest gaps | Add date params to match_chunks? | SQL migration for date_from/date_to in match_chunks |
| T-06d | Parser alternatives | Investigation | [06d](06d-parser-alternatives.md) | LlamaParse replaceable with PyMuPDF4LLM (free) + Gemini Flash ($0.00033/page); total cost $164→$5 possible | Switch parsers before ingest? | Phase 1: local pre-filter + pypdf; Phase 2: Gemini Flash for complex |
| T-09 | Test validation plan | Proposal | [09](09-test-validation-plan.md) | 15 test docs selected; 11 scenarios; critical: no test for reranker bug | Approve test subset? | Implement test additions; run subset ingest |

## Cost Summary

### Reconciliation of Two Cost Estimates

Task 01 and Task 06a both estimate full ingest costs but use different methodologies:

| Cost Center | Task 01 Estimate | Task 06a Estimate | Reconciled |
|-------------|-----------------|-------------------|------------|
| LlamaParse (document parsing) | $75.54 | $157 (4 pages/email avg) | **$75-$157** -- Task 01 uses measured sample (243 emails, 1.92 pages/email avg); Task 06a uses higher estimate. Actual depends on attachment mix. |
| Vision API (image processing) | $83.73 | $63 ($0.01/image x meaningful) | **$63-$84** -- Task 01 counts all images for classification; Task 06a counts only meaningful. Both are upper/lower bounds. |
| LLM calls (context + topics) | $4.54 | $3.15 | **$3-$5** -- Negligible either way |
| Embeddings | $0.27 | $0.63 | **$0.30-$0.65** -- Negligible either way |
| **TOTAL (no optimization)** | **$164** | **$225** | **$140-$225** |

### With Proposed Optimizations (06a)

| Optimization | Savings | New Total Range |
|-------------|---------|-----------------|
| Tiered PDF parsing (local for simple PDFs) | $63-$94 (28-42%) | $131-$162 |
| + Local image pre-filtering | +$19-$32 (8-14%) | $99-$143 |
| + LLM call consolidation | +$1.26 (<1%) | $98-$142 |
| + All cost optimizations | **$82-$126 saved** | **$99-$143** |

### Re-ingest Cost (with caching, 06a Proposal 7)

With per-artifact caching (images, context, topics, embeddings) in place, re-ingests of unchanged content would cost near **$0**. This is high-priority for operational cost since version bumps trigger reprocessing.

## Optimization Roadmap

### Phase 0: Critical Fixes (before any ingest)

| Item | Source | File(s) | Effort |
|------|--------|---------|--------|
| Fix reranker bug: agent top_k | 07a (P0) | `src/mtss/api/agent.py` line 460 | Trivial -- change to `top_k=20` |
| Fix DATA_SOURCE_DIR | 03 | `.env` | Trivial -- set to `./data/emails` |
| Probe DB state | 03 | CLI: `mtss stats` | 5 minutes |
| Decide recovery scenario (A/B/C) | 03 | N/A | Decision only |

### Phase 1: Quick Wins (high impact, low effort, 1-2 weeks)

| Item | Source | Est. Savings/Impact | Effort |
|------|--------|-------------------|--------|
| Apply image pre-filter in pipeline | 06a (P2) | $19-$32 cost savings | Low -- reuse estimator heuristic |
| Increase MAX_CONCURRENT_FILES to 8-10 | 06c (P4) | 30-50% throughput | Trivial -- config change |
| Parallelize attachments within email | 06c (P1) | 40-60% speedup per multi-attachment email | Low -- ~15 lines in pipeline.py |
| Parallelize context + topic generation | 06c (P2) | 10-15% speedup | Low -- ~10 lines |
| Attachment context inheritance | 06b (P6) | +15-25% recall for attachment queries | Low -- pass parent context |
| Minimum content filter for chunks | 06b (P1-A) | +15-25% precision | Low -- skip chunks <30 tokens |
| Date prepending in embedding text | 06b (P8-A) | +10-15% recall for temporal queries | Low -- prepend date to embedding_text |
| Enriched rerank context (email subject) | 07a (P1, 7a) | Medium quality improvement | Low -- ~5 lines in reranker.py |
| Increase rerank_top_n to 8 | 07a (P1, 8c) | Better coverage for complex queries | Trivial -- config change |
| Increase max_tokens to 2000-3000 | 07a (P1, 8d) | Prevent truncated responses | Trivial -- config change |
| Increase top_k to 40 | 07a (P2, 1) | Better reranking candidate pool | Trivial -- after P0 fix |

### Phase 2: Medium Effort Improvements (2-4 weeks)

| Item | Source | Impact | Effort |
|------|--------|--------|--------|
| Tiered PDF parsing (local + LlamaParse) | 06a (P1) | $63-$94 cost savings (28-42%) | Medium -- new complexity classifier, local parser |
| Date filtering in match_chunks | 07b (#1) | Correct temporal queries | Medium -- SQL migration + query preprocessing |
| Adaptive chunk sizes for attachments (1024 tokens) | 06b (P1-B/C) | Better context per chunk, fewer chunks | Low-Medium |
| Structured metadata extraction | 06b (P2) | +20-30% precision via faceted filtering | Medium |
| Vessel alias support + IMO matching | 06b (P4) | +10-20% recall for vessel queries | Low-Medium |
| Hybrid search (BM25 + vector) | 07a (P3, 3) | Technical term matching (equipment codes) | Medium -- migration, SQL, RRF merge |
| Sibling chunk expansion | 07a (P3, 8a) | Adjacent context for top results | Medium |
| Parent email context for attachment chunks | 07a (P3, 8b) | Critical incident context for PDFs | Medium |
| Pipeline body + attachment overlap | 06c (P3) | 20-30% additional speedup | Medium -- pipeline restructuring |
| Batch topic matching embeddings | 06c (P7) | 3-5% speedup | Low |
| Caching (images, context, embeddings) | 06a (P3) | ~$0 re-ingest cost | Medium |
| Local storage: extend LocalStorageClient | 02 (Phase 1) | 10 missing methods for local-only mode | Medium |
| Embedding dimensions: 512 instead of 1536 | 06a (P4) | ~67% DB vector storage reduction | Low -- config change, requires re-embed |

### Phase 3: Major Architectural Changes (4-8 weeks)

| Item | Source | Impact | Effort |
|------|--------|--------|--------|
| Query intent classification | 07b (#2) | Route aggregation/comparison/temporal queries | Medium-High |
| Aggregation SQL path | 07b (#5) | Correct count/frequency answers | Medium |
| Document relationship linking (email threads) | 06b (P5) | +20-30% recall for incident timeline queries | High |
| Topic taxonomy (controlled vocabulary) | 06b (P3) | Consistent categorization, hierarchical browsing | Medium-High |
| Multi-document incident aggregation | 06b (P7) | Coherent incident narratives | Medium (depends on P5) |
| Diversity-aware retrieval (MMR) | 07b (#7) | Better cross-vessel comparisons | Medium |
| Migrate hot-path DB to asyncpg | 06c (P10) | 5-10% speedup, cleaner architecture | High |
| Local storage: full local-only pipeline | 02 (Phase 2-4) | Provider-independent ingest | High |

## Decision Log

Decisions requiring user input, ordered by urgency:

### D-1: Recovery Scenario Selection (Blocking)

**Context:** Supabase Storage archive bucket has been auto-cleared (03). DB state is unknown.

**Options:**
- **Scenario A (Archive-Only Rebuild):** If DB is intact. Cost: ~$0 (no LLM/API calls). Risk: Low.
- **Scenario B (Incremental Recovery):** If DB is partially intact. Cost: LlamaParse for new docs only. Risk: Medium.
- **Scenario C (Full Re-ingest):** If DB is empty. Cost: $140-$225. Risk: Highest cost.
- **Nuclear Option:** `mtss clean --yes` then full re-ingest. Safest but most expensive.

**Next step:** Run `mtss stats` to probe DB state and determine which scenario applies.

### D-2: Cost Optimization Before Ingest

**Context:** Tiered PDF parsing (06a P1) saves $63-$94 but requires Medium implementation effort.

**Options:**
- **Option A:** Implement tiered parsing + image pre-filter before ingest. Saves $82-$126. Delays ingest by 1-2 weeks.
- **Option B:** Implement only image pre-filter (Low effort). Saves $19-$32. Delays ingest by 1-2 days.
- **Option C:** Ingest now at full cost ($140-$225). Implement optimizations for future re-ingests.

**Trade-off:** Time vs. money. At $140-$225 per run, savings matter most if multiple re-ingests are expected.

### D-3: Embedding Dimensions (512 vs 1536)

**Context:** Reducing from 1536 to 512 dimensions saves ~67% vector storage (06a P4). At production scale (~100 GB DB), this is significant. Quality loss is ~2% MTEB, likely unnoticeable.

**Options:**
- **512 dims:** Smaller DB, faster search. Requires re-embedding if switching.
- **1536 dims (current):** Maximum quality. Larger storage footprint.

**Trade-off:** Storage cost vs. retrieval quality. The 2% quality loss is within noise for maritime incident reports.

### D-4: Chunk Size (512 vs 1024 tokens)

**Context:** Larger chunks provide more context per result (06a P5, 06b P1-B). Combined with 512 dims, reduces vector storage by ~73%. May improve retrieval for narrative content.

**Options:**
- **512 tokens (current):** More granular, better for specific lookups.
- **1024 tokens:** Better context, fewer chunks, lower storage. May reduce precision for very specific queries.

**Trade-off:** Granularity vs. context. Recommend testing on a sample before full ingest.

### D-5: Local Storage Priority

**Context:** Task 02 proposes a local-only ingest mode with JSONL output.

**Options:**
- **Before ingest:** Build local storage, ingest locally first, then import to cloud. Provides a portable backup.
- **After ingest:** Ingest to cloud first, add local export later. Faster to get production data.
- **Parallel development:** Build local storage while running cloud ingest.

**Trade-off:** Data portability vs. time to production.

### D-6: Search Quality Improvements Scope

**Context:** Tasks 07a and 07b identify 20+ search quality proposals.

**Options:**
- **Minimal (P0 only):** Fix the reranker bug. Immediate improvement.
- **Quick wins (P0-P2):** Bug fix + enriched reranking + config tuning. 1-2 days.
- **Comprehensive (P0-P3):** Add hybrid search, sibling expansion, parent context. 2-4 weeks.
- **Full (all proposals):** Include query classification, aggregation, MMR. 6-8 weeks.

**Trade-off:** Time vs. response quality. The reranker bug fix alone restores significant quality.

## Dependencies

```
Phase 0 (Critical Fixes)
  |
  +-> Phase 1 (Quick Wins) -- can start immediately after Phase 0
  |     |
  |     +-> T-06c P1 (parallel attachments) -> T-06c P3 (body+attachment overlap)
  |     +-> T-07a P0 (reranker fix) -> T-07a P2 (increase top_k) -> T-07a P3 (hybrid search)
  |
  +-> D-1 (Recovery Scenario) -- must be decided before ingest
  |     |
  |     +-> Scenario A/B/C -> Ingest Run -> Phase 2 improvements
  |
  +-> Phase 2 (Medium Effort)
  |     |
  |     +-> T-06a P1 (tiered parsing) -- can develop in parallel with ingest
  |     +-> T-07b date filtering -> T-07b query classification
  |     +-> T-06b P2 (structured metadata) -- requires re-ingest to apply
  |     +-> T-02 LocalStorageClient extension -- independent
  |
  +-> Phase 3 (Major Changes)
        |
        +-> T-06b P5 (thread linking) -> T-06b P7 (incident aggregation)
        +-> T-07b aggregation SQL -> T-07b query classification
        +-> T-06b P3 (topic taxonomy) -- independent, needs domain expertise
```

Key dependency chains:

1. **Reranker fix (07a P0)** is prerequisite for all reranking improvements (top_k tuning, enriched context, rerank score floor).
2. **Date filtering (07b)** is prerequisite for temporal query optimization and aggregation queries.
3. **Document relationship linking (06b P5)** is prerequisite for multi-document incident aggregation (06b P7) and post-retrieval incident grouping (07b).
4. **DB state probe (03)** is prerequisite for choosing recovery scenario and estimating ingest scope.
5. **DATA_SOURCE_DIR fix (03)** is prerequisite for any ingest or estimate command.
6. **Embedding dimension decision (D-3)** should be made before ingest -- changing later requires full re-embedding.

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Source |
|------|-----------|--------|------------|--------|
| DB is empty or corrupted after Supabase downtime | Medium | High -- full re-ingest at $140-$225 | Probe with `mtss stats` first; use `--lenient` flag during ingest | 03 |
| LlamaParse re-parsing cost due to lost cache | Confirmed | High -- $75-$157 unavoidable if Scenario C | Tiered parsing (06a P1) can reduce by 40-60% if implemented first | 03, 06a |
| Reranker bug degrades production search quality | Confirmed | High -- 20-35% accuracy loss in agent path | Trivial fix: change `top_k` in agent.py | 07a |
| Estimator underreports actual ingest cost | Confirmed | Medium -- $32 gap (24%) could exceed budget | Run full estimate; add missing cost centers to estimator | 01 |
| Aggregation queries return misleading counts | High | High -- users trust LLM-generated numbers | Add query classification + SQL aggregation path | 07b |
| Temporal queries return arbitrary results | High | High -- "last 2 years" has no date filtering | Add date_from/date_to to match_chunks | 07b |
| Topic taxonomy fragmentation over time | Medium | Medium -- duplicate topics degrade filtering | Controlled vocabulary (06b P3) or raise dedup threshold | 06b |
| Interrupted ingest leaves partial data | Low | Medium -- pipeline is resumable | Use `--resume` (default); `mtss reset-stale` for stuck entries | 03 |
| Attachment content underrepresented in search | High | Medium -- PDF inspection reports miss top-5 | Attachment context inheritance (06b P6); sibling expansion (07a 8a) | 06b, 07a |
| Local storage JSONL size with embeddings | Low | Low -- ~1.2 GB for 100K chunks at 1536 dims | Reduce to 512 dims (06a P4); optional `--skip-embeddings` flag | 02 |
| Short vessel names cause false positive matches | Medium | Low -- 3-4 letter names match common words | Minimum name length threshold; contextual disambiguation | 06b |
| `max_tokens=1000` truncates detailed responses | Medium | Medium -- procedural and comparison answers cut short | Increase to 2000-3000 (07a P1, 8d) | 07a, 07b |

## Processing Time Estimates (06c)

| Configuration | Time per Email | Throughput | Total (6,289 files) |
|--------------|---------------|------------|---------------------|
| Current (5 concurrent, sequential attachments) | ~30s avg | ~10 files/min | ~10.5 hours |
| After Phase 1 quick wins (8-10 concurrent, parallel attachments) | ~15-20s avg | ~25-35 files/min | ~3-4 hours |
| After all speed optimizations | ~10-15s avg | ~40-60 files/min | ~1.5-2.5 hours |

## Document Index

| Document | Title | Status | One-Line Summary |
|----------|-------|--------|-----------------|
| [01](01-cost-estimation.md) | Cost Estimation: Full Ingest of 6,289 EML Files | Research complete | Full pipeline cost breakdown: $140-$200 total; LlamaParse ($75) and Vision API ($84) dominate; estimator misses $32 in hidden LLM costs. |
| [02](02-local-storage-design.md) | Local Storage Design: Backup & Provider-Independent Ingest | Proposal | JSONL + archive folder design for local-only ingest; 10 missing methods in LocalStorageClient; 4-phase implementation plan. |
| [03](03-data-recovery-plan.md) | Data Recovery Plan | Investigation complete | Supabase Storage bucket auto-cleared; 3 recovery scenarios (archive-only rebuild, incremental, full re-ingest); DATA_SOURCE_DIR misconfigured. |
| [06a](06a-optimization-cost-reduction.md) | Ingest Pipeline Cost Reduction | Proposal | 8 proposals; tiered PDF parsing saves 28-42%; image pre-filter saves 8-14%; combined first-run savings of 36-56%. |
| [06b](06b-optimization-retrieval-quality.md) | Retrieval Quality Optimization Proposals | Proposal | 8 proposals; top priority: attachment context inheritance (low effort, +15-25% recall); chunking strategy; temporal context preservation. |
| [06c](06c-optimization-processing-speed.md) | Processing Speed Optimization Proposals | Proposal | 10 proposals; parallel attachments (40-60% speedup) and increased concurrency (30-50%) are top wins; total time reducible from 10.5h to 1.5-2.5h. |
| [07a](07a-response-search-optimization.md) | Search & Retrieval Performance Optimization | Investigation | CRITICAL: reranker silently disabled in agent path (line 460); 11 proposals across 5 priority tiers; hybrid search and context assembly improvements. |
| [07b](07b-response-scenario-analysis.md) | Response Quality Analysis Across Query Scenarios | Investigation | 6 query scenarios analyzed; missing date filtering and aggregation queries are biggest gaps; 7 cross-cutting issues identified with proposed fixes. |
| [06d](06d-parser-alternatives.md) | Parser Alternatives & LLM-Native Document Processing | Investigation | Current parsers replaceable: PyMuPDF4LLM for simple PDFs (free), Gemini Flash for complex (11x cheaper than LlamaParse). Total cost reducible from $164 to $5 (97%). |
| [09](09-test-validation-plan.md) | Test Subset Identification & Validation Plan | Proposal | 15 test documents selected; 11 test scenarios; 7 sample queries; critical gap: no test catches reranker bug; 6-phase validation procedure. |

## Key File Reference

| File | Relevance |
|------|-----------|
| `src/mtss/api/agent.py` (line 460) | CRITICAL BUG: reranker bypass |
| `src/mtss/ingest/pipeline.py` (lines 342-376) | Attachment processing loop (parallelize target) |
| `src/mtss/ingest/pipeline.py` (lines 199-276) | Context + topic generation (parallelize target) |
| `src/mtss/ingest/estimator.py` | Cost estimator (missing cost centers) |
| `src/mtss/rag/retriever.py` (line 76) | Rerank skip condition |
| `src/mtss/rag/query_engine.py` (line 218) | `max_tokens=1000` limit |
| `src/mtss/parsers/preprocessor.py` | Image classification routing (add pre-filter) |
| `src/mtss/processing/image_processor.py` | Vision API calls (pre-filter target) |
| `src/mtss/rag/reranker.py` | Reranker (enrich context target) |
| `tests/local_storage.py` | Existing LocalStorageClient (extend for 02) |
| `.env` | DATA_SOURCE_DIR (must fix to `./data/emails`) |
