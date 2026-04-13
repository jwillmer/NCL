---
purpose: Track all decisions made and progress across investigation documents for session continuity
status: active
date: 2026-04-13
last_updated: 2026-04-14T01:00:00
---

# Decisions & Progress Tracker

## Context for New Agents

**What is MTSS?** Maritime Technical Support System -- a RAG (Retrieval-Augmented Generation) pipeline that ingests maritime incident report emails (EML files with PDF/image attachments), chunks and embeds them, stores them in a vector database, and provides a chat UI for searching and querying incident history.

**Tech stack:** Python 3.12, FastAPI, Supabase (PostgreSQL + pgvector), LiteLLM (wraps OpenAI APIs), Cohere reranker, Vite + React chat frontend. Project root: `C:/Projects/GitHub/NCL/`, source code at `src/mtss/`, tests at `tests/`, data at `data/emails/` (6,289 EML files).

**What this document is:** A tracker of every decision made during a multi-document investigation of the MTSS ingest pipeline. The investigation analyzed cost, storage, parsers, speed, and search quality across documents 01-09 in `docs/investigation/`. Each decision references its source investigation document and rationale.

**How to use this document:** Read the Quick Status table for an overview, then check Implementation Order at the bottom for what to do next. All active plans are in this same `plans/` directory. Execute `01-critical-fixes.md` first, then `02-implementation.md`, then `03-test-validation.md`.

---

This document captures every decision made during the ingest pipeline investigation so a new agent or session can pick up exactly where we left off.

## Quick Status

| Document | Reviewed | Decision Made | Plan Created | Implementation |
|----------|----------|---------------|--------------|----------------|
| 01 — Cost estimation | Yes | Use real estimate ($230.56 baseline) | Yes (`optimization-plan.md`) | Pending |
| 02 — Local storage design | Yes | JSONL (not SQLite), implement local-only first | Yes (`02-implementation.md`) | Pending |
| 03 — Data recovery | Yes | Clean start, skip recovery | N/A | `.env` fixed |
| 06a — Cost reduction | Yes | Proposals 1,2,4,5 approved; merged into impl plan | Yes (`02-implementation.md`) | Pending |
| 06b — Retrieval quality | Yes | P6, P1-A, P8-A approved; P4-A noted; rest deferred | Updated `optimization-plan.md` | Pending |
| 06c — Processing speed | Yes | P1, P4 approved; P7 low-priority; P2, P3, P5 deferred; P6, P9, P10 irrelevant | `06c-review-findings.md` | Pending |
| 06d — Parser alternatives | Reviewed (during investigation) | Stay with OpenAI (LiteLLM), local parsers for simple docs | Covered in `optimization-plan.md` | Pending |
| 06e — LLM provider comparison | Reviewed (during investigation) | GPT-4.1-mini batch for complex PDFs, no new provider | Covered in `optimization-plan.md` | Pending |
| 07a — Search optimization | Yes | P0 bug fix + P1 quick wins approved; P2-P5 deferred post-ingest; tsvector auto-generates | Yes (`01-critical-fixes.md`) | **Done** (d710f5f) |
| 07b — Scenario analysis | Yes | Query-side completeness transparency included in Plan 01; remaining deferred | Partial (`01-critical-fixes.md` Fix 2.5) | **Done** (d710f5f) |
| 09 — Test validation plan | Yes | Plan created (`03-test-validation.md`) | Yes | Execution pending |

## Decisions Log

### D-01: Local storage format — JSONL over SQLite
- **Date:** 2026-04-13
- **Context:** Needed local-only ingest before production DB is ready
- **Options:** SQLite + sqlite-vec, JSONL flat files, hybrid
- **Decision:** JSONL flat files
- **Reasoning:** 80% of code already exists in `tests/local_storage.py`; sqlite-vec has Windows loading issues; only ~200 topics need similarity search (brute-force <1ms); no full vector search needed locally
- **Document:** `plans/reference/local-storage-sqlite-vs-files.md`

### D-02: Implement local-only ingest before cloud
- **Date:** 2026-04-13
- **Context:** Production system being reset, not all services reachable
- **Decision:** Build local-only ingest first, import to production later
- **Reasoning:** Parse once (expensive), store locally, apply to any provider. Production isn't ready anyway.
- **Document:** `02-local-storage-design.md`

### D-03: Clean start over data recovery
- **Date:** 2026-04-13
- **Context:** Supabase Storage bucket auto-cleared; DB state unknown; production being reset
- **Decision:** Skip recovery (Scenarios A/B/C), do fresh ingest with optimized pipeline
- **Reasoning:** Rebuilding production anyway; optimized pipeline ($6 vs $230) makes fresh ingest cheap; recovery dance adds complexity for uncertain benefit
- **Action taken:** Fixed `.env` DATA_SOURCE_DIR from `./data/source` to `./data/emails`
- **Document:** `03-data-recovery-plan.md`

### D-04: Parser stack — stay with OpenAI via LiteLLM
- **Date:** 2026-04-13
- **Context:** Evaluated Gemini, Claude, Mistral as LlamaParse replacements
- **Decision:** Use local parsers (PyMuPDF4LLM) for simple PDFs + GPT-4.1-mini batch for complex. No new provider.
- **Reasoning:** LiteLLM already abstracts providers; OpenAI already integrated; absolute cost difference between providers is <$1 for the complex PDF corpus; zero integration overhead
- **Documents:** `06d-parser-alternatives.md`, `06e-llm-provider-comparison.md`

### D-05: Implement cost optimization Phase 1+2 before first ingest
- **Date:** 2026-04-13
- **Context:** Baseline cost $230.56; Phase 1+2 reduces to $6.34
- **Decision:** Implement both phases before running ingest
- **Reasoning:** $224 savings worth the 5-6 day implementation effort; building local-only ingest anyway (need to wire parsers regardless)
- **Document:** `plans/reference/optimization-plan.md`

### D-06: Full estimate completed and stored
- **Date:** 2026-04-13
- **Context:** Previously only had 3.9% sample (243/6,289 emails)
- **Result:** All 6,289 emails scanned. Actual numbers: 20,995 pages, 20,684 images (9,860 meaningful), 24,974 total attachments
- **Baseline cost:** $230.56 (LlamaParse $131.22 + Vision $98.60 + LLM $0.12 + Embeddings $0.63)
- **Document:** `plans/reference/estimate-full-results.md`

### D-07: Retrieval quality proposals (06b) — batch decision
- **Date:** 2026-04-13
- **Context:** Reviewed 8 proposals from `06b-optimization-retrieval-quality.md`
- **Approved for Phase 1 (before first ingest):**
  - **P6 — Attachment Context Inheritance:** APPROVED. Zero additional LLM cost, low effort. Parent email `context_summary` is already computed before attachment processing in `pipeline.py:199-206`. Implementation: pass `context_summary` to `process_attachment()`, prepend to attachment embedding text via `build_embedding_text(email_context + "\n\n" + attach_context, chunk_content)`. ~15 lines changed.
  - **P1-A — Minimum content filter (skip <30 token chunks):** APPROVED. Trivial. Add token count check after `remove_boilerplate_from_message()` in `pipeline.py:289`. ~5 lines changed.
  - **P8-A — Prepend date to embedding text:** APPROVED. Trivial. In `pipeline.py:304-306`, prepend `[Date: YYYY-MM-DD]` from `email_doc.email_metadata.date_start` before context+content. Same for attachment chunks. ~5 lines changed.
- **Noted (low-effort, implement if time permits):**
  - **P4-A — Vessel aliases:** Low effort, but requires extending `Vessel` model and `vessel-list.csv` schema with an `aliases` column. Not blocking ingest, can be added later without re-embedding (vessel IDs are metadata, not embedding text).
- **Deferred to Phase 2:**
  - **P2 — Structured metadata extraction:** Needs prompt engineering, schema design, and query-time filter logic. Not blocking ingest.
  - **P1-B/C — Adaptive chunk sizes:** Requires testing with real data to validate 1024 vs 512. Linked to PD-01.
  - **P8-B — Per-message date preservation:** Needs `split_into_messages` refactor to return date+content pairs instead of plain strings.
- **Deferred to Phase 3:**
  - **P5 — Document relationship linking:** High complexity, needs incident entity design.
  - **P3 — Topic taxonomy:** Needs domain expertise to define maritime categories.
  - **P7 — Multi-doc aggregation:** Depends on P5 for grouping signal.
  - **P8-C — Temporal query parsing:** Needs NLP/LLM date extraction at query time.
- **Reasoning:** P6, P1-A, P8-A affect embedding text and must be implemented before the first ingest run (changing them later requires re-embedding all chunks). All three are zero-cost, low-effort changes. Deferred proposals either add LLM cost, need testing, or require new DB schema.

### D-08: Chunk size -- 1024 tokens (was PD-01)
- **Date:** 2026-04-13
- **Context:** 1024 may improve retrieval for narrative incident reports, halves chunk count
- **Decision:** Switch to 1024 tokens, overlap to 100 (config-only change in `src/mtss/config.py`)
- **Reasoning:** Maritime incident reports are narrative content where larger chunks provide more context for retrieval. Combined with 512 embedding dims (D-09), total vector storage drops ~73%. Must be set before first ingest (changing later requires re-embedding all chunks).
- **Document:** `plans/02-implementation.md` Phase 0.1

### D-09: Embedding dimensions -- 512 (was PD-02)
- **Date:** 2026-04-13
- **Context:** 512 retains 98% quality per OpenAI MTEB benchmarks, saves 67% DB storage at production scale (~100GB)
- **Decision:** Switch to 512 dimensions (config-only change in `src/mtss/config.py`)
- **Reasoning:** 2% MTEB quality drop is unlikely to be noticeable for this maritime RAG use case. At production scale, 67% vector storage reduction is significant. `text-embedding-3-small` supports native Matryoshka reduction via the `dimensions` parameter. Must be set before first ingest.
- **Document:** `plans/02-implementation.md` Phase 0.2

### D-10: Processing speed proposals (06c) -- batch decision
- **Date:** 2026-04-13
- **Context:** Reviewed 10 proposals from `06c-optimization-processing-speed.md`. Critical finding: with local parsers replacing LlamaParse and local-only ingest replacing Supabase, the bottleneck profile changes dramatically. LlamaParse (5-30s/doc) becomes local parsing (<10ms/doc). DB round-trips (50-200ms each) become in-memory dict lookups (<1ms). The dominant bottleneck shifts to LLM API calls (context + topics, 1-3s/email) and the Embedding API (0.2-0.5s/batch).
- **Approved:**
  - **P1 -- Parallel attachments within email:** Verified attachments are independent (each creates own attach_doc, email_doc is read-only, no shared mutable state). Use `asyncio.gather()` with semaphore(3). Revised speedup: ~2-4s for emails with cloud-parsed attachments. ~15 lines in `pipeline.py`.
  - **P4 -- Increase MAX_CONCURRENT_FILES to 8:** Trivial config change. No global state issues (JSONL writes serialized by asyncio event loop). Helps saturate LLM API quotas.
  - **P7 -- Batch topic embeddings:** Low priority. Saves ~800ms/email during initial topic-building phase, negligible after cache warms. ~20 lines in TopicMatcher.
- **Deferred:**
  - **P2 -- Parallel context+topics:** Likely moot if context + topic extraction consolidated into one LLM call.
  - **P3 -- Pipeline body+attachment overlap:** Diminishing returns with local parsers + P1.
  - **P5 -- Progressive embedding:** Irrelevant for local-only (JSONL output, no early searchability).
- **Skipped (irrelevant for local-only):** P6 (batch storage uploads), P9 (batch pre-fetch), P10 (asyncpg migration).
- **Revised total ingest estimate:** ~53 minutes for 6,289 emails with 8 workers (vs 06c's original 10.5 hours).
- **Document:** `plans/reference/06c-review-findings.md`

### D-11: Search/retrieval proposals (07a) -- batch decision
- **Date:** 2026-04-13
- **Context:** Reviewed 11 proposals from `07a-response-search-optimization.md` covering search strategy, filtering, reranking, context assembly, and infrastructure.
- **Approved (pre-ingest, immediate):**
  - **P0 -- Reranker bug fix:** CRITICAL. `agent.py:461` sets `top_k=settings.rerank_top_n` (5), causing retriever to get exactly 5 candidates, which matches `effective_top_n` at `retriever.py:76`, so reranking is silently skipped. Fix: change to `top_k=20`. One-line change.
  - **P1 -- Enriched rerank context (7a):** Prepend `email_subject` and `source_title` to chunk text for the reranker. ~5 lines in `reranker.py`. Data already available in `RetrievalResult`.
  - **P1 -- max_tokens 1000 to 2000 (8d):** One-line change in `query_engine.py:218`. With 8 chunks at ~1024 tokens, total context is ~9000 tokens. 2000 output tokens stays well under 128K model limit. Negligible cost increase (~$0.0006/query at gpt-4o-mini rates).
  - **P1 -- rerank_top_n 5 to 8 (8c):** Config-only change in `config.py:147`. 3 more chunks adds ~3072 input tokens/query (~$0.008/query at gpt-4o rates). Negligible. Same Cohere rerank cost (per-search pricing for <100 docs).
- **Deferred (all query-time only, no ingest blockers):**
  - **P2 -- top_k to 40 (Proposal 1):** Config change, after P0 fix.
  - **P2 -- HNSW ef_search tuning (Proposal 9):** `SET LOCAL` at query time, no schema change.
  - **P2 -- Parallel embed+topic (Proposal 11):** asyncio.gather refactor in agent.
  - **P2 -- Rerank score floor (Proposal 2):** Filter in `Reranker.rerank_results()`.
  - **P3 -- Sibling expansion (8a), parent context (8b), topic loosening (5):** Medium complexity, query-time.
  - **P3 -- Hybrid search/BM25 (Proposal 3):** tsvector column can be added as PostgreSQL generated column (`GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`). Zero ingest code changes -- auto-populated on INSERT. Can be added anytime via migration.
  - **P4 -- Rerank v3.5 (7b):** Config change, test first.
  - **P5 -- Query expansion (4), Caching (10):** Skip for now.
- **No changes needed:** Proposal 6 (vessel pre-filtering already correct), Proposal 11 streaming (already well-structured).
- **Key finding on tsvector:** The `tsvector` column for hybrid search does NOT require ingest code changes. PostgreSQL generated columns auto-populate from `content` on INSERT. A trivial migration can be added now or later. JSONL local output stores raw `content` text; tsvector auto-populates when imported to production. No reason to block ingest.
- **Document:** `plans/reference/07a-review-findings.md`

### D-12: Scenario analysis (07b) -- batch decision
- **Date:** 2026-04-13
- **Context:** Reviewed 07b scenario analysis covering query-side improvements for search result quality, trust, and completeness.
- **Included in Plan 00 (Fix 2.5):**
  - **Result completeness transparency:** Surface `total_candidate_count` and a note to the LLM context so it can say "showing top 8 of 47 results." Low effort (~10 lines in `agent.py`), high trust impact. Already available from `TopicFilterResult.total_chunk_count`.
- **Deferred (post-ingest, query-time only):**
  - All remaining 07b proposals are query-time improvements that don't depend on ingest changes. They will be evaluated during the "remaining optimizations" phase (step 10 in implementation order).
- **Document:** `07b-response-scenario-analysis.md`

## Pending Decisions

### PD-03: Remaining document decisions (09)
- **Status:** 06b, 06c, 07a, 07b complete. Only 09 (test validation) remains.

## Critical Bugs Found

| Bug | Location | Severity | Status |
|-----|----------|----------|--------|
| Reranker silently disabled | `src/mtss/api/agent.py:461` — `top_k=settings.rerank_top_n` (5) means retriever gets 5 candidates, skips reranking | P0 | **Fixed** (d710f5f) |
| DATA_SOURCE_DIR mismatch | `.env` pointed to `./data/source`, files at `./data/emails` | P0 | **Fixed** |
| Image pre-filter not in pipeline | `_is_meaningful_image()` exists in estimator but not used in actual ingest | P1 | Not fixed |
| Estimator underestimates by 24% | Misses LLM calls, undercounts image processing | P2 | Documented |

## Implementation Order (Planned)

**Plan 00 runs FIRST** -- fixes critical search bugs and quick wins that affect production NOW.
See `plans/02-implementation.md` for the full merged ingest plan.

1. ~~**Plan 01: Critical fixes & search quick wins**~~ -- **DONE** (d710f5f, 879b010). Reranker bug fixed, enriched rerank, max_tokens 2000, rerank_top_n 8, retrieval_top_k 40, ef_search 100, parallel embed+topic, score floor 0.2, completeness transparency. Also fixed pre-existing test_topic_filter assertion.
2. **Phase 0: Config quick wins** -- chunk 512->1024, dims 1536->512 (15 min)
3. **Phase 1: Image pre-filtering + model switch** -- port estimator heuristic + filename filter + GPT-4.1-nano (2-3 hrs)
4. **Phase 2: Local parsers** (parallel with Phase 3) -- PDF classifier, PyMuPDF4LLM, DOCX/XLSX/CSV/HTML (4-5 days)
5. **Phase 3: Local storage backend** (parallel with Phase 2) -- extend LocalStorageClient, progress tracker, loggers (5-7 hrs)
6. **Phase 4: Pipeline wiring + speed + quality** -- component factory, CLI --local-only, manifest, P1 parallel attachments, P4 concurrent files to 8, 06b quality wins (P6/P1-A/P8-A) (5-6 hrs, quality wins MUST be before first ingest)
7. **Phase 5: Validation** -- unit tests, integration test, cost verification (4-6 hrs)
8. **Test subset validation** (`03-test-validation.md`) -- run small ingest, validate via UI
9. **Remaining optimizations** -- 06c P7 batch topic embeddings; 07a P3-P5 query-time improvements; 07b remaining proposals; 06b Phase 2 items
10. **Full ingest** -- local-only, all 6,289 emails (~$6-10 estimated cost)
11. **Production import** -- when production system is ready

## Files Changed

| File | Change | Date |
|------|--------|------|
| `.env` | `DATA_SOURCE_DIR=./data/source` → `./data/emails` | 2026-04-13 |
| `src/mtss/api/agent.py` | Fix reranker bug (top_k), parallel embed+topic, completeness transparency | 2026-04-13 |
| `src/mtss/config.py` | Add `retrieval_top_k=40`, `rerank_score_floor=0.2`, rerank_top_n 5→8 | 2026-04-13 |
| `src/mtss/rag/reranker.py` | Enriched rerank context, score floor with fallback | 2026-04-13 |
| `src/mtss/rag/retriever.py` | Unified retrieve() with optional query_embedding, embed_query() | 2026-04-13 |
| `src/mtss/rag/query_engine.py` | max_tokens 1000→2000, pass query_embedding through | 2026-04-13 |
| `src/mtss/storage/repositories/search.py` | HNSW ef_search=100 via SET LOCAL in transaction | 2026-04-13 |
| `tests/test_reranker.py` | Tests for enriched context, score floor, all-below-keeps-one | 2026-04-13 |
| `tests/test_topic_filter.py` | Fix assertion: unmatched topic correctly skips RAG | 2026-04-13 |

## Plan Documents Index

> **For implementation, follow `01-critical-fixes.md` first, then `02-implementation.md`.**

### Active Plans

| Plan | Purpose | Status |
|------|---------|--------|
| `01-critical-fixes.md` | Critical search/retrieval bug fixes + quick wins | **Completed** (d710f5f) |
| `02-implementation.md` | **Merged plan**: local-only ingest + cost optimizations (Phases 0-5) | **Active** |
| `03-test-validation.md` | Test validation (execute after implementation) | **Ready** |
| `decisions-and-progress.md` | This document | **Active** |

### Reference Documents (`reference/`)

| Plan | Purpose | Status |
|------|---------|--------|
| `reference/optimization-plan.md` | Cost analysis with before/after numbers | Reference (cost analysis) |
| `reference/estimate-full-results.md` | Raw output from full 6,289-email estimate | Reference |
| `reference/local-storage-sqlite-vs-files.md` | SQLite vs JSONL decision rationale | Reference (JSONL decided) |
| `reference/06b-review-findings.md` | Code-level review of 06b proposals, feasibility verification | Reference |
| `reference/06c-review-findings.md` | Code-level review of 06c speed proposals, bottleneck re-analysis | Reference |
| `reference/07a-review-findings.md` | Code-level review of 07a search/retrieval proposals, bug verification, tsvector decision | Reference |
| `reference/consolidation-proposal.md` | Plan consolidation rationale | Reference |

### Archived Plans (`archive/`)

| Plan | Purpose | Status |
|------|---------|--------|
| `archive/local-only-ingest-plan.md` | Detailed local-only ingest steps | Archived (superseded by `02-implementation.md`) |
