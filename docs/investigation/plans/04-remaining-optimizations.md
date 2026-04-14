---
purpose: Implementation plan for remaining pipeline and query-time optimizations
status: completed
completed: 2026-04-14
date: 2026-04-14
depends_on: [01-critical-fixes.md, 02-implementation.md, 03-test-validation.md]
execute_after: Plan 03 phases 1-4 complete (automated validation passed)
---

# Plan 04: Remaining Optimizations

## Context for New Agents

**What is MTSS?** Maritime Technical Support System -- a RAG pipeline that ingests maritime incident report emails (EML files with PDF/image attachments), chunks and embeds them, stores in a vector database (Supabase/pgvector), and provides a chat UI for querying incident history.

**Tech stack:** Python 3.12, FastAPI, Supabase (PostgreSQL + pgvector), LiteLLM, Cohere reranker, Vite + React. Build via `uv`. CLI: `uv run mtss <command>`.

**Project root:** `C:/Projects/GitHub/NCL/`

**What came before:** Plans 01-03 are complete. Critical reranker bug fixed, local-only ingest mode built (tiered parsers, image pre-filtering, JSONL output), test validation passed (322 tests, 87 chunks, 0 failures). Full ingest of 6,289 emails is pending.

**What this plan accomplishes:** Implements remaining optimizations that were deferred from earlier plans. Split into two phases: pre-ingest (affects stored data) and post-ingest (query-time only, safe to add anytime).

---

## Phase A: Pre-Ingest Optimizations

These should be implemented BEFORE the full production ingest because they affect stored data or ingest performance.

### A1: Batch Topic Embeddings (06c P7)

**Goal:** Reduce ingest time by batching topic embedding calls instead of one-at-a-time.

**Current state:**
- `src/mtss/processing/topics.py:346` — `TopicMatcher.get_or_create_topic()` calls `self.embeddings.generate_embedding(name)` per topic (single call)
- `src/mtss/processing/embeddings.py:129` — `EmbeddingGenerator.generate_embeddings_batch()` already exists but TopicMatcher doesn't use it

**Changes:**
- Add `async def get_or_create_topics_batch(self, names: List[str]) -> List[UUID]` to `TopicMatcher`
- Collect all new topic names, call `generate_embeddings_batch()` once, then insert all
- Call site: `src/mtss/ingest/pipeline.py` where topics are created during email processing

**Estimated impact:** ~800ms saved per email during topic-building phase. ~84 min total across 6,289 emails.

**Files:** `src/mtss/processing/topics.py`, `src/mtss/ingest/pipeline.py`

### A2: Vessel Aliases (06b P4-A)

**Goal:** Improve vessel matching by supporting alternative names (e.g., "MARAN CANOPUS" vs "M/V MARAN CANOPUS").

**Current state:**
- `src/mtss/models/vessel.py:11-28` — Vessel model has `name`, `vessel_type`, `vessel_class` only
- DB schema (`migrations/001_initial_schema.sql:267-276`) already has `aliases TEXT[]` and `imo TEXT` columns
- `src/mtss/processing/vessel_matcher.py:29-36` — `_build_lookup()` only indexes by `name`
- `data/vessel-list.csv` format: `NAME;TYPE;CLASS` (3 columns)

**Changes:**
1. Add `aliases: List[str] = []` and `imo: Optional[str] = None` to `Vessel` model
2. Update `VesselMatcher._build_lookup()` to also index each alias
3. Update CSV import in `cli/entities_cmd.py:102-194` to read optional 4th (IMO) and 5th (aliases, comma-separated) columns
4. Update `vessel-list.csv` with known aliases if any exist

**Estimated impact:** Better vessel matching for emails using abbreviated/formal names. No re-embedding needed — vessel IDs are metadata, not embedding text.

**Files:** `src/mtss/models/vessel.py`, `src/mtss/processing/vessel_matcher.py`, `src/mtss/cli/entities_cmd.py`, `data/vessel-list.csv`

---

## Phase B: Post-Ingest Optimizations (Query-Time)

These only affect query processing. Safe to implement before or after ingest — no stored data changes.

### B1: Cohere Rerank v3.5 (07a P4)

**Goal:** Upgrade reranker model for better relevance scoring.

**Current state:**
- `src/mtss/config.py:144-145` — default `rerank_model: str = "cohere/rerank-english-v3.0"`
- `src/mtss/rag/reranker.py:35` — reads model from config, passes to LiteLLM `rerank()`

**Changes:**
- Update default in `config.py` to `"cohere/rerank-english-v3.0"` -> `"cohere/rerank-v3.5"`
- Verify v3.5 is available via LiteLLM (check LiteLLM docs for supported Cohere models)
- If v3.5 not supported by LiteLLM, keep v3.0

**Estimated impact:** Better reranking quality. Config-only change, zero risk.

**Files:** `src/mtss/config.py`

### B2: Sibling Chunk Expansion (07a P3)

**Goal:** When a chunk scores high, also fetch adjacent chunks from the same document for better context.

**Current state:**
- `src/mtss/rag/retriever.py:65-70` — retrieves chunks independently, no expansion
- DB schema has `chunk_index INTEGER NOT NULL` on chunks table with index on `(document_id, chunk_index)`
- `match_chunks()` SQL function returns individual chunks only

**Changes:**
1. Add `expand_siblings: bool = False` and `sibling_window: int = 1` params to `Retriever.retrieve()`
2. After initial retrieval, for each result fetch `chunk_index ± sibling_window` from same `document_id`
3. Deduplicate (a sibling of result A might be result B itself)
4. Merge sibling content into the result's context for the LLM prompt
5. Add a repository method: `async def fetch_sibling_chunks(document_id, chunk_index, window)`

**Estimated impact:** Better context assembly for multi-paragraph answers. Medium effort.

**Files:** `src/mtss/rag/retriever.py`, `src/mtss/storage/repositories/search.py`, `src/mtss/rag/query_engine.py`

### B3: Hybrid Search / BM25 (07a P3)

**Goal:** Combine vector similarity with keyword matching for better recall on exact terms (vessel names, part numbers, IMO codes).

**Current state:**
- `src/mtss/storage/repositories/search.py:43` — pure vector search via `match_chunks()`
- No `tsvector` column on chunks table
- Decision D-11: tsvector can be added as PostgreSQL generated column, auto-populates on INSERT

**Changes:**
1. Add migration: `ALTER TABLE chunks ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;`
2. Add GIN index: `CREATE INDEX idx_chunks_content_tsv ON chunks USING gin(content_tsv);`
3. Update `match_chunks()` SQL function to accept optional `query_text` parameter
4. When `query_text` provided: compute hybrid score = `0.7 * vector_score + 0.3 * ts_rank(content_tsv, query)`
5. Update `SearchRepository.search_similar_chunks()` to pass query text
6. Update `Retriever.retrieve()` to pass raw query alongside embedding

**Estimated impact:** Better keyword recall for exact matches. Requires DB migration. Medium-high effort.

**Files:** `migrations/` (new migration), `src/mtss/storage/repositories/search.py`, `src/mtss/rag/retriever.py`

### B4: Topic Loosening (07a P3)

**Goal:** Automatically broaden search when strict topic matching returns too few results.

**Current state:**
- `src/mtss/rag/topic_filter.py:185-328` — `TopicFilter.analyze_query()` with strict matching
- `src/mtss/api/agent.py:407-412` — `skip_topic_filter` option exists, user can request broader search
- Topic matching threshold: 0.70 at query time

**Changes:**
1. In `TopicFilter.analyze_query()`, after initial match with threshold 0.70:
   - If matched topics return < 3 chunks, retry with threshold 0.55
   - If still < 3, set `should_skip_rag = False` with a note suggesting broader search
2. Add `topic_match_threshold_loose: float = 0.55` to config

**Estimated impact:** Fewer dead-end queries. Low effort.

**Files:** `src/mtss/rag/topic_filter.py`, `src/mtss/config.py`

### B5: Query Expansion (07a P5) — DEFERRED

**Goal:** Reformulate queries for better recall.

**Status:** Skip for now. The combination of B1-B4 should significantly improve recall. Query expansion adds LLM latency per query and complexity. Revisit after measuring B1-B4 impact.

---

## Implementation Order

| Step | Optimization | Phase | Effort | Blocks Ingest? |
|------|-------------|-------|--------|---------------|
| 1 | A1: Batch topic embeddings | Pre-ingest | Low (~20 lines) | No, but saves ~84 min |
| 2 | A2: Vessel aliases | Pre-ingest | Low (~30 lines) | No, metadata only |
| 3 | B1: Cohere rerank v3.5 | Post-ingest | Trivial (config) | No |
| 4 | B4: Topic loosening | Post-ingest | Low (~15 lines) | No |
| 5 | B2: Sibling chunk expansion | Post-ingest | Medium (~50 lines) | No |
| 6 | B3: Hybrid search / BM25 | Post-ingest | Medium-high (~80 lines + migration) | No |

**None of these block ingest.** A1 saves significant time if done first. A2 improves stored metadata quality. B1-B4 can be layered on anytime.

## Verification

### Phase A verification:
```bash
uv run pytest tests/ -x -v
# Run test subset ingest and verify topics are batched (check logs for batch calls)
# Verify vessel aliases match in test emails
```

### Phase B verification:
```bash
uv run pytest tests/ -x -v
# Start API server: uv run mtss serve
# Run test queries from Plan 03 Phase 6
# Verify reranker logs show v3.5 model
# Verify hybrid search finds exact keyword matches
```

---

## Implementation Summary

**Completed: 2026-04-14** | 344 tests passing (22 new) | 0 failures

### What was implemented:
- **A1:** `get_or_create_topics_batch()` in TopicMatcher — batch-embeds uncached topics in one API call (~84 min saved across 6,289 emails)
- **A2:** Vessel aliases — `aliases: List[str]` field, `_build_lookup()` indexes aliases, CSV import reads optional ALIASES column, fixed `_vessels_list()` crash (referenced removed `imo`/`dwt` fields)
- **B1:** Rerank model upgraded to `cohere/rerank-v3.5`
- **B2:** `context_summary` returned from `match_chunks()` and prepended to LLM context (replaces sibling expansion — low value with 1024-token chunks)
- **B3:** Hybrid search via `content_tsv` generated column + `ts_rank` blending (0.7 vector + 0.3 BM25)
- **B4:** Topic loosening — retries at threshold 0.55 when initial match (0.70) yields < 3 chunks

### Key files modified:
| File | Change |
|------|--------|
| `src/mtss/processing/topics.py` | `get_or_create_topics_batch()`, threshold param on find methods |
| `src/mtss/ingest/pipeline.py` | Switched to batch topic creation |
| `src/mtss/models/vessel.py` | Added `aliases` field |
| `src/mtss/processing/vessel_matcher.py` | Aliases in `_build_lookup()` |
| `src/mtss/cli/entities_cmd.py` | CSV aliases import, fixed `_vessels_list()` |
| `src/mtss/config.py` | rerank v3.5, hybrid_search_enabled, topic loosening thresholds |
| `src/mtss/rag/topic_filter.py` | Topic loosening retry logic |
| `src/mtss/rag/retriever.py` | Passes query_text for hybrid search, captures context_summary |
| `src/mtss/rag/citation_processor.py` | Prepends context_summary to LLM context |
| `src/mtss/storage/repositories/search.py` | query_text parameter for hybrid scoring |
| `src/mtss/models/chunk.py` | context_summary on RetrievalResult |
| `migrations/006_vessel_aliases.sql` | Re-adds aliases TEXT[] column |
| `migrations/007_hybrid_search.sql` | tsvector column, updated match_chunks() |

### Architectural decision:
Sibling chunk expansion (B2 original) was replaced with context_summary return. With 1024-token chunks, adjacent chunks add marginal value. The stored `context_summary` already provides document-level framing at zero extra query cost.
