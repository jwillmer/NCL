---
purpose: Review findings for 07a (Search & Retrieval Performance Optimization)
status: complete
date: 2026-04-13
source: 07a-response-search-optimization.md
---

# 07a Review Findings: Search & Retrieval Optimization

## P0: Reranker Bug Fix -- CONFIRMED, APPROVED

### Bug Verification

**Location:** `src/mtss/api/agent.py` line 461

```python
top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
```

When `rerank_enabled=True` (the default), `top_k` is set to `rerank_top_n` (default 5). The retriever then calls `search_similar_chunks(match_count=5)`, gets 5 results, and at `retriever.py` line 76:

```python
if use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n:
```

Since `len(retrieval_results) == 5 == effective_top_n`, the condition is `False` and reranking is skipped entirely. The reranker, documented to improve accuracy by 20-35%, is silently disabled in the production agent path.

**Note:** `RAGQueryEngine.query()` and `query_with_citations()` both default `top_k=20`, so the non-agent paths work correctly. But the agent is the primary production path.

### Fix

```python
# Before:
top_k=settings.rerank_top_n if settings.rerank_enabled else 10,

# After:
top_k=20,
```

The `use_rerank=settings.rerank_enabled` parameter on the next line already controls whether reranking happens. The `top_k` parameter should always request a large candidate pool for the vector search, regardless of reranking configuration. The value 20 is consistent with `RAGQueryEngine.query()` and `search_only()` defaults.

### Edge Cases

- When reranking is disabled (`rerank_enabled=False`), the old code used `top_k=10`. The fix changes this to `top_k=20`. This is fine because `Retriever.retrieve()` returns all 20 results when reranking is off -- slightly more results for the LLM context, but the agent's context builder already handles variable result counts. If we want to limit non-reranked results, a separate `search_top_k` config parameter could be added later.
- The `retriever.py` line 76 condition `len(retrieval_results) > effective_top_n` should arguably be `>=` (rerank even when we have exactly top_n+1 results), but this is a marginal case that does not block the fix.

**Verdict:** Fix is correct and safe. One-line change, no risk.

---

## P1: Quick Wins -- CONFIRMED, APPROVED

### P1a: Enriched Rerank Context (Proposal 7a)

**Current code in `reranker.py` line 64:**

```python
documents = [r.text for r in results]
```

**Proposed change:**

```python
documents = [
    f"[{r.email_subject or ''}] {r.source_title or ''}: {r.text}"
    for r in results
]
```

**Review:** The `RetrievalResult` model already carries `email_subject` and `source_title` fields, populated by `_convert_to_retrieval_results()` in `retriever.py` lines 106-107. These come from the `match_chunks` SQL function which joins the root document for email metadata. The data is already there -- this change just includes it in the reranker input.

**Edge cases:**
- When both `email_subject` and `source_title` are None, the prefix becomes `[] :` -- harmless but slightly noisy. Consider: `f"{r.email_subject + ': ' if r.email_subject else ''}{r.text}"` for a cleaner fallback, or simply skip the prefix when both are None.
- Added prefix text increases per-document input size by ~50-100 chars. Cohere rerank-english-v3.0 has a 4096 token limit per document. With 1024-token chunks (~4000 chars) plus a ~100 char prefix, we stay well under the limit.

**Verdict:** Approved. Small code change, meaningful quality improvement for incident disambiguation.

### P1b: Increase max_tokens from 1000 to 2000 (Proposal 8d)

**Current code in `query_engine.py` line 218:**

```python
max_tokens=1000,
```

**Review:** The RAG LLM model is `rag_llm_model` which defaults to `llm_model` (gpt-4o-mini, 128K context). The agent path uses gpt-4o (also 128K). With 8 chunks at ~1024 tokens each = ~8192 tokens input + system prompt (~500 tokens), the total input is ~9000 tokens. Adding 2000 output tokens = ~11,000 total, well under 128K.

**Cost impact:** Worst case, output doubles from 1000 to 2000 tokens. At gpt-4o-mini pricing ($0.60/MTok output), cost per query increases by ~$0.0006. At gpt-4o pricing ($10/MTok output), it increases by ~$0.01. Both are negligible at expected query volumes.

**Risk:** The max_tokens parameter is a ceiling, not a target. Short answers will still be short. Only complex multi-incident analysis answers benefit from the higher ceiling.

**Note:** This change affects the `query_engine.py` path (direct API queries). The agent path uses LangGraph streaming where max_tokens is set elsewhere (in the LLM configuration, not in this file). The query_engine.py fix is still valuable for the API path.

**Verdict:** Approved. No risk, negligible cost increase.

### P1c: Increase rerank_top_n from 5 to 8 (Proposal 8c)

**Current code in `config.py` line 147:**

```python
rerank_top_n: int = Field(default=5, validation_alias="RERANK_TOP_N")
```

**Review:** Config-only change. Affects how many results the reranker returns (and thus how many chunks the LLM receives as context).

**Cost impact:** 3 additional chunks at ~1024 tokens = ~3072 more input tokens per query. At gpt-4o pricing ($2.50/MTok input), that is ~$0.008/query. At gpt-4o-mini pricing ($0.15/MTok input), it is ~$0.0005/query. Both negligible.

**Quality impact:** More context for the LLM means better coverage for complex queries that span multiple documents or incidents. The reranker ensures the additional chunks are still relevant.

**Reranker cost impact:** Cohere rerank pricing is per-search (not per-document for <100 docs), so requesting top_n=8 vs top_n=5 costs the same.

**Verdict:** Approved. Config change only, negligible cost, meaningful quality improvement for complex queries.

---

## P2-P5: Deferred Proposals -- CONFIRMED, NO INGEST BLOCKERS

### P2: Increase top_k to 40 (Proposal 1)

Query-time only. No ingest impact. Can be implemented alongside or after P0 fix. Depends on P0 being fixed first (otherwise top_k is irrelevant since it gets overridden to rerank_top_n).

### P2: HNSW ef_search tuning (Proposal 9)

Query-time `SET LOCAL` statement. No schema change needed. The `ef_construction` increase (64 to 128-200) would benefit from a re-index, but `REINDEX CONCURRENTLY` can be done post-ingest without downtime.

### P2: Parallel embedding + topic extraction (Proposal 11)

Query-time code change in the agent. No ingest impact.

### P2: Rerank score floor (Proposal 2)

Query-time filter in `Reranker.rerank_results()`. No ingest impact.

### P3: Sibling chunk expansion (Proposal 8a)

Query-time DB lookup. No ingest impact. Chunks already have `chunk_index` and `document_id` for adjacency queries.

### P3: Parent email context for attachment chunks (Proposal 8b)

Query-time JOIN. No ingest impact. The `documents` table already has `parent_id` and `root_id` for hierarchy traversal.

### P3: Hybrid search / BM25 (Proposal 3) -- **SPECIAL CASE, SEE BELOW**

### P3: Topic filter loosening (Proposal 5)

Query-time logic change. No ingest impact.

### P4: Rerank model upgrade v3.5 (Proposal 7b)

Config change. No ingest impact.

### P5: Query expansion (Proposal 4), Caching (Proposal 10)

Both deferred. No ingest impact.

**Verdict:** All P2-P5 proposals are query-time only. None block ingest.

---

## P3 Special Case: tsvector for Hybrid Search (Proposal 3)

### The Question

Hybrid search requires a `tsvector` column on the `chunks` table. Should we add it during ingest to avoid re-processing later?

### Analysis

**Option A: Auto-generated tsvector via trigger (recommended)**

Add a generated column or trigger that populates `tsvector` from `content` automatically on INSERT/UPDATE:

```sql
-- Option A1: Generated column (PostgreSQL 12+, Supabase supports this)
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);
```

Or via trigger if generated columns cause issues with the vector extension:

```sql
-- Option A2: Trigger-based
ALTER TABLE chunks ADD COLUMN content_tsv tsvector;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);

CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
```

**Pros:** Zero code changes to ingest pipeline. The column is populated automatically by PostgreSQL whenever a chunk is inserted. Works for both Supabase (production) and any future PostgreSQL deployment.

**Cons:** Adds ~20-30% storage overhead for the tsvector column + GIN index. At production scale (~100K chunks), this is small (~50-100 MB).

**Option B: Populate during ingest code**

Add tsvector computation to `DocumentRepository.insert_chunks()` and include in the JSONL local output.

**Cons:** Requires code changes in insert_chunks, JSONL schema, and DB import script. Pre-computing tsvector in Python is error-prone (must match PostgreSQL's `to_tsvector('english', ...)` tokenization exactly).

### Decision: Option A1 (generated column)

- **For production (Supabase):** Add a migration with the generated column. No code changes. Chunks inserted during ingest will have `content_tsv` populated automatically.
- **For local-only ingest (JSONL):** No change needed. The JSONL stores raw `content` text. When chunks are imported to production, the generated column auto-populates.
- **Migration timing:** Can be applied now as part of the ingest schema setup, or later before hybrid search is implemented. Since it is a non-breaking `ALTER TABLE ADD COLUMN`, it can be done at any time without re-ingesting.

**Verdict:** The tsvector column does NOT require any ingest code changes. A simple migration adds it as a generated column. It can be added now (free, no risk) or later (equally easy). There is no reason to defer it, but also no urgency to include it before ingest.

**Recommendation:** Add the migration now since it is trivial and avoids forgetting. But do NOT block ingest on it.

---

## Proposal-by-Proposal Summary

| # | Proposal | Decision | Timing | Notes |
|---|----------|----------|--------|-------|
| P0 | Bug fix: agent top_k | **APPROVED** | Pre-ingest | One-line fix at agent.py:461, critical |
| P1 | 7a: Enriched rerank context | **APPROVED** | Pre-ingest | ~5 lines in reranker.py |
| P1 | 8d: max_tokens 1000 to 2000 | **APPROVED** | Pre-ingest | One-line in query_engine.py |
| P1 | 8c: rerank_top_n 5 to 8 | **APPROVED** | Pre-ingest | Config change in config.py |
| P2 | 1: top_k 20 to 40 | Deferred | Post-ingest | After P0 fix; config-only |
| P2 | 9: HNSW ef_search tuning | Deferred | Post-ingest | SET LOCAL, no schema change |
| P2 | 11: Parallel embed+topic | Deferred | Post-ingest | asyncio.gather refactor |
| P2 | 2: Rerank score floor | Deferred | Post-ingest | Filter in reranker |
| P3 | 8a: Sibling chunk expansion | Deferred | Post-ingest | New DB query logic |
| P3 | 8b: Parent email context | Deferred | Post-ingest | JOIN query |
| P3 | 3: Hybrid search (BM25) | Deferred | Post-ingest | tsvector migration can be added anytime (generated column) |
| P3 | 5: Topic filter loosening | Deferred | Post-ingest | Score boost vs hard filter |
| P4 | 7b: Rerank v3.5 upgrade | Deferred | Post-ingest | Config change, test first |
| P5 | 4: Query expansion | Deferred | Post-ingest | Skip unless hybrid insufficient |
| P5 | 10: Caching | Deferred | Post-ingest | Skip for now |
| -- | 6: Vessel pre-filtering | No change | N/A | Current approach is correct |
| -- | 11: Streaming | No change | N/A | Already well-structured |

## Implementation Notes for Approved Items

### P0 Fix (agent.py)

```python
# File: src/mtss/api/agent.py, line 461
# Change:
top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
# To:
top_k=20,
```

### P1a: Enriched rerank context (reranker.py)

```python
# File: src/mtss/rag/reranker.py, line 64
# Change:
documents = [r.text for r in results]
# To:
documents = []
for r in results:
    prefix = ""
    if r.email_subject:
        prefix += f"[{r.email_subject}] "
    if r.source_title:
        prefix += f"{r.source_title}: "
    documents.append(f"{prefix}{r.text}" if prefix else r.text)
```

### P1b: max_tokens (query_engine.py)

```python
# File: src/mtss/rag/query_engine.py, line 218
# Change:
max_tokens=1000,
# To:
max_tokens=2000,
```

### P1c: rerank_top_n (config.py)

```python
# File: src/mtss/config.py, line 147
# Change:
rerank_top_n: int = Field(default=5, validation_alias="RERANK_TOP_N")
# To:
rerank_top_n: int = Field(default=8, validation_alias="RERANK_TOP_N")
```

### Optional: tsvector migration

```sql
-- File: migrations/006_tsvector_hybrid_search.sql
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);
```
