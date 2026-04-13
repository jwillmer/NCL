---
purpose: Fix critical search/retrieval bugs and quick wins BEFORE any ingest work
priority: 1 (execute first)
source: 07a search optimization, 07b scenario analysis (completeness transparency)
status: completed
completed_date: 2026-04-13
commit: d710f5f (fixes) + 879b010 (test fix)
date: 2026-04-13
estimated_time: ~3-4 hours total
---

# Plan 01: Critical Fixes & Search Quick Wins

**Execute BEFORE all other plans.**

## Context for New Agents

**What is MTSS?** Maritime Technical Support System -- a RAG (Retrieval-Augmented Generation) pipeline that ingests maritime incident report emails (EML files with PDF/image attachments), chunks and embeds them, stores them in a vector database, and provides a chat UI for searching and querying incident history.

**Tech stack:** Python 3.12, FastAPI, Supabase (PostgreSQL + pgvector), LiteLLM (wraps OpenAI APIs), Cohere reranker, Vite + React chat frontend.

**Project root:** `C:/Projects/GitHub/NCL/`

**Key directories:**
- `src/mtss/` -- main application code
  - `api/agent.py` -- LLM agent with tool-calling search node
  - `rag/retriever.py` -- vector search + reranking
  - `rag/reranker.py` -- Cohere cross-encoder reranking
  - `rag/query_engine.py` -- standalone RAG query engine
  - `config.py` -- Pydantic settings (env-based configuration)
  - `storage/repositories/search.py` -- database search queries (pgvector HNSW)
- `tests/` -- pytest test suite
- `data/emails/` -- 6,289 source EML files
- `docs/investigation/plans/` -- all execution plans and decision tracking

**What this plan accomplishes:** Fixes a critical production bug (reranker silently disabled) and applies low-effort search quality improvements. These are all query-side fixes -- they do not touch the ingest pipeline and can be deployed independently.

**Prerequisites / what came before:** The codebase has a working ingest pipeline and search system, but investigation documents `07a` and `07b` uncovered bugs and improvement opportunities. This plan extracts the urgent fixes. See `decisions-and-progress.md` for the full decision trail (D-11, D-12).

**Key decisions affecting this plan:**
- D-11: Reranker bug confirmed (P0), quick wins approved (P1), deferred items identified (P2+)
- D-12: Completeness transparency (Fix 2.5) pulled from 07b scenario analysis
- This plan modifies `config.py` (adds `retrieval_top_k`, `rerank_score_floor`); the implementation plan also modifies `config.py` (different fields) -- changes are additive and non-conflicting

**What NOT to do:** Do not change chunk size, embedding dimensions, or any ingest pipeline code. Those belong to `02-implementation.md`.

---

## Overview

The 07a investigation found a critical bug where the reranker is silently disabled in
production, plus several low-effort improvements that materially improve search quality.
These must be fixed before any ingest work because:

1. The reranker bug means production search is degraded RIGHT NOW (only vector similarity, no cross-encoder reranking)
2. The quick wins are trivial config/code changes that improve existing functionality
3. None of these depend on ingest pipeline changes
4. Fix 2.5 (from 07b) is a low-effort transparency improvement that fits naturally here

## Prerequisites

- Access to the codebase at `src/mtss/`
- Python environment with project dependencies installed
- Access to a running instance (local or staging) for verification
- Database access for HNSW tuning verification (Phase 2)

> **Note:** This plan modifies `src/mtss/config.py` (adds `retrieval_top_k` and `rerank_score_floor`).
> `02-implementation.md` also modifies `config.py` (changes `chunk_size_tokens`, `chunk_overlap_tokens`,
> `embedding_dimensions`, `max_concurrent_files`). The changes are additive and do not conflict.

---

## Phase 0: Critical Bug Fix (~5 minutes)

### Fix 0.1: Restore two-stage retrieval (reranker silently disabled)

**Severity:** P0 -- production search is degraded  
**Root cause:** `agent.py` passes `top_k=settings.rerank_top_n` (which is 5) to `search_only()`. The retriever then fetches 5 candidates. In `retriever.py` line 76, reranking is skipped when `len(retrieval_results) <= effective_top_n` (also 5). So the reranker is never called despite being enabled.

**File:** `src/mtss/api/agent.py` line 459-465  
**Current code:**
```python
# Get raw search results (no LLM generation)
retrieval_results = await engine.search_only(
    question=question,
    top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
    use_rerank=settings.rerank_enabled,
    metadata_filter=metadata_filter,
    on_progress=on_progress,
)
```

**File:** `src/mtss/config.py`  
**Add new config field** after `rerank_top_n` (line 147):
```python
retrieval_top_k: int = Field(default=20, validation_alias="RETRIEVAL_TOP_K")
```

**File:** `src/mtss/api/agent.py` line 461  
**New code:**
```python
# Get raw search results (no LLM generation)
retrieval_results = await engine.search_only(
    question=question,
    top_k=settings.retrieval_top_k,
    use_rerank=settings.rerank_enabled,
    metadata_filter=metadata_filter,
    on_progress=on_progress,
)
```

**Why this is correct:**
- `retrieval_top_k` (20) controls how many vector-search candidates to fetch
- `rerank_top_n` (5, later 8) controls how many the reranker returns
- These are fundamentally different parameters: candidate pool vs. final output
- With 20 candidates and top_n=5, `len(results) > effective_top_n` is True, so reranking proceeds
- The old `else 10` fallback for disabled reranking is also absorbed: when reranking is off, the retriever returns all `top_k` results (truncated by the disabled reranker's `results[:top_n]`)
- A dedicated config value (`RETRIEVAL_TOP_K`) is better than a hardcoded 20 because it can be tuned per deployment without code changes

**Test:**
1. Run existing test: `pytest tests/test_reranker.py -v` -- should still pass
2. Manual verification: add temporary logging to `retriever.py` line 76:
   ```python
   logger.info("Rerank check: %d results, effective_top_n=%d, will_rerank=%s",
               len(retrieval_results), effective_top_n,
               use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n)
   ```
3. Run a search query and confirm log shows `will_rerank=True`
4. Remove temporary logging

**Risk:** Low. This restores intended behavior. The reranker was already wired up correctly in `retriever.py` and `reranker.py`; only the caller was passing the wrong `top_k`.

---

## Phase 1: Quick Wins (~30 minutes)

### Fix 1.1: Enriched rerank context

**Severity:** P1 -- improves reranker accuracy  
**Rationale:** The reranker currently only sees raw chunk text. Prepending email subject and source title gives the cross-encoder more semantic signal for relevance scoring, especially for short chunks.

**File:** `src/mtss/rag/reranker.py` lines 63-64  
**Current code:**
```python
        # Extract document texts for reranking
        documents = [r.text for r in results]
```

**New code:**
```python
        # Build enriched documents for reranking (subject + title provide context)
        documents = []
        for r in results:
            prefix_parts = []
            if r.email_subject:
                prefix_parts.append(r.email_subject)
            if r.source_title and r.source_title != r.email_subject:
                prefix_parts.append(r.source_title)
            prefix = " | ".join(prefix_parts)
            documents.append(f"{prefix}\n{r.text}" if prefix else r.text)
```

**Why this is correct:**
- The LiteLLM `rerank()` function accepts `documents` as a list of strings (plain text). Each string is a document to score against the query. There is no structured format requirement -- the reranker model reads the full string. Adding a subject prefix is a standard RAG technique.
- The `r.email_subject != r.source_title` check avoids redundant duplication (emails often have `source_title == email_subject`).
- The `RetrievalResult` dataclass already has `email_subject` and `source_title` fields populated from the database (see `retriever.py` lines 112-113 in `_convert_to_retrieval_results`).

**Test:**
1. Update `test_rerank_results_extracts_text` in `tests/test_reranker.py` to verify enriched documents:
   ```python
   # Verify rerank was called with enriched documents
   call_kwargs = mock_rerank_fn.call_args.kwargs
   # First result has email_subject="Project Update", source_title="Project Update" (same, no dup)
   assert call_kwargs["documents"][0].startswith("Project Update\n")
   # Third result has email_subject="Project Update", source_title="Budget Report" (different)
   assert "Project Update | Budget Report" in call_kwargs["documents"][2]
   ```
2. Run full reranker test suite: `pytest tests/test_reranker.py -v`

**Risk:** Low. The reranker model handles variable-length input. A few extra tokens of context won't affect latency or cost meaningfully. Cohere rerank models are trained on documents with titles/headers.

### Fix 1.2: Increase max_tokens from 1000 to 2000

**Severity:** P1 -- prevents answer truncation  
**Rationale:** 1000 tokens is tight for multi-source answers with citations. Maritime incident responses often need to synthesize multiple emails and attachments.

**File:** `src/mtss/rag/query_engine.py` line 216  
**Current code:**
```python
            max_tokens=1000,
```

**New code:**
```python
            max_tokens=2000,
```

**Why this is correct:**
- This only affects the `_generate_answer_with_citations` method in `RAGQueryEngine`, which is used by the standalone `query` and `query_with_citations` endpoints (not the agent path).
- The agent path (`agent.py`) generates answers through the LLM chat node, which has its own token settings.
- The LLM model (`rag_llm_model`, defaults to `gpt-4o-mini`) has a 16K output token limit, so 2000 is well within bounds.
- Context window: the system prompt (~300 tokens) + user context (8 results * ~600 tokens each = ~4800) + question (~50 tokens) = ~5150 input tokens. With a 128K context model, 2000 output tokens is safely within limits.

**Test:**
1. Run a query that previously produced truncated output (look for answers ending mid-sentence).
2. Verify response completes naturally.

**Risk:** Minimal. Doubles potential output cost per query (~$0.001 extra at GPT-4o-mini rates). No functional risk.

### Fix 1.3: Increase rerank_top_n from 5 to 8

**Severity:** P1 -- more diverse search results  
**Rationale:** With the reranker restored (Fix 0.1), 5 final results is conservative. 8 gives the LLM more source material without significantly increasing context length.

**File:** `src/mtss/config.py` line 147  
**Current code:**
```python
    rerank_top_n: int = Field(default=5, validation_alias="RERANK_TOP_N")
```

**New code:**
```python
    rerank_top_n: int = Field(default=8, validation_alias="RERANK_TOP_N")
```

**Why this is correct:**
- `rerank_top_n` controls the number of results returned after cross-encoder reranking.
- 8 results * ~600 tokens = ~4800 tokens of context, well within any model's context window.
- The Cohere rerank API handles any `top_n` up to the number of input documents.
- This affects both the standalone RAG query engine and the agent path (both use the reranker).

**Test:**
1. Update `tests/test_reranker.py` `mock_settings` fixture: change `settings.rerank_top_n = 3` to `settings.rerank_top_n = 3` (keep test value at 3 since it tests the mechanism, not the production default).
2. Run: `pytest tests/test_reranker.py -v`
3. No test changes needed -- tests use their own mock settings.

**Risk:** Low. Marginal increase in context tokens (~1800 extra). Cohere rerank cost is per-search, not per-result.

---

## Phase 2: Short-term Improvements (~2-3 hours)

### Fix 2.1: Increase candidate pool (top_k 20 to 40)

**Severity:** P2 -- broader candidate pool for reranker  
**Rationale:** With the reranker restored, feeding it more candidates improves its ability to find the most relevant results. 40 is a good balance between recall and latency.

**File:** `src/mtss/config.py`  
**Change the default** added in Fix 0.1:
```python
retrieval_top_k: int = Field(default=40, validation_alias="RETRIEVAL_TOP_K")
```

**Why this is correct:**
- More candidates = better reranker coverage. The reranker can find good results that vector search ranked lower.
- Vector search latency scales sub-linearly with HNSW. 40 vs 20 adds ~2-5ms.
- Cohere rerank cost is per-query (not per-document), so 40 documents costs the same as 20.
- The `match_chunks` SQL function's `match_count` parameter directly controls the LIMIT clause.

**Note:** Implement this AFTER Fix 0.1 is verified working with top_k=20. If 20 works well, bump to 40. This is a config-only change.

**Test:**
1. Run a search query, verify 40 candidates are fetched (add logging to retriever if needed).
2. Check latency: should be <50ms increase over top_k=20.

**Risk:** Low. Slightly more data transferred from DB, but chunks are small. Reranker handles up to 1000 documents per Cohere docs.

### Fix 2.2: HNSW ef_search tuning

**Severity:** P2 -- improves vector search recall  
**Rationale:** The default `ef_search` in pgvector is 40. Increasing to 100 explores more candidates during the HNSW graph traversal, improving recall at the cost of ~10-20ms latency.

**File:** `src/mtss/storage/repositories/search.py` lines 32-44  
**Current code:**
```python
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            rows = await conn.fetch(
                """
                SELECT * FROM match_chunks($1, $2, $3, $4::jsonb)
                """,
                query_embedding,
                match_threshold,
                match_count,
                filter_json,
            )
```

**New code:**
```python
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Increase HNSW search quality for this query
            # Default ef_search=40; 100 improves recall by ~5-10% with ~10-20ms cost
            await conn.execute("SET LOCAL hnsw.ef_search = 100")
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            rows = await conn.fetch(
                """
                SELECT * FROM match_chunks($1, $2, $3, $4::jsonb)
                """,
                query_embedding,
                match_threshold,
                match_count,
                filter_json,
            )
```

**Why this is correct:**
- `SET LOCAL` only applies within the current transaction. Since `asyncpg` runs each `fetch` in an implicit transaction by default, this needs to be wrapped in an explicit transaction to take effect. **REVIEW FINDING: `SET LOCAL` requires an explicit transaction block.** See updated code below.

**Revised new code (with explicit transaction):**
```python
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            # Use explicit transaction so SET LOCAL applies to the query
            async with conn.transaction():
                # Increase HNSW search quality for this query
                # Default ef_search=40; 100 improves recall by ~5-10% with ~10-20ms cost
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
                rows = await conn.fetch(
                    """
                    SELECT * FROM match_chunks($1, $2, $3, $4::jsonb)
                    """,
                    query_embedding,
                    match_threshold,
                    match_count,
                    filter_json,
                )
```

**Compatibility check:**
- The HNSW index was created in `migrations/001_initial_schema.sql` line 185-187 with `(m = 16, ef_construction = 64)`.
- `hnsw.ef_search` is a GUC parameter in pgvector (available since pgvector 0.5.0). Supabase uses pgvector >= 0.7.0.
- `SET LOCAL` is standard PostgreSQL and works with `asyncpg`.

**Test:**
1. Run a search query and verify it returns results (functional correctness).
2. Compare result quality: run the same query with ef_search=40 (default) and ef_search=100, compare overlap in top-8 results.
3. Measure latency impact: should be <20ms increase.

**Risk:** Low. `SET LOCAL` is transaction-scoped, so it cannot affect other connections. The only risk is the explicit transaction wrapper changing error handling behavior -- but `conn.transaction()` is the standard asyncpg pattern.

### Fix 2.3: Parallel query embedding + topic extraction

**Severity:** P2 -- reduces search latency by ~200-500ms  
**Rationale:** Currently in `agent.py`, topic extraction runs first (LLM call, ~500ms), then `search_only` runs (which embeds the query, ~200ms, then searches). The query embedding and topic extraction are independent and can run concurrently.

**File:** `src/mtss/api/agent.py` lines 406-465

This is a structural change. The current flow is:
```
1. topic_filter.analyze_query(question)    # LLM call (~500ms) + embedding (~200ms)
2. engine.search_only(question, ...)       # embedding (~200ms) + DB search (~50ms)
```

The problem is that `analyze_query` internally generates embeddings for topic matching, and `search_only` generates a separate embedding for the query. These are different embeddings (topic name vs full query text), so they can't be shared. But the two operations are independent.

**Proposed approach:** Use `asyncio.gather` to run topic filtering and query embedding concurrently, then run the DB search with the pre-computed embedding.

However, this requires refactoring `Retriever.retrieve()` to accept a pre-computed embedding, or splitting it into `embed` + `search` steps.

**Current `retriever.py` flow:**
```python
async def retrieve(self, query, ...):
    query_embedding = await self.embeddings.generate_embedding(query)  # Step A
    results = await self.db.search_similar_chunks(query_embedding, ...)  # Step B
    # ... rerank ...
```

**Revised approach -- split retrieve into two phases:**

**File:** `src/mtss/rag/retriever.py`  
Add a method to generate embedding separately:
```python
    async def embed_query(self, query: str) -> list[float]:
        """Generate query embedding for later use in search.

        Use this when you want to run embedding concurrently with other async work.
        """
        return await self.embeddings.generate_embedding(query)

    async def retrieve_with_embedding(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
        similarity_threshold: float = 0.3,
        rerank_top_n: int | None = None,
        use_rerank: bool = True,
        metadata_filter: dict | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[RetrievalResult]:
        """Search with pre-computed embedding, then rerank.

        Same as retrieve() but skips the embedding step.
        """
        if on_progress:
            await on_progress("Searching documents")

        results = await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=similarity_threshold,
            match_count=top_k,
            metadata_filter=metadata_filter,
        )

        if not results:
            return []

        retrieval_results = _convert_to_retrieval_results(results)

        # Stage 2: Rerank if enabled (skip if too few results)
        effective_top_n = rerank_top_n or self.reranker.top_n
        if use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n:
            if on_progress:
                await on_progress("Reranking results...")
            retrieval_results = self.reranker.rerank_results(
                query=query, results=retrieval_results, top_n=rerank_top_n
            )

        return retrieval_results
```

**File:** `src/mtss/rag/query_engine.py` `search_only` method  
Add a `query_embedding` parameter:
```python
    async def search_only(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.3,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
        vessel_id: Optional[str] = None,
        vessel_type: Optional[str] = None,
        vessel_class: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
```
And in the body:
```python
        if query_embedding is not None:
            return await self.retriever.retrieve_with_embedding(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                rerank_top_n=rerank_top_n,
                use_rerank=use_rerank,
                metadata_filter=metadata_filter,
                on_progress=on_progress,
            )
        return await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            rerank_top_n=rerank_top_n,
            use_rerank=use_rerank,
            metadata_filter=metadata_filter,
            on_progress=on_progress,
        )
```

**File:** `src/mtss/api/agent.py` (in `search_node`)  
Replace the sequential flow with concurrent execution:
```python
        import asyncio

        if skip_filter:
            filter_result = TopicFilterResult()
            query_embedding = None
        else:
            # Run topic analysis and query embedding concurrently
            await on_progress("Analyzing query")
            topic_filter = TopicFilter(
                topic_extractor=TopicExtractor(),
                topic_matcher=TopicMatcher(engine.db, engine.embeddings),
                db=engine.db,
            )
            filter_task = topic_filter.analyze_query(question, vessel_filter)
            embed_task = engine.retriever.embed_query(question)
            filter_result, query_embedding = await asyncio.gather(
                filter_task, embed_task
            )

        # ... (early return checks remain the same) ...

        # Get raw search results (pass pre-computed embedding)
        retrieval_results = await engine.search_only(
            question=question,
            top_k=settings.retrieval_top_k,
            use_rerank=settings.rerank_enabled,
            metadata_filter=metadata_filter,
            on_progress=on_progress,
            query_embedding=query_embedding,
        )
```

**Why this is correct:**
- `embed_query` and `analyze_query` are fully independent: they use different data (full query text vs extracted topic names), different models (embedding model vs LLM), and no shared mutable state.
- The `EmbeddingGenerator` is stateless (calls OpenAI API).
- The `TopicExtractor` and `TopicMatcher` don't modify engine state.
- When `skip_filter=True`, we pass `query_embedding=None` which falls back to the standard `retrieve()` path (embedding happens inside).

**Test:**
1. Run search queries with and without `skip_topic_filter=True`.
2. Verify identical results (correctness).
3. Measure latency improvement (expect ~200-400ms savings).

**Risk:** Medium. This adds concurrency to the search path. If either task fails, `asyncio.gather` will raise the first exception. Consider wrapping in try/except or using `return_exceptions=True` with explicit error checking. However, both tasks already have their own error handling (topic extraction returns empty result on failure, embedding raises on API error which should propagate).

### Fix 2.4: Rerank score floor

**Severity:** P2 -- filters out irrelevant results  
**Rationale:** After reranking, some results may have very low relevance scores. Passing these to the LLM dilutes context quality and may cause hallucination.

**File:** `src/mtss/rag/reranker.py` after line 79  
**Current code (end of `rerank_results`):**
```python
        # Reorder results by rerank scores
        reranked = []
        for item in response.results:
            result = results[item.index]
            result.rerank_score = item.relevance_score
            reranked.append(result)

        return reranked
```

**New code:**
```python
        # Reorder results by rerank scores
        reranked = []
        for item in response.results:
            result = results[item.index]
            result.rerank_score = item.relevance_score
            reranked.append(result)

        # Filter out results below score floor (but keep at least 1 result)
        min_score = 0.2
        filtered = [r for r in reranked if r.rerank_score >= min_score]
        return filtered if filtered else reranked[:1]
```

**File:** `src/mtss/config.py`  
Optionally add a config field (for tuning without code changes):
```python
rerank_score_floor: float = Field(default=0.2, validation_alias="RERANK_SCORE_FLOOR")
```
And reference it in the reranker's `__init__`:
```python
self.score_floor = settings.rerank_score_floor
```

**Why this is correct:**
- The `reranked[:1]` fallback ensures we never return zero results (the LLM always has something to work with, even if poorly scored).
- Score 0.2 is a reasonable floor for Cohere's rerank models (scores range 0.0 to 1.0, with 0.2 indicating weak relevance).
- This is applied after reranking, so it doesn't affect the candidate pool.

**Test:**
1. Add test to `tests/test_reranker.py`:
   ```python
   def test_rerank_score_floor(self, mock_settings, sample_results):
       """Test that low-scoring results are filtered out."""
       mock_response = MagicMock()
       # All results score below 0.2 except one
       results_data = [
           MagicMock(index=0, relevance_score=0.05),
           MagicMock(index=1, relevance_score=0.45),
           MagicMock(index=2, relevance_score=0.10),
       ]
       mock_response.results = results_data

       with patch("mtss.rag.reranker.get_settings", return_value=mock_settings):
           with patch("mtss.rag.reranker.rerank", return_value=mock_response):
               reranker = Reranker()
               results = reranker.rerank_results(
                   query="test", results=sample_results,
               )
               # Only result with score >= 0.2 should remain
               assert len(results) == 1
               assert results[0].rerank_score == 0.45
   ```

**Risk:** Medium. If all results have low rerank scores (e.g., the query is unusual), the floor might filter out everything. The `reranked[:1]` fallback mitigates this -- the user always gets at least one result. However, that single result might be irrelevant. An alternative is to log a warning when all results are below the floor, so operators can tune the threshold.

### Fix 2.5: Result completeness transparency

**Severity:** P2 -- improves LLM answer trustworthiness  
**Source:** 07b scenario analysis  
**Rationale:** The LLM doesn't know how many total chunks matched the query. Surfacing this count helps it calibrate confidence and say "there are 47 relevant documents; I'm showing the top 8" rather than implying it saw everything.

**File:** `src/mtss/api/agent.py` lines 500-509  
**Current code:**
```python
        # Return context to agent (agent will generate answer with citations)
        tool_response = ToolMessage(
            content=json.dumps({
                "context": enhanced_context,
                "available_chunk_ids": list(citation_map.keys()),
                "incident_count": len(retrieval_results),
                "unique_incidents": len(incident_groups),
            }),
            tool_call_id=tool_call["id"],
        )
```

**New code:**
```python
        # Return context to agent (agent will generate answer with citations)
        tool_response = ToolMessage(
            content=json.dumps({
                "context": enhanced_context,
                "available_chunk_ids": list(citation_map.keys()),
                "incident_count": len(retrieval_results),
                "unique_incidents": len(incident_groups),
                "total_candidate_count": candidate_count,
                "note": (
                    f"Showing top {len(retrieval_results)} results out of "
                    f"{candidate_count} candidates. If the answer seems incomplete, "
                    f"the user can ask for a broader search."
                ) if candidate_count > len(retrieval_results) else None,
            }),
            tool_call_id=tool_call["id"],
        )
```

**To get `candidate_count`**, modify the retriever to return it. The simplest approach is to capture it from the retriever's vector search step.

**File:** `src/mtss/rag/retriever.py`  
Change `retrieve` to also return candidate count:

Actually, this is simpler than refactoring the retriever return type. We can get the count from the `TopicFilterResult` which already has `total_chunk_count`:

```python
        candidate_count = filter_result.total_chunk_count or len(retrieval_results)
```

Add this line before building the tool response (after `enhanced_context` is built).

**Why this is correct:**
- `filter_result.total_chunk_count` already counts all chunks matching the topic filter (computed in `topic_filter.py` line 264). This is the total pool before vector search narrows it down.
- When no topic filter is active (broad search), `total_chunk_count` is 0, so we fall back to `len(retrieval_results)` (the note won't display since `candidate_count == len(retrieval_results)`).
- The `note` field is only included when there are more candidates than shown, giving the LLM appropriate context.
- The JSON structure just gains two new optional fields. The agent's system prompt reads the full tool response, so it will naturally incorporate this information.

**Test:**
1. Run a topic-filtered query and verify `total_candidate_count` appears in the tool response.
2. Run a broad query (no topic match) and verify the note is absent.
3. Verify the agent's answer references the breadth of available data when appropriate.

**Risk:** Very low. Adding fields to a JSON response is backward-compatible. The LLM handles new fields gracefully.

---

## Verification Checklist

After all fixes are applied, run through this checklist:

- [ ] `pytest tests/test_reranker.py -v` -- all tests pass
- [ ] `pytest tests/ -v` -- full test suite passes (no regressions)
- [ ] Manual search test: query returns reranked results (check `rerank_score` is populated on results)
- [ ] Manual search test: query returns 8 results (not 5)
- [ ] Verify `RETRIEVAL_TOP_K` can be overridden via `.env`
- [ ] Verify `RERANK_SCORE_FLOOR` can be overridden via `.env`
- [ ] Check latency: search queries complete in <3 seconds
- [ ] Check that `ef_search=100` doesn't cause errors with the database
- [ ] Verify parallel embedding doesn't break topic filtering
- [ ] Verify completeness note appears in agent tool responses

---

## Rollback Plan

All changes are isolated and independently reversible:

1. **Fix 0.1 (top_k):** Revert `agent.py` line 461 to use `settings.rerank_top_n`. Remove `retrieval_top_k` from `config.py`.
2. **Fix 1.1 (enriched rerank):** Revert `reranker.py` to `documents = [r.text for r in results]`.
3. **Fix 1.2 (max_tokens):** Change `max_tokens=2000` back to `1000`.
4. **Fix 1.3 (rerank_top_n):** Change default back to `5`.
5. **Fix 2.1 (top_k=40):** Change `retrieval_top_k` default back to `20`.
6. **Fix 2.2 (ef_search):** Remove the `SET LOCAL` and `conn.transaction()` wrapper.
7. **Fix 2.3 (parallel):** Revert to sequential flow in `agent.py`.
8. **Fix 2.4 (score floor):** Remove the filtering code after reranking.
9. **Fix 2.5 (completeness):** Remove the `total_candidate_count` and `note` fields.

Each fix is in a separate file or a clearly delineated code section. If any fix causes issues, it can be reverted independently without affecting the others.

---

## Review Findings

The following issues were identified during review and resolved in the plan:

### RF-1: `SET LOCAL` requires explicit transaction (Fix 2.2)
**Issue:** The initial plan used `SET LOCAL hnsw.ef_search = 100` without an explicit transaction. In asyncpg, each `fetch` call runs in its own implicit transaction, so `SET LOCAL` in a separate `execute` call would have no effect on the subsequent `fetch`.  
**Resolution:** Wrapped both the `SET LOCAL` and `fetch` in an explicit `async with conn.transaction()` block. This is the standard asyncpg pattern and ensures the GUC setting applies to the query.

### RF-2: Enriched rerank context format (Fix 1.1)
**Issue:** Need to verify that the LiteLLM `rerank()` function accepts plain text strings with embedded newlines.  
**Finding:** Confirmed. LiteLLM's `rerank()` passes `documents` as-is to the provider API (Cohere, Azure AI, etc.). Cohere's rerank API accepts arbitrary text strings. Newlines in documents are standard and handled correctly.

### RF-3: max_tokens and model context window (Fix 1.2)
**Issue:** Does increasing max_tokens risk exceeding the model context window?  
**Finding:** No. The RAG context is 8 results * ~600 tokens = ~4800 tokens. System prompt ~300 tokens. Total input ~5100 tokens. With GPT-4o-mini's 128K context window, 2000 output tokens is far within limits. Even with GPT-4o (128K) or GPT-4.1-mini (1M context), this is safe.

### RF-4: Rerank score floor edge case (Fix 2.4)
**Issue:** What if ALL results have low scores? The `reranked[:1]` fallback returns one result, but it might be irrelevant.  
**Resolution:** The fallback is acceptable -- returning one low-confidence result is better than returning nothing. Added logging recommendation: when all results fall below the floor, log a warning so operators can investigate and tune the threshold. The config-based `RERANK_SCORE_FLOOR` allows per-deployment tuning.

### RF-5: Parallel embedding exception handling (Fix 2.3)
**Issue:** If `embed_query` fails (API error), `asyncio.gather` will cancel the other task and raise.  
**Finding:** This is acceptable behavior. If we can't embed the query, search cannot proceed. The existing try/except in `search_node` (line 523) catches all exceptions and returns an error message. No additional handling needed.

### RF-6: `retrieve_with_embedding` code duplication (Fix 2.3)
**Issue:** `retrieve_with_embedding` duplicates most of `retrieve`.  
**Resolution:** This is intentional to avoid changing the existing `retrieve` method signature, which is used by `RAGQueryEngine.query()` and `query_with_citations()`. A cleaner refactor would make `retrieve` call `retrieve_with_embedding` internally, but that's a larger change. The duplication is 15 lines and acceptable for this fix. Can be refactored later.

### RF-7: `total_chunk_count` for broad searches (Fix 2.5)
**Issue:** When no topic filter is active, `filter_result.total_chunk_count` is 0 (TopicFilterResult default). The note won't display.  
**Finding:** This is acceptable for now. Broad searches (no topic detected) already return the full `top_k` candidates. The completeness note is most valuable for topic-filtered searches where the user might not realize there's more data. For broad searches, the LLM naturally handles the "showing top N" framing.

### RF-8: Changing top_k in agent.py -- side effects (Fix 0.1)
**Issue:** Does changing the `top_k` parameter break anything else?  
**Finding:** No. The `top_k` parameter flows through `search_only` -> `retrieve` -> `search_similar_chunks` (DB query). It only controls the SQL `LIMIT` clause. No other code depends on the specific value of `top_k`. The reranker independently uses `rerank_top_n` to control its output count.

### RF-9: Adding fields to citation_processor context (Fix 2.5)
**Issue:** Does adding `total_candidate_count` and `note` to the tool response break citation processing?  
**Finding:** No. The `tool_response` is a JSON blob consumed by the LLM agent (via `ToolMessage`). Citation processing happens separately via `citation_map` in the state. The two systems don't interact -- `CitationProcessor` only processes the LLM's *output* (looking for `[C:chunk_id]` markers), not the tool response input.
