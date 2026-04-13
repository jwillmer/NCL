---
title: "Search & Retrieval Performance Optimization Proposals"
status: investigation
created: 2026-04-13
scope: response quality, latency, cost
affects: src/mtss/rag/, src/mtss/api/agent.py, migrations/
---

# Search & Retrieval Performance Optimization

Investigation of improvements to the RAG search and retrieval pipeline, covering search strategy, filtering, reranking, context assembly, and infrastructure performance.

## Current Pipeline Summary

```
User query
  -> [TopicFilter] LLM extracts topics from query, matches to DB topics (semantic)
  -> [TopicFilter] Pre-check: skip RAG if no chunks exist for matched topics
  -> [Retriever]   Embed query (text-embedding-3-small, 1536d)
  -> [match_chunks] Vector search (cosine similarity, HNSW index)
                    Metadata filter: topic_ids (OR), vessel_ids/types/classes (AND)
                    threshold=0.3, top_k=20 (query_engine) or top_k=5 (agent)
  -> [Reranker]    Cohere rerank-english-v3.0, top_n=5
  -> [Agent]       Build context with citation headers, incident grouping
  -> [LLM]         gpt-4o generates answer with [C:chunk_id] citations
  -> [CitationProcessor] Validate citations, retry if >50% invalid, format <cite> tags
```

---

## CRITICAL BUG: Agent Bypasses Two-Stage Retrieval

**File:** `src/mtss/api/agent.py`, line 460

```python
top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
```

When reranking is enabled (the default), the agent sets `top_k` to `rerank_top_n` (default 5), meaning it retrieves exactly 5 candidates and then skips reranking because `len(retrieval_results) <= effective_top_n` (see `retriever.py` line 76). The two-stage retrieval (retrieve 20, rerank to 5) never actually happens via the agent path.

The `RAGQueryEngine.query()` and `query_with_citations()` methods correctly default to `top_k=20`, but the agent -- which is the primary production path -- does not use them.

**Impact:** High. The reranker, documented to improve accuracy by 20-35%, is silently disabled in production.

**Fix:** Change the agent's `search_only` call to use the intended two-stage values:
```python
top_k=20,  # Candidate pool for reranking
use_rerank=settings.rerank_enabled,
```

This fix is a prerequisite for all other proposals -- without it, reranking improvements and candidate pool tuning are irrelevant.

---

## Proposal 1: Increase Candidate Pool (top_k)

### Current State
- `top_k=20` in `query_engine.py` (unused by agent)
- `top_k=5` in `agent.py` (effective in production, bypasses reranking)
- `rerank_top_n=5` (final output count)

### Recommendation
After fixing the bug above, increase `top_k` from 20 to 40-50.

**Rationale:** With ~100K+ chunks at production scale, top_k=20 gives the reranker a small window. Cross-encoder rerankers work best when they can re-score a larger, noisier candidate set and promote truly relevant results that the embedding model ranked lower. Research consistently shows diminishing returns past ~50 candidates for a target of 5-10 final results.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Medium. Recall improvement, especially for technical terms where embedding similarity is unreliable |
| Latency | +50-100ms for additional vector search rows; Cohere rerank latency scales linearly (~10ms per doc) |
| Cost | Negligible. Cohere rerank-english-v3.0 pricing is per-search, not per-document for <100 docs |
| Complexity | Low. Single parameter change |
| Ingest impact | None |

---

## Proposal 2: Adaptive Similarity Threshold

### Current State
- `match_threshold=0.3` in `query_engine.py` (very permissive)
- `match_threshold=0.5` in `query_with_citations()` (stricter, but unused by agent)
- The threshold is static regardless of query specificity

### Recommendation
Keep the low threshold (0.3) as-is when reranking is enabled, since the reranker will filter noise. When reranking is disabled, use a higher threshold (0.5-0.6).

An adaptive threshold adds complexity with minimal benefit when the reranker is doing its job. The real quality gate should be a **minimum rerank score cutoff** (e.g., discard results with `rerank_score < 0.2`) to prevent irrelevant results from reaching the LLM.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Low-Medium. Prevents irrelevant chunks from polluting context |
| Latency | Neutral to slight improvement (fewer chunks to rerank if threshold is raised) |
| Cost | Neutral |
| Complexity | Low (add rerank score floor to `Reranker.rerank_results`) |
| Ingest impact | None |

---

## Proposal 3: Hybrid Search (Vector + Keyword/BM25)

### Current State
Pure vector search via `match_chunks()`. The `content` column has no full-text search index. Technical maritime terms (equipment model numbers, error codes, part IDs) may have poor embedding representation.

### Recommendation
Add PostgreSQL full-text search as a secondary retrieval path using `tsvector` and combine with vector results via Reciprocal Rank Fusion (RRF).

**Implementation approach:**
1. Add a `tsvector` GIN index on `chunks.content` (migration)
2. Create a `keyword_search_chunks()` SQL function
3. In the `Retriever`, run both searches in parallel (asyncio.gather)
4. Merge results using RRF: `score = sum(1 / (k + rank_i))` across both result lists
5. Feed merged candidates to the reranker

**Why not Supabase full-text search natively?** PostgreSQL's built-in `ts_rank` and `tsvector` are sufficient. No need for an external search engine (Elasticsearch, Typesense) at this scale (~100K chunks).

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +High for technical/keyword queries. Equipment codes like "ME-B 9S50" or error codes will match exactly instead of relying on semantic similarity |
| Latency | +30-50ms (parallel keyword search + RRF merge); offset by better reranking input |
| Cost | None (PostgreSQL built-in) |
| Complexity | Medium. New migration for tsvector index, new SQL function, RRF merge logic in retriever |
| Ingest impact | Minor. Populate tsvector column during chunk insertion (can be auto-generated via trigger) |

---

## Proposal 4: Query Expansion / Multi-Query

### Current State
Single query embedding is generated from the user's exact question. No reformulation.

### Recommendation
**Skip for now.** The topic filter already provides a form of query understanding (extracting structured topic names from natural language). Adding multi-query generation would add an LLM call (~300-500ms) per search for modest recall improvement.

If implemented later, the most cost-effective approach would be a single LLM call that produces 2-3 query variants, embed them all in one batch, and union the vector search results before reranking. But the hybrid search (Proposal 3) addresses the same recall gap more directly for technical terms.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Low-Medium. Marginal when combined with hybrid search |
| Latency | +300-500ms (additional LLM call for query variants) |
| Cost | +1 LLM call per query (~$0.001-0.003) |
| Complexity | Medium |
| Ingest impact | None |

---

## Proposal 5: Topic Pre-Filtering Loosening

### Current State
The `TopicFilter` extracts topics from the query via LLM, then matches them semantically to stored topics (threshold 0.70). If topics are detected but none match, it returns an early "skip RAG" response, preventing any vector search from running.

Topic IDs are passed as metadata filters to `match_chunks()`, which uses OR logic (match ANY topic). This means results are restricted to chunks tagged with at least one matched topic.

### Recommendation
The current approach is sound but has a risk: **over-filtering.** If topic extraction misidentifies the query intent or the topic taxonomy is incomplete, relevant chunks may be excluded.

**Proposed changes:**
1. **Fallback to broad search on low confidence:** If topic extraction returns topics but the topic matcher finds low similarity (e.g., best match is 0.70-0.75), run the search without topic filtering rather than with a marginal topic filter.
2. **Topic filter as boost, not hard filter:** Instead of filtering chunks by topic_id in the SQL WHERE clause, use topic match as a score boost. This preserves recall while still favoring on-topic results. Implementation: remove topic_ids from metadata_filter, retrieve a larger candidate pool, and add a topic-match bonus to the similarity score before reranking.

Option 2 is more complex but significantly reduces the risk of missing relevant cross-topic results (e.g., "engine cooling" tagged under "Engine Maintenance" vs "Cooling Systems").

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Medium. Prevents false-negative filtering from incomplete topic taxonomy |
| Latency | Neutral (option 1) to +slight (option 2, larger candidate pool) |
| Cost | Neutral |
| Complexity | Low (option 1) to Medium (option 2) |
| Ingest impact | None |

---

## Proposal 6: Pre-/Post-Filtering for Vessel Metadata

### Current State
Vessel filtering (`vessel_ids`, `vessel_types`, `vessel_classes`) is applied as a WHERE clause inside `match_chunks()` via JSONB `@>` containment. This means the filter is applied **during** vector search, which is correct for PostgreSQL/pgvector -- the HNSW index scan is filtered inline, and Supabase handles this efficiently.

### Recommendation
The current pre-filtering approach is correct and performant. Post-filtering (retrieve, then filter) would waste candidate slots on irrelevant vessels. The GIN index on `chunks.metadata` supports the JSONB containment queries.

No changes needed for vessel filtering order.

**One improvement:** Add a composite partial index for common vessel filter patterns if query latency becomes an issue at scale:
```sql
CREATE INDEX idx_chunks_vessel_ids ON chunks
    USING GIN ((metadata->'vessel_ids'))
    WHERE metadata->'vessel_ids' IS NOT NULL;
```

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | Neutral |
| Latency | Neutral (current) to +slight improvement (specialized index) |
| Cost | None |
| Complexity | Low |
| Ingest impact | None |

---

## Proposal 7: Reranker Improvements

### Current State
- Cohere `rerank-english-v3.0` via LiteLLM
- Reranks on `result.text` (chunk content only, no metadata)
- No consideration of document hierarchy (email + attachments)

### Recommendations

#### 7a. Rerank with enriched context
Currently, the reranker sees only `result.text`. Prepend the email subject and source title to give the cross-encoder more context:

```python
documents = [
    f"[{r.email_subject or ''}] {r.source_title or ''}: {r.text}"
    for r in results
]
```

This helps the reranker distinguish between chunks from different incidents that have similar content but different contexts.

#### 7b. Upgrade to rerank-v3.5
Cohere's `rerank-english-v3.5` (or Azure AI's `cohere-rerank-v3.5`) shows improved accuracy in benchmarks, particularly for technical content. The LiteLLM abstraction makes this a config change.

#### 7c. Keep Cohere over alternatives
Jina Reranker v2 and open-source cross-encoders (e.g., `ms-marco-MiniLM-L-12-v2`) are alternatives, but Cohere remains the best quality/latency tradeoff for a hosted solution. Self-hosted cross-encoders would add infrastructure complexity.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Medium (enriched context), +Low (model upgrade) |
| Latency | Neutral (same API, slightly larger input) |
| Cost | Neutral (same pricing tier for v3.5) |
| Complexity | Low (both are small code changes) |
| Ingest impact | None |

---

## Proposal 8: Context Assembly Improvements

### Current State
- 5 chunks passed to LLM after reranking (configurable via `RERANK_TOP_N`)
- No sibling chunk expansion (adjacent chunks from the same document)
- No parent email context injection when an attachment chunk is retrieved
- Incident grouping happens (`_group_by_incident`) but only as a summary appended to context
- `max_tokens=1000` for LLM response

### Recommendations

#### 8a. Sibling chunk expansion
When a chunk scores high, its neighboring chunks (same document, adjacent `chunk_index`) often contain relevant continuation or setup context. After reranking, for each top result, fetch `chunk_index - 1` and `chunk_index + 1` from the same `document_id` and include them in the context (deduplicated).

**Implementation:** Query `chunks` table for adjacent indexes after reranking, before context building. Cap expanded context at ~10 chunks total to stay within LLM context budget.

#### 8b. Parent email context for attachment chunks
When an attachment chunk is retrieved (e.g., a PDF page), the parent email's body often contains crucial context about why the attachment was sent. After retrieval, for any result where `document_type` starts with `attachment_`, fetch the parent email document's first chunk and prepend it as context.

**Implementation:** Use `document_id` -> `documents.parent_id` -> fetch parent's first chunk. This is a single JOIN query.

#### 8c. Increase rerank_top_n to 8-10
With ~100K chunks, 5 results may miss relevant context, especially for multi-incident queries ("how was X handled across the fleet?"). Increasing to 8-10 results combined with gpt-4o's 128K context window is well within budget.

**Trade-off:** More chunks = more tokens = higher LLM cost (~$0.005/query increase) but significantly better coverage for complex queries.

#### 8d. Increase max_tokens for LLM response
The current `max_tokens=1000` (in `query_engine.py` line 218) may truncate detailed responses with multiple incident comparisons. Increase to 2000-3000 for the agent path where the LLM is generating structured incident analysis.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +High (sibling expansion + parent context), +Medium (more results) |
| Latency | +20-50ms (additional DB queries for siblings/parents) |
| Cost | +$0.003-0.010/query (more input tokens to LLM) |
| Complexity | Medium (sibling expansion), Low (parent context, top_n increase) |
| Ingest impact | None |

---

## Proposal 9: HNSW Index Tuning

### Current State
```sql
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Default pgvector HNSW parameters. No `ef_search` override at query time (uses PostgreSQL default of 40).

### Recommendation
The current parameters are conservative. For ~100K chunks:

- **`m = 16`**: Adequate. Increasing to 24-32 improves recall at the cost of index size and build time. Not worth changing unless recall issues are observed.
- **`ef_construction = 64`**: Low for production. Increase to 128-200 for better index quality. This only affects index build time (during ingest), not query time.
- **`ef_search`**: Set to 100-200 at query time for better recall. This is the most impactful tuning parameter:

```sql
SET LOCAL hnsw.ef_search = 100;
```

Add this before the `match_chunks` call in the search repository, or set it as a session default.

**At 100K chunks, expected query latency:**
- Current (ef_search=40): ~5-15ms
- With ef_search=100: ~10-25ms
- With ef_search=200: ~15-40ms

All are well within acceptable latency for a RAG pipeline where the LLM call takes 1-3 seconds.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | +Low-Medium (better recall from HNSW index) |
| Latency | +5-20ms |
| Cost | None |
| Complexity | Low (SET LOCAL before query, migration for ef_construction) |
| Ingest impact | Re-index needed for ef_construction change (REINDEX CONCURRENTLY) |

---

## Proposal 10: Query Result Caching

### Current State
No caching. Every query generates a new embedding, runs vector search, reranks, and calls the LLM.

### Recommendation
**Skip for now.** The user base is small (internal maritime operations team), query patterns are diverse (incident-specific), and caching stale results in a knowledge base that grows with new incidents could return outdated information.

If latency becomes an issue, the most effective cache is an **embedding cache** (cache query text -> embedding vector) since the same question phrased identically will always produce the same embedding. This avoids the OpenAI API call (~100-200ms) but still runs fresh vector search.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | Neutral to -Low (risk of stale cached results) |
| Latency | -100-200ms (embedding cache only) |
| Cost | -$0.0001/query (skip embedding API call) |
| Complexity | Low (LRU dict or Redis for embedding cache) |
| Ingest impact | Cache invalidation needed on new ingests |

---

## Proposal 11: Response Streaming Efficiency

### Current State
Streaming is implemented via Vercel AI SDK v1 format (`streaming.py`). LangGraph events are streamed as `on_chat_model_stream` deltas. Progress updates are sent as data annotations.

The streaming implementation is well-structured. LLM token streaming works correctly.

### Recommendation
The streaming path is efficient. The main latency bottleneck is not streaming but the **serial execution of pre-search steps:**

```
TopicExtractor LLM call (~300ms)
  -> TopicMatcher embedding + DB lookup (~200ms)
  -> Pre-check count queries (~50ms each, sequential per topic)
  -> Query embedding (~100-200ms)
  -> Vector search (~10-30ms)
  -> Rerank (~200-300ms)
  -> LLM generation (1-3s, streamed)
```

**Parallelization opportunity:** The query embedding generation can run in parallel with topic extraction/matching. Both are independent operations. Use `asyncio.gather`:

```python
topic_task = topic_filter.analyze_query(question, vessel_filter)
embedding_task = embeddings.generate_embedding(question)
filter_result, query_embedding = await asyncio.gather(topic_task, embedding_task)
```

This saves ~100-200ms by overlapping the embedding API call with the topic extraction LLM call.

### Assessment
| Dimension | Impact |
|-----------|--------|
| Response quality | Neutral |
| Latency | -100-200ms (parallel embedding + topic extraction) |
| Cost | Neutral |
| Complexity | Low-Medium (refactor Retriever to accept pre-computed embedding) |
| Ingest impact | None |

---

## Priority Ranking

Ordered by impact-to-effort ratio:

| Priority | Proposal | Impact | Effort | Notes |
|----------|----------|--------|--------|-------|
| **P0** | Bug fix: agent top_k | Critical | Trivial | Reranking is silently disabled in production |
| **P1** | 7a: Enriched rerank context | Medium | Low | Small code change in `reranker.py` |
| **P1** | 8d: Increase max_tokens | Medium | Low | Config change |
| **P1** | 8c: Increase rerank_top_n to 8 | Medium | Low | Config change |
| **P2** | 1: Increase top_k to 40 | Medium | Low | After P0 fix |
| **P2** | 9: HNSW ef_search tuning | Low-Medium | Low | SET LOCAL in search repo |
| **P2** | 11: Parallel embedding + topic | Low-Medium | Low-Medium | asyncio.gather refactor |
| **P2** | 2: Rerank score floor | Low-Medium | Low | Filter low-confidence results |
| **P3** | 8a: Sibling chunk expansion | High | Medium | New DB query + dedup logic |
| **P3** | 8b: Parent email context | High | Medium | JOIN query + context assembly |
| **P3** | 3: Hybrid search (BM25) | High | Medium | New migration, SQL function, RRF merge |
| **P3** | 5: Topic filter loosening | Medium | Medium | Score boost vs hard filter trade-off |
| **P4** | 7b: Upgrade to rerank-v3.5 | Low | Low | Config change, test first |
| **P5** | 4: Query expansion | Low-Medium | Medium | Skip unless hybrid search is insufficient |
| **P5** | 10: Caching | Low | Low-Medium | Skip for now |

---

## Estimated Combined Impact

Implementing P0 through P3 proposals would yield:

- **Response quality:** Significant improvement. The reranker bug fix alone restores 20-35% accuracy gain. Enriched reranking context, sibling expansion, and parent context will meaningfully improve answer completeness for complex multi-document incidents.
- **Latency:** Net neutral to +50ms. Parallelization (P2) offsets the additional DB queries (P3). The dominant latency factor (LLM generation at 1-3s) is unchanged.
- **Cost:** +$0.005-0.015/query. Primarily from additional LLM input tokens (more context chunks). Marginal at expected query volume.
