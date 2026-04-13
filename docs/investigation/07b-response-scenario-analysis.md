---
title: "Response Quality Analysis Across Query Scenarios"
status: investigation
date: 2026-04-13
scope: RAG pipeline behavior for six representative maritime query types
---

# 07b - Response Quality Analysis Across Query Scenarios

Analysis of how the current MTSS RAG pipeline handles six representative query types from maritime operations. Each scenario is evaluated end-to-end: query reception, topic extraction, retrieval, reranking, LLM generation, and citation processing.

## Pipeline Architecture Summary

The current pipeline follows this flow:

1. **Agent (LangGraph)** receives user message, decides to call `search_documents` tool
2. **Topic pre-filter**: LLM extracts topic(s) from query, matches to existing topics via embedding similarity, checks chunk counts for early return
3. **Vector search**: `match_chunks` PostgreSQL function performs cosine similarity search with optional metadata filters (topic_ids, vessel_ids/types/classes)
4. **Reranking**: Cross-encoder reranks top-K candidates down to top-N (default 5)
5. **Context assembly**: Citation headers built per chunk, incident grouping by root email thread
6. **LLM generation**: Agent generates answer with `[C:chunk_id]` citations using full conversation context
7. **Citation validation**: Validates chunk_ids, replaces with `<cite>` tags, retries if >50% invalid

Key constraints discovered during analysis:
- `max_tokens=1000` on the standalone `query_engine.py` LLM call (agent path uses ChatOpenAI defaults)
- `rerank_top_n` defaults to 5 -- only 5 chunks reach the LLM
- No date filtering in `match_chunks` -- email_date is returned but never used as a filter
- No aggregation queries -- the system can only do vector similarity search
- Metadata filter supports: `topic_ids` (OR logic), `vessel_ids`, `vessel_types`, `vessel_classes` (AND logic via `@>`)

---

## Scenario 1: Specific Incident Lookup

> "Vessel X had a cargo hold ventilation failure -- what was done before?"

### What goes right

- **Topic extraction** will detect "Cargo Hold Ventilation Failure" or similar, matching to a stored topic if one exists.
- **Vessel filter** can be applied if the user has selected Vessel X via the UI dropdown (stored in `selected_vessel_id` state). The `match_chunks` function filters by `vessel_ids` in metadata.
- **Semantic search** will find chunks describing ventilation failures because the query is specific and technical.
- **Reranking** improves precision by scoring query-document pairs with a cross-encoder.
- **Incident grouping** (`_group_by_incident`) correctly groups chunks from the same email thread, giving the LLM awareness of "3 chunks from 1 incident" vs "3 chunks from 3 incidents".
- **Citation headers** include email subject, date, and initiator -- enough for the LLM to reference the original incident report.
- **System prompt** instructs structured output with Component, Issue, Resolution Steps -- ideal for this query type.

### What goes wrong or is suboptimal

- **Vessel name in query vs. vessel filter**: If the user types "Vessel X" in the query text but has not selected a vessel in the UI, there is no vessel filter applied. The query relies entirely on semantic similarity. The topic filter and vector search have no mechanism to extract a vessel name from the query text and resolve it to a vessel_id.
- **Topic filter false negative**: If the extracted topic ("Cargo Hold Ventilation") doesn't match any stored topic at the 0.70 similarity threshold, the system returns an early "category not in records" message and skips RAG entirely. The user must then confirm "search across all categories" which adds a round-trip.
- **5-chunk limit**: With `rerank_top_n=5`, a complex incident with multiple email replies, PDF attachments, and image attachments may have its context truncated. The LLM only sees 5 chunks, potentially missing the resolution steps in a later email reply.
- **No temporal ordering within results**: Chunks are ordered by similarity score, not chronologically. For an incident that evolved over days (problem report -> diagnosis -> fix -> follow-up), the LLM receives chunks in relevance order, which may interleave the timeline.

### Where data might be missed

- Attachments (PDFs, images) from the incident email thread may not surface in the top 5 if their embedding similarity is lower than the email body chunks. A PDF maintenance report attached to the ventilation failure email could be ranked below unrelated but semantically similar chunks from other incidents.

---

## Scenario 2: Aggregation / Counting Query

> "How many times has this type of engine issue occurred across our fleet?"

### What goes right

- **Topic extraction** will detect "Engine Issues" and match to stored topics.
- **Vector search** will find engine-related chunks.
- The LLM can attempt to count distinct incidents from the context it receives.

### What goes wrong or is suboptimal

- **Fundamental mismatch**: This is an aggregation query, but the pipeline is designed for retrieval. `match_chunks` returns at most `match_count` chunks (default 20 pre-rerank, 5 post-rerank). If there are 50 engine incidents in the database, the LLM only sees 5 chunks and will answer "I found 5 relevant incidents" -- a severely incomplete answer.
- **No COUNT capability**: There is no SQL path that counts incidents by topic or type. The `get_chunks_count_for_topic` function counts chunks, not incidents. Chunks-per-topic is not the same as incidents-per-topic (one incident may have 10+ chunks).
- **No incident-level deduplication**: Multiple chunks from the same email thread will consume slots in the top-K results, reducing coverage of distinct incidents.
- **"Across our fleet" implies no vessel filter**: The system handles this correctly by not applying a vessel filter when none is selected. But it also means no structured fleet-wide aggregation.

### Where the response might be misleading

- The LLM will confidently state a count based on the 5 chunks it sees, giving the user a false sense of completeness. There is no mechanism to indicate "I found 5 but there may be more" unless the system prompt instructs this, which it currently does not.
- The incident summary says "Found X relevant chunks from Y unique incident(s)" but this is limited to the retrieved set, not the full database.

### Proposed optimization

- **Query classification**: Detect aggregation intent ("how many", "how often", "count", "frequency") and route to a SQL aggregation path instead of vector search.
- **Incident-level counting**: A new database function that counts distinct `root_id` values per topic, with optional vessel/type/class filters.
- **Hybrid response**: Return the count from SQL, then retrieve a sample of representative incidents via vector search for the LLM to describe.

---

## Scenario 3: Procedural / Best-Practice Query

> "What are the standard procedures for ballast water treatment system failures?"

### What goes right

- **Topic extraction** correctly identifies "Ballast Water Treatment System" as the topic.
- **Semantic search** finds chunks describing ballast water treatment procedures.
- **System prompt** instructs the LLM to provide actionable steps and resolution procedures -- well-suited for procedural queries.
- **Cross-encoder reranking** helps distinguish between chunks that describe procedures vs. chunks that merely mention ballast water in passing.

### What goes wrong or is suboptimal

- **Procedure synthesis across incidents**: Standard procedures may not exist as a single document. They may be implicitly described across multiple incident reports where different vessels handled the same failure differently. The LLM sees 5 chunks from potentially 5 different incidents and must synthesize a coherent procedure, which is challenging.
- **No "official procedure" signal**: There is no metadata distinguishing official SOPs from ad-hoc fixes in incident emails. The LLM treats all sources equally, potentially mixing authoritative procedures with one-off workarounds.
- **max_tokens=1000 in standalone path**: For procedural queries requiring multi-step answers, 1000 tokens may truncate the response mid-procedure. The agent path (via ChatOpenAI) does not have this explicit limit but may still truncate on some model configurations.
- **"Standard procedures" expectation vs. incident-report reality**: The knowledge base contains incident reports (emails + attachments), not procedure manuals. The user may expect a canonical answer, but the system can only synthesize from how past incidents were handled. The system prompt does not set this expectation clearly.

### Where data might be missed

- If ballast water procedures are documented in PDF attachments rather than email bodies, and those PDF chunks have lower similarity scores, they may not surface in the top 5.

---

## Scenario 4: Temporal + Categorical Filter Query

> "Show me all incidents on bulk carriers in the last 2 years"

### What goes right

- **Vessel type filter**: If the user selects "Bulk Carrier" via the UI, the `vessel_types` metadata filter is applied in `match_chunks`, correctly restricting results.
- **Broad query**: Without a specific topic, topic extraction returns an empty list, and the system proceeds with broad search (no topic filter). This is the correct behavior for a browse-style query.

### What goes wrong or is suboptimal

- **No date filtering**: This is the single biggest gap for this query type. The `match_chunks` function has no date parameters. `email_date` is returned in results but never used as a filter criterion. The query "last 2 years" is treated purely as semantic text -- the embedding of "last 2 years" has no meaningful relationship to actual dates.
- **"All incidents" is impossible**: Vector search returns at most `match_count` results. "Show me all" is an enumeration query that vector search cannot satisfy. The user expects a list, not a semantic similarity ranking.
- **"Bulk carriers" in text vs. filter**: If the user types "bulk carriers" in the chat without selecting a vessel type in the UI, the system has no way to extract "bulk carrier" from the query text and convert it to a `vessel_types` metadata filter. It falls back to semantic search, which may match chunks mentioning "bulk carrier" in their content but will also miss incidents that don't use that exact phrase.
- **No pagination**: The system returns a fixed number of results (top 5 after reranking). For "show me all", users expect scrollable results or at least a complete list.
- **Vector search requires a meaningful query**: "Show me all incidents on bulk carriers in the last 2 years" has low semantic specificity. The embedding may not effectively retrieve diverse incidents because it's not looking for a specific topic.

### Where the response might be misleading

- The LLM will present whatever 5 chunks the vector search returns as if they represent "all incidents." The user has no way to know if they're seeing 5 out of 5 or 5 out of 500.
- Any temporal claims are fabricated -- the system has no mechanism to verify whether retrieved incidents actually fall within the last 2 years, beyond whatever date metadata happens to appear in the chunk text.

### Proposed optimization

- **Date range filter**: Add `date_from` and `date_after` parameters to `match_chunks`, filtering on `COALESCE(d.email_date_start, root_doc.email_date_start)`.
- **Query intent detection**: Detect temporal expressions ("last 2 years", "since 2024", "recent") and convert to date filters before search.
- **Vessel type extraction from query text**: Use NLP or LLM to extract vessel type mentions from the query and auto-apply vessel_types filter.
- **List/browse mode**: For "show me all" queries, switch to a paginated SQL query ordered by date, bypassing vector search entirely.

---

## Scenario 5: Symptom-Based Troubleshooting Query

> "The main engine turbocharger is leaking oil -- what do our records show?"

### What goes right

- **Highly specific query**: The symptom "turbocharger leaking oil" generates a focused embedding that should match well against incident reports describing the same or similar symptoms.
- **Topic extraction** will detect "Engine Issues" or "Turbocharger" and narrow search.
- **Reranking** excels here because cross-encoder models are good at matching symptom descriptions to resolution descriptions.
- **System prompt** is well-designed for this use case -- it instructs the LLM to find past cases with similar symptoms and how they were resolved, provide resolution steps, and note patterns.
- **Incident grouping** helps the LLM distinguish between multiple turbocharger incidents.

### What goes wrong or is suboptimal

- **Synonym gap**: "Leaking oil" vs. "oil leak" vs. "lubricant seepage" vs. "oil drip from T/C" -- different vessels may describe the same symptom differently. Embedding similarity handles this reasonably well but not perfectly. The reranker helps compensate.
- **Component hierarchy**: "Main engine turbocharger" is specific, but the topic system may only have "Engine Issues" as a broad category. This means the topic filter includes all engine-related chunks (fuel injectors, governors, etc.), consuming slots in the top-K results with irrelevant engine issues. The reranker must then distinguish turbocharger issues from other engine issues within the broader category.
- **Resolution may be in attachments**: Service engineer reports (often PDFs) contain the actual fix details. These may not surface if the email body chunks rank higher by similarity.

### Where data might be missed

- If a vessel reported a turbocharger oil leak but used highly technical jargon or model-specific terminology (e.g., "MET42SC T/C lube oil drain clogged"), the embedding may not match well against the more general query.

---

## Scenario 6: Cross-Vessel Comparison Query

> "Compare how different vessels handled similar anchor windlass problems"

### What goes right

- **Topic extraction** correctly identifies "Anchor Windlass" as the topic.
- **No vessel filter**: Without a specific vessel selected, the search runs across all vessels, which is correct for a comparison query.
- **Incident grouping** provides the LLM with awareness of which chunks come from which incident/vessel.

### What goes wrong or is suboptimal

- **5-chunk limit severely constrains comparison**: A meaningful comparison needs at least 2-3 chunks per vessel, across at least 2-3 vessels. With `rerank_top_n=5`, the LLM may receive chunks from only 1-2 vessels, making comparison impossible.
- **No vessel diversity in retrieval**: Vector search ranks by similarity. If one vessel's incident report is highly detailed and well-written, it will dominate the top-5 results. There is no mechanism to ensure diversity across vessels.
- **LLM synthesis challenge**: Even with good chunks, comparing approaches across vessels requires the LLM to identify differences in methodology, which is hard from decontextualized chunks.
- **No structured comparison format**: The system prompt defines a single-incident response format. It does not instruct the LLM on how to structure a comparison (e.g., table format, vessel-by-vessel breakdown).
- **Missing vessel names in chunks**: If the chunk content does not mention the vessel name (it's only in the email metadata), the LLM may not be able to attribute approaches to specific vessels, even though `email_subject` in the citation header may contain it.

### Where the response might be misleading

- The LLM may present a comparison based on 2 incidents as if it represents fleet-wide patterns. There is no indication of how many total windlass incidents exist or how representative the sample is.

### Proposed optimization

- **Diversity-aware retrieval**: After initial vector search, apply diversity sampling: ensure at least N distinct `root_file_path` values (i.e., distinct incidents) are represented. Consider MMR (Maximal Marginal Relevance) to reduce redundancy.
- **Increased top-K for comparison queries**: Detect comparison intent and increase `rerank_top_n` from 5 to 10-15.
- **Structured comparison prompt**: Add a comparison-specific section to the system prompt with instructions for tabular or vessel-by-vessel output format.

---

## Cross-Cutting Issues

### 1. Temporal Queries

**Current state**: No date filtering exists in the retrieval pipeline. `email_date` is stored in the `documents` table (`email_date_start`) and returned by `match_chunks`, but it is never used as a filter criterion. Dates appear in citation headers (`date:{email_date}`) so the LLM can reference them, but the retrieval set is not constrained by date.

**Impact**: Any query with temporal intent ("last 2 years", "recent", "since January") relies entirely on the LLM interpreting dates in the retrieved chunks and self-filtering, which is unreliable and limited to the chunks that happen to be retrieved.

**Recommendation**: Add `date_from TIMESTAMPTZ DEFAULT NULL` and `date_to TIMESTAMPTZ DEFAULT NULL` parameters to `match_chunks`, filtering on `COALESCE(d.email_date_start, root_doc.email_date_start)`. Add temporal expression extraction (regex or LLM) to the query preprocessing step to populate these parameters automatically.

### 2. Aggregation Queries

**Current state**: The pipeline has no aggregation capability. Every query goes through embed -> search -> rerank -> LLM generate. Count queries receive at most `rerank_top_n` results and the LLM fabricates counts from this incomplete sample.

**Impact**: Users asking "how many", "how often", "what percentage" get misleading answers. This is potentially the most dangerous gap because the user trusts the numbers.

**Recommendation**:
- Add a query classifier that detects aggregation intent.
- Create database functions for incident counting by topic, vessel, date range.
- For aggregate queries: run the SQL count first, then retrieve a representative sample for the LLM to describe qualitatively. Include the true count in the context.

### 3. Comparison / Multi-Document Synthesis

**Current state**: The pipeline retrieves 5 chunks ranked by similarity. For comparison queries, this provides insufficient diversity across incidents/vessels.

**Impact**: Comparisons are superficial, often based on 1-2 incidents instead of the full fleet's experience.

**Recommendation**:
- Implement diversity-aware retrieval (MMR or post-hoc deduplication by `root_file_path`).
- Increase `rerank_top_n` dynamically based on query intent (e.g., comparison -> 15, lookup -> 5).
- Add a comparison response template to the system prompt.

### 4. Broad vs. Narrow Queries

**Current state**: The topic filter handles this reasonably well. Narrow queries ("turbocharger oil leak") extract a specific topic and narrow the search. Broad queries ("tell me about the Maran") return an empty topic list, triggering unfiltered search.

**Impact**: Broad queries work but return unpredictable results because vector search requires a meaningful embedding. Very broad queries may return results that are semantically similar to the query text but not what the user intended.

**Improvement**: For broad queries that mention a vessel name, auto-resolve the vessel name to a vessel_id and apply as a filter rather than relying on semantic similarity. This would require a vessel name resolution step (fuzzy match against the vessels table).

### 5. Follow-Up / Multi-Turn Queries

**Current state**: LangGraph persists conversation state via checkpoints. When continuing a thread, only the latest user message is sent as input. The full conversation history is available in the checkpoint, and the LLM receives all prior messages as context.

**What works**: The LLM can reference prior search results from the conversation. Follow-up questions like "tell me more about the second incident" or "what about a different vessel?" benefit from conversation context.

**What doesn't work**: Each search is independent. If the user asks "tell me more about incident #2", the agent must reformulate a new search query. There is no mechanism to carry forward previous search results or refine them. The `citation_map` is cleared after each response (`citation_map: None`), so prior citations cannot be re-referenced.

**Improvement**: Consider retaining the `citation_map` across turns for the duration of the conversation, allowing the LLM to reference previously retrieved chunks without re-searching. Add a "drill down" search mode that filters within the previously retrieved document set.

### 6. Missing Data Scenarios

**Current state**: Well-handled at the topic filter level. When no matching topic exists, the system returns a clear message ("this category isn't in our records yet") and offers to search more broadly. When topics exist but have no chunks for the selected vessel, it offers alternatives.

**What works well**:
- `TopicMessages` provides helpful, actionable messages.
- The `skip_topic_filter` parameter allows the user to retry with broader search.
- Related topic suggestions guide the user to available data.

**What could improve**:
- When RAG search (after topic filter passes) returns zero results, the response is a generic "No relevant documents found." It does not suggest alternative queries, related topics, or explain why no results matched. This contrasts with the richer topic-filter messages.
- When the vector search returns results but they're all low-quality (low similarity scores, all below a meaningfulness threshold), the LLM still generates an answer from marginally relevant chunks, potentially producing a misleading response.

### 7. Vessel Name Extraction from Query Text

**Current state**: Vessel filtering only works via the UI dropdown. If a user mentions a vessel by name in their query text, the system does not extract it and apply it as a metadata filter. The system prompt for the agent mentions IMO numbers but the Vessel model in the codebase does not have an `imo` field (though `agent.py` references `vessel.imo`).

**Impact**: Users who type "What happened on MARAN CANOPUS?" without using the dropdown get unfiltered results. The semantic similarity to "MARAN CANOPUS" may help but is unreliable.

**Recommendation**: Add a vessel name/IMO extraction step in query preprocessing. Fuzzy-match extracted names against the vessels table and auto-apply the filter.

### 8. Result Completeness Transparency

**Current state**: The incident summary says "Found X relevant chunks from Y unique incident(s)" but does not indicate whether this is comprehensive. The user has no way to know if they're seeing 3 out of 3 incidents or 3 out of 300.

**Recommendation**: For topic-filtered searches, include the total chunk count for the topic (already available from `TopicFilterResult.total_chunk_count`) in the context provided to the LLM. This allows the LLM to say "I found 3 relevant incidents out of 47 total records in the Engine Issues category."

---

## Scenario-Specific Optimization Summary

| Query Type | Key Gap | Proposed Strategy |
|---|---|---|
| Specific incident lookup | Vessel name not extracted from text; 5-chunk limit | Vessel name extraction; dynamic top-N |
| Aggregation / counting | No SQL aggregation path | Query classifier + incident count SQL function |
| Procedural / best-practice | No distinction between SOPs and ad-hoc fixes | Source type metadata; synthesis-aware prompt |
| Temporal + categorical | No date filtering in search | Date parameters in match_chunks; temporal expression parser |
| Symptom-based troubleshooting | Attachment content underrepresented | Boost attachment chunks; component-level topics |
| Cross-vessel comparison | No diversity in retrieval; too few results | MMR diversity; dynamic top-N; comparison prompt template |

## Priority Ranking

1. **Date filtering in match_chunks** -- Required by multiple query types (scenarios 4, 2). Straightforward SQL change.
2. **Query intent classification** -- Routes aggregation, comparison, temporal, and lookup queries to appropriate strategies. Enables all other optimizations.
3. **Dynamic rerank_top_n** -- Comparison and aggregation queries need 10-15 results; lookup queries are fine with 5. Low-effort, high-impact.
4. **Vessel name extraction from query text** -- Solves the common case where users type vessel names instead of using the dropdown.
5. **Aggregation SQL path** -- Prevents misleading counts. Moderate implementation effort.
6. **Result completeness transparency** -- Include total counts so the LLM can caveat its answers. Low effort, high trust impact.
7. **Diversity-aware retrieval (MMR)** -- Benefits comparison queries and reduces redundancy. Moderate effort.
