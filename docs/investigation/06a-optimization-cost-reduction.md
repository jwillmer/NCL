---
purpose: Investigate cost reduction opportunities in the MTSS ingest pipeline
status: proposal
date: 2026-04-13
---

# 06a — Ingest Pipeline Cost Reduction

## Current Cost Baseline

### Per-email API calls (6,289 emails)

| Cost center        | Service            | Unit cost                      | Estimate per email | Full run estimate |
|--------------------|--------------------|-------------------------------|--------------------|-------------------|
| Document parsing   | LlamaParse         | $0.00625/page                 | ~$0.025 (est. 4 pages avg) | ~$157 |
| Image classification + description | OpenAI Vision (gpt-4o-mini) | ~$0.01/image | ~$0.01 (est. 1 meaningful image avg) | ~$63 |
| Context summary    | gpt-4o-mini        | ~$0.00015/call (600 input + 100 output tokens) | ~$0.00030 (email + 1 attachment) | ~$1.89 |
| Topic extraction   | gpt-4o-mini        | ~$0.00020/call (1000 input + 200 output tokens) | ~$0.00020 | ~$1.26 |
| Topic dedup embeddings | text-embedding-3-small | ~$0.00002/call | ~$0.00010 (5 topics) | ~$0.63 |
| Chunk embeddings   | text-embedding-3-small | $0.020/1M tokens | ~$0.00010/email (5 chunks avg) | ~$0.63 |
| **Total estimate** |                    |                               | **~$0.035/email** | **~$225** |

LlamaParse dominates at **~70% of total cost**. Vision API is second at ~28%.

---

## Proposal 1: Tiered PDF Parsing (LlamaParse only for complex docs)

### Current behavior
All PDFs, Office docs, CSVs, RTFs, and HTML files are routed to LlamaParse via `LlamaParseParser`. The parser is configured with aggressive features: `high_res_ocr`, `adaptive_long_table`, `outlined_table_extraction`, `specialized_image_parsing`, and an agentic auto-mode trigger for pages with tables/charts/images.

### Observation
Many email attachments are simple text PDFs (incident reports, letters, memos) that do not contain tables, charts, or scanned images. For these, a free local parser (PyPDF2, pdfplumber, or PyMuPDF) extracts text with equivalent quality. LlamaParse's premium features only add value for:
- Scanned PDFs requiring OCR
- PDFs with complex tables (e.g., inspection checklists)
- PDFs with embedded charts/diagrams
- Office documents with complex layouts

### Proposed approach
Add a PDF complexity classifier before parsing:
1. Open PDF with `pypdf` (already a dependency for the estimator).
2. Check each page: does it contain extractable text (`page.extract_text()`)? Any images? Any form fields?
3. **Simple PDF** (all pages have extractable text, no images/forms): parse with `pdfplumber` or `PyMuPDF` locally for free.
4. **Complex PDF** (scanned pages, images, forms, tables): route to LlamaParse.
5. For Office formats (.docx, .xlsx, .pptx): use `python-docx`, `openpyxl`, `python-pptx` for simple extraction; reserve LlamaParse for legacy formats (.doc, .xls, .ppt) and complex layouts.
6. For CSV, RTF, HTML, plain text: always parse locally (trivial extraction).

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | 40-60% of LlamaParse costs ($63-$94) |
| **Savings %** | 28-42% of total pipeline cost |
| **Retrieval quality** | Neutral for simple docs; LlamaParse still used where it adds value |
| **Implementation complexity** | Medium — new complexity classifier, local parser integration, fallback logic |
| **Risk** | Low-medium — need quality validation on sample set; some edge cases may get mis-classified |

### Implementation notes
- `TextParser` already exists for `.txt` files. Extend the pattern with a `LocalPDFParser` and `LocalOfficeParser`.
- CSV and HTML are sent to LlamaParse today; these should always use local parsing (Python `csv` module, `beautifulsoup4`).
- The `ParserRegistry` already supports multiple parsers. Add a routing layer in `AttachmentProcessor.process_attachment()` that checks complexity before selecting parser.

---

## Proposal 2: Local Image Pre-filtering Before Vision API

### Current behavior
`DocumentPreprocessor.preprocess()` calls `ImageProcessor.classify_and_describe()` for every email-level image attachment. This sends every image to the Vision API for classification (logo/banner/signature/icon/meaningful) and, if meaningful, generates a description. This is **two capabilities in one API call** (classification + description), but the API call itself costs ~$0.01 per image.

The estimator's `_is_meaningful_image()` already implements a free heuristic: skip files < 15KB, skip images < 100px, skip banner-shaped images (wide and short). However, **this heuristic is only used in the estimator, not in the actual pipeline**.

### Proposed approach
1. Apply the estimator's `_is_meaningful_image()` heuristic in `DocumentPreprocessor.preprocess()` *before* calling the Vision API.
2. Add filename-based filtering: skip files matching patterns like `logo*`, `banner*`, `signature*`, `icon*`, `image001.*` (Outlook inline images are almost always signatures/logos).
3. Only send images that pass both heuristics to the Vision API.

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | 30-50% of Vision API costs ($19-$32) |
| **Savings %** | 8-14% of total pipeline cost |
| **Retrieval quality** | Neutral — logos/banners have no retrieval value |
| **Implementation complexity** | Low — reuse existing `_is_meaningful_image()` from estimator |
| **Risk** | Very low — false positives only skip non-content images; meaningful images are typically large files |

### Implementation notes
- Extract `_is_meaningful_image()` from `IngestEstimator` to a shared utility (e.g., `utils.py` or `image_processor.py`).
- Call it in `DocumentPreprocessor.preprocess()` before `classify_and_describe()`.

---

## Proposal 3: LLM Call Consolidation (Context + Topics in Single Call)

### Current behavior
Per email, the pipeline makes these sequential LLM calls:
1. **Context summary** (`ContextGenerator.generate_context()`): "Summarize this document in 2-3 sentences" -> gpt-4o-mini
2. **Topic extraction** (`TopicExtractor.extract_topics()`): "Extract 1-5 topics from this email" -> gpt-4o-mini
3. **Per-attachment context summary**: Another `generate_context()` call per document attachment

These are independent calls to the same model with overlapping input (email body text).

### Proposed approach
Combine context summary and topic extraction into a single LLM call for the email body:

```
Analyze this email and return JSON:
{
  "context_summary": "2-3 sentence summary...",
  "topics": [{"name": "...", "description": "..."}]
}
```

This halves the LLM calls for the email body. Attachment context summaries remain separate (they have different input).

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | ~$1.26 (topic extraction calls eliminated; context call absorbs it) |
| **Savings %** | <1% of total pipeline cost |
| **Retrieval quality** | Neutral — same information extracted, just in one call |
| **Implementation complexity** | Medium — need to refactor `ContextGenerator` and `TopicExtractor` to support combined mode, update pipeline orchestration |
| **Risk** | Low — single-call structured output is well-supported by gpt-4o-mini |

### Verdict
**Low priority.** LLM call cost is <2% of total. The consolidation saves negligible money. Only worth doing if it also reduces latency (fewer sequential calls).

---

## Proposal 4: Embedding Model Optimization

### Current behavior
- Model: `text-embedding-3-small` (OpenAI)
- Dimensions: 1536
- Cost: $0.020 per 1M tokens
- Chunk size: 512 tokens with 50 token overlap
- Batch size: 100 (configurable)

### Observation
`text-embedding-3-small` supports `dimensions` parameter for Matryoshka reduction. OpenAI's own benchmarks show that 512 dimensions retains ~98% of MTEB performance vs 1536 dimensions. The model at 512 dims outperforms `text-embedding-ada-002` at 1536 dims.

Reducing dimensions does NOT change the API cost (same tokens processed), but it:
- Reduces database storage by ~67% (512 floats vs 1536 floats per chunk = ~4KB vs ~12KB per vector)
- Speeds up vector similarity search
- Reduces memory usage in production

### API cost alternatives

| Model | Cost/1M tokens | MTEB score | Notes |
|-------|---------------|------------|-------|
| text-embedding-3-small (current) | $0.020 | 62.3 | Current model |
| text-embedding-3-small @ 512 dims | $0.020 | ~61.0 | Same cost, smaller vectors |
| text-embedding-3-large @ 1024 dims | $0.130 | ~64.0 | 6.5x more expensive |
| Gemini text-embedding-004 | Free (rate-limited) / $0.00625/1M | ~66.0 | Cheaper, potentially better |
| Cohere embed-v4 | $0.100/1M | ~65.0 | 5x more expensive |

### Cost impact

| Metric | Value |
|--------|-------|
| **Embedding API savings** | Negligible — embeddings are <1% of cost |
| **DB storage savings** | ~67% vector storage reduction at 512 dims (significant at 100GB scale) |
| **Retrieval quality** | Slightly negative at 512 dims (2% MTEB drop), but likely unnoticeable for this use case |
| **Implementation complexity** | Low — change `EMBEDDING_DIMENSIONS=512` in config |
| **Risk** | Low — easily reversible; requires re-embedding on dimension change |

### Recommendation
- Change `embedding_dimensions` from 1536 to 512. This primarily saves **database storage** (~67% reduction in vector size), which matters at the documented production scale of ~100GB DB.
- The embedding API cost is already negligible ($0.63 total), so model switching does not meaningfully reduce cost.
- If pursuing cheaper embedding models, Gemini's `text-embedding-004` is worth evaluating but introduces a new API dependency.

---

## Proposal 5: Larger Chunk Size

### Current behavior
- Chunk size: 512 tokens
- Chunk overlap: 50 tokens
- Result: More chunks per document, more embedding calls, more DB storage

### Analysis
Maritime incident reports are typically narrative descriptions ("the engine failed because...") and technical details. For this RAG use case:
- Users ask high-level questions ("What happened with engine failures on tankers?")
- The LLM needs enough context in each chunk to understand the incident
- 512 tokens is relatively small for narrative content

Increasing chunk size to 1024 tokens with 100 overlap would:
- Reduce chunk count by ~45-50%
- Reduce embedding API calls proportionally
- Provide more context per chunk for the LLM
- Reduce total vector storage

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | ~$0.30 in embeddings, significant DB storage reduction |
| **Savings %** | <1% of API cost, ~40-50% reduction in chunk count |
| **Retrieval quality** | Potentially **positive** — more context per chunk improves LLM responses for narrative content |
| **Implementation complexity** | Low — change `CHUNK_SIZE_TOKENS=1024`, `CHUNK_OVERLAP_TOKENS=100` |
| **Risk** | Low-medium — larger chunks may reduce precision for very specific queries; needs A/B testing |

### Recommendation
Worth testing at 1024 tokens. Combined with 512 embedding dimensions (Proposal 4), this would reduce vector storage by ~73% (half the chunks, each with 1/3 the vector size).

---

## Proposal 6: Batch Optimizations

### Current behavior
- Embedding batches: Already optimized (batch_size=100, OpenAI supports up to 2048)
- LlamaParse: One API call per document (unavoidable — each doc is a separate file)
- Vision API: One call per image via OpenAI Agents SDK
- LLM calls: One call per context summary, one per topic extraction

### Observation
Most batch-level optimization is already implemented. The main area for improvement:

1. **Embedding batch size increase**: Could increase from 100 to 500-1000. This reduces HTTP round-trip overhead but does not change API cost.
2. **Parallel LlamaParse**: Already handled by `max_concurrent_files=5`. Increasing parallelism may help throughput but not cost.

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | $0 (batching reduces latency, not cost) |
| **Implementation complexity** | Low |
| **Risk** | Low |

### Verdict
**No cost reduction opportunity.** Batching is already well-implemented. Could improve throughput/latency but not API costs.

---

## Proposal 7: Caching Improvements

### Current behavior
- **Parsed content caching**: Already implemented. The archive stores `.md` files for parsed attachments. On re-ingest, cached markdown is used instead of re-calling LlamaParse (`_extract_content_from_cached_markdown` in `attachment_handler.py`).
- **Progress tracking**: Already implemented. Files with matching hashes are skipped (`version_manager.check_document()`).
- **Image results**: NOT cached. Every image is re-classified and re-described on re-ingest.
- **Context summaries**: NOT cached (regenerated per email and per attachment on re-ingest).
- **Topic extraction**: NOT cached (regenerated on re-ingest).
- **Embeddings**: NOT cached (regenerated on re-ingest).

### Proposed approach
1. **Cache image classification + description**: Store results alongside the image in the archive. On re-ingest, load from cache.
2. **Cache embeddings**: Store embedding vectors in the archive or a sidecar file. On re-ingest, reuse if chunk content unchanged.
3. **Context/topic caching**: Store per-document context summaries and topics in archive metadata.

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings on re-ingest** | 90%+ (almost all API calls eliminated on re-run) |
| **Savings on first run** | $0 (caching only helps repeat runs) |
| **Implementation complexity** | Medium — need cache storage, invalidation logic, hash-based freshness checks |
| **Risk** | Low — cache miss falls back to fresh processing |

### Recommendation
**High priority for operational cost**, especially since the version manager already triggers `reprocess` on version bumps. Adding per-artifact caching (images, context, topics, embeddings) would make version-bump re-ingests nearly free for unchanged content.

---

## Proposal 8: Model Selection for LLM Calls

### Current behavior
All LLM calls use `gpt-4o-mini` (or a configured override via `context_llm_model`, `email_cleaner_model`, `image_llm_model`).

### Comparison

| Model | Input $/1M | Output $/1M | Quality | Notes |
|-------|-----------|-------------|---------|-------|
| gpt-4o-mini (current) | $0.15 | $0.60 | Good | Well-tested for this use case |
| gpt-4.1-nano | $0.10 | $0.40 | Good | 33% cheaper, newer |
| Claude 3.5 Haiku | $0.80 | $4.00 | Very good | 5x more expensive |
| Gemini 2.0 Flash | $0.10 | $0.40 | Good | Comparable price, good quality |
| Gemini 2.5 Flash | $0.15 | $3.50 (thinking) / $0.60 (non-thinking) | Very good | Same input cost, better quality |

### Cost impact

| Metric | Value |
|--------|-------|
| **Estimated savings** | ~$1 by switching to gpt-4.1-nano or Gemini Flash (33% cheaper on already-cheap calls) |
| **Savings %** | <1% of total pipeline cost |
| **Retrieval quality** | Neutral — these are simple extraction/summarization tasks |
| **Implementation complexity** | Low — change config: `LLM_MODEL=gpt-4.1-nano` |
| **Risk** | Low — litellm already abstracts the provider; needs prompt validation |

### Verdict
**Marginal savings.** LLM calls are <2% of total cost. Switching models saves ~$1 total. Only worth doing if also pursuing latency improvements.

---

## Summary: Prioritized Recommendations

| Priority | Proposal | Est. Savings | % of Total | Complexity | Quality Impact |
|----------|----------|-------------|------------|------------|----------------|
| **1** | Tiered PDF parsing (local + LlamaParse) | $63-$94 | 28-42% | Medium | Neutral |
| **2** | Local image pre-filtering | $19-$32 | 8-14% | Low | Neutral |
| **3** | Caching (images, context, embeddings) | $0 first run / 90%+ on re-runs | Major for ops | Medium | Neutral |
| **4** | 512 embedding dimensions | $0 API / ~67% DB storage | 0% API, big storage | Low | Minimal negative |
| **5** | Larger chunks (1024 tokens) | ~40-50% fewer chunks | Storage savings | Low | Potentially positive |
| **6** | LLM call consolidation | ~$1.26 | <1% | Medium | Neutral |
| **7** | Model selection (gpt-4.1-nano) | ~$1 | <1% | Low | Neutral |
| **8** | Batch optimizations | $0 | 0% | Low | N/A |

### Top 3 actions by cost impact

1. **Tiered PDF parsing** delivers the largest single saving. LlamaParse is 70% of cost. Routing simple PDFs to local parsers could save $63-$94 per full ingest run.

2. **Local image pre-filtering** is low-effort, low-risk, and saves 8-14% by reusing the heuristic that already exists in the estimator.

3. **Caching for re-ingests** delivers massive savings on the (frequent) scenario of re-processing after version bumps or partial failures. First-run cost is unchanged, but operational cost drops dramatically.

### Combined savings estimate (first run)
Proposals 1 + 2 combined: **$82-$126 saved per full ingest** (36-56% reduction), reducing total from ~$225 to ~$99-$143.

### Combined savings estimate (re-ingest after caching)
With Proposal 3 (caching) in place, re-ingests of unchanged content cost near $0.
