---
purpose: Before/after optimization plan for the MTSS ingest pipeline based on actual estimate data
status: actionable
date: 2026-04-13
depends_on: [estimate-full-results, 06a-optimization-cost-reduction, 06c-optimization-processing-speed, 06d-parser-alternatives, 06e-llm-provider-comparison]
---

# Optimization Plan: Before & After

## Baseline (Current Pipeline)

Based on the full estimate run of 6,289 EML files (24,974 attachments) on 2026-04-13.

| Category | Count | Unit Cost | Total |
|----------|-------|-----------|-------|
| LlamaParse PDF pages | 19,432 pages | $0.00625 | $121.45 |
| LlamaParse DOCX pages | 479 pages | $0.00625 | $2.99 |
| LlamaParse XLSX pages (sheets) | 734 pages | $0.00625 | $4.59 |
| LlamaParse DOC pages | 51 pages | $0.00625 | $0.32 |
| LlamaParse XLS pages (sheets) | 299 pages | $0.00625 | $1.87 |
| Vision API (meaningful images) | 9,860 images | $0.01000 | $98.60 |
| LLM text processing | 115 files | $0.00100 | $0.12 |
| Embeddings (~1.5x pages) | ~31,492 chunks | $0.00002 | $0.63 |
| **TOTAL** | | | **$230.56** |

### Cost Distribution

- **LlamaParse: $131.22 (56.9%)** -- dominant cost center
- **Vision API: $98.60 (42.8%)** -- second largest
- **LLM text + embeddings: $0.75 (0.3%)** -- negligible

### Pricing Note

The estimator uses $0.00625/page (5 avg credits at $0.00125/credit). The pipeline is configured for the `cost_effective` tier at 3 credits/page = $0.00375/page. At the actual tier price, LlamaParse cost would be $78.73 instead of $131.22, and the total would be $178.07. This plan uses the estimator's conservative $0.00625/page as the baseline for worst-case planning.

---

## Optimization Phase 1: Quick Wins (minimal code changes)

### 1a. Image pre-filtering (port estimator heuristic to pipeline)

The estimator already skips 10,824 non-content images (logos, icons, banners) using size/dimension heuristics in `_is_meaningful_image()`. The live pipeline does NOT use this heuristic -- it sends all images to the Vision API. Porting the heuristic to `DocumentPreprocessor.preprocess()` eliminates 52% of all images before any API call.

Additionally, add filename-based filtering (`logo*`, `banner*`, `image001.*`) to catch another ~5% of non-content images.

**Implementation:** Extract `_is_meaningful_image()` from `IngestEstimator` to a shared utility. Call it in `DocumentPreprocessor.preprocess()` before `classify_and_describe()`. (~20 lines changed)

### 1b. Switch image model to GPT-4.1-nano

Replace gpt-4o-mini ($0.15/$0.60 per MTok) with GPT-4.1-nano ($0.05/$0.20 per MTok) for image classification. For simple classify+describe tasks, the cheaper model is sufficient. Per-image cost drops from ~$0.01 to ~$0.0002.

**Implementation:** Change `IMAGE_LLM_MODEL` in config. (~1 line)

### 1c. Local parsing for trivial formats

CSV, RTF, HTML files are currently routed to LlamaParse unnecessarily. Use Python `csv`, `striprtf`, and `html2text` (or existing `html_to_plain_text()`) for free local parsing.

**Implementation:** Add format checks in `ParserRegistry` routing. (~30 lines)

### Phase 1 Cost Impact

| Category | Before | After | Savings |
|----------|--------|-------|---------|
| LlamaParse (documents) | $131.22 (20,995 pages) | $131.22 (unchanged*) | $0.00 |
| Vision API (images) | $98.60 (9,860 images) | $1.18 (5,900 images x $0.0002) | $97.42 |
| LLM text processing | $0.12 | $0.12 | $0.00 |
| Embeddings | $0.63 | $0.63 | $0.00 |
| **TOTAL** | **$230.56** | **$133.15** | **$97.41** |

*Trivial format savings (CSV/HTML/RTF) are minor and lumped into rounding. The main savings come from image cost reduction.

The estimator already pre-filters to 9,860 "meaningful" images. The additional savings in Phase 1 come from switching the per-image model cost from $0.01 to $0.0002 for those 9,860 images, plus applying filename-based filtering to reduce the count further to ~5,900.

---

## Optimization Phase 2: Parser Replacement

### 2a. Tiered PDF parsing (local for simple, LlamaParse for complex)

Use pypdf (already a dependency) to classify each PDF:
- **Simple text PDFs** (extractable text, no images): Parse with PyMuPDF4LLM or pdfplumber locally for free
- **Complex PDFs** (scanned, images, tables, forms): Keep LlamaParse (or switch to GPT-4.1-mini batch)

Estimated split based on maritime email corpus analysis: ~50% simple text, ~20% basic tables, ~30% complex/scanned.

### 2b. Local Office document parsers

- **DOCX:** python-docx (free, local) -- 154 files, 479 pages
- **XLSX:** openpyxl (free, local) -- 288 files, 734 pages
- **DOC/XLS:** Keep LlamaParse (15 + 174 files = $2.19, not worth optimizing)

### 2c. GPT-4.1-mini batch for complex/scanned PDFs (optional, replaces LlamaParse)

For the ~30% of PDFs that need cloud parsing (~5,830 pages), use GPT-4.1-mini batch API instead of LlamaParse: $0.000381/page batch vs $0.00625/page LlamaParse. If quality validation fails, fall back to Gemini 2.5 Flash at $0.000414/page batch.

### Phase 2 Cost Impact

| Category | Before (Phase 1) | After | Savings |
|----------|-------------------|-------|---------|
| LlamaParse -- simple PDFs (~9,716 pages) | $60.73 | $0.00 (local PyMuPDF4LLM) | $60.73 |
| LlamaParse -- table PDFs (~3,886 pages) | $24.29 | $0.00 (local Marker/pdfplumber) | $24.29 |
| LlamaParse -- complex/scanned PDFs (~5,830 pages) | $36.44 | $2.22 (GPT-4.1-mini batch) | $34.22 |
| LlamaParse -- DOCX (479 pages) | $2.99 | $0.00 (python-docx) | $2.99 |
| LlamaParse -- XLSX (734 pages) | $4.59 | $0.00 (openpyxl) | $4.59 |
| LlamaParse -- DOC + XLS (350 pages) | $2.19 | $2.19 (keep LlamaParse) | $0.00 |
| Vision API | $1.18 | $1.18 | $0.00 |
| LLM text + embeddings | $0.75 | $0.75 | $0.00 |
| **TOTAL** | **$133.15** | **$6.34** | **$126.81** |

---

## Optimization Phase 3: LLM Consolidation

### 3a. Single LLM call for summary + topics + metadata

Combine context summary and topic extraction into one structured output call per email body. Eliminates one LLM call per email (6,289 calls saved).

Current: 2 calls/email x 6,289 = 12,578 LLM calls
After: 1 call/email x 6,289 = 6,289 LLM calls

### 3b. Batch API for non-urgent processing

Use OpenAI Batch API (50% discount) for all LLM enrichment calls during bulk ingest. Context summaries and topic extraction are not latency-sensitive during batch ingest.

### 3c. Switch LLM to GPT-4.1-nano

Replace gpt-4o-mini with GPT-4.1-nano for context/topic calls. Same quality for simple extraction tasks, 33% cheaper.

### Phase 3 Cost Impact

| Category | Before (Phase 1+2) | After | Savings |
|----------|---------------------|-------|---------|
| Complex PDF parsing (GPT-4.1-mini batch) | $2.22 | $2.22 | $0.00 |
| Vision API (images) | $1.18 | $1.18 | $0.00 |
| LLM context + topics (6,289 emails) | ~$1.89 | ~$0.63 (consolidated + batch) | $1.26 |
| LLM attachment context (~3,863 docs) | ~$1.16 | ~$0.39 (batch discount) | $0.77 |
| Embeddings | $0.63 | $0.63 | $0.00 |
| DOC/XLS LlamaParse | $2.19 | $2.19 | $0.00 |
| LLM text files | $0.12 | $0.06 (batch) | $0.06 |
| **TOTAL** | **$6.34** | **$4.25** | **$2.09** |

---

## Total Before & After

| Phase | Total Cost | Savings vs Baseline | % Reduction |
|-------|-----------|--------------------:|------------:|
| Baseline (current pipeline) | $230.56 | -- | -- |
| After Phase 1 (image filtering + model switch) | $133.15 | $97.41 | 42% |
| After Phase 1+2 (+ parser replacement) | $6.34 | $224.22 | 97% |
| After Phase 1+2+3 (+ LLM consolidation) | $4.25 | $226.31 | 98% |

### At actual LlamaParse tier pricing ($0.00375/page)

| Phase | Total Cost | Savings vs Baseline | % Reduction |
|-------|-----------|--------------------:|------------:|
| Baseline (cost_effective tier) | $178.07 | -- | -- |
| After Phase 1 | $80.66 | $97.41 | 55% |
| After Phase 1+2 | $6.34 | $171.73 | 96% |
| After Phase 1+2+3 | $4.25 | $173.82 | 98% |

---

## Processing Time Before & After

> **Note:** These estimates pre-date the local-only architecture decision. See
> [06c-review-findings.md](06c-review-findings.md) for revised estimates (~53 minutes with local parsers
> and local storage).

Based on analysis from 06c-optimization-processing-speed.

### Current Pipeline

- 6,289 files at ~30s avg/file with 5 concurrent workers
- Effective throughput: ~10 files/min
- **Total: ~10.5 hours**

### After Speed Phase 1 (quick wins, 1-2 days effort)

Parallelize attachments within email (P1), increase MAX_CONCURRENT_FILES to 8-10 (P4), parallelize context+topic generation (P2).

- Avg/file drops to ~15-20s, concurrency rises to 8-10
- Effective throughput: ~25-35 files/min
- **Total: ~3-4 hours**

### After Speed Phase 2 (medium effort, 2-3 days)

Local image filtering (P8), batch topic embeddings (P7), pre-fetch skip data (P9).

- Additional 10-20% speedup
- **Total: ~2.5-3.5 hours**

### After All Speed Phases

- Avg/file drops to ~10-15s, optimized concurrency
- Effective throughput: ~40-60 files/min
- **Total: ~1.5-2.5 hours**

### Speed Improvement with Cost Optimization

Replacing LlamaParse API calls (5-30s per document) with local parsers (< 0.1s per document) for 70% of documents further reduces per-file time. Combined with concurrency improvements:

- **Estimated total: ~1-2 hours** for full ingest of 6,289 emails

---

## Implementation Effort

### Phase 1: Image Cost Reduction

| Task | Files to Change | Estimated Effort | Risk |
|------|----------------|-----------------|------|
| Extract `_is_meaningful_image()` to shared utility | `src/mtss/ingest/estimator.py`, new `src/mtss/utils/image_utils.py` | 2 hours | Very low |
| Add heuristic call in `DocumentPreprocessor.preprocess()` | `src/mtss/parsers/preprocessor.py` | 1 hour | Very low |
| Add filename-based filtering | `src/mtss/parsers/preprocessor.py` | 1 hour | Very low |
| Switch IMAGE_LLM_MODEL to GPT-4.1-nano | `src/mtss/config.py` or `.env` | 10 min | Low |
| Add local parsers for CSV/HTML/RTF | `src/mtss/parsers/registry.py`, new parser classes | 4 hours | Low |
| **Total Phase 1** | | **~1 day** | **Low** |

### Phase 2: Parser Replacement

| Task | Files to Change | Estimated Effort | Risk |
|------|----------------|-----------------|------|
| PDF complexity classifier (pypdf-based) | New `src/mtss/parsers/pdf_classifier.py` | 4 hours | Medium |
| Local PDF parser (PyMuPDF4LLM) | New `src/mtss/parsers/local_pdf_parser.py` | 4 hours | Low |
| Routing logic in AttachmentProcessor | `src/mtss/parsers/attachment_processor.py` | 4 hours | Medium |
| python-docx parser for DOCX | New `src/mtss/parsers/local_office_parser.py` | 4 hours | Low |
| openpyxl parser for XLSX | Same file as above | 3 hours | Low |
| Quality validation on 100-doc sample | Testing | 4 hours | N/A |
| GPT-4.1-mini batch integration for complex PDFs | `src/mtss/parsers/llamaparse_parser.py` replacement | 8 hours | Medium |
| **Total Phase 2** | | **~4-5 days** | **Medium** |

### Phase 3: LLM Consolidation

| Task | Files to Change | Estimated Effort | Risk |
|------|----------------|-----------------|------|
| Combine context + topic extraction prompt | `src/mtss/parsers/chunker.py`, context/topic generators | 4 hours | Low |
| Batch API integration | Pipeline orchestration | 8 hours | Medium |
| Switch to GPT-4.1-nano for enrichment | Config change | 10 min | Low |
| **Total Phase 3** | | **~2 days** | **Low-Medium** |

---

---

## Retrieval Quality Quick Wins (from 06b review)

> **Implementation:** See [02-implementation.md](../02-implementation.md) for code-level details.
> See [06b-review-findings.md](06b-review-findings.md) for edge case analysis (e.g., ZIP attachment handling for RQ-1).

These affect embedding text and **must be implemented before the first ingest run**. Changing them later requires re-embedding all chunks (costly). All three have zero additional LLM/API cost.

### RQ-1: Attachment Context Inheritance (P6)

**Problem:** Attachment chunks get their own context summary generated from attachment content alone. A PDF inspection report attached to "RE: Main Engine Cylinder 3 Crack - MARAN CASTOR" gets summarized as "This is a PDF inspection report with tables and measurements" — no mention of the vessel or incident.

**Implementation:**

1. Add `email_context_summary: str | None = None` parameter to `process_attachment()` in `attachment_handler.py`
2. In `pipeline.py:355`, pass `email_context_summary=context_summary` to `process_attachment()`
3. In `attachment_handler.py:222-228`, when building attachment embedding text, prepend the email context:
   ```python
   if email_context_summary and attach_context:
       full_context = f"{email_context_summary}\n\n{attach_context}"
   elif email_context_summary:
       full_context = email_context_summary
   else:
       full_context = attach_context
   chunk.embedding_text = components.context_generator.build_embedding_text(
       full_context, chunk.content
   )
   ```
4. For image chunks (lines 152-167), also apply the email context summary

**Files changed:** `pipeline.py` (~2 lines), `attachment_handler.py` (~15 lines)
**Effort:** 30 minutes
**Risk:** Very low. Worst case: slightly longer embedding text (well within 8000 token limit)

### RQ-2: Minimum Content Filter (P1-A)

**Problem:** Short messages like "Noted, thanks" or "Please see attached" become standalone chunks with full context summaries prepended, making them appear semantically rich when they carry no incident information.

**Implementation:**

1. In `pipeline.py`, after line 289 (`cleaned_message = remove_boilerplate_from_message(message)`), add:
   ```python
   # Skip chunks below minimum meaningful content threshold
   if components.context_generator:
       token_count = components.context_generator.count_tokens(cleaned_message)
       if token_count < 30:
           continue
   ```
   Note: `ContextGenerator` does not currently have `count_tokens`. Use the `DocumentChunker.count_tokens()` method instead, or add a simple `len(cleaned_message.split()) < 20` word-count heuristic to avoid a dependency.

**Simpler alternative (recommended):** Use word count instead of token count to avoid needing a tokenizer:
   ```python
   if len(cleaned_message.split()) < 20:
       continue
   ```
   20 words approximates 25-30 tokens. This avoids adding a chunker dependency to the pipeline.

**Files changed:** `pipeline.py` (~3 lines)
**Effort:** 15 minutes
**Risk:** Very low. Edge case: a short but critical message like "Engine room flooded" (3 words) would be skipped. Mitigation: the parent email context summary already captures the incident; the short message itself adds no retrieval value that the subject line doesn't already provide.

### RQ-3: Date in Embedding Text (P8-A)

**Problem:** Embedding text for "we replaced the pump" has no temporal signal. Queries like "what was done about the pump last year" rely entirely on vector similarity without date context.

**Implementation:**

1. In `pipeline.py`, between lines 304-306, prepend date to embedding text:
   ```python
   # Prepend date for temporal search relevance
   date_prefix = ""
   if email_doc.email_metadata and email_doc.email_metadata.date_start:
       date_prefix = f"[Date: {email_doc.email_metadata.date_start.strftime('%Y-%m-%d')}]\n"

   embedding_text = cleaned_message
   if context_summary:
       embedding_text = components.context_generator.build_embedding_text(context_summary, cleaned_message)
   embedding_text = date_prefix + embedding_text
   ```
2. In `attachment_handler.py`, apply the same date prefix to attachment chunk embedding text. The email date is available via `email_doc.email_metadata.date_start`.

**Files changed:** `pipeline.py` (~5 lines), `attachment_handler.py` (~5 lines)
**Effort:** 20 minutes
**Risk:** Very low. The date prefix adds ~15 characters, well within embedding model limits. Embedding models handle structured date formats well.

### Retrieval Quality Implementation Summary

| Task | Files | Effort | Risk |
|------|-------|--------|------|
| RQ-1: Attachment context inheritance | `pipeline.py`, `attachment_handler.py` | 30 min | Very low |
| RQ-2: Minimum content filter | `pipeline.py` | 15 min | Very low |
| RQ-3: Date in embedding text | `pipeline.py`, `attachment_handler.py` | 20 min | Very low |
| **Total** | | **~1 hour** | **Very low** |

---

## Recommendation

### Before the first ingest run: implement Cost Phase 1 + Retrieval Quality

Cost Phase 1 saves $97.41 (42%) with only ~1 day of effort and very low risk. The image heuristic already exists in the estimator and just needs to be ported to the live pipeline. The model switch is a config change.

Retrieval quality quick wins (RQ-1, RQ-2, RQ-3) add ~1 hour of effort with zero additional cost. They **must** be done before the first ingest because they change embedding text. Doing them after requires re-embedding all chunks.

**Do not delay the first ingest for Phase 2.** Phase 2 requires quality validation of local parsers on a representative sample. Run the first ingest with Phase 1 + RQ optimizations at ~$133 cost, then implement Phase 2 for subsequent runs.

### After the first ingest run: implement Phase 2

Phase 2 delivers the largest absolute savings ($126.81) but requires a PDF complexity classifier and quality validation. Use the first ingest run's parsed output as a quality benchmark for local parser comparison.

### Phase 3 is optional

Phase 3 saves only $2.09. Implement only if the codebase is being refactored anyway or if re-ingest frequency makes the cumulative savings worthwhile.

### Summary

| Recommendation | Cost | Effort | When |
|---------------|------|--------|------|
| **Phase 1 + RQ: Do before first ingest** | $133.15 | 1 day + 1 hour | Now |
| **Phase 2: Do before second ingest** | $6.34 | 4-5 days | After first run |
| Phase 3: Optional | $4.25 | 2 days | When convenient |

The pipeline cost drops from **$230.56 to $133.15 (Phase 1)** with minimal effort, and to **$6.34 (Phase 1+2)** with moderate effort -- a **97% total reduction**. Retrieval quality quick wins improve search precision and recall at no additional cost.

---

## Search/Retrieval Fixes

See [01-critical-fixes.md](../01-critical-fixes.md) for the authoritative implementation plan covering:
- SR-0/Fix 0.1: Reranker bug fix (P0 CRITICAL)
- SR-1/Fix 1.1: Enriched rerank context
- SR-2/Fix 1.2: max_tokens increase
- SR-3/Fix 1.3: rerank_top_n increase
- Fix 2.1-2.5: Additional search improvements (top_k, HNSW, parallel, score floor, transparency)
