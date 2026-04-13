---
purpose: Evaluate whether dedicated parsers (LlamaParse, Vision API) are optimal or if modern LLMs and open-source tools can replace them
status: research-complete
date: 2026-04-13
depends_on: [01-cost-estimation, 06a-optimization-cost-reduction]
---

# 06d -- Parser Alternatives: LLMs vs Dedicated Parsers

## Current Parser Inventory

The MTSS ingest pipeline uses five parsing mechanisms:

| Parser | Used For | Cost | Speed | Runs Locally |
|--------|----------|------|-------|-------------|
| **LlamaParse** (cost_effective tier) | PDF, DOCX, PPTX, XLSX, DOC, XLS, PPT, CSV, RTF, HTML, ODF | $0.003-0.004/page (3 credits) | 5-30s/doc (API latency) | No |
| **EML parser** (Python email stdlib) | Email files | Free | <100ms | Yes |
| **Vision API** (gpt-4o-mini via OpenAI Agents SDK) | Image classification + description | ~$0.004/image | 1-3s/image | No |
| **Text chunker** (LangChain splitters) | Markdown/text/HTML chunking | Free | <10ms | Yes |
| **pypdf** (estimator only) | Page counting for cost estimation | Free | <100ms | Yes |

### Current cost breakdown (6,289 emails, from 01-cost-estimation)

| Component | Cost | % of Total |
|-----------|------|-----------|
| LlamaParse (~12,087 pages) | $75.54 | 46% |
| Vision API (~20,783 images) | ~$83.73 | 51% |
| LLM calls (context + topics) | ~$4.54 | 3% |
| Embeddings | ~$0.27 | <1% |
| **Total** | **~$164.08** | |

LlamaParse and Vision API together account for **97%** of ingest cost.

### Current LlamaParse configuration

The parser (`llamaparse_parser.py`) uses aggressive settings:
- `tier="cost_effective"` (3 credits/page = ~$0.00375/page)
- `high_res_ocr=True`
- `adaptive_long_table=True`, `outlined_table_extraction=True`
- `output_tables_as_HTML=True`
- `specialized_image_parsing=True`
- Agentic auto-mode trigger for pages with tables/charts/images

This configuration is overkill for many simple text documents in the maritime email corpus.

---

## Alternative Comparison Matrix

### PDF Parsing Alternatives

| Alternative | Cost/Page | Speed | Quality (text) | Quality (tables) | Quality (scanned/OCR) | Local | Notes |
|-------------|-----------|-------|----------------|-------------------|----------------------|-------|-------|
| **LlamaParse** (current, cost_effective) | $0.00375 | 5-30s | Excellent | Excellent | Excellent | No | Production-proven, but external dependency |
| **LlamaParse** (agentic) | $0.0125 | 10-60s | Excellent | Excellent | Excellent | No | 3.3x more expensive, marginal quality gain |
| **Gemini 2.5 Flash** (PDF native) | ~$0.00008 | 1-5s | Very good | Good | Good | No | 47x cheaper; 258 tokens/page input + output |
| **Gemini 2.5 Flash-Lite** (batch) | ~$0.00003 | async (24h) | Good | Good | Fair | No | 125x cheaper; batch mode 50% discount |
| **Gemini 2.5 Pro** | ~$0.00032 | 2-10s | Excellent | Very good | Very good | No | 12x cheaper than LlamaParse |
| **Claude Haiku 4.5** (PDF native) | ~$0.002-0.004 | 2-5s | Very good | Good | Good | No | Similar cost to LlamaParse; dual text+image tokenization |
| **Claude Sonnet 4.6** (PDF native) | ~$0.005-0.009 | 2-5s | Excellent | Very good | Very good | No | More expensive than LlamaParse |
| **GPT-4.1-mini** (vision) | ~$0.001 | 2-5s | Good | Good | Good | No | Competitive; image token multiplier 1.62x |
| **Marker** (open-source) | Free | 0.5s/page (GPU) | Very good | Good | Good (Surya OCR) | Yes | Optional LLM flag for complex docs |
| **Docling** (IBM, open-source) | Free | 1-3s/page | Very good | Very good | Fair | Yes | Best structured output (DoclingDocument) |
| **PyMuPDF4LLM** | Free | <0.01s/page | Good | Poor (0.401 bench) | None (no OCR) | Yes | Extremely fast; text-only PDFs |
| **LiteParse** (LlamaIndex, open-source) | Free | ~500 pages/2s | Good | Fair | None | Yes | New (Mar 2026); no GPU, no Python deps |
| **Surya** (OCR toolkit) | Free | 1-5s/page (GPU) | Good | Fair | Good | Yes | 90+ languages; GPU recommended |
| **pdfplumber** | Free | <0.1s/page | Good | Good (simple tables) | None | Yes | Already a Python ecosystem tool |

#### PDF cost per page calculation details

- **Gemini 2.5 Flash**: 258 tokens/page input x $0.30/1M = $0.000077 + ~100 output tokens x $2.50/1M = $0.00025 total ~$0.00033 (using markdown extraction prompt). For simple text extraction with minimal output: ~$0.00008.
- **Gemini 2.5 Flash-Lite batch**: 258 tokens/page x $0.05/1M = $0.000013 + output ~$0.00002. Total ~$0.00003.
- **Claude Haiku 4.5**: Each PDF page = text tokens (~1,500-3,000) + image rendering. At $1/1M input: ~$0.002-0.004/page for input alone.
- **GPT-4.1-mini**: Image tokens x 1.62 multiplier at $0.40/1M input. A page image ~1,610 tokens x 1.62 = ~2,608 tokens = ~$0.001/page.

### Office Document Alternatives

| Alternative | Formats | Cost | Quality | Local | Notes |
|-------------|---------|------|---------|-------|-------|
| **LlamaParse** (current) | DOCX, PPTX, XLSX, DOC, XLS, PPT | $0.00375/page | Excellent | No | Handles all formats |
| **python-docx** | DOCX only | Free | Good (text + tables) | Yes | No style/layout inference |
| **openpyxl** | XLSX only | Free | Good (data extraction) | Yes | Direct cell access |
| **python-pptx** | PPTX only | Free | Good (text + shapes) | Yes | No visual layout |
| **Marker** | DOCX, PPTX, XLSX | Free | Good | Yes | Converts to markdown |
| **LiteParse** | DOCX, XLSX | Free | Fair | Yes | Newer, less tested |
| **LibreOffice headless** | All Office formats | Free | Good | Yes | Converts to PDF, then PDF pipeline |
| **Pandoc** | DOCX, ODT, HTML, etc. | Free | Good (text) | Yes | Universal converter |

### Image Processing Alternatives

| Alternative | Cost/Image | Speed | Classification | Description | Local | Notes |
|-------------|-----------|-------|----------------|-------------|-------|-------|
| **gpt-4o-mini via Agents SDK** (current) | ~$0.004 | 1-3s | Yes | Yes | No | Structured output via Pydantic |
| **Gemini 2.5 Flash** | ~$0.0003 | 1-2s | Yes | Yes | No | 13x cheaper; native multimodal |
| **Gemini 2.5 Flash-Lite** | ~$0.0001 | 1-2s | Yes | Yes | No | 40x cheaper |
| **GPT-4.1-nano** | ~$0.0002 | <1s | Yes | Yes | No | 20x cheaper than current |
| **Claude Haiku 4.5** | ~$0.002 | 1-3s | Yes | Yes | No | Higher quality but 2x cheaper than Sonnet |
| **Local heuristic pre-filter** | Free | <1ms | Partial (size/dim) | No | Yes | Already in estimator; catches ~73% non-content |
| **BLIP-2 / LLaVA** (local) | Free | 0.5-2s (GPU) | Limited | Yes | Yes | Requires GPU; quality varies |

---

## Analysis by Document Type

### 1. PDFs (largest cost center: ~$65 for 10,354 pages)

**Current approach**: All PDFs go to LlamaParse at $0.00375/page.

**Recommended hybrid approach** (aligns with 06a Proposal 1):

| PDF Complexity | Detection Method | Parser | Cost/Page |
|---------------|-----------------|--------|-----------|
| Simple text (extractable, no images) | pypdf `page.extract_text()` length > 0, no images | PyMuPDF4LLM or pdfplumber | Free |
| Text with tables | pypdf text check + table heuristic | Marker (local, no LLM flag) | Free |
| Scanned / image-only | pypdf text check fails | LlamaParse or Gemini 2.5 Flash | $0.00033-0.00375 |
| Complex (charts, mixed layout) | Image count > 0 + sparse text | LlamaParse (current config) | $0.00375 |

**Estimated savings**: If 50% of PDFs are simple text and 20% have basic tables:
- 5,177 simple pages x $0 = $0 (was $19.41)
- 2,071 table pages x $0 = $0 (was $7.77)
- 3,106 complex/scanned pages x $0.00375 = $11.65 (unchanged)
- **New PDF cost: ~$12 vs current ~$39** (savings: ~$27, or 70%)

If using Gemini 2.5 Flash for scanned PDFs instead of LlamaParse:
- 3,106 complex pages x $0.00033 = $1.02
- **New PDF cost: ~$1 vs current ~$39** (savings: ~$38, or 97%)

### 2. Office Documents (DOCX/XLSX/PPTX: ~942 pages, ~$3.53)

**Current approach**: All go to LlamaParse.

**Recommended approach**:

| Format | Parser | Cost | Notes |
|--------|--------|------|-------|
| DOCX | python-docx (text + tables) | Free | Falls back to LlamaParse for complex layouts |
| XLSX | openpyxl (cell data extraction) | Free | Spreadsheet data is structured; LLM parsing unnecessary |
| PPTX | python-pptx (text + shapes) | Free | Slide text extraction is straightforward |

**Estimated savings**: ~$3.53 (100% -- these are all simple extractions).

### 3. Legacy Office (DOC/XLS/PPT: ~25 pages, ~$0.09)

**Current approach**: LlamaParse (only parser that handles these).

**Recommended approach**: Keep LlamaParse OR use LibreOffice headless to convert to modern format, then use local parser. The cost is negligible ($0.09 total), so optimization is not worthwhile.

**Verdict**: Keep as-is. Not worth the complexity of adding LibreOffice as a dependency for $0.09.

### 4. CSV/RTF/HTML (trivial formats)

**Current approach**: Routed to LlamaParse (unnecessary).

**Recommended approach**:
- CSV: Python `csv` module (free, deterministic)
- RTF: `striprtf` library or pandoc (free)
- HTML: BeautifulSoup / html2text (free, already have html_to_plain_text in EML parser)

**Estimated savings**: Small (few files), but eliminates unnecessary API calls for trivial formats.

### 5. Images (~20,783 images, ~$83 -- largest actual cost center)

**Current approach**: All email-level images sent to gpt-4o-mini Vision API for classification + description in one call.

**Recommended three-tier approach**:

| Tier | Method | Cost | Catches |
|------|--------|------|---------|
| 1. Local heuristic (from estimator) | Size < 15KB, dims < 100px, banner ratio | Free | ~73% (15,140 images) |
| 2. Filename pattern filter | Skip `logo*`, `banner*`, `image00*`, etc. | Free | ~5% additional |
| 3. LLM classification + description | Gemini 2.5 Flash-Lite for remaining | ~$0.0001/image | ~22% (4,572 images) |

**Estimated savings**:
- Current: 20,783 x $0.004 = $83.13
- New: 4,572 x $0.0001 = $0.46
- **Savings: ~$83** (99.4% reduction)

Even using GPT-4.1-nano instead of Gemini:
- 4,572 x $0.0002 = $0.91
- **Savings: ~$82** (98.9% reduction)

### 6. EML Parsing (free, local -- no change needed)

The custom EML parser is well-implemented, handles encoding edge cases, conversation threading, and attachment extraction. No LLM alternative would improve on this deterministic parsing. **Keep as-is.**

---

## Unified LLM Approach Analysis

### Concept: Single LLM call for parse + summarize + extract

Instead of the current pipeline (LlamaParse -> chunk -> context LLM -> topic LLM), send the entire document to one LLM call that returns:

```json
{
  "markdown_text": "...",
  "context_summary": "2-3 sentence summary",
  "topics": [{"name": "...", "description": "..."}],
  "document_type": "inspection_report",
  "key_entities": ["vessel_name", "port", "date"]
}
```

### Evaluation

| Criterion | Assessment |
|-----------|-----------|
| **Quality** | Mixed. LLMs are good at summarization and extraction but may hallucinate or miss text in dense documents. Dedicated parsers are deterministic for text extraction. |
| **Cost** | Potentially cheaper for simple documents. For a 4-page PDF with Gemini 2.5 Flash: ~$0.0013 total (input + output) vs current $0.015 (LlamaParse) + $0.0003 (context) + $0.0002 (topics) = $0.0155. Savings: 92%. |
| **Reliability** | Lower. LLM output format may vary. Structured output helps but is not guaranteed to capture all text faithfully. |
| **Batch processing** | Gemini batch API supports async processing at 50% discount. 12,087 pages at ~$0.00003/page = $0.36 total. |
| **Edge cases** | Scanned PDFs with handwriting, maritime technical diagrams, classification society stamps -- these need OCR that LLMs alone cannot provide. LLMs process the rendered page image, which works for printed text but fails on very small text or complex layouts. |

### Verdict on unified approach

**Not recommended as the sole approach.** The risk of text extraction errors is too high for a RAG pipeline where faithful reproduction of document content is critical. However, a hybrid is compelling:

1. **Local parser for text extraction** (PyMuPDF4LLM, pdfplumber, Marker)
2. **LLM for enrichment** in a single call: context summary + topics + metadata extraction
3. **LLM as fallback** for documents where local parsing produces poor output (scanned, complex layout)

This preserves deterministic text extraction while consolidating the LLM enrichment calls.

---

## Recommended Migration Path

### Phase 1: Quick wins (low effort, high impact)

**Target savings: ~$83 (image costs)**

1. **Port estimator heuristic to preprocessor** (06a Proposal 2, already planned)
   - Move `_is_meaningful_image()` from `IngestEstimator` to `DocumentPreprocessor`
   - Add filename-based filtering
   - Eliminates ~78% of Vision API calls

2. **Switch image model from gpt-4o-mini to GPT-4.1-nano** (or Gemini 2.5 Flash-Lite)
   - Change `IMAGE_LLM_MODEL` in config
   - GPT-4.1-nano: $0.10/$0.40 per 1M tokens (vs gpt-4o-mini: $0.15/$0.60)
   - Gemini 2.5 Flash-Lite: $0.10/$0.40 (or $0.05/$0.20 batch)
   - For classification + description, quality is sufficient

3. **Add local parsers for trivial formats**
   - CSV: Python `csv` -> plain text
   - HTML: `html2text` or existing `html_to_plain_text()`
   - RTF: `striprtf` library
   - No LlamaParse calls needed for these

### Phase 2: PDF tiering (medium effort, significant impact)

**Target savings: ~$27-38 (PDF costs)**

4. **Add PDF complexity classifier**
   - Use pypdf (already a dependency) to check text extractability and image presence
   - Route simple PDFs to PyMuPDF4LLM (free, local, ~500 pages/sec)
   - Route complex/scanned PDFs to LlamaParse (or Gemini 2.5 Flash for cost savings)

5. **Add local Office document parsers**
   - python-docx for DOCX, openpyxl for XLSX, python-pptx for PPTX
   - Keep LlamaParse for DOC/XLS/PPT legacy formats only

6. **Evaluate Marker as universal local parser**
   - Handles PDF, DOCX, PPTX, XLSX, HTML, EPUB
   - GPU optional (Surya OCR for scanned docs)
   - Could replace multiple local parsers with one tool

### Phase 3: LLM provider optimization (medium effort, further cost reduction)

**Target savings: variable, depends on Phase 2 choices**

7. **Evaluate Gemini 2.5 Flash as LlamaParse replacement for complex PDFs**
   - Native PDF support at ~$0.00033/page vs $0.00375/page (11x cheaper)
   - Test on sample of maritime documents for quality validation
   - Use batch API for non-urgent processing (50% further discount)

8. **Consolidate LLM enrichment calls** (06a Proposal 3)
   - Combine context summary + topic extraction into single LLM call
   - Marginal cost savings but reduces latency

### Phase 4: Full optimization (optional, diminishing returns)

9. **Evaluate Gemini 2.5 Flash-Lite batch for all document parsing**
   - $0.00003/page in batch mode
   - 12,087 pages = $0.36 total (vs current $75.54)
   - Quality validation required; not suitable for OCR-heavy documents

10. **Consider Marker with --use_llm as unified parser**
    - Local Marker for structure + LLM for accuracy on complex pages
    - Eliminates LlamaParse dependency entirely

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Local parser misses text in complex PDFs | Medium | High (data loss) | Quality validation on 100-document sample; LlamaParse fallback for low-extraction scores |
| Gemini API quality insufficient for maritime docs | Low-Medium | Medium | Test on representative sample (inspection reports, damage photos, classification letters) |
| LLM hallucinations in text extraction | Medium | High | Use LLMs for enrichment only, not primary text extraction |
| New dependencies increase maintenance burden | Low | Low | Choose well-maintained libraries (PyMuPDF, Marker have active communities) |
| LlamaParse pricing changes | Low | Medium | Phased migration reduces dependency; local parsers as fallback |
| Gemini batch API latency (up to 24h) unacceptable | Low | Low | Use real-time API for urgent processing; batch for bulk ingest |

---

## Cost Projection Summary

| Scenario | PDF Cost | Image Cost | LLM Cost | Embed Cost | Total | vs Current |
|----------|----------|-----------|----------|-----------|-------|-----------|
| **Current** | $75.54 | $83.73 | $4.54 | $0.27 | **$164.08** | -- |
| **Phase 1** (image heuristic + model switch) | $75.54 | ~$1.83 | $4.54 | $0.27 | **$82.18** | -50% |
| **Phase 1+2** (+ PDF tiering, local office) | ~$12.00 | ~$1.83 | $4.54 | $0.27 | **$18.64** | -89% |
| **Phase 1+2+3** (+ Gemini for complex PDFs) | ~$1.00 | ~$0.46 | $3.50 | $0.27 | **$5.23** | -97% |
| **Maximum** (all Gemini Flash-Lite batch) | ~$0.36 | ~$0.46 | $3.50 | $0.27 | **$4.59** | -97% |

### Key takeaways

1. **Image processing is the biggest cost** ($83.73) and the easiest to fix. Local heuristic pre-filtering alone (already built in the estimator) eliminates 73% of API calls. Switching to a cheaper model handles the rest.

2. **LlamaParse is the second biggest cost** ($75.54) and can be reduced 84-99% through tiered local parsing. Most maritime email attachments are simple text PDFs that do not need cloud AI parsing.

3. **The EML parser should not be changed.** It is free, fast, deterministic, and handles edge cases well. No LLM can improve on structured email parsing.

4. **A unified single-LLM-call approach is risky for text extraction** but attractive for enrichment (context + topics + metadata in one call). The recommended path is deterministic local parsing for text extraction + LLM for enrichment.

5. **Gemini 2.5 Flash-Lite batch mode is the cheapest LLM option** ($0.05/1M input, $0.20/1M output) for cases where an LLM is needed. It is 75-125x cheaper than LlamaParse per page.

6. **Total pipeline cost can drop from ~$164 to ~$5** with full optimization, a 97% reduction. Even conservative Phase 1+2 achieves an 89% reduction to ~$19.

---

## Key Files

| File | Role |
|------|------|
| `src/mtss/parsers/attachment_processor.py` | Routing logic, ZIP handling |
| `src/mtss/parsers/llamaparse_parser.py` | LlamaParse integration (primary change target) |
| `src/mtss/parsers/eml_parser.py` | EML parsing (keep as-is) |
| `src/mtss/parsers/preprocessor.py` | File routing + image classification trigger |
| `src/mtss/parsers/registry.py` | Parser plugin registry (add new parsers here) |
| `src/mtss/parsers/chunker.py` | Text chunking + context generation |
| `src/mtss/processing/image_processor.py` | Vision API for images (model switch target) |
| `src/mtss/ingest/estimator.py` | Has `_is_meaningful_image()` heuristic to port |
| `src/mtss/config.py` | Model configuration, LlamaParse settings |

## Sources

- [Claude PDF Support Documentation](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Gemini Document Processing](https://ai.google.dev/gemini-api/docs/document-processing)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini 2.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite)
- [LlamaParse V2 Pricing](https://www.llamaindex.ai/blog/introducing-llamaparse-v2-simpler-better-cheaper)
- [LlamaParse Pricing Page](https://www.llamaindex.ai/pricing)
- [LiteParse (LlamaIndex open-source)](https://github.com/run-llama/liteparse)
- [Marker PDF Parser](https://github.com/datalab-to/marker)
- [Best Open-Source PDF-to-Markdown Tools in 2026](https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026)
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [5 Best Document Parsers in 2026](https://www.f22labs.com/blogs/5-best-document-parsers-in-2025-tested/)
- [GPT-4.1 Introduction](https://openai.com/index/gpt-4-1/)
- [OpenAI API Pricing](https://platform.openai.com/docs/pricing/)
- [MDKeyChunker Single-Call LLM Enrichment](https://arxiv.org/html/2603.23533)
- [6,000 Pages per Dollar: Gemini 2.0 Flash PDF Processing](https://medium.com/ai-simplified-in-plain-english/6-000-pages-per-dollar-how-gemini-2-0-flash-crushes-pdf-processing-costs-19637618243a)
- [Gemini 2.5 Cost and Quality Comparison](https://www.leanware.co/insights/gemini-2-5-cost-quality-comparison)
