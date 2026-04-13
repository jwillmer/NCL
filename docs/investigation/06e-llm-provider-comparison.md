---
purpose: Compare LLM providers for document parsing — cost, quality, and integration tradeoffs
status: research-complete
date: 2026-04-13
depends_on: [06d-parser-alternatives]
---

# 06e -- LLM Provider Comparison for Document Parsing

## Context

The MTSS pipeline needs an LLM provider for parsing complex/scanned PDFs and classifying images — tasks where local parsers fall short. The existing stack already uses **OpenAI** heavily (text-embedding-3-small, gpt-4o-mini, gpt-4o). Adding a second provider increases API key management, SDK dependencies, and operational complexity.

**Use case**: Parse maritime incident report attachments — inspection reports, survey reports, classification society letters, damage reports, equipment photos. Output: markdown for chunking and embedding.

**Volume**: ~25,000 document pages + ~5,600 meaningful images (after local heuristic pre-filtering).

---

## Provider Pricing Summary (April 2026)

### Document Parsing Models (PDF pages as input)

| Provider / Model | Input $/MTok | Output $/MTok | Batch Input | Batch Output | PDF Input Method | Tokens/Page |
|------------------|-------------|--------------|-------------|-------------|-----------------|-------------|
| **Gemini 2.5 Flash** | $0.30 | $2.50 | $0.15 | $1.25 | Native PDF (258 tok/page) | ~258 |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.05 | $0.20 | Native PDF (258 tok/page) | ~258 |
| **Gemini 3 Flash** | $0.50 | $3.00 | $0.25 | $1.50 | Native PDF (258 tok/page) | ~258 |
| **GPT-4.1-mini** | $0.20 | $0.80 | $0.10 | $0.40 | Vision (image tiles, ~1,610 tok x 1.62 = ~2,608) | ~2,608 |
| **GPT-4.1-nano** | $0.05 | $0.20 | $0.025 | $0.10 | Vision (image tiles, ~1,610 tok x 2.46 = ~3,961) | ~3,961 |
| **GPT-4o-mini** | $0.15 | $0.60 | $0.075 | $0.30 | Vision (image tiles) | ~2,608 |
| **Claude Haiku 4.5** | $1.00 | $5.00 | $0.50 | $2.50 | Native PDF (text + image, ~1,500-3,000) | ~2,000 |
| **Claude Haiku 3.5** | $0.80 | $4.00 | $0.40 | $2.00 | Native PDF (text + image) | ~2,000 |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | $1.50 | $7.50 | Native PDF (text + image) | ~2,000 |
| **Mistral OCR 3** | $2.00/1K pages | -- | $1.00/1K pages | -- | Dedicated OCR API | N/A |

### Image Classification Models

| Provider / Model | Input $/MTok | Output $/MTok | Batch Input | Batch Output |
|------------------|-------------|--------------|-------------|-------------|
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.05 | $0.20 |
| **GPT-4.1-nano** | $0.05 | $0.20 | $0.025 | $0.10 |
| **GPT-4o-mini** (current) | $0.15 | $0.60 | $0.075 | $0.30 |
| **Claude Haiku 4.5** | $1.00 | $5.00 | $0.50 | $2.50 |

---

## Cost Per Page Calculation

Assumptions: Each page produces ~300 output tokens (markdown extraction). Prompt/system tokens: ~200 tokens overhead.

### PDF Parsing Cost Per Page

| Provider / Model | Input Cost | Output Cost | **Total/Page** | **Batch/Page** |
|------------------|-----------|-------------|---------------|---------------|
| **Gemini 2.5 Flash-Lite** | 258 tok x $0.10/M = $0.000026 | 300 tok x $0.40/M = $0.000120 | **$0.000146** | **$0.000073** |
| **Gemini 2.5 Flash** | 258 tok x $0.30/M = $0.000077 | 300 tok x $2.50/M = $0.000750 | **$0.000827** | **$0.000414** |
| **GPT-4.1-nano** | 3,961 tok x $0.05/M = $0.000198 | 300 tok x $0.20/M = $0.000060 | **$0.000258** | **$0.000129** |
| **GPT-4.1-mini** | 2,608 tok x $0.20/M = $0.000522 | 300 tok x $0.80/M = $0.000240 | **$0.000762** | **$0.000381** |
| **Gemini 3 Flash** | 258 tok x $0.50/M = $0.000129 | 300 tok x $3.00/M = $0.000900 | **$0.001029** | **$0.000515** |
| **Mistral OCR 3** | flat rate | -- | **$0.002000** | **$0.001000** |
| **GPT-4o-mini** | 2,608 tok x $0.15/M = $0.000391 | 300 tok x $0.60/M = $0.000180 | **$0.000571** | **$0.000286** |
| **Claude Haiku 3.5** | 2,000 tok x $0.80/M = $0.001600 | 300 tok x $4.00/M = $0.001200 | **$0.002800** | **$0.001400** |
| **Claude Haiku 4.5** | 2,000 tok x $1.00/M = $0.002000 | 300 tok x $5.00/M = $0.001500 | **$0.003500** | **$0.001750** |
| **LlamaParse** (current) | flat rate | -- | **$0.003750** | N/A |
| **Claude Sonnet 4.6** | 2,000 tok x $3.00/M = $0.006000 | 300 tok x $15.00/M = $0.004500 | **$0.010500** | **$0.005250** |

**Sorted by batch cost**, cheapest first.

---

## Full Ingest Cost Projection

Volume: 25,000 pages (complex/scanned PDFs only — simple text PDFs handled locally for free).

If only ~3,100 pages need LLM parsing (after local parser tiering from 06d), costs scale proportionally.

| Provider / Model | 25K pages (all) | 25K batch | 3.1K pages | 3.1K batch |
|------------------|----------------|-----------|-----------|-----------|
| **Gemini 2.5 Flash-Lite** | $3.65 | $1.83 | $0.45 | $0.23 |
| **GPT-4.1-nano** | $6.45 | $3.23 | $0.80 | $0.40 |
| **GPT-4o-mini** | $14.28 | $7.15 | $1.77 | $0.89 |
| **GPT-4.1-mini** | $19.05 | $9.53 | $2.36 | $1.18 |
| **Gemini 2.5 Flash** | $20.68 | $10.35 | $2.56 | $1.28 |
| **Gemini 3 Flash** | $25.73 | $12.88 | $3.19 | $1.60 |
| **Mistral OCR 3** | $50.00 | $25.00 | $6.20 | $3.10 |
| **Claude Haiku 3.5** | $70.00 | $35.00 | $8.68 | $4.34 |
| **Claude Haiku 4.5** | $87.50 | $43.75 | $10.85 | $5.43 |
| **LlamaParse** (current) | $93.75 | N/A | $11.63 | N/A |
| **Claude Sonnet 4.6** | $262.50 | $131.25 | $32.55 | $16.28 |

---

## Quality Comparison for Document Parsing

Based on the Applied AI benchmark (800+ documents, 7 frontier LLMs) and other sources.

| Criterion | Gemini Flash/Pro | GPT-4.1-mini/nano | Claude Haiku | Mistral OCR 3 |
|-----------|-----------------|-------------------|-------------|---------------|
| **Text extraction accuracy** | Very good (88% edit similarity for Pro; Flash close behind) | Good (75% for 4o-mini; 4.1-mini likely similar) | Good (but variable; Haiku collapsed to 8% on academic papers) | Excellent (dedicated OCR; 74% win rate over OCR 2) |
| **Structure preservation** | Good (native PDF understands layout) | Poor-Fair (4o-mini: 13% structure preservation — "actively harmful" for RAG) | Fair-Good (dual text+image tokenization) | Very good (HTML table output with colspan/rowspan) |
| **Table extraction** | Good (native PDF sees table structure) | Fair (image-based, loses structure) | Good (sees both text and image layers) | Very good (reconstructs headers, merged cells, multi-row blocks) |
| **Scanned/OCR documents** | Good (processes rendered page image) | Good (processes page as image) | Good (renders each page as image) | Excellent (dedicated OCR engine, handles low DPI, skew, noise) |
| **Handwritten content** | Fair-Good | Fair | Fair | Very good (improved cursive + mixed-content detection) |
| **Maritime technical docs** | Not benchmarked specifically; native PDF likely handles inspection reports well | Not benchmarked; image-only approach may miss fine details | Not benchmarked; dual tokenization helps with mixed content | Not benchmarked; strong OCR should handle stamps and annotations |

### Key Quality Findings

1. **GPT-4o-mini has a critical flaw**: 13% structure preservation despite 75% text extraction. It extracts words but loses document structure, which is "actively harmful" for RAG pipelines. GPT-4.1-mini may improve on this but is not yet benchmarked independently.

2. **Gemini's native PDF support is a major advantage**: At 258 tokens/page, Gemini processes PDFs natively rather than converting to images. This preserves text fidelity and costs far less in tokens than image-based approaches (2,000-4,000 tokens/page for OpenAI/Claude).

3. **Mistral OCR 3 is purpose-built for document parsing**: Dedicated OCR API with excellent table and handwriting support, but at $2/1K pages it is 14x more expensive than Gemini 2.5 Flash-Lite.

4. **Claude's PDF processing is expensive**: Each page is tokenized as both text AND image, resulting in ~2,000 tokens/page — 8x more than Gemini's 258 tokens/page.

---

## Integration Effort Analysis

| Provider | Already in Stack? | SDK/Library | LiteLLM Support | Additional Config |
|----------|------------------|-------------|----------------|-------------------|
| **OpenAI** | Yes (embeddings, summaries, RAG) | openai SDK installed | Full support | None — zero effort |
| **Google Gemini** | No | google-genai SDK needed | Full support via `gemini/` prefix | New API key, new dependency |
| **Anthropic Claude** | No | anthropic SDK needed | Full support via `anthropic/` prefix | New API key, new dependency |
| **Mistral** | No | mistral SDK or REST API | Supported via `mistral/` prefix | New API key, dedicated OCR endpoint |

### "Stay with OpenAI" Argument

Advantages of consolidating on OpenAI:
- **Zero integration overhead**: API key already configured, SDK already installed
- **LiteLLM already routes** all LLM calls through a unified interface — but the OpenAI key is already there
- **Single billing relationship**: One invoice, one usage dashboard
- **Consolidation opportunity**: Parse + summarize + extract topics in ONE call (saves latency)
- **GPT-4.1-nano batch** at $0.000129/page is the second cheapest option overall
- **Operational simplicity**: One fewer secret to manage, one fewer SDK to update

Disadvantages:
- **Quality risk**: GPT-4o-mini's 13% structure preservation is concerning; GPT-4.1-mini/nano not independently benchmarked for PDF parsing yet
- **No native PDF support**: OpenAI processes PDFs as images only (higher token cost per page)
- **Token efficiency**: 2,608-3,961 tokens/page vs Gemini's 258 tokens/page

---

## The Gemini Advantage Explained

The previous investigation (06d) recommended Gemini 2.5 Flash for complex PDF parsing. Here is why:

### 1. Native PDF tokenization (258 vs 2,000-4,000 tokens/page)

Gemini processes PDFs natively at 258 tokens per page regardless of content density. OpenAI and Claude must convert each page to an image, consuming 2,000-4,000 tokens per page. This 8-15x token efficiency gap translates directly to cost savings.

### 2. Cheapest batch processing

Gemini 2.5 Flash-Lite batch at $0.000073/page is **51x cheaper than LlamaParse** and **1.8x cheaper than GPT-4.1-nano batch**. Even Gemini 2.5 Flash standard at $0.000827/page is 4.5x cheaper than LlamaParse.

### 3. Strong quality for document understanding

Gemini Flash achieved 95% edit similarity on legal contracts in the Applied AI benchmark. For the MTSS use case (inspection reports with tables, typed text, some stamps), Gemini's native PDF understanding should perform well.

### 4. 50% batch API discount

All Gemini models offer a 50% batch discount for async processing — ideal for bulk ingest where latency is not critical.

---

## Scenario Analysis

### Scenario A: Stay with OpenAI (GPT-4.1-nano batch)

- **Cost**: 3,100 pages x $0.000129 = **$0.40** (complex PDFs only)
- **Quality**: Unverified for document parsing; nano model may struggle with fine details
- **Integration**: Zero additional effort
- **Risk**: Medium — nano is the smallest model; quality for tables/stamps unknown

### Scenario B: Stay with OpenAI (GPT-4.1-mini batch)

- **Cost**: 3,100 pages x $0.000381 = **$1.18** (complex PDFs only)
- **Quality**: Better than nano; 4.1-mini is a capable model but still image-based PDF processing
- **Integration**: Zero additional effort
- **Risk**: Low-Medium — good general model but no native PDF understanding

### Scenario C: Add Gemini 2.5 Flash-Lite batch (cheapest overall)

- **Cost**: 3,100 pages x $0.000073 = **$0.23** (complex PDFs only)
- **Quality**: Good for text extraction; may struggle with very complex layouts
- **Integration**: New API key + `google-genai` SDK (but LiteLLM handles routing)
- **Risk**: Low — well-established model; native PDF support; cheapest option

### Scenario D: Add Gemini 2.5 Flash (best cost/quality balance)

- **Cost**: 3,100 pages x $0.000414 = **$1.28** (complex PDFs only, batch)
- **Quality**: Very good; native PDF; strong benchmark results
- **Integration**: Same as Scenario C
- **Risk**: Low — recommended in 06d for good reason

### Scenario E: Mistral OCR 3 batch (best quality for OCR-heavy docs)

- **Cost**: 3,100 pages x $0.001 = **$3.10** (complex PDFs only)
- **Quality**: Excellent for OCR, handwriting, tables; purpose-built
- **Integration**: New API key + dedicated OCR endpoint (different from chat API)
- **Risk**: Low — dedicated tool; but overkill if most pages are typed text

### Scenario F: Add Claude Haiku 3.5 batch

- **Cost**: 3,100 pages x $0.001400 = **$4.34** (complex PDFs only)
- **Quality**: Good-Very good; native PDF with dual tokenization
- **Integration**: New API key + `anthropic` SDK
- **Risk**: Low-Medium — good model but the Haiku line has shown variable quality on complex documents

---

## Recommendation

### Primary: GPT-4.1-mini batch (Scenario B) with quality validation

**Rationale:**

1. **Already in the stack** — zero integration overhead; the OpenAI API key and SDK are already configured
2. **Competitive cost** — $1.18 for the entire complex PDF corpus; even without batch API, $2.36 is trivial
3. **Good baseline quality** — GPT-4.1-mini is a significantly better model than GPT-4o-mini; the 4.1 series was specifically improved for instruction following and long-context tasks
4. **Simplicity** — one provider for everything (embeddings, parsing, summarization, RAG generation)
5. **Batch API** — 50% discount for non-urgent ingest processing
6. **Consolidation** — can combine parse + context summary + topic extraction in a single call

### Fallback: Add Gemini 2.5 Flash if OpenAI quality is insufficient

If quality validation on a 50-100 document sample shows GPT-4.1-mini struggles with:
- Table structure preservation
- Scanned document OCR
- Maritime-specific content (stamps, handwritten annotations)

Then add Gemini 2.5 Flash as the parsing provider:
- Native PDF at 258 tokens/page is fundamentally more efficient
- $1.28 batch cost for the entire complex PDF corpus
- LiteLLM supports Gemini via `gemini/gemini-2.5-flash` prefix — integration is straightforward

### Not recommended

- **Claude** for parsing: Too expensive ($4-$44 for the corpus), no unique quality advantage for this use case
- **Gemini 2.5 Flash-Lite** as sole parser: Cheapest at $0.23 but quality may be insufficient for complex maritime documents; use only if budget is the primary constraint
- **Mistral OCR 3**: Excellent quality but $3.10 for the corpus is 2.6x GPT-4.1-mini batch; consider only if handwritten content is a significant portion of the corpus
- **Claude Sonnet 4.6** or **GPT-4.1**: Premium models; the quality improvement does not justify 10-20x cost for bulk parsing

### Decision Framework

```
Is document parsing quality critical? (tables, stamps, handwriting)
├── No → GPT-4.1-nano batch ($0.40 total, already in stack)
├── Somewhat → GPT-4.1-mini batch ($1.18 total, already in stack) ← START HERE
└── Very much → Run quality validation on 50-doc sample
    ├── GPT-4.1-mini passes → Keep GPT-4.1-mini
    ├── GPT-4.1-mini fails → Add Gemini 2.5 Flash ($1.28 total)
    └── All LLMs fail → Mistral OCR 3 ($3.10) or keep LlamaParse ($11.63)
```

---

## Action Items

1. **Run quality validation**: Take 50 representative maritime PDFs (mix of inspection reports, survey reports, classification letters, scanned documents) and compare GPT-4.1-mini vs Gemini 2.5 Flash vs current LlamaParse output
2. **Test consolidation**: Try a single GPT-4.1-mini call that returns both parsed markdown AND context summary + topics (saves separate LLM calls)
3. **Implement batch processing**: Use OpenAI Batch API for non-urgent ingest to get 50% discount
4. **Monitor GPT-4.1-mini benchmarks**: Independent PDF parsing benchmarks for 4.1-mini are not yet available; watch for Applied AI or similar evaluations

---

## Sources

- [OpenAI API Pricing](https://platform.openai.com/docs/pricing)
- [OpenAI GPT-4.1 Models](https://openai.com/index/gpt-4-1/)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Document Understanding](https://ai.google.dev/gemini-api/docs/document-processing)
- [Claude Models & Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Claude PDF Support](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3)
- [Mistral Document AI](https://docs.mistral.ai/capabilities/document_ai)
- [Applied AI PDF Parsing Benchmark (800+ docs)](https://www.applied-ai.com/briefings/pdf-parsing-benchmark/)
- [Gemini 3 Flash](https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/)
- [Gemini 3.1 Flash-Lite](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-lite/)
- [OpenAI Vision/Image Token Calculation](https://developers.openai.com/api/docs/guides/images-vision)
- [GPT-4.1-mini Pricing](https://pricepertoken.com/pricing-page/provider/openai)
- [Claude vs GPT vs Gemini Invoice Extraction](https://www.koncile.ai/en/ressources/claude-gpt-or-gemini-which-is-the-best-llm-for-invoice-extraction)
