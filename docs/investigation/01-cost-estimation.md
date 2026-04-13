---
purpose: Cost estimation for ingesting ~7GB / 6,289 EML files through the MTSS pipeline
status: research-complete
date: 2026-04-13
---

# Cost Estimation: Full Ingest of 6,289 EML Files

## Summary

The existing `mtss estimate` command covers **LlamaParse** (document parsing), **Vision API** (image description), **text file processing**, and **embedding generation**. It does **not** account for several LLM calls that happen during ingest: context summary generation, topic extraction, image classification, and topic embedding lookups. This document provides a complete cost breakdown including those hidden costs.

## Data Source

- **6,289 EML files** in `data/emails/` (~7 GB total)
- **243 emails scanned** in cached estimate data (`data/processed/estimate/`, 432 MB)
- Extrapolation factor: 6,289 / 243 = **25.88x**

## Sample Analysis (243 Scanned Emails)

| Category       | Files | Pages | Unknown Pages | Meaningful Images | Skipped Images |
|----------------|------:|------:|--------------:|------------------:|---------------:|
| PDF            |   131 |   400 |             1 |                 - |              - |
| DOCX           |    10 |    35 |             0 |                 - |              - |
| DOC            |     4 |    19 |             0 |                 - |              - |
| XLSX           |     7 |     7 |             0 |                 - |              - |
| XLS            |     3 |     6 |             0 |                 - |              - |
| Images         |   803 |     - |             - |               218 |            585 |
| Text/Markdown  |     1 |     - |             - |                 - |              - |
| Other          |    13 |     - |             - |                 - |              - |
| **TOTAL**      | **972** | **467** |         **1** |           **218** |        **585** |

- **82%** of emails have attachments (199/243)
- **73%** of images are non-content (logos, banners, signatures) -- filtered by size/dimension heuristic
- Average **4.0 attachments** per email, **1.92 document pages** per email

## Extrapolated Totals (6,289 Emails)

| Metric                    | Extrapolated Value |
|---------------------------|-------------------:|
| Document pages (LlamaParse) |           ~12,087 |
| Meaningful images (Vision)  |            ~5,642 |
| Skipped images (filtered)   |           ~15,140 |
| Document files total         |            ~4,013 |
| Text files                   |               ~26 |

## What the Estimator Covers

The `mtss estimate` CLI command calculates costs for:

| Service                   | How It Estimates                          |
|---------------------------|-------------------------------------------|
| LlamaParse (documents)    | page_count x $0.00625/page                |
| Vision API (images)       | meaningful_images x $0.01/image           |
| LLM text (text files)     | text_file_count x $0.001/file             |
| Embeddings                | ~1.5x pages x $0.00002/chunk              |

## What the Estimator Misses

The following LLM calls happen during ingest but are **not included** in the estimate:

| Missing Cost                      | When It Fires                                         | Model Used        |
|-----------------------------------|-------------------------------------------------------|--------------------|
| Context summary (email body)      | 1x per email with body text                           | gpt-4o-mini        |
| Context summary (attachments)     | 1x per attachment with parsed content                 | gpt-4o-mini        |
| Topic extraction                  | 1x per email                                          | gpt-4o-mini        |
| Topic dedup embedding             | 1-5x per email (one per extracted topic)              | text-embedding-3-small |
| Image classification + description| 1x per image attachment at email level (not in ZIPs)  | gpt-4o-mini (vision) |
| Email body cleaning (regex only)  | **Not an LLM call** -- uses regex patterns            | N/A                |

### Important: Image costs are double-counted

The estimator's "Vision API" line already accounts for meaningful images at $0.01/image. However, during actual ingest, **all** image attachments (not just meaningful ones) go through the Vision API for classification first. The classification call determines if the image is meaningful. So the real cost is:

- **All 803 images** (in the 243 sample) get a classification call, not just the 218 meaningful ones
- Only the 218 meaningful ones also get a description (combined in the same call via structured output)
- Images inside ZIPs get description-only (no classification) since they are assumed meaningful

## Full Cost Breakdown

### Tier 1: Document Parsing (LlamaParse)

| Item         | Units   | Unit Cost  | Total     |
|--------------|--------:|-----------:|----------:|
| PDF pages    |  10,354 |   $0.00625 |    $64.71 |
| DOCX pages   |     906 |   $0.00625 |     $5.66 |
| DOC pages    |     492 |   $0.00625 |     $3.07 |
| XLSX sheets  |     181 |   $0.00625 |     $1.13 |
| XLS sheets   |     155 |   $0.00625 |     $0.97 |
| **Subtotal** | **12,087** |         | **$75.54** |

### Tier 2: Image Processing (Vision API via gpt-4o-mini)

All image attachments at email level go through classification (one Vision API call each). Images inside ZIPs get describe-only calls.

Pricing: gpt-4o-mini vision -- ~85 tokens input detail per image + ~150 output tokens.
Estimated cost per image call: ~$0.003-0.005 (much less than the $0.01 default in estimator).

| Item                         | Units   | Est. Unit Cost | Total     |
|------------------------------|--------:|---------------:|----------:|
| Image classify+describe (email-level) | ~20,783 | $0.004    | ~$83.13  |
| Image describe-only (ZIP contents)    |    ~200 | $0.003    |  ~$0.60  |
| **Subtotal**                 |         |                | **~$83.73** |

Note: The estimator assumes $0.01/image for only meaningful images (~5,642 = $56.42). The actual cost is higher because classification runs on ALL images (~20,783), not just meaningful ones.

### Tier 3: LLM Calls (gpt-4o-mini)

gpt-4o-mini pricing: $0.15/M input tokens, $0.60/M output tokens.

| Call Type                | Count   | Est. Input Tokens | Est. Output Tokens | Cost     |
|--------------------------|--------:|------------------:|-------------------:|---------:|
| Context summary (emails) |   6,289 |              ~800 |               ~100 |    $1.13 |
| Context summary (attachments) | ~4,013 |           ~1,500 |               ~100 |    $1.14 |
| Topic extraction         |   6,289 |            ~1,200 |               ~300 |    $2.27 |
| **Subtotal**             |         |                   |                    | **$4.54** |

### Tier 4: Embeddings (text-embedding-3-small)

Pricing: $0.02/M tokens (1536 dimensions).

| Item                      | Units    | Est. Tokens/Unit | Total Tokens | Cost     |
|---------------------------|----------|-----------------:|-------------:|---------:|
| Email body chunks         |  ~12,578 |             ~400 |    5,031,200 |    $0.10 |
| Document chunks (1.5x pages) | ~18,131 |           ~400 |    7,252,400 |    $0.15 |
| Image description chunks  |   ~5,642 |             ~200 |    1,128,400 |    $0.02 |
| Topic dedup embeddings    |  ~18,867 |              ~10 |      188,670 |    $0.00 |
| **Subtotal**              |          |                  |              | **$0.27** |

Note: Email bodies are split into individual thread messages (~2 chunks/email on average).

### Tier 5: Supabase / Infrastructure

| Item                    | Estimate   | Notes                              |
|-------------------------|------------|-------------------------------------|
| Database storage        | ~5-10 GB   | Chunks + embeddings + metadata     |
| Supabase Storage        | ~7 GB      | Archive bucket (EMLs + markdown)   |
| Database compute        | Free tier  | Depends on plan                    |

### Total Estimated Cost

| Category              | Estimate        |
|-----------------------|----------------:|
| LlamaParse            |         $75.54  |
| Image Processing      |        ~$83.73  |
| LLM Calls (context + topics) | ~$4.54 |
| Embeddings            |         ~$0.27  |
| **TOTAL**             |    **~$164.08** |

**Cost range: $140 - $200** (accounting for estimation uncertainty from 3.9% sample)

## Comparison: Estimator Output vs Actual

| Line Item                   | Estimator Would Show | Actual Expected |
|-----------------------------|---------------------:|----------------:|
| LlamaParse                  |              $75.54  |         $75.54  |
| Vision API (images)         |              $56.42  |        ~$83.73  |
| LLM text (text files)       |               $0.03  |          $0.03  |
| Embeddings                  |               $0.36  |         ~$0.27  |
| Context summaries           |           *(missed)* |         ~$2.27  |
| Topic extraction            |           *(missed)* |         ~$2.27  |
| **TOTAL**                   |         **~$132.35** |    **~$164.08** |

The estimator underestimates by approximately **$32** (~24%), mainly due to:
1. Image classification running on ALL images, not just meaningful ones
2. Context summaries and topic extraction LLM calls not being counted

## Recommendations

1. **Run the full estimate first** -- Execute `mtss estimate --source ./data/emails` to get exact page counts for all 6,289 emails. The 243 cached results will be instant; the remaining ~6,046 require extraction only (no API calls). This takes ~10-30 minutes but produces exact numbers.

2. **LlamaParse is the largest single cost** ($75) -- Consider whether all document types need parsing. Legacy formats (.doc, .ppt, .xls) are a small fraction (~7 files in the sample) and could be skipped initially.

3. **Image classification is expensive in aggregate** -- The heuristic filter in the estimator (file size < 15KB, dimensions < 100px) already skips many images before the API call. But ~20K images still go through Vision API. Consider tightening the heuristic (e.g., skip images < 30KB) to reduce classification calls.

4. **Batch processing is already optimized** -- Embeddings use batch API calls (100 per batch), and the pipeline has concurrent file processing (5 workers). No changes needed here.

5. **Incremental ingestion is supported** -- The pipeline tracks progress and supports `--resume`. If budget is a concern, ingest in batches of 500-1000 emails and monitor costs.

6. **Topic extraction adds minimal cost** ($2.27) but significant value for filtering. Keep it enabled.

7. **Email cleaner LLM calls are NOT active** -- The `clean_email_body()` function with LLM-based boundary detection exists in the code but is not called during ingest. Only regex-based `split_into_messages()` and `remove_boilerplate_from_message()` are used. If enabled in the future, it would add ~$1-2 (one gpt-4o-mini call per email).

## Key Files

| File | Purpose |
|------|---------|
| `src/mtss/ingest/estimator.py` | Estimator: extracts attachments, counts pages |
| `src/mtss/cli/ingest_cmd.py` | CLI command + cost display logic |
| `src/mtss/ingest/pipeline.py` | Main ingest pipeline (all LLM calls) |
| `src/mtss/ingest/components.py` | Component factory (models, processors) |
| `src/mtss/parsers/chunker.py` | ContextGenerator (LLM summary calls) |
| `src/mtss/processing/topics.py` | TopicExtractor (LLM topic calls) |
| `src/mtss/processing/embeddings.py` | EmbeddingGenerator |
| `src/mtss/processing/image_processor.py` | ImageProcessor (Vision API calls) |
| `src/mtss/parsers/email_cleaner.py` | Email cleaner (regex-based, LLM not active) |
| `src/mtss/parsers/preprocessor.py` | Routing: classify images, select parsers |
