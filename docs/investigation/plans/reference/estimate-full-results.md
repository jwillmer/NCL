---
purpose: Full cost estimate results for 6,289 EML files in data/emails
status: complete
date: 2026-04-13
command: "uv run mtss estimate --source ./data/emails --verbose"
runtime: 189.8 seconds
cached: 243 emails (from prior runs)
newly_extracted: 6,046 emails
---

# Full Ingest Cost Estimate Results

## Run Details

- **Date:** 2026-04-13
- **Command:** `uv run mtss estimate --source ./data/emails --verbose`
- **Runtime:** 189.8 seconds
- **EML files scanned:** 6,289 (243 cached, 6,046 newly extracted)
- **Default pricing:** LlamaParse $0.00625/page, Vision $0.01/image, LLM $0.001/file, Embedding $0.00002/chunk

## Ingest File Inventory

| Category      | Files  | Pages             | Unknown        |
|---------------|--------|-------------------|----------------|
| PDF           | 3,232  | 19,432            | 1 (0%)         |
| DOCX          | 154    | 479               | 7 (5%)         |
| XLSX          | 288    | 734               | 0              |
| DOC           | 15     | 51                | 0              |
| XLS           | 174    | 299               | 0              |
| Images        | 20,684 | ~9,860 meaningful | 10,824 skipped |
| Text/Markdown | 115    | --                | --             |
| Other         | 312    | --                | --             |
| **TOTAL**     | **24,974** | **20,995**    | **8 (0%)**     |

## Estimated Ingest Cost

| Service                  | Units                               | Unit Cost  | Cost     |
|--------------------------|-------------------------------------|------------|----------|
| LlamaParse (documents)   | 20,995 pages                        | $0.00625   | $131.22  |
| Vision API (images)      | ~9,860 images (10,824 skipped)      | $0.01000   | $98.60   |
| LLM text (text files)    | 115 files                           | $0.00100   | $0.12    |
| Embeddings (~1.5x pages) | ~31,492 chunks                      | $0.00002   | $0.63    |
| **TOTAL ESTIMATED**      |                                     |            | **$230.56** |

### Pricing Notes

- LlamaParse: 5 avg credits/page at $0.00125/credit ($50 / 40k credits)
- Embeddings: ~1.5 chunks/page via text-embedding-3-small ($0.02/M tokens)
- Images: 10,824 likely non-content (logos, icons, banners) excluded by size/dimension heuristic
- 8 files had unknown page counts (counted as 1 page each)

## Files with Issues

| File | Issue | Detail |
|------|-------|--------|
| attachments\5415DA512A10... | parse_error | could not determine page count (pypdf + regex fallbacks failed) |
| attachments\PURIF.zip | zip_error | Corrupt ZIP: File is not a zip file |
| attachments\Manifolds_Pr... | zip_error | Encrypted ZIP (Hanla.zip), password required |
| attachments\Friday_Report... MARAN ARCTURUS DECK WEEK 49.docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\Friday_Report... MARAN ARCTURUS (1).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\Friday_Report... MARAN DIONE (1).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\Friday_Report... MARAN APOLLO (36).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\Friday_Report... MARAN APOLLO (39).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\Friday_Report... MARAN MARS (15).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |
| attachments\MTM_RA_JHA00... MARAN PYTHIA- WK25 DECK.zip | zip_error | ZIP exceeds max file count (100) |
| attachments\Friday_Report... MARAN ARETE DECK(11) (1).docx | page_count_unknown | no metadata in docProps/app.xml, no manual page breaks found |

## Breakdown by Document Type

### Documents Sent to LlamaParse

| Format | Files | Pages | Cost @ $0.00625 |
|--------|-------|-------|-----------------|
| PDF    | 3,232 | 19,432 | $121.45         |
| DOCX   | 154   | 479   | $2.99           |
| XLSX   | 288   | 734   | $4.59           |
| DOC    | 15    | 51    | $0.32           |
| XLS    | 174   | 299   | $1.87           |
| **Total** | **3,863** | **20,995** | **$131.22** |

### Images

| Category | Count | Cost |
|----------|-------|------|
| Meaningful (sent to Vision API) | 9,860 | $98.60 |
| Skipped (logos, icons, banners) | 10,824 | $0.00 |
| **Total images** | **20,684** | **$98.60** |

### Text Files

| Category | Count | Cost |
|----------|-------|------|
| Text/Markdown (LLM processing) | 115 | $0.12 |
| Other (not processed) | 312 | $0.00 |
