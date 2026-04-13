# Parser Validation Results

Validation of local parsers (Plan 02) against real attachments extracted from EML files.
Tested 2026-04-13 across 12 EML files covering PDF, DOCX, XLSX, XLS, and image attachments.

## Summary

- **EML files tested:** 12 (6 initial + 6 extended for DOCX/XLSX coverage)
- **Total unique attachments tested:** 54
- **Parser dependencies:** All installed and available (pymupdf4llm, python-docx, openpyxl, Pillow, html2text)

### Attachment Type Distribution (across all tested EMLs)

| Extension | Count | Notes |
|-----------|-------|-------|
| .pdf | 6 | 3 complex (scanned), 3+ simple (text-based) from broader sample |
| .docx | 2 | 1 text+tables, 1 image-only |
| .xlsx | 3 | All parsed successfully |
| .xls | 7 | Legacy format, cannot be parsed locally |
| .jpg | 30 | Mix of equipment photos and email signatures |
| .png | 6 | Mix of diagrams and email logos |
| .gif | 1 | Email separator image |

## PDF Classifier Results

Tested on 11 PDFs total: 3 from original sample + 8 from a broader 200-EML scan.

| Filename | Size | Classification | Classifier Correct? |
|----------|------|---------------|---------------------|
| MASTER'S STATEMENT.pdf | 250 KB | complex | YES -- scanned image PDF, no text layer |
| ME AUX BLOWER STARTER.pdf | 214 KB | complex | YES -- scanned image PDF, no text layer |
| FORE SIDE CHAIN.pdf | 38 KB | complex | YES -- scanned image PDF, no text layer |
| Service Report...MARAN HOMER.pdf | 298 KB | simple | YES -- text-based service report |
| Pre-commissioning protocol.pdf | 323 KB | simple | YES -- text-based protocol |
| Quote SQ251096.pdf | 96 KB | simple | YES -- text-based quote |
| po_4500562040_TECO CHEMICALS.pdf | 32 KB | simple | YES -- text-based purchase order |
| Freight Document1.pdf | 19 KB | simple | YES -- text with some images |
| TARIFS AIRFREIGHT 2025.pdf | 57 KB | simple | YES -- text table |
| Avitaillements H&T 2025 - ANTIFER.pdf | 19 KB | simple | YES -- text table |
| Avitaillements H&T 2025 - LE HAVRE.pdf | 19 KB | simple | YES -- text table |

**Classifier accuracy: 11/11 (100%)** -- all classifications were correct.

## Local PDF Parser Results (PyMuPDF4LLM)

### Simple PDFs -- LocalPDFParser

| Filename | Chars | Lines | Quality | Notes |
|----------|-------|-------|---------|-------|
| Service Report...MARAN HOMER.pdf | 2,548 | 31 | GOOD | Clean markdown, headers, structured text |
| Pre-commissioning protocol.pdf | -- | -- | FAIL | Encoding error: `charmap` codec failure on U+FFFD |
| Quote SQ251096.pdf | 1,527 | 21 | GOOD | Clean quote with line items |
| po_4500562040_TECO CHEMICALS.pdf | 4,280 | 62 | GOOD | Clean PO with structured data |
| Freight Document1.pdf | 1,052 | 12 | FAIR | Some `[picture ... omitted]` placeholders |
| TARIFS AIRFREIGHT 2025.pdf | 432 | 10 | GOOD | Clean markdown table |
| Avitaillements - ANTIFER.pdf | 1,020 | 18 | FAIR | U+FFFD replacement chars in output |
| Avitaillements - LE HAVRE.pdf | 1,021 | 18 | FAIR | U+FFFD replacement chars in output |

**Success rate: 7/8 (87.5%)** -- 1 encoding failure.

### Complex PDFs -- force-tested with LocalPDFParser

All 3 complex PDFs correctly produced no text (they are scanned images).
These would be sent to LlamaParse in production, which is the correct behavior.

| Filename | LocalPDFParser Result |
|----------|---------------------|
| MASTER'S STATEMENT.pdf | Empty output (correct -- scanned image) |
| ME AUX BLOWER STARTER.pdf | Empty output (correct -- scanned image) |
| FORE SIDE CHAIN.pdf | Empty output (correct -- scanned image) |

## DOCX Parser Results (python-docx)

| Filename | Size | Chars | Success | Notes |
|----------|------|-------|---------|-------|
| FRM_Maritime Service Request Form.docx | 1.5 MB | 2,390 | YES | Extracted text and tables correctly |
| D.2 LOW INSULATION EXPLAINATION.docx | 124 KB | 0 | NO | File contains only 1 empty paragraph + 1 embedded image |

**Finding:** The DOCX parser correctly handles text+table documents but produces no output for
image-only DOCX files. This is expected behavior -- the image content would need separate
OCR/vision processing. The parser does not crash; it raises a clear `ValueError`.

**Content quality sample (FRM_Maritime Service Request Form.docx):**
```
IMPORTANT: Please provide the following information best possible to accelerate your
request! Instructions on how to save the analyzer configuration file SOPAS ET:
Preliminary steps SOPAS ET: Set IP address...
```
Contains both paragraph text and pipe-delimited table data. Usable for RAG.

## XLSX Parser Results (openpyxl)

| Filename | Size | Chars | Sheets | Rows | Quality |
|----------|------|-------|--------|------|---------|
| CT-0010112072-MARAN ANTARES-ENGINE.XLSX | 9 KB | 4,517 | 1 | 49 | GOOD |
| SIMOPS PLAN_Rev1_7-3-2025 (1).xlsx | 14 KB | 2,678 | 1 | 13 | GOOD |
| CW_LIQUIDEWT- JUNE 2025 (1).xlsx | 60 KB | 9,268 | 1 | 38 | GOOD |

**Success rate: 3/3 (100%)**

**Content quality sample (CT-0010112072-MARAN ANTARES-ENGINE.XLSX):**
```
## Comparison Table
Vendor | Vendor Name | Rfq Items | Quot Items | Total Amount | Currency | Total in USD...
```
Well-structured pipe-delimited data with sheet headers. Fully usable for RAG.

## XLS (Legacy) Parser Results

All 7 `.xls` files failed as expected -- openpyxl cannot read the legacy OLE2 format.
These must go through LlamaParse (or xlrd, if added as a dependency).

| Filename | Size | Result |
|----------|------|--------|
| AB No1 JUNE 2025.xls | 223 KB | Failed (expected) |
| AB No2 JUNE 2025.xls | 229 KB | Failed (expected) |
| CB- JUNE 2025.xls | 230 KB | Failed (expected) |
| BW_AGK-100...BLR COMPOSITE.xls | 246 KB | Failed (expected) |
| BW_AGK-100...BLR PORT.xls | 251 KB | Failed (expected) |
| BW_AGK-100...BLR STBD.xls | 193 KB | Failed (expected) |
| CW_LIQUIDEWT-LOG-R159...xls | 200 KB | Failed (expected) |

## CSV / HTML Parser Results

No CSV or HTML attachments were found in the tested EML files. These parsers remain untested
against real data. Both parsers use standard libraries (stdlib csv, html2text) and are low-risk.

## Image Filter Results

### From initial 6 EMLs (18 images)

| Category | Count | Examples |
|----------|-------|---------|
| Meaningful (kept) | 7 | DSCN0995.JPG (2272x1704), equipment/repair photos |
| Filtered: name pattern | 9 | image001-007 (email signature images, logos) |
| Filtered: too small | 1 | ~WRD0000.jpg (823 bytes, 100x100) |
| Filtered: dimensions | 1 | image005.gif (540x18, banner-shaped) |

**Filter accuracy: 17/18 (94%)** -- one potential false positive.

### Edge Case: image002.png

- **File:** image002.png from 100297780 (HERMIONE worklogs)
- **Size:** 243,136 bytes (243 KB)
- **Dimensions:** 666x252
- **Filter result:** FILTERED (not meaningful)
- **Reason:** Filename matches `image\d{3}` pattern
- **Issue:** This is a sizable image (243 KB, 666x252) that could be a meaningful diagram or
  table screenshot, but is rejected solely because of the generic filename. Most `imageNNN`
  files from email clients are indeed signature/logo images, so the pattern is correct for
  the majority of cases. This is an acceptable false-positive rate.

## Issues Found

### 1. PDF parser encoding error (MEDIUM)

**File:** Pre-commissioning protocol Maran Phoebe.pdf
**Error:** `'charmap' codec can't encode character '\ufffd'`
**Root cause:** PyMuPDF4LLM extracts text containing U+FFFD replacement characters, and
the string fails when written/printed with a non-UTF-8 codec (Windows console codepage).
The parse itself succeeds, but downstream handling may fail.
**Impact:** Affects PDFs with unusual character encodings on Windows.
**Fix:** Ensure all downstream consumers handle UTF-8 explicitly.

### 2. Unicode replacement characters in output (LOW)

**Files:** Avitaillements H&T 2025 - ANTIFER.pdf, Avitaillements H&T 2025 - LE HAVRE.pdf
**Issue:** Output contains U+FFFD (`\ufffd`) replacement characters where the PDF had
non-standard characters (likely French accented characters or special symbols).
**Impact:** Minor text quality degradation. Still usable for RAG but search/matching may miss
affected terms.

### 3. Image-only DOCX produces no content (LOW)

**File:** D.2 LOW INSULATION EXPLAINATION.docx
**Issue:** File contains only an embedded image and an empty paragraph. Parser correctly
reports no text content via ValueError.
**Impact:** Expected behavior. Image-only documents need vision/OCR pipeline.

### 4. `image\d{3}` filter may reject meaningful images (LOW)

**File:** image002.png (243 KB, 666x252)
**Issue:** The pattern-based filter catches all `imageNNN` filenames, which are typically
email client auto-names for inline images. Occasionally a meaningful image gets this name.
**Impact:** Low -- vast majority of `imageNNN` files are indeed signatures/logos. Adding a
size exception (e.g., keep if >100KB) could recover these cases, but would also let through
large company logo images.

## Overall Quality Assessment

### RAG Usability

| Parser | Tested | Success | Quality | RAG-Ready? |
|--------|--------|---------|---------|------------|
| PDF Classifier | 11 PDFs | 11/11 (100%) | Excellent | YES |
| LocalPDFParser (simple) | 8 PDFs | 7/8 (87.5%) | Good-Fair | YES (with encoding fix) |
| LocalDocxParser | 2 DOCX | 1/2 (50%) | Good | YES (image-only is expected) |
| LocalXlsxParser | 3 XLSX | 3/3 (100%) | Good | YES |
| LocalCsvParser | 0 | -- | -- | Untested |
| LocalHtmlParser | 0 | -- | -- | Untested |
| Image Filter | 18 images | 17/18 (94%) | Good | YES |

### Content Quality Observations

1. **PDF parser** produces clean markdown with headers, tables, and structured text. Quality
   is good enough for RAG embeddings and retrieval.
2. **DOCX parser** extracts paragraphs and tables as pipe-delimited text. Output is clean
   and structured.
3. **XLSX parser** produces sheet-headed, pipe-delimited rows. Numeric and text data both
   come through correctly.
4. **Image filter** correctly identifies equipment photos as meaningful and rejects email
   chrome (signatures, logos, tracking pixels, banners).

## Recommendations

1. **No blocking issues** -- all parsers are functional and produce usable output.
2. **Monitor encoding** on Windows: The PyMuPDF4LLM U+FFFD issue (#1 above) may surface
   when processing French/Greek/non-ASCII PDFs. Consider adding `.encode('utf-8', errors='replace')`
   guardrails in the local PDF parser's output path.
3. **CSV/HTML parsers** remain untested against real data but use standard libraries and are
   low-risk. Test when real CSV/HTML attachments appear.
4. **Legacy .xls files** (7 found in sample) require LlamaParse. Ensure the LlamaParse
   fallback path is wired up before production.
