---
purpose: Merged implementation plan — local-only ingest with cost optimizations
status: approved
date: 2026-04-13
scope: config changes, image pre-filtering, tiered parsing, local storage backend, CLI wiring, speed optimizations
depends_on:
  - 02-local-storage-design.md
  - 06a-optimization-cost-reduction.md
  - 06c-optimization-processing-speed.md
  - local-storage-sqlite-vs-files.md
  - local-only-ingest-plan.md
  - optimization-plan.md
decisions:
  - "D-07: chunk_size_tokens 512 -> 1024, overlap 50 -> 100"
  - "D-08: embedding_dimensions 1536 -> 512"
  - "D-05: implement cost optimization Phase 1+2 before first ingest"
---

# Implementation Plan: Local-Only Ingest with Cost Optimizations

## Overview

This plan merges two workstreams into a single implementation order:

1. **Local-only ingest** -- extend `LocalStorageClient`, JSONL output, `--local-only` CLI flag, no Supabase dependency.
2. **Cost optimizations** -- approved proposals 1, 2, 4, 5 from `06a-optimization-cost-reduction.md`.

**Expected outcome:** A pipeline that ingests 6,289 EML files to local JSONL files, costing approximately $6 instead of $230, with 73% smaller vector storage.

**Key decisions already made:**
- JSONL over SQLite for local storage (`local-storage-sqlite-vs-files.md`)
- Chunk size: 1024 tokens, overlap: 100 tokens (was 512/50) -- D-07
- Embedding dimensions: 512 (was 1536) -- D-08
- Local parsers for simple PDFs + GPT-4.1-mini batch for complex -- D-04
- Image pre-filtering with estimator heuristic before Vision API -- D-05

---

## Phase 0: Configuration Quick Wins

Config-only changes that affect ALL subsequent work. Do these first so every test
run, embedding, and chunk produced from this point forward uses the final settings.

### 0.1 Increase chunk size (Proposal 5)

**File:** `src/mtss/config.py` (lines 74-75)

```python
# Before:
chunk_size_tokens: int = Field(default=512, validation_alias="CHUNK_SIZE_TOKENS")
chunk_overlap_tokens: int = Field(default=50, validation_alias="CHUNK_OVERLAP_TOKENS")

# After:
chunk_size_tokens: int = Field(default=1024, validation_alias="CHUNK_SIZE_TOKENS")
chunk_overlap_tokens: int = Field(default=100, validation_alias="CHUNK_OVERLAP_TOKENS")
```

**Impact:** Halves chunk count (~31,492 -> ~15,746 chunks). More context per chunk improves retrieval for narrative incident reports.

**Test strategy:** Run existing chunker tests; verify output chunk count is approximately halved on test documents.

### 0.2 Reduce embedding dimensions (Proposal 4)

**File:** `src/mtss/config.py` (line 46)

```python
# Before:
embedding_dimensions: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSIONS")

# After:
embedding_dimensions: int = Field(default=512, validation_alias="EMBEDDING_DIMENSIONS")
```

**Impact:** 67% reduction in vector storage per chunk. Combined with 0.1, total vector storage drops ~73%. At production scale (~100 GB DB), this is significant.

**Dependencies:** Update `local-only-ingest-plan.md` Step 2 comment from "1536-dim" to "512-dim" in the embedding round-trip test.

**Test strategy:** Verify `EmbeddingGenerator` passes `dimensions=512` to the API. Verify output embeddings have length 512.

### 0.3 Update manifest template

**File:** `src/mtss/storage/local_client.py` (when created in Phase 3)

The `manifest.json` writer (Step 9 of local-only plan) references `embedding_dimensions: 1536`. Update to read from `settings.embedding_dimensions` dynamically so the manifest always reflects the actual config.

**Effort:** Phase 0 total: 15 minutes (two config line changes).

---

## Phase 1: Image Pre-filtering (Proposal 2)

Port the estimator's `_is_meaningful_image()` heuristic to the live pipeline so
non-content images (logos, icons, banners) are filtered BEFORE any Vision API call.
This is the lowest-effort, lowest-risk cost optimization.

### 1.1 Extract `_is_meaningful_image()` to shared utility

**File to create:** `src/mtss/utils/image_filter.py`
**Source:** `src/mtss/ingest/estimator.py` lines 669-700

```python
"""Shared image filtering heuristics for estimator and pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Filename patterns for non-content images (case-insensitive)
_SKIP_FILENAME_PATTERNS = [
    re.compile(r"^logo", re.IGNORECASE),
    re.compile(r"^banner", re.IGNORECASE),
    re.compile(r"^signature", re.IGNORECASE),
    re.compile(r"^icon", re.IGNORECASE),
    re.compile(r"^image\d{3}\.", re.IGNORECASE),  # image001.png etc. (Outlook inline)
]


def is_meaningful_image(path: Path) -> bool:
    """Heuristic: return True if the image is likely meaningful content.

    Filters out tracking pixels, icons, logos, banners, and email
    signature images based on file size, dimensions, and filename.
    """
    # Filename-based filtering
    name = path.name
    for pattern in _SKIP_FILENAME_PATTERNS:
        if pattern.match(name):
            logger.debug(f"Skipping image by filename pattern: {name}")
            return False

    try:
        file_size = path.stat().st_size
    except OSError:
        return True  # can't check, assume meaningful

    # Very small files are almost always tracking pixels or tiny icons
    if file_size < 15_000:
        return False

    # Check dimensions with PIL for more accurate filtering
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        # Tiny images: icons, tracking pixels, small logos
        if max(w, h) < 100:
            return False
        # Banner-shaped: wide and short (email separators, signature strips)
        if h < 50 and w > 3 * h:
            return False
    except Exception:
        pass  # can't read dimensions, file size check is enough

    return True
```

**Test strategy:**
- Unit test with fixture images of varying sizes (1KB icon, 50KB logo, 200KB photo).
- Test filename patterns: `logo.png`, `banner_top.jpg`, `image001.png`, `report_photo.jpg`.

### 1.2 Update estimator to use shared utility

**File:** `src/mtss/ingest/estimator.py`

Replace the `_is_meaningful_image` static method (lines 669-700) with a delegation to the shared utility:

```python
# At top of file, add:
from ..utils.image_filter import is_meaningful_image

# Replace the @staticmethod _is_meaningful_image with:
@staticmethod
def _is_meaningful_image(path: Path) -> bool:
    """Delegate to shared image filter utility."""
    return is_meaningful_image(path)
```

The estimator's public interface (`_is_meaningful_image(path)`) is preserved for backward compatibility. Internally it now delegates to the shared function.

**Test strategy:** Run existing estimator tests; behavior should be identical except that filename-based filtering now also applies.

### 1.3 Add image pre-filtering to `DocumentPreprocessor.preprocess()`

**File:** `src/mtss/parsers/preprocessor.py` (lines 142-159)

Insert a local heuristic check BEFORE the Vision API call:

```python
# At top of file, add:
from ..utils.image_filter import is_meaningful_image

# In preprocess(), replace the image classification block (lines 142-167):
# Check if it's an image that needs classification
if self.is_image(actual_type):
    if classify_images:
        # Phase 1 optimization: local heuristic BEFORE Vision API call
        if not is_meaningful_image(file_path):
            return PreprocessResult(
                should_process=False,
                skip_reason=f"filtered_by_heuristic: {file_path.name}",
                is_image=True,
                content_type=actual_type,
            )

        # Passed heuristic -- send to Vision API for classification + description
        result = await self.image_processor.classify_and_describe(file_path)
        if result.should_skip:
            return PreprocessResult(
                should_process=False,
                skip_reason=result.skip_reason,
                is_image=True,
                content_type=actual_type,
            )
        # Image is meaningful - store description for later use
        return PreprocessResult(
            should_process=True,
            parser_name="image",
            is_image=True,
            content_type=actual_type,
            image_description=result.description,
        )
    else:
        # No classification - just describe
        return PreprocessResult(
            should_process=True,
            parser_name="image",
            is_image=True,
            content_type=actual_type,
        )
```

**Impact:** Eliminates ~52% of images before any API call ($19-32 savings). The remaining images still go through Vision API classification as a second filter.

**Dependencies:** Phase 1.1 (shared utility must exist).

**Test strategy:**
- Unit test: mock `is_meaningful_image` returning False -> verify `PreprocessResult.should_process == False`.
- Unit test: mock returning True -> verify `classify_and_describe` is called.
- Integration test: pass a small icon file, verify no Vision API call is made.

### 1.4 Create `src/mtss/utils/__init__.py` if needed

**File:** `src/mtss/utils/__init__.py`

Check if the `utils` directory needs an `__init__.py`. Currently `src/mtss/utils.py` exists as a module (not a package). We need to decide:

- Option A: Put `image_filter.py` in the existing `src/mtss/` directory as `src/mtss/image_filter.py`.
- Option B: Convert `utils.py` to a package `utils/` with `__init__.py` re-exporting existing functions.

**Recommendation:** Option A is simpler. Create `src/mtss/image_filter.py` instead of `src/mtss/utils/image_filter.py` to avoid refactoring the existing `utils.py` module.

If Option A is chosen, update all imports in 1.1-1.3 from `..utils.image_filter` to `..image_filter`.

**Effort:** Phase 1 total: 2-3 hours.

---

## Phase 2: Local PDF/Office Parsers (Proposal 1)

Replace LlamaParse with free local parsers for simple documents. This is the largest cost saving ($125+ from baseline) but requires the most code.

### 2.1 PDF complexity classifier

**File to create:** `src/mtss/parsers/pdf_classifier.py`

```python
"""PDF complexity classifier for tiered parsing."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFComplexity(str, Enum):
    SIMPLE = "simple"       # Extractable text, no images/forms -> local parser
    COMPLEX = "complex"     # Scanned, images, tables, forms -> cloud parser


def classify_pdf(file_path: Path) -> PDFComplexity:
    """Classify a PDF as simple or complex using pypdf (already a dependency).

    Simple: all pages have extractable text, no images, no form fields.
    Complex: any page is scanned (no text), has images, or has form fields.

    Args:
        file_path: Path to the PDF file.

    Returns:
        PDFComplexity enum value.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        logger.warning(f"Cannot open PDF {file_path.name} for classification: {e}")
        return PDFComplexity.COMPLEX  # Can't classify -> assume complex

    if not reader.pages:
        return PDFComplexity.COMPLEX

    # Check for form fields (interactive forms need cloud parsing)
    if reader.get_fields():
        return PDFComplexity.COMPLEX

    for page in reader.pages:
        # Check if page has extractable text
        text = page.extract_text() or ""
        if len(text.strip()) < 50:
            # Page has very little text -> likely scanned
            return PDFComplexity.COMPLEX

        # Check for images on the page
        if "/XObject" in (page.get("/Resources") or {}):
            xobjects = page["/Resources"]["/XObject"].get_object()
            for obj_name in xobjects:
                xobj = xobjects[obj_name].get_object()
                if xobj.get("/Subtype") == "/Image":
                    return PDFComplexity.COMPLEX

    return PDFComplexity.SIMPLE
```

**Dependencies:** `pypdf` (already installed for estimator).

**Test strategy:**
- Test with a simple text-only PDF -> returns `SIMPLE`.
- Test with a scanned PDF (no extractable text) -> returns `COMPLEX`.
- Test with a PDF containing embedded images -> returns `COMPLEX`.
- Test with a corrupt/unreadable PDF -> returns `COMPLEX` (safe fallback).

### 2.2 Local PDF parser

**File to create:** `src/mtss/parsers/local_pdf_parser.py`

```python
"""Local PDF parser using PyMuPDF4LLM for simple text PDFs."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser

logger = logging.getLogger(__name__)


class LocalPDFParser(BaseParser):
    """Parser for simple text PDFs using PyMuPDF4LLM (free, local).

    Only handles PDFs classified as 'simple' by the PDF classifier.
    Complex PDFs should be routed to LlamaParse or GPT-4.1-mini batch.
    """

    name = "local_pdf"

    # This parser is NOT registered in the global registry.
    # It is invoked explicitly by the routing logic in AttachmentProcessor
    # after the PDF complexity classifier determines the PDF is simple.
    supported_mimetypes: set[str] = set()
    supported_extensions: set[str] = set()

    @property
    def is_available(self) -> bool:
        """Check if pymupdf4llm is installed."""
        try:
            import pymupdf4llm  # noqa: F401
            return True
        except ImportError:
            return False

    async def parse(self, file_path: Path) -> str:
        """Parse a simple PDF to markdown using PyMuPDF4LLM.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text in markdown format.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If parsing fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pymupdf4llm

            markdown = pymupdf4llm.to_markdown(str(file_path))

            if not markdown or not markdown.strip():
                raise ValueError(f"PyMuPDF4LLM produced no content for {file_path}")

            logger.info(
                f"Local PDF parser extracted {len(markdown)} chars from {file_path.name}"
            )
            return markdown

        except ImportError:
            raise ValueError(
                "pymupdf4llm is not installed. "
                "Install with: pip install pymupdf4llm"
            )
        except Exception as e:
            raise ValueError(f"Local PDF parsing failed for {file_path}: {e}") from e
```

**Dependencies:** Add `pymupdf4llm` to project dependencies.

**Test strategy:**
- Parse a known simple PDF and verify markdown output contains expected text.
- Verify the parser returns valid markdown (headings, paragraphs preserved).
- Test error handling when file doesn't exist.

### 2.3 Local Office parser

**File to create:** `src/mtss/parsers/local_office_parser.py`

```python
"""Local parsers for modern Office formats using python-docx and openpyxl."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

from .base import BaseParser

logger = logging.getLogger(__name__)


class LocalDocxParser(BaseParser):
    """Parser for DOCX files using python-docx (free, local)."""

    name = "local_docx"
    supported_mimetypes: set[str] = set()  # Not auto-registered
    supported_extensions: set[str] = set()

    @property
    def is_available(self) -> bool:
        try:
            import docx  # noqa: F401
            return True
        except ImportError:
            return False

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        from docx import Document

        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))

        text = "\n\n".join(paragraphs)
        if not text.strip():
            raise ValueError(f"python-docx produced no content for {file_path}")

        logger.info(f"Local DOCX parser extracted {len(text)} chars from {file_path.name}")
        return text


class LocalXlsxParser(BaseParser):
    """Parser for XLSX files using openpyxl (free, local)."""

    name = "local_xlsx"
    supported_mimetypes: set[str] = set()
    supported_extensions: set[str] = set()

    @property
    def is_available(self) -> bool:
        try:
            import openpyxl  # noqa: F401
            return True
        except ImportError:
            return False

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        from openpyxl import load_workbook

        wb = load_workbook(str(file_path), read_only=True, data_only=True)
        parts = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            parts.append(f"## {sheet_name}\n")

            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    parts.append(" | ".join(cells))

        wb.close()
        text = "\n".join(parts)

        if not text.strip():
            raise ValueError(f"openpyxl produced no content for {file_path}")

        logger.info(f"Local XLSX parser extracted {len(text)} chars from {file_path.name}")
        return text


class LocalCsvParser(BaseParser):
    """Parser for CSV files (free, local)."""

    name = "local_csv"
    supported_mimetypes: set[str] = set()
    supported_extensions: set[str] = set()

    @property
    def is_available(self) -> bool:
        return True  # csv is a stdlib module

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try common encodings
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = file_path.read_text(encoding="utf-8", errors="replace")

        reader = csv.reader(io.StringIO(content))
        rows = []
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append(" | ".join(row))

        text = "\n".join(rows)
        if not text.strip():
            raise ValueError(f"CSV parser produced no content for {file_path}")

        logger.info(f"Local CSV parser extracted {len(text)} chars from {file_path.name}")
        return text


class LocalHtmlParser(BaseParser):
    """Parser for HTML files using html2text or basic stripping (free, local)."""

    name = "local_html"
    supported_mimetypes: set[str] = set()
    supported_extensions: set[str] = set()

    @property
    def is_available(self) -> bool:
        return True

    async def parse(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                html_content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            html_content = file_path.read_text(encoding="utf-8", errors="replace")

        # Try html2text first (better markdown output)
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            text = h.handle(html_content)
        except ImportError:
            # Fallback: use the existing html_to_plain_text utility
            from ..parsers.email_cleaner import html_to_plain_text
            text = html_to_plain_text(html_content)

        if not text.strip():
            raise ValueError(f"HTML parser produced no content for {file_path}")

        logger.info(f"Local HTML parser extracted {len(text)} chars from {file_path.name}")
        return text
```

**Dependencies:** `python-docx`, `openpyxl` (likely already installed or easy to add). `html2text` (optional fallback exists).

**Test strategy:**
- Test each parser with a fixture file of its type.
- Verify DOCX table extraction produces pipe-delimited text.
- Verify XLSX multi-sheet output includes sheet name headers.
- Verify CSV handles different encodings.
- Verify HTML strips tags and produces readable text.

### 2.4 Add tiered routing to `AttachmentProcessor.process_attachment()`

**File:** `src/mtss/parsers/attachment_processor.py` (lines 178-233)

Add complexity-based routing before the default parser lookup:

```python
async def process_attachment(
    self,
    file_path: Path,
    document_id: UUID,
    content_type: Optional[str] = None,
) -> List[Chunk]:
    if not file_path.exists():
        raise FileNotFoundError(f"Attachment not found: {file_path}")

    # --- Tiered routing: try local parsers first ---
    parser = self._get_tiered_parser(file_path, content_type)

    if not parser:
        raise ValueError(f"No parser available for {file_path}")

    # Parse document to markdown text
    logger.info(f"Processing {file_path.name} with {parser.name} parser")
    text = await parser.parse(file_path)
    # ... rest unchanged ...
```

New private method:

```python
def _get_tiered_parser(self, file_path: Path, content_type: Optional[str] = None):
    """Select parser using tiered routing: local first, LlamaParse fallback.

    Routing rules:
    - PDF: classify complexity -> simple=local_pdf, complex=llamaparse
    - DOCX: local_docx
    - XLSX: local_xlsx
    - CSV: local_csv
    - HTML/HTM: local_html
    - DOC/XLS/PPT (legacy): llamaparse (no free local parser)
    - Everything else: ParserRegistry default lookup
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        from .pdf_classifier import classify_pdf, PDFComplexity
        complexity = classify_pdf(file_path)
        if complexity == PDFComplexity.SIMPLE:
            from .local_pdf_parser import LocalPDFParser
            local = LocalPDFParser()
            if local.is_available:
                return local
            logger.info(f"PyMuPDF4LLM not available, falling back to LlamaParse for {file_path.name}")
        # Complex PDF or local parser unavailable -> fall through to registry

    elif ext == ".docx":
        from .local_office_parser import LocalDocxParser
        local = LocalDocxParser()
        if local.is_available:
            return local

    elif ext == ".xlsx":
        from .local_office_parser import LocalXlsxParser
        local = LocalXlsxParser()
        if local.is_available:
            return local

    elif ext == ".csv":
        from .local_office_parser import LocalCsvParser
        return LocalCsvParser()

    elif ext in (".html", ".htm"):
        from .local_office_parser import LocalHtmlParser
        return LocalHtmlParser()

    # Legacy Office (.doc, .xls, .ppt), RTF, or unknown -> registry lookup
    return ParserRegistry.get_parser_for_file(file_path, content_type)
```

**Impact:** Eliminates LlamaParse for ~70% of documents ($94+ savings).

**Dependencies:** Phases 2.1, 2.2, 2.3.

**Test strategy:**
- Mock `classify_pdf` to return SIMPLE -> verify `LocalPDFParser` is selected.
- Mock to return COMPLEX -> verify `LlamaParseParser` is selected via registry.
- Verify `.docx` routes to `LocalDocxParser`.
- Verify `.doc` routes to `LlamaParseParser` (via registry).
- Verify `.csv` routes to `LocalCsvParser`.

### 2.5 Update LlamaParse parser scope

**File:** `src/mtss/parsers/llamaparse_parser.py`

Remove MIME types and extensions now handled by local parsers from `supported_mimetypes` and `supported_extensions`. LlamaParse should only claim formats where it is the sole option:

```python
# Keep only formats that NEED LlamaParse:
supported_mimetypes = {
    "application/pdf",  # Complex PDFs still route here via registry fallback
    "application/msword",              # .doc (legacy, no local parser)
    "application/vnd.ms-excel",        # .xls
    "application/vnd.ms-powerpoint",   # .ppt
    "application/rtf",                 # .rtf
    "text/rtf",
    "application/epub+zip",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
}

supported_extensions = {
    ".pdf",  # Complex PDFs
    ".doc", ".xls", ".ppt",
    ".rtf", ".epub", ".odt", ".ods", ".odp",
}
```

Remove from `supported_mimetypes`: DOCX, PPTX, XLSX, CSV, HTML.
Remove from `supported_extensions`: `.docx`, `.pptx`, `.xlsx`, `.csv`, `.html`, `.htm`.

These formats are now handled by local parsers (Phase 2.3) or the tiered router (Phase 2.4). LlamaParse remains as a fallback for complex PDFs and legacy formats.

**Note:** `.pptx` could also get a local parser (`python-pptx`), but there are only 27 PPTX files in the corpus. Keeping LlamaParse for PPTX is acceptable; add a local PPTX parser later if needed.

**Test strategy:** Verify `ParserRegistry.get_parser_for_file(Path("test.docx"))` returns `None` (or a local parser if registered), not `LlamaParseParser`.

### 2.6 Quality validation on sample set

Before proceeding to Phase 3, validate local parser output quality:

1. Select 20 PDFs (10 simple, 10 complex per classifier).
2. Parse each with both LlamaParse and the local parser.
3. Compare output: text completeness, structure preservation, table extraction.
4. Accept if local parser output is >= 90% as complete as LlamaParse for simple PDFs.

**Effort:** Phase 2 total: 4-5 days.

---

## Phase 3: Local-Only Storage Backend

Extend `LocalStorageClient` from `tests/local_storage.py` to support the full
ingest pipeline. This phase implements Steps 1-6 and 10-13 from the
`local-only-ingest-plan.md`.

### 3.1 Move LocalStorageClient to src/

**Source:** `tests/local_storage.py`
**Target:** `src/mtss/storage/local_client.py`

Copy the file and update imports. Keep `tests/local_storage.py` as a re-export
for backward compatibility if tests import from it.

**Effort:** 15 minutes.

### 3.2 Extend LocalStorageClient with missing methods

**File:** `src/mtss/storage/local_client.py`

Add all missing methods documented in `local-only-ingest-plan.md` Step 1:

| Method | Purpose |
|---|---|
| In-memory indexes (`_documents_by_hash`, etc.) | Fast lookups |
| `get_document_by_hash(file_hash)` | Legacy skip check |
| `get_document_by_id(doc_id)` | Status check |
| `delete_document_for_reprocess(doc_id)` | Cleanup + tombstone |
| `get_document_children(doc_id)` | Hierarchy traversal |
| `insert_topic(topic)` | Topic persistence |
| `get_topic_by_name(name)` | Topic dedup |
| `get_topic_by_id(topic_id)` | Topic lookup |
| `find_similar_topics(embedding, threshold, limit)` | Cosine similarity (brute-force) |
| `increment_topic_counts(topic_ids, chunk_delta, document_delta)` | Counter updates |
| `update_chunks_topic_ids(document_id, topic_ids)` | Chunk metadata |
| `update_chunks_topics_checked(document_id)` | Chunk metadata |
| `get_all_vessels()` | Vessel registry |

Also update existing methods:
- `update_document_status` -- also update in-memory doc object
- `update_document_archive_uris` -- also update in-memory doc object
- `_doc_to_dict` -- include all fields from design doc schema

**Dependencies:** None (can start immediately alongside Phase 2).

**Effort:** 2-3 hours.

### 3.3 Add embeddings to chunk serialization

**File:** `src/mtss/storage/local_client.py`

In `_chunk_to_dict`, add:
```python
"embedding": chunk.embedding,  # 512-dim float vector (after Phase 0.2)
"page_number": chunk.page_number,
```

Add `_topic_to_dict` method for topic JSONL serialization.

**Effort:** 15 minutes.

### 3.4 In-memory topic similarity search

**File:** `src/mtss/storage/local_client.py`

Implement `find_similar_topics` using brute-force cosine similarity. With ~200
topics and 512-dim vectors (after Phase 0.2), this is ~100K FLOPs per query --
well under 1ms in pure Python.

**Effort:** 30 minutes.

### 3.5 Flush and validation logic

**File:** `src/mtss/storage/local_client.py`

Add `flush()` method that:
1. Rewrites `topics.jsonl` with final counts.
2. Rewrites `documents.jsonl` with final statuses merged.
3. Writes `vessels.json` snapshot if vessels exist.

The `close()` method should call `flush()` internally.

**Effort:** 30 minutes.

### 3.6 Create LocalProgressTracker

**File to create:** `src/mtss/storage/local_progress_tracker.py`

Implements the `ProgressTracker` interface using `processing_log.jsonl`:
- `compute_file_hash`, `mark_started`, `mark_completed`, `mark_failed`
- `get_pending_files`, `get_failed_files`, `get_processing_stats`
- `reset_stale_processing`, `get_outdated_files`

See `local-only-ingest-plan.md` Step 3 for full implementation.

**Effort:** 1-2 hours.

### 3.7 Create LocalUnsupportedFileLogger

**File to create:** `src/mtss/storage/local_unsupported_logger.py`

Single method: `log_unsupported_file(...)` -> appends to `ingest_events.jsonl`.

See `local-only-ingest-plan.md` Step 4 for full implementation.

**Effort:** 30 minutes.

### 3.8 Type relaxation for dependency injection

**Files to modify:**

1. `src/mtss/ingest/hierarchy_manager.py` (line 18, 33):
   - Move `from ..storage.supabase_client import SupabaseClient` into `TYPE_CHECKING` block.
   - Change `db_client: SupabaseClient` -> `db_client` (or `db_client: Any`).

2. `src/mtss/ingest/archive_generator.py` (constructor):
   - Add `storage=None` parameter, default to `ArchiveStorage()`.
   - `self.storage = storage or ArchiveStorage()`

3. `src/mtss/ingest/version_manager.py` (constructor):
   - Lazy-import `SupabaseClient` only when `db is None`.

All changes are backward-compatible.

**Effort:** 30 minutes total.

**Effort:** Phase 3 total: 5-7 hours.

---

## Phase 4: Pipeline Integration

Wire everything together: component factory, CLI flag, manifest writer.

### 4.1 Create `create_local_ingest_components()` factory

**File:** `src/mtss/ingest/components.py`

Add a new factory function (see `local-only-ingest-plan.md` Step 7 for full code):

```python
def create_local_ingest_components(
    output_dir: Path,
    source_dir: Path,
    vessels: Optional[list] = None,
    enable_topics: bool = True,
) -> IngestComponents:
    """Create ingest components backed by local storage (JSONL + local files)."""
    # Uses LocalStorageClient, LocalBucketStorage
    # See local-only-ingest-plan.md Step 7 for full implementation
```

**Dependencies:** Phase 3 (all local storage components must exist).

**Effort:** 1 hour.

### 4.2 Add `--local-only` flag to CLI

**File:** `src/mtss/cli/ingest_cmd.py`

Add parameters:
```python
local_only: bool = typer.Option(False, "--local-only", help="Write to local JSONL files")
output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output dir for local-only")
```

Add a branch in `_ingest()` (around line 179) that uses `create_local_ingest_components`
instead of `SupabaseClient` when `local_only=True`. See `local-only-ingest-plan.md`
Step 8 for full implementation.

**Key differences from Supabase mode:**
- Uses `LocalStorageClient` instead of `SupabaseClient`.
- Uses `LocalProgressTracker` instead of `ProgressTracker`.
- Uses `LocalUnsupportedFileLogger` instead of `UnsupportedFileLogger`.
- Uses `LocalVersionManager` or `VersionManager(db=local_db)` with lazy import.
- Cleanup calls `db.flush()` instead of `db.close()`.
- Writes `manifest.json` at end.

**Dependencies:** Phase 4.1.

**Effort:** 1-2 hours.

### 4.3 Add `manifest.json` writer

**File:** `src/mtss/cli/ingest_cmd.py` or `src/mtss/storage/local_client.py`

Write `manifest.json` after successful ingest with:
- Version info, creation timestamp.
- Embedding model and dimensions (from `settings`).
- Chunk size and overlap settings.
- Record counts for documents, chunks, topics.

See `local-only-ingest-plan.md` Step 9 for full implementation. Note: use
`settings.embedding_dimensions` (now 512) rather than hardcoded 1536.

**Dependencies:** Phase 4.2.

**Effort:** 30 minutes.

### 4.4 LocalVersionManager

**File to create:** `src/mtss/ingest/local_version_manager.py`

Thin wrapper with same `check_document()` logic as `VersionManager` but without
requiring Supabase imports. See `local-only-ingest-plan.md` Step 11.

Alternatively, modify `VersionManager.__init__` to lazy-import `SupabaseClient`
(simpler, no new file needed).

**Effort:** 30 minutes.

### 4.5 Speed optimization: Increase MAX_CONCURRENT_FILES to 8 (06c P4)

**File:** `src/mtss/config.py` (line 134)

```python
# Before:
max_concurrent_files: int = Field(default=5, validation_alias="MAX_CONCURRENT_FILES")

# After:
max_concurrent_files: int = Field(default=8, validation_alias="MAX_CONCURRENT_FILES")
```

This is safe for local-only ingest: JSONL writes are serialized by the asyncio event
loop (single-threaded), in-memory indexes are dict operations, and archive file writes
go to unique paths per email. Helps saturate LLM and embedding API quotas.

**Effort:** 1 minute (config change only).

### 4.6 Speed optimization: Parallelize attachments within email (06c P1)

**File:** `src/mtss/ingest/pipeline.py` (lines 342-376)

Replace the sequential attachment loop with `asyncio.gather()`:

```python
# Before (sequential):
for i, attachment in enumerate(parsed_email.attachments):
    attachment_chunks = await process_attachment(...)
    ...

# After (parallel with semaphore):
import asyncio

sem = asyncio.Semaphore(3)  # limit per-email concurrency to avoid API rate limits

async def _process_one_attachment(i, attachment):
    async with sem:
        archive_file_result = _find_archive_file(attachment, archive_result)
        return await process_attachment(
            attachment=attachment,
            email_doc=email_doc,
            ...
        )

if parsed_email.attachments:
    tasks = [
        _process_one_attachment(i, att)
        for i, att in enumerate(parsed_email.attachments)
    ]
    results = await asyncio.gather(*tasks)
    for att_chunks in results:
        attachment_chunk_count += len(att_chunks)
        email_chunks.extend(att_chunks)
```

**Why this is safe** (verified via code review):
- `email_doc` is read-only at this point (created at line 173, not modified after)
- Each attachment creates its own `attach_doc` via `hierarchy_manager.create_attachment_document()`
  with unique IDs derived from `parent_doc.source_id + filename`
- `components` is shared but all called methods are stateless or handle their own state
- `email_chunks` list extension happens after `gather()` returns, not concurrently
- Progress callback needs adjustment (use attachment completion count instead of index)

**Note:** The progress reporting (line 374-376) needs adaptation since attachments
complete in arbitrary order. Track completed count with an atomic counter or report
after gather completes.

**Effort:** 1 hour (including progress reporting adjustment and testing).

**Effort:** Phase 4 total: 4-5 hours.

---

## Phase 5: Validation

### 5.1 Unit tests

| Test | Phase | Priority |
|---|---|---|
| Config defaults (chunk=1024, overlap=100, dims=512) | 0 | High |
| `is_meaningful_image` with fixtures | 1 | High |
| `classify_pdf` with simple/complex PDFs | 2 | High |
| `LocalPDFParser.parse` with text PDF | 2 | High |
| `LocalDocxParser.parse` with DOCX | 2 | Medium |
| `LocalXlsxParser.parse` with XLSX | 2 | Medium |
| `LocalCsvParser.parse` with CSV | 2 | Medium |
| `_get_tiered_parser` routing logic | 2 | High |
| `LocalStorageClient` CRUD methods | 3 | High |
| `find_similar_topics` cosine similarity | 3 | High |
| `LocalProgressTracker` lifecycle | 3 | Medium |
| `create_local_ingest_components` factory | 4 | High |
| `manifest.json` output | 4 | Medium |
| Parallel attachment processing (gather + semaphore) | 4 | Medium |
| `max_concurrent_files` default is 8 | 4 | Low |

### 5.2 Integration test

1. Create a small fixture with 3 EML files:
   - One with only body text (no attachments).
   - One with a simple PDF attachment.
   - One with an image attachment (small logo + meaningful photo).
2. Run `_ingest()` with `local_only=True`.
3. Verify output files:
   - `documents.jsonl` -- correct record count, all status=completed.
   - `chunks.jsonl` -- embeddings present, length=512.
   - `topics.jsonl` -- entries exist.
   - `processing_log.jsonl` -- all 3 files completed.
   - `archive/` folder -- expected structure.
   - `manifest.json` -- counts match JSONL line counts.

### 5.3 Cost verification

After Phase 2, run `estimate` command with updated routing to verify cost reduction:
- LlamaParse pages should drop from ~20,995 to ~6,000 (complex + legacy only).
- Vision API images should drop from ~9,860 to ~5,900 (after filename filtering).
- Expected total: ~$6-10 (vs $230 baseline).

### 5.4 Regression safety

- All existing tests must pass (changes are additive or backward-compatible).
- `ArchiveGenerator` DI change does not affect default behavior.
- `HierarchyManager` type relaxation does not affect default behavior.
- Parser registry changes only remove formats from LlamaParse that are now
  handled by local parsers -- no format goes unhandled.

**Effort:** Phase 5 total: 4-6 hours.

---

## Implementation Order (Dependencies)

```
Phase 0: Config changes                          [15 min, no deps]
    0.1 chunk_size_tokens 512 -> 1024
    0.2 embedding_dimensions 1536 -> 512

Phase 1: Image pre-filtering                     [2-3 hrs, no deps]
    1.1 Create image_filter.py (shared utility)
    1.2 Update estimator to delegate
    1.3 Add heuristic to preprocessor.py
    1.4 Resolve utils module structure

Phase 2: Local parsers (parallel with Phase 3)   [4-5 days]
    2.1 PDF complexity classifier
    2.2 Local PDF parser (PyMuPDF4LLM)
    2.3 Local Office parsers (DOCX, XLSX, CSV, HTML)
    2.4 Tiered routing in AttachmentProcessor
    2.5 Update LlamaParse scope
    2.6 Quality validation on sample set

Phase 3: Local storage backend (parallel w/ 2)   [5-7 hrs]
    3.1 Move LocalStorageClient to src/
    3.8 Type relaxation (DI prep)                   <- do early
    3.2 Extend LocalStorageClient methods
    3.3 Embeddings in chunk serialization
    3.4 Topic similarity search
    3.5 Flush/validation logic
    3.6 LocalProgressTracker
    3.7 LocalUnsupportedFileLogger

Phase 4: Pipeline wiring + speed (after 2+3)     [4-5 hrs]
    4.1 create_local_ingest_components()
    4.4 LocalVersionManager / VersionManager lazy import
    4.2 CLI --local-only flag
    4.3 Manifest writer
    4.5 MAX_CONCURRENT_FILES 5 -> 8 (06c P4)
    4.6 Parallel attachments with asyncio.gather (06c P1)

Phase 5: Validation (after 4)                     [4-6 hrs]
    5.1 Unit tests
    5.2 Integration test
    5.3 Cost verification
    5.4 Regression safety
```

**Total estimated effort:** 8-10 days.

**Phases 2 and 3 can run in parallel** since they modify different files and have
no shared dependencies. Phase 0 and Phase 1 should be done first because all
subsequent work benefits from correct config values and image filtering.

---

## Files Modified/Created Summary

### Modified files

| File | Phase | Change |
|---|---|---|
| `src/mtss/config.py` | 0 | chunk_size 512->1024, overlap 50->100, dims 1536->512 |
| `src/mtss/ingest/estimator.py` | 1 | Delegate to shared `is_meaningful_image` |
| `src/mtss/parsers/preprocessor.py` | 1 | Add heuristic before Vision API call |
| `src/mtss/parsers/attachment_processor.py` | 2 | Add `_get_tiered_parser()` routing |
| `src/mtss/parsers/llamaparse_parser.py` | 2 | Remove formats handled by local parsers |
| `src/mtss/parsers/__init__.py` | 2 | Register new local parsers (if auto-registered) |
| `src/mtss/ingest/components.py` | 4 | Add `create_local_ingest_components()` |
| `src/mtss/config.py` | 4 | `max_concurrent_files` default 5 -> 8 (06c P4) |
| `src/mtss/ingest/pipeline.py` | 4 | Parallel attachment processing with asyncio.gather (06c P1) |
| `src/mtss/cli/ingest_cmd.py` | 4 | Add `--local-only` and `--output-dir` flags |
| `src/mtss/ingest/hierarchy_manager.py` | 3 | Type relaxation (move import to TYPE_CHECKING) |
| `src/mtss/ingest/archive_generator.py` | 3 | Add `storage` DI parameter |
| `src/mtss/ingest/version_manager.py` | 3/4 | Lazy import of SupabaseClient |

### New files

| File | Phase | Purpose |
|---|---|---|
| `src/mtss/image_filter.py` | 1 | Shared image heuristic (size, dims, filename) |
| `src/mtss/parsers/pdf_classifier.py` | 2 | PDF complexity classification (simple/complex) |
| `src/mtss/parsers/local_pdf_parser.py` | 2 | PyMuPDF4LLM-based PDF parser |
| `src/mtss/parsers/local_office_parser.py` | 2 | DOCX, XLSX, CSV, HTML parsers |
| `src/mtss/storage/local_client.py` | 3 | Extended LocalStorageClient (from tests/) |
| `src/mtss/storage/local_progress_tracker.py` | 3 | JSONL-based progress tracking |
| `src/mtss/storage/local_unsupported_logger.py` | 3 | JSONL-based event logging |
| `src/mtss/ingest/local_version_manager.py` | 4 | Version manager without Supabase dep (optional) |

### New dependencies

| Package | Phase | Purpose |
|---|---|---|
| `pymupdf4llm` | 2 | Local PDF-to-markdown conversion |
| `python-docx` | 2 | DOCX text extraction |
| `openpyxl` | 2 | XLSX text extraction |
| `html2text` | 2 | HTML-to-markdown (optional, fallback exists) |

---

## Cost Impact Summary

| Optimization | Savings | Phase |
|---|---|---|
| Chunk size 512 -> 1024 | ~50% fewer chunks, storage savings | 0 |
| Embedding dims 1536 -> 512 | ~67% vector storage savings | 0 |
| Image pre-filtering (heuristic) | $19-32 (30-50% Vision API) | 1 |
| Local PDF/Office parsers | $94+ (60-75% LlamaParse) | 2 |
| **Combined first-run savings** | **$113-126** (49-55% of baseline) | 1+2 |
| **Combined with model switch** | **$220+** (95%+ of baseline) | 1+2 |

With all optimizations, estimated first-run cost drops from **$230 to ~$6-10**.
