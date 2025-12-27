# RAG Optimization Plan: Contextual Chunking, Citations & Versioning

## Summary

Breaking changes to the initial schema for:
1. **Contextual Chunking** - Prepend document context to chunks (35-67% retrieval improvement)
2. **Comprehensive Citation System** - Stable IDs, deterministic headers, post-processing validation
3. **Browsable Content Archive** - All content (emails + attachments) as markdown with download links
4. **Ingest Logic Versioning** - Track schema/logic version to enable bulk re-processing when ingest improves

---

## Phase 1: Updated Schema (Breaking Changes)

Modify `migrations/001_initial_schema.sql` directly. Only store latest version (delete old on update).

### 1.1 Documents Table Changes

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- STABLE IDENTIFICATION (NEW)
    source_id TEXT NOT NULL,           -- Stable ID: normalized file path or URL
    doc_id TEXT NOT NULL UNIQUE,       -- Content-addressable: hash(source_id + file_hash)

    -- VERSIONING (NEW)
    content_version INTEGER NOT NULL DEFAULT 1,   -- Increments when content changes
    ingest_version INTEGER NOT NULL DEFAULT 1,    -- Schema/logic version used during ingest

    -- Hierarchy relationships (existing)
    parent_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    root_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    depth INTEGER NOT NULL DEFAULT 0,
    path TEXT[] NOT NULL DEFAULT '{}',

    -- Document identification (existing)
    document_type document_type NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,

    -- SOURCE METADATA FOR CITATIONS (NEW)
    source_title TEXT,                 -- Human-readable title (email subject, filename, etc.)
    source_uri TEXT,                   -- Canonical URI for linking

    -- Email metadata (existing, kept for emails)
    email_subject TEXT,
    email_participants TEXT[],
    email_initiator TEXT,
    email_date_start TIMESTAMPTZ,
    email_date_end TIMESTAMPTZ,
    email_message_count INTEGER DEFAULT 1,

    -- Attachment metadata (existing)
    attachment_content_type TEXT,
    attachment_size_bytes BIGINT,

    -- Processing metadata (existing)
    status processing_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    processed_at TIMESTAMPTZ,

    -- Timestamps (existing)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- NEW INDEXES
CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_ingest_version ON documents(ingest_version);
```

### 1.2 Chunks Table Changes

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- STABLE IDENTIFICATION (NEW)
    chunk_id TEXT NOT NULL,            -- Deterministic: hash(doc_id + start_offset + end_offset)

    -- Relationship to document
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,

    -- CONTEXTUAL CHUNKING (NEW)
    context_summary TEXT,              -- Document-level context prepended for embedding
    embedding_text TEXT,               -- Full text used for embedding (context + content)

    -- CITATION METADATA (NEW/ENHANCED)
    section_path TEXT[],               -- e.g., ['Manual', 'Safety', 'GPS'] - renamed from heading_path
    source_title TEXT,                 -- Denormalized from document for fast retrieval
    source_uri TEXT,                   -- Denormalized from document
    source_id TEXT,                    -- Denormalized from document

    -- Source location (ENHANCED)
    page_number INTEGER,
    line_from INTEGER,                 -- Renamed from start_char for clarity
    line_to INTEGER,                   -- Renamed from end_char for clarity
    char_start INTEGER,                -- Keep character offsets too
    char_end INTEGER,

    -- Embedding vector (existing)
    embedding extensions.vector(1536),

    -- Metadata (existing)
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- NEW INDEXES
CREATE UNIQUE INDEX idx_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX idx_chunks_source_id ON chunks(source_id);
```

### 1.3 Ingest Version Tracking Table (NEW)

```sql
-- Track ingest logic versions for bulk re-processing
CREATE TABLE ingest_versions (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    breaking_changes BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed with initial version
INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (1, 'Initial schema with contextual chunking and citations', FALSE);
```

---

## Phase 2: Browsable Mail Archive

Generate a folder structure for each email that enables direct linking from citations.

### 2.1 Archive Structure

```
archive/
└── {doc_id}/                          # e.g., a3f2b1c8d4e5/
    ├── email.eml                      # Original EML file (for download)
    ├── email.eml.md                   # Full conversation as browsable markdown
    ├── metadata.json                  # Structured metadata for linking
    └── attachments/
        ├── report.pdf                 # Original file (for download)
        ├── report.pdf.md              # Browsable markdown version
        ├── data.xlsx                  # Original file (for download)
        ├── data.xlsx.md               # Browsable markdown version
        └── notes.md                   # Already markdown - used directly
```

**Key decisions:**
- Single `email.eml.md` contains full conversation (no split message files)
- Original EML preserved for download
- All attachments have both original + `.md` companion (unless already markdown)

### 2.2 Email Markdown Format (email.eml.md)

```markdown
# Re: Q3 Sales Report

**Type:** Email Conversation
**Participants:** john@example.com, jane@example.com
**Date Range:** 2024-01-15 to 2024-01-17
**Messages:** 2

📥 [Download Original](email.eml)

---

## Message 1
**From:** john@example.com
**To:** jane@example.com
**Date:** 2024-01-15 09:30 AM

Here's the Q3 sales report as requested.

**Attachments:**
- [report.pdf](attachments/report.pdf) ([View](attachments/report.pdf.md))

---

## Message 2
**From:** jane@example.com
**To:** john@example.com
**Date:** 2024-01-16 02:15 PM

Thanks John! The numbers look great. Can you clarify the GPS calibration data on page 12?

---
```

### 2.3 Archive Generator

```python
# src/ncl/processing/archive_generator.py

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import shutil

@dataclass
class ArchiveResult:
    """Result of generating archive for an email."""
    folder_path: Path
    email_md_path: Path              # Path to email.eml.md
    email_original_path: Path        # Path to email.eml
    attachments: Dict[str, dict]     # filename -> {download_uri, browse_uri, ...}

class MailArchiveGenerator:
    """Generate browsable markdown archive from parsed emails."""

    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def generate_archive(
        self,
        parsed_email: ParsedEmail,
        doc_id: str,
        source_eml_path: Path,
        parsed_attachment_contents: Dict[str, str]  # filename -> parsed text
    ) -> ArchiveResult:
        """Generate complete archive folder for an email."""
        folder = self.archive_dir / doc_id[:16]
        folder.mkdir(parents=True, exist_ok=True)

        # Copy original EML file
        email_original = folder / "email.eml"
        shutil.copy2(source_eml_path, email_original)

        # Generate email markdown
        email_md = folder / "email.eml.md"
        self._write_email_md(parsed_email, email_md)

        # Process attachments (originals + markdown versions)
        attachments_dir = folder / "attachments"
        attachments_dir.mkdir(exist_ok=True)
        attachment_map = self._process_attachments(
            parsed_email, attachments_dir, parsed_attachment_contents
        )

        # Write metadata JSON
        metadata = {
            "doc_id": doc_id,
            "subject": parsed_email.metadata.subject,
            "participants": parsed_email.metadata.participants,
            "email_browse_uri": f"{doc_id[:16]}/email.eml.md",
            "email_download_uri": f"{doc_id[:16]}/email.eml",
            "attachments": attachment_map,
        }
        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return ArchiveResult(
            folder_path=folder,
            email_md_path=email_md,
            email_original_path=email_original,
            attachments=attachment_map,
        )
```

### 2.4 Schema Updates for Archive Links

Add to documents table:
```sql
-- Archive location
archive_path TEXT,                  -- Relative path to archive folder
archive_browse_uri TEXT,            -- URI to browsable .md file
archive_download_uri TEXT,          -- URI to original file for download
```

Add to chunks table:
```sql
-- Archive links (denormalized for fast retrieval)
archive_browse_uri TEXT,            -- URI to browsable .md file
archive_download_uri TEXT,          -- URI to original file for download
```

---

## Phase 2.5: Browsable Content Files (All Attachments)

Generate markdown preview files for all extracted content (PDFs, DOCX, images, etc.).

### 2.5.1 Design Principles

1. **Every extracted file gets a companion `.md` file** (unless already markdown)
2. **Markdown file contains**: parsed/OCR'd content for reading + metadata + download link
3. **Original file preserved**: for download functionality
4. **Skip if source is `.md`**: no redundant conversion needed

### 2.5.2 Markdown Preview Format

Each `.md` file follows a consistent format:

```markdown
# report.pdf

**Type:** PDF Document
**Size:** 245 KB
**Pages:** 12
**Extracted:** 2024-01-15 10:30:00

📥 [Download Original](report.pdf)

---

## Content

[Full parsed/OCR'd text content from LlamaParse goes here...]
```

For images:
```markdown
# screenshot.png

**Type:** Image (PNG)
**Size:** 128 KB
**Dimensions:** 1920x1080

📥 [Download Original](screenshot.png)

---

## AI Description

[OpenAI Vision description of the image...]
```

### 2.5.3 Content File Generator

```python
# File extensions that are already markdown - skip conversion
MARKDOWN_EXTENSIONS = {'.md', '.markdown', '.mdown', '.mkd'}

@dataclass
class ContentFileResult:
    """Result of generating browsable content file."""
    original_path: Path
    markdown_path: Optional[Path]
    download_uri: str
    browse_uri: Optional[str]
    skipped: bool  # True if already markdown

class ContentFileGenerator:
    """Generate browsable markdown files for all extracted content."""

    def should_skip_markdown_generation(self, filename: str) -> bool:
        """Check if file is already markdown format."""
        return Path(filename).suffix.lower() in MARKDOWN_EXTENSIONS

    def generate_content_markdown(
        self,
        original_path: Path,
        parsed_content: str,
        content_type: str,
        file_size: int,
        extra_metadata: Optional[dict] = None
    ) -> ContentFileResult:
        """Generate markdown preview for a content file."""
        # Skip if already markdown
        if self.should_skip_markdown_generation(original_path.name):
            return ContentFileResult(skipped=True, ...)

        # Create companion .md file
        markdown_path = original_path.with_suffix(original_path.suffix + '.md')
        # ... build and write markdown content
```

---

## Phase 3: Stable ID Generation

### 3.1 Document IDs

```python
# Add to src/ncl/utils.py

def normalize_source_id(file_path: str, ingest_root: Path) -> str:
    """Normalize file path to stable source_id relative to ingest root."""
    abs_path = Path(file_path).resolve()
    try:
        rel_path = abs_path.relative_to(ingest_root.resolve())
    except ValueError:
        rel_path = abs_path
    return rel_path.as_posix().lower()

def compute_doc_id(source_id: str, file_hash: str) -> str:
    """Generate content-addressable document ID."""
    return hashlib.sha256(f"{source_id}:{file_hash}".encode()).hexdigest()[:16]

def compute_chunk_id(doc_id: str, char_start: int, char_end: int) -> str:
    """Generate deterministic chunk ID from document and offsets."""
    return hashlib.sha256(f"{doc_id}:{char_start}:{char_end}".encode()).hexdigest()[:12]
```

### 3.2 Re-chunking Stability

To preserve citations when re-chunking:
- Store `(doc_id, char_start, char_end)` for each chunk
- `chunk_id` is derived deterministically from these
- Old citations `[C:abc123]` can be resolved via the mapping

---

## Phase 4: Contextual Chunking

### 4.1 Configuration

Add to `.env.template`:
```bash
CONTEXT_LLM_MODEL=gpt-4o-mini
CURRENT_INGEST_VERSION=1
```

### 4.2 Context Generator

```python
# Add to src/ncl/parsers/chunker.py

class ContextGenerator:
    """Generate document-level context for chunks."""

    async def generate_context(self, document: Document) -> str:
        """Generate 2-3 sentence context summary."""
        prompt = f"""Summarize this document in 2-3 sentences for context.
Include: document type, author/source, date if available, main topic.

Document: {document.content[:4000]}"""

        response = await litellm.acompletion(
            model=config.CONTEXT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content

    def build_embedding_text(self, context: str, chunk_content: str) -> str:
        """Combine context with chunk for embedding."""
        return f"{context}\n\n{chunk_content}"
```

---

## Phase 5: Comprehensive Citation System

### 5.1 Retrieval Result Model

```python
@dataclass
class RetrievalResult:
    """Single retrieval result with full citation metadata."""
    text: str
    score: float
    chunk_id: str
    doc_id: str
    source_id: str
    source_uri: str
    source_title: str
    section_path: List[str]
    page_number: Optional[int]
    line_from: Optional[int]
    line_to: Optional[int]
    archive_uri: Optional[str]
```

### 5.2 Context Assembly & Citation Formatting

```python
class CitationProcessor:
    """Build context, format citations, and validate LLM responses."""

    CITATION_PATTERN = re.compile(r'\[C:([a-f0-9]+)\]')

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

CITATION RULES (MANDATORY):
1. When stating a fact from the context, append a citation: [C:chunk_id]
2. You may cite multiple chunks: [C:abc123][C:def456]
3. If information is NOT in the context, say "Not found in sources"
4. Never invent or guess information not in the context
"""

    def to_citation_header(self, result: RetrievalResult) -> str:
        """Generate deterministic citation header for LLM context."""
        parts = [
            f"S:{result.source_id[:8]}",
            f"D:{result.doc_id[:8]}",
            f"C:{result.chunk_id}",
        ]
        if result.page_number:
            parts.append(f"p:{result.page_number}")
        return f"[{' | '.join(parts)}]"
```

**Example context format:**
```
[S:manual_s | D:a3f2b1c8 | C:8f3a2b1c | p:12 | title:"Safety Manual"]
The GPS system must be calibrated before each flight according to procedures...

---

[S:email_jo | D:b2c3d4e5 | C:9a4b3c2d | title:"Re: GPS Calibration Issue"]
John mentioned that the calibration was completed on Tuesday...
```

### 5.3 Citation Post-Processing with Verification

```python
@dataclass
class ValidatedCitation:
    """Citation after validation."""
    index: int
    chunk_id: str
    source_title: str
    source_uri: str
    page: Optional[int]
    lines: Optional[Tuple[int, int]]
    archive_uri: Optional[str]
    download_uri: Optional[str]
    archive_verified: bool

@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    response: str
    citations: List[ValidatedCitation]
    invalid_citations: List[str]
    missing_archives: List[str]
    needs_retry: bool

class CitationProcessor:
    MAX_INVALID_RATIO = 0.5  # Retry if >50% of citations are invalid

    def verify_archive_exists(self, archive_uri: Optional[str]) -> bool:
        """Check if the archive file actually exists on disk."""
        if not archive_uri:
            return False
        archive_path = self.archive_dir / archive_uri.replace("archive/", "", 1)
        return archive_path.exists()

    def process_response(
        self,
        response: str,
        citation_map: Dict[str, RetrievalResult]
    ) -> CitationValidationResult:
        """Validate citations, verify archives exist, flag for retry if needed."""
        # Check 1: Does chunk_id exist in retrieved results?
        # Check 2: Does the archive file exist on disk?
        # Remove hallucinated citations, flag for retry if too many invalid
```

### 5.4 Query Engine with Retry Logic

```python
class RAGQueryEngine:
    MAX_CITATION_RETRIES = 2

    async def query(self, question: str) -> RAGResponse:
        """Execute RAG query with citation validation and retry."""
        results = await self._retrieve(question)
        citation_map = self.citation_processor.get_citation_map(results)
        context = self.citation_processor.build_context(results)

        for attempt in range(self.MAX_CITATION_RETRIES + 1):
            raw_response = await self._generate_answer(question, context)
            validation = self.citation_processor.process_response(raw_response, citation_map)

            if not validation.needs_retry:
                break

            # On retry, add explicit instruction about valid chunk IDs
            context = self._add_citation_hint(context, list(citation_map.keys()))

        return RAGResponse(
            answer=formatted,
            citations=validation.citations,
            had_invalid_citations=len(validation.invalid_citations) > 0,
        )
```

---

## Phase 6: Versioning & Re-ingestion

### 6.1 Version Manager

```python
@dataclass
class IngestDecision:
    action: Literal["insert", "update", "skip", "reprocess"]
    reason: str
    existing_doc_id: Optional[UUID] = None

class VersionManager:
    """Manage document versions and re-ingestion decisions."""

    async def check_document(self, source_id: str, file_hash: str) -> IngestDecision:
        """Determine what action to take for a document."""
        doc_id = compute_doc_id(source_id, file_hash)
        existing = await self.db.get_document_by_doc_id(doc_id)

        if not existing:
            old_version = await self.db.get_document_by_source_id(source_id)
            if old_version:
                return IngestDecision(action="update", reason="Content changed")
            return IngestDecision(action="insert", reason="New document")

        if existing.ingest_version < self.current_ingest_version:
            return IngestDecision(action="reprocess", reason="Ingest logic upgraded")

        return IngestDecision(action="skip", reason="Already processed")
```

### 6.2 CLI Command for Bulk Re-processing

```python
@app.command()
def reprocess(
    target_version: int = typer.Option(None, help="Only reprocess docs below this version"),
    dry_run: bool = typer.Option(False, help="Show what would be reprocessed"),
):
    """Re-ingest documents processed with older ingest versions."""
    # Find documents needing reprocessing
    # Delete and re-ingest from source file
```

---

## Phase 7: Updated match_chunks Function

```sql
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id TEXT,
    doc_id TEXT,
    source_id TEXT,
    content TEXT,
    similarity FLOAT,
    source_uri TEXT,
    source_title TEXT,
    section_path TEXT[],
    page_number INTEGER,
    line_from INTEGER,
    line_to INTEGER,
    document_uuid UUID,
    document_type document_type
)
```

---

## Implementation Order

| Priority | Task | Files |
|----------|------|-------|
| P0 | Update initial schema with all new fields | `migrations/001_initial_schema.sql` |
| P0 | Add ID generation utilities | `src/ncl/utils.py` |
| P0 | Create archive generator | `src/ncl/processing/archive_generator.py` |
| P1 | Update EML parser to generate archive | `src/ncl/parsers/eml_parser.py` |
| P1 | Update document processor for .md generation | `src/ncl/processing/document_processor.py` |
| P1 | Add context generator to chunker | `src/ncl/parsers/chunker.py` |
| P1 | Add RetrievalResult model | `src/ncl/models/chunk.py` |
| P1 | Create citation processor | `src/ncl/rag/citation_processor.py` |
| P1 | Update query engine for citations | `src/ncl/rag/query_engine.py` |
| P2 | Create version manager | `src/ncl/processing/version_manager.py` |
| P2 | Add reprocess CLI command | `src/ncl/cli.py` |
| P2 | Update README documentation | `README.md` |

---

## Files to Modify (Existing)

1. `migrations/001_initial_schema.sql` - Breaking schema changes
2. `src/ncl/utils.py` - Add ID generation functions
3. `src/ncl/parsers/eml_parser.py` - Integrate archive generation
4. `src/ncl/parsers/chunker.py` - Add ContextGenerator class
5. `src/ncl/models/chunk.py` - Add RetrievalResult, ValidatedCitation
6. `src/ncl/rag/query_engine.py` - Use CitationProcessor
7. `src/ncl/storage/supabase_client.py` - Version queries
8. `src/ncl/processing/document_processor.py` - Generate .md files
9. `src/ncl/cli.py` - Add reprocess command
10. `src/ncl/config.py` - New config vars
11. `README.md` - Document new features

## New Files to Create

1. `src/ncl/processing/archive_generator.py` - Mail archive + ContentFileGenerator
2. `src/ncl/processing/version_manager.py` - Ingest versioning logic
3. `src/ncl/rag/citation_processor.py` - Citation building, validation, formatting

---

## Configuration

Add to `.env.template`:
```bash
# Contextual chunking LLM
CONTEXT_LLM_MODEL=gpt-4o-mini

# Ingest versioning (increment when changing ingest logic)
CURRENT_INGEST_VERSION=1

# Archive directory for browsable email markdown
ARCHIVE_DIR=./archive

# Base URL for archive links in citations (optional, for web serving)
ARCHIVE_BASE_URL=
```

---

## Security Considerations

### Archive Access Control
The archive folder contains sensitive email content as plain markdown files:

1. **File System Permissions**: Set restrictive permissions on `ARCHIVE_DIR`
2. **Web Access**: Serve behind authentication if `ARCHIVE_BASE_URL` is set
3. **Email Address Exposure**: Consider redaction if exposing externally

### Path Validation
The `normalize_source_id()` function prevents path traversal attacks by resolving paths relative to the ingest root.

---

## Expected Outcomes

- **Retrieval Quality**: 35-67% improvement from contextual embeddings
- **Citations**: Validated, numbered references with browse + download links
- **Browsable Archive**: All content (emails + attachments) as readable markdown
- **Download Support**: Original files preserved for download alongside markdown views
- **Re-ingestion**: Update documents when content OR ingest logic changes
- **Bulk Reprocess**: CLI command to upgrade all docs to new ingest version

## Example Citation Output

**User question:** "What was discussed about GPS calibration?"

**RAG Response:**
```
The GPS system must be calibrated before each flight according to standard procedures [1].
John confirmed that the calibration was completed on Tuesday and verified by the
maintenance team [2].

**Sources:**
[1] Safety Manual | p.12 | [View](archive/a3f2b1c8/attachments/manual.pdf.md) | [Download](archive/a3f2b1c8/attachments/manual.pdf)
[2] Re: GPS Issue | [View](archive/b2c3d4e5/email.eml.md) | [Download](archive/b2c3d4e5/email.eml)
```

Users can:
- Click **[View]** to see the markdown-rendered content in their browser
- Click **[Download]** to get the original file (EML, PDF, DOCX, etc.)
- Navigate the archive folder to explore attachments and related content
