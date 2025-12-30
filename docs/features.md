# NCL Features

This document provides a comprehensive overview of NCL's features and capabilities.

## Core Features

### 1. Email Parsing with Conversation Support

NCL treats emails as conversations, not just individual messages.

**Conversation Parsing:**
- Splits threaded emails into individual messages
- Extracts participants from headers and quoted replies
- Identifies conversation initiator (first sender)
- Tracks date range of conversation
- Preserves message threading via In-Reply-To/References

**Supported Patterns:**
- Outlook-style: `From: ... Sent: ... To: ...`
- Gmail-style: `On Mon, Jan 1, John wrote:`
- Original Message markers: `-----Original Message-----`
- Quoted replies with `>` prefix

**Example Output:**
```python
ParsedEmail(
    metadata=EmailMetadata(
        subject="Project Update",
        participants=["alice@example.com", "bob@example.com", "carol@example.com"],
        initiator="alice@example.com",
        date_start=datetime(2024, 1, 15, 9, 0),
        date_end=datetime(2024, 1, 16, 14, 30),
        message_count=5
    ),
    messages=[...],  # 5 individual messages
    attachments=[...]
)
```

### 2. Multi-Format Attachment Processing

NCL processes a wide variety of document formats using Docling.

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | .pdf | OCR, tables, images, structure preservation |
| Word | .docx | Full text, headers, lists, tables |
| PowerPoint | .pptx | Slides, speaker notes, embedded content |
| Excel | .xlsx | Cell data, sheet names, formulas as values |
| Images | .png, .jpg, .jpeg, .tiff, .bmp | OCR text extraction, AI description |
| HTML | .html | Content extraction, link handling |
| ZIP | .zip | Recursive extraction, nested archives |

### 3. Intelligent Image Understanding

When `ENABLE_PICTURE_DESCRIPTION=true`, NCL uses AI to describe images.

**How it works:**
1. SmolVLM model analyzes each image
2. Generates detailed text description
3. Description becomes searchable content
4. Includes: text, diagrams, charts, visual elements

**Example Description:**
```
"A bar chart showing quarterly revenue from Q1 to Q4 2024.
Q1: $2.3M, Q2: $2.8M, Q3: $3.1M, Q4: $3.5M.
The chart has a blue color scheme with a 'Revenue Growth' title.
An upward trend arrow indicates 52% year-over-year growth."
```

**Use Case:** Find charts, diagrams, or screenshots by describing their content.

### 4. ZIP Archive Support

NCL automatically extracts and processes ZIP attachments.

**Features:**
- Recursive extraction (nested ZIPs)
- Security: Path traversal prevention
- Filters: Hidden files, macOS metadata
- Directory structure preservation
- Only extracts supported file types

**Security Limits (DoS Protection):**

NCL enforces resource limits to prevent denial-of-service attacks from malicious ZIP files:

| Limit | Default | Environment Variable | Description |
|-------|---------|---------------------|-------------|
| Max Files | 100 | `ZIP_MAX_FILES` | Maximum files to extract from a single ZIP |
| Max Depth | 3 | `ZIP_MAX_DEPTH` | Maximum nested ZIP depth (zip-in-zip) |
| Max Size | 500 MB | `ZIP_MAX_TOTAL_SIZE_MB` | Maximum total extracted size |

When limits are exceeded, extraction stops and a `ZipExtractionError` is raised. The email is still processed, but the offending ZIP is logged as failed.

**Example Processing:**
```
email.eml
└── documents.zip
    ├── report.pdf         → Processed
    ├── images/
    │   ├── chart1.png     → Processed with AI description
    │   └── chart2.png     → Processed with AI description
    ├── data.xlsx          → Processed
    └── nested.zip         → Recursively extracted (depth 2)
        └── more_docs.pdf  → Processed
```

### 5. Semantic Chunking with Structure Preservation

NCL uses HybridChunker for intelligent text segmentation.

**Features:**
- OpenAI tokenizer for accurate token counting
- Heading hierarchy preservation
- Merge undersized peer chunks
- Configurable chunk size (default: 512 tokens)

**Heading Path Example:**
```
heading_path: ["Chapter 1", "Section 1.2", "Subsection 1.2.3"]
```

This allows queries like "What does Section 1.2 say about X?"

### 6. Two-Stage Retrieval with Reranking

NCL implements state-of-the-art retrieval for better accuracy.

**Stage 1: Vector Search (Bi-encoder)**
- Fast approximate nearest neighbor search
- Retrieves 20 candidates
- Uses pgvector with cosine similarity

**Stage 2: Cross-Encoder Reranking**
- Examines query+document pairs together
- More accurate semantic understanding
- Returns top 5 most relevant results

**Accuracy Improvement:** 20-35% over vector search alone

**Supported Reranking Providers:**
- Cohere (recommended): `cohere/rerank-english-v3.0`
- Azure AI: `azure_ai/cohere-rerank-v3.5`
- AWS Bedrock: `bedrock/rerank`
- Self-hosted: `infinity/<model>`

### 7. Document Hierarchy

NCL maintains parent-child relationships for context.

**Benefits:**
- Know which email an attachment came from
- Trace content back to source
- Filter by document type
- Navigate from chunk to root email

**Hierarchy Fields:**
- `parent_id`: Direct parent document
- `root_id`: Root email (always an email)
- `depth`: 0 for emails, 1+ for attachments
- `path`: Full ancestry array

### 8. File Registry for Fast Lookup

NCL maintains a file registry for efficient processing.

**Features:**
- Hash-based file deduplication
- Subdirectory support for organizing data
- Processing status tracking (pending, processing, completed, failed)
- Automatic detection of changed files via hash comparison

**File States:**
| Status | Description |
|--------|-------------|
| pending | File found, not yet processed |
| processing | Currently being processed |
| completed | Successfully processed |
| failed | Processing failed (will retry) |

### 9. Unsupported File Tracking

NCL logs files it cannot process for visibility.

**Tracked Information:**
- File name and path
- MIME type and extension
- Source email path
- Source ZIP path (if extracted from archive)
- Reason for being unsupported

**Reasons:**
| Reason | Description |
|--------|-------------|
| unsupported_format | File type not supported by Docling |
| too_large | File exceeds size limit |
| corrupted | File is corrupted or invalid |
| extraction_failed | Processing failed despite supported format |

**View unsupported files:**
```bash
ncl stats  # Shows unsupported files breakdown
```

### 10. Resumable Processing

NCL tracks progress for reliable batch processing.

**Features:**
- Skip already-processed files
- Retry failed files with `--retry-failed`
- Reset stale processing with `ncl reset-stale`
- Hash-based deduplication

**Progress Tracking:**
```bash
# Resume from where you left off
ncl ingest --source ./data/source --resume

# Retry files that failed
ncl ingest --retry-failed

# Reset stuck files (processing > 60 min)
ncl reset-stale --max-age 60
```

**Progress Display:**

During ingestion, NCL shows real-time progress for each concurrent worker:

```
⠙ [1] email_001.eml               ━━━━━━━━━━  3/5
⠙ [2] email_002.eml               ━━━━━━━━━━  7/12
⠙ [3] email_003.eml               ━━━━━━━━━━  1/3
⠙ [4] email_004.eml (embeddings)  ━━━━━━━━━━  2/2
⠙ [5] (idle)
⠙ Total                           ━━━━━━━━━━  42/100
```

Each worker slot shows:
- Current email file name
- Attachment processing progress (e.g., 3/5 = 3 of 5 attachments processed)
- Status indicator for embedding generation phase
- Idle state when waiting for work

### 11. Vessel Filtering

NCL supports filtering search results by vessel. Documents are automatically tagged with vessel references during ingestion.

**How Vessel Tagging Works:**

During ingestion, NCL scans the **email subject and body** for vessel names and aliases from the vessel registry. Matched vessel IDs are stored in chunk metadata for filtering.

```
Email arrives → Parse subject + body → Match vessel names → Store vessel_ids in chunks
```

**Current Tagging Scope:**
| Content | Scanned | Notes |
|---------|---------|-------|
| Email subject | Yes | Primary source of vessel references |
| Email body | Yes | Includes all messages in thread |
| Attachments | No | Inherit vessel tags from parent email |

Attachments inherit the `vessel_ids` from their parent email. This means if an email mentions "MARAN THALEIA" in the subject or body, all attachments (PDFs, images, etc.) are also tagged with that vessel.

**Why Email-Only Tagging:**
- **Performance:** Scanning parsed attachment text would slow ingestion significantly
- **Reliability:** Email subject/body usually names the vessel explicitly
- **Simplicity:** Avoids false positives from vessel names appearing in unrelated document content

**Future Enhancement:** Attachment content scanning can be added if email-level tagging proves insufficient. This would scan parsed text from PDFs, DOCX, etc. and merge found vessels with email-level matches. See `VesselMatcher.find_vessels()` in [vessel_matcher.py](../src/ncl/processing/vessel_matcher.py).

**Vessel Matching:**
- Case-insensitive matching
- Word boundary detection (prevents "MARAN" matching "AMARANTO")
- Supports vessel aliases (configured in registry)

**Query Flow with Vessel Filter:**
```
User selects vessel → Frontend passes vessel_id → Agent builds metadata filter
    → match_chunks filters by chunks.metadata @> '{"vessel_ids":["uuid"]}'
    → Only matching chunks returned
```

**CLI Commands:**
```bash
# Import vessel registry from CSV
uv run ncl vessels import data/vessel-list.csv

# List all vessels
uv run ncl vessels list
```

**CSV Format (semicolon-delimited):**
```csv
IMO_Number;Vessel_Name;Vessel_type;DWT
9527295;MARAN THALEIA;VLCC;321225
```

### 12. Source Attribution

Every answer includes traced sources with interactive citations in the web UI.

**Citation Features:**
- Inline citation badges `[1]`, `[2]` rendered as clickable elements
- Tooltip preview showing source title on hover
- Collapsible "Sources" accordion below each response
- Full source viewer dialog with markdown content
- Download original files directly from the UI

**Web UI Interaction:**
1. Citations appear as small numbered badges inline with the response text
2. Hover over a citation to see the source title
3. Click a citation or use the Sources accordion to view details
4. The source dialog displays the full markdown content
5. Download the original file (PDF, email, etc.) with one click

**Source Information:**
- Source document title
- Page number (if applicable)
- Line range within the document
- Archive links for viewing and downloading

**Example Response:**
```
The GPS system requires calibration before each flight [1].
This was confirmed by the maintenance team [2].
```

Where `[1]` and `[2]` are interactive badges that reveal source details on click.

## CLI Commands

### `ncl ingest`

Process EML files into the RAG system.

```bash
ncl ingest [OPTIONS]

Options:
  -s, --source PATH      Directory containing EML files
  -b, --batch-size INT   Files per batch (default: 10)
  --resume/--no-resume   Resume from progress (default: resume)
  --retry-failed         Retry previously failed files
```

**Examples:**
```bash
# Ingest all emails
ncl ingest --source ./data/emails

# Process in smaller batches
ncl ingest --source ./emails --batch-size 5

# Start fresh (ignore progress)
ncl ingest --source ./emails --no-resume

# Retry failed files only
ncl ingest --retry-failed
```

### `ncl query`

Ask questions with AI-generated answers.

```bash
ncl query "Your question here" [OPTIONS]

Options:
  -k, --top-k INT          Candidates for reranking (default: 20)
  -t, --threshold FLOAT    Similarity threshold (default: 0.5)
  -n, --rerank-top-n INT   Final results (default: 5)
  --no-rerank              Disable reranking
```

**Examples:**
```bash
# Basic query
ncl query "What was discussed about the budget?"

# More candidates for better recall
ncl query "Find all mentions of Project X" --top-k 50

# Lower threshold for broader results
ncl query "Meeting notes from January" --threshold 0.3

# Disable reranking for faster results
ncl query "Quick search" --no-rerank
```

### `ncl search`

Search without generating an answer.

```bash
ncl search "search terms" [OPTIONS]

Options:
  -k, --top-k INT          Candidates (default: 20)
  -t, --threshold FLOAT    Similarity threshold (default: 0.5)
  -n, --rerank-top-n INT   Final results (default: 10)
  --no-rerank              Disable reranking
```

**Examples:**
```bash
# Find relevant documents
ncl search "quarterly report 2024"

# Get more results
ncl search "invoice" --rerank-top-n 20
```

### `ncl stats`

View processing statistics and unsupported files breakdown.

```bash
ncl stats
```

**Output:**
```
┌─────────────────────────────────┐
│     Processing Statistics       │
├─────────────────┬───────────────┤
│ Status          │         Count │
├─────────────────┼───────────────┤
│ Completed       │         1,234 │
│ Processing      │             3 │
│ Failed          │            12 │
│ Pending         │            45 │
└─────────────────┴───────────────┘

┌─────────────────────────────────┐
│ Unsupported Files (23 total)    │
├─────────────────┬───────────────┤
│ Reason          │         Count │
├─────────────────┼───────────────┤
│ Unsupported Format │          18 │
│ Extraction Failed  │           3 │
│ Corrupted          │           2 │
└─────────────────┴───────────────┘

┌─────────────────────────────────┐
│ Unsupported Files by MIME Type  │
├─────────────────┬───────────────┤
│ MIME Type       │         Count │
├─────────────────┼───────────────┤
│ video/mp4       │             8 │
│ application/exe │             5 │
│ audio/mpeg      │             5 │
│ unknown         │             5 │
└─────────────────┴───────────────┘
```

### `ncl reset-stale`

Reset files stuck in processing state.

```bash
ncl reset-stale [OPTIONS]

Options:
  -m, --max-age INT   Max age in minutes (default: 60)
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.template`:

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://...
OPENAI_API_KEY=sk-...

# Optional - Models
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
LLM_MODEL=gpt-4o-mini

# Optional - Chunking
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=50

# Optional - Paths (source = user data, processed = NCL-generated)
DATA_SOURCE_DIR=./data/source          # EML files (supports subdirectories)
DATA_PROCESSED_DIR=./data/processed    # Extracted attachments and ZIP contents

# Optional - Processing
BATCH_SIZE=10
MAX_CONCURRENT_EMBEDDINGS=5
ENABLE_OCR=true
ENABLE_PICTURE_DESCRIPTION=true

# Optional - Reranking
RERANK_ENABLED=true
RERANK_MODEL=cohere/rerank-english-v3.0
RERANK_TOP_N=5
COHERE_API_KEY=your-cohere-key
```

## Performance Considerations

### Embedding Costs

| Component | Approximate Cost |
|-----------|-----------------|
| Email body | ~$0.0001 |
| PDF page | ~$0.0002 |
| Image with description | ~$0.0003 |
| Total per email (avg) | ~$0.001 |

### Query Latency

| Stage | Time |
|-------|------|
| Embedding generation | 100-200ms |
| Vector search | 50-100ms |
| Reranking | 200-500ms |
| LLM generation | 500-2000ms |
| **Total (with rerank)** | **~1-3 seconds** |
| **Total (no rerank)** | **~0.5-2 seconds** |

### Scaling Tips

1. **Large archives:** Use `--batch-size 5` to avoid memory issues
2. **Many attachments:** Consider disabling picture description
3. **Fast queries:** Use `--no-rerank` for time-sensitive searches
4. **Better accuracy:** Increase `--top-k` for reranking
