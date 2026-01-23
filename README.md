# MTSS - Email RAG Pipeline

RAG pipeline for processing EML email files with attachments, preserving document hierarchy for context-aware question answering with source links.

## Features

- **Email Parsing:** Conversation-aware parsing with participant tracking
- **Multi-Format Support:** PDF, DOCX, PPTX, XLSX, images, ZIP archives, legacy formats (DOC, XLS, PPT)
- **Image Understanding:** AI-powered image classification and descriptions via OpenAI Vision
- **Document Parsing:** LlamaParse for all document types with high-res OCR and table extraction
- **Contextual Chunking:** LLM-generated document summaries prepended to chunks (35-67% retrieval improvement)
- **Vector Storage:** Supabase with pgvector for similarity search
- **Two-Stage Retrieval:** Vector search + cross-encoder reranking (20-35% accuracy improvement)
- **Citation System:** Validated citations with chunk-level references and archive links
- **Browsable Archive:** Markdown versions of all content with download links via API
- **Ingest Versioning:** Track schema version for bulk re-processing capability
- **Vessel Filtering:** Filter search results by vessel - documents are automatically tagged during ingest

## Installation

```bash
# Using uv (recommended)
uv sync
```

> **Note:** LlamaParse handles all document processing (PDFs, Office files) with built-in OCR. Requires `LLAMA_CLOUD_API_KEY`.

## Running the CLI

After installation, use `uv run` to execute CLI commands:

```bash
uv run MTSS --help
```

Alternatively, activate the virtual environment to use `MTSS` directly:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Now you can use MTSS directly
MTSS --help
```

## Quick Start

```bash
# 1. Copy environment template
cp .env.template .env

# 2. Configure your credentials in .env
#    - SUPABASE_URL, SUPABASE_KEY, SUPABASE_DB_URL
#    - OPENAI_API_KEY
#    - COHERE_API_KEY (for reranking)
#    - LLAMA_CLOUD_API_KEY (required for document parsing)
#
# Optional new settings:
#    - CONTEXT_LLM_MODEL=gpt-4o-mini (for contextual chunking)
#    - ARCHIVE_BUCKET=archive (Supabase Storage bucket for browsable archive)
#    - ARCHIVE_BASE_URL= (custom base URL for archive links)
#    - CURRENT_INGEST_VERSION=1 (increment when upgrading processing logic)

# 3. Run database migrations in Supabase
#    (see migrations/001_initial_schema.sql)
#
#    ⚠️ BREAKING CHANGE: The schema has been updated with new fields
#    for stable IDs, contextual chunking, and archive links. Existing
#    databases must be recreated (MTSS clean) or migrated manually.

# 4. Import vessel register (optional - enables vessel filtering)
uv run MTSS vessels import data/vessel-list.csv

# 5. Ingest emails
uv run MTSS ingest --source ./data/emails

# 6. Query the system
uv run MTSS query "What did John say about the project deadline?"

# 7. Search without generating answer
uv run MTSS search "budget allocation" --top-k 10
```

## Documentation

See the [docs/](docs/) folder for detailed documentation:

- [Processing Flow](docs/processing-flow.md) - Visual flowcharts of the data pipeline
- [Ingest Flow](docs/ingest-flow.md) - Detailed ingest and update flow documentation
- [Architecture](docs/architecture.md) - Technical architecture and components
- [Features](docs/features.md) - Comprehensive feature guide with CLI examples

## CLI Commands

| Command | Description |
|---------|-------------|
| `uv run MTSS ingest` | Process EML files into the RAG system |
| `uv run MTSS query` | Ask questions with AI-generated answers |
| `uv run MTSS search` | Search without generating an answer |
| `uv run MTSS stats` | View processing statistics |
| `uv run MTSS failures` | View/export ingest reports |
| `uv run MTSS reset-stale` | Reset files stuck in processing |
| `uv run MTSS reprocess` | Re-ingest documents with older ingest version |
| `uv run MTSS vessels import` | Import vessel register from CSV |
| `uv run MTSS vessels list` | List all vessels in registry |
| `uv run MTSS vessels retag` | Re-tag existing chunks with vessel IDs |
| `uv run MTSS clean` | Delete all data (database + processed files) |

### Ingest Options

```bash
# Verbose mode - show detailed processing info
uv run MTSS ingest --source ./data/emails --verbose

# Retry failed files
uv run MTSS ingest --retry-failed

# Process without resuming from previous state
uv run MTSS ingest --no-resume

# Reprocess files ingested with an older version
uv run MTSS ingest --reprocess-outdated

# Note: By default, ingest reuses previously parsed attachment content
# from the archive bucket. Use --reprocess-outdated to force re-parsing.

# Process 10 emails concurrently (default: 5)
MAX_CONCURRENT_FILES=10 uv run MTSS ingest

# Lenient mode - continue processing on errors instead of failing documents
uv run MTSS ingest --lenient
```

### Data Integrity

The ingest command now fails documents by default when critical data loss is detected:
- Attachment parsing failures
- ZIP member extraction failures

Use `--lenient` to continue processing despite errors (logs to `ingest_events` table).

**Monitoring Ingest Quality:**

```sql
-- View all events by type
SELECT event_type, severity, COUNT(*)
FROM ingest_events
GROUP BY event_type, severity;

-- Find documents with errors
SELECT parent_document_id, event_type, reason
FROM ingest_events
WHERE severity = 'error'
ORDER BY discovered_at DESC;
```

**Event Types:**
- `unsupported_file` - File format not supported
- `encoding_fallback` - Character replacement used during decoding
- `parse_failure` - Parser returned empty or error
- `archive_failure` - Archive generation failed
- `context_failure` - LLM context generation failed
- `empty_content` - Content empty after processing

### Graceful Shutdown

Press `Ctrl+C` once during ingest to stop gracefully - all in-progress emails will complete before stopping. Press `Ctrl+C` twice to force exit immediately.

### Ingest Reports

After each ingest run, a report is exported to `data/reports/`:

```bash
# View recent reports
uv run MTSS failures

# Show details of latest report
uv run MTSS failures --latest

# Export fresh report from current database state
uv run MTSS failures --export
```

Reports are saved as JSON and CSV. The system keeps the last 30 reports.

### Reprocess Command

Re-ingest documents that were processed with an older ingest version. Useful after upgrading the processing logic:

```bash
# Show documents needing reprocessing (dry run)
uv run MTSS reprocess --dry-run

# Reprocess documents below current version
uv run MTSS reprocess

# Reprocess documents below a specific version
uv run MTSS reprocess --target-version 2

# Limit number of documents processed
uv run MTSS reprocess --limit 50
```

### Vessel Commands

Import and manage the vessel register for document filtering:

```bash
# Import vessels from CSV (semicolon-delimited)
uv run MTSS vessels import data/vessel-list.csv

# Clear existing vessels before import
uv run MTSS vessels import --clear

# List all vessels in registry
uv run MTSS vessels list

# Re-tag existing chunks after adding new vessels (without re-ingesting)
uv run MTSS vessels retag

# Preview what would be updated
uv run MTSS vessels retag --dry-run

# Limit to first N documents
uv run MTSS vessels retag --limit 100
```

CSV format (semicolon-delimited, 3 required columns):
```csv
NAME;TYPE;CLASS
MARAN THALEIA;VLCC;Canopus Class
```

The vessel type/class are used for filtering RAG results. Only one filter can be active at a time (vessel OR type OR class).

### Clean Command

Delete all data from the database and local processed files. Useful for testing:

```bash
# Clean all data (prompts for confirmation)
uv run MTSS clean

# Skip confirmation prompt
uv run MTSS clean --yes

# Clean with verbose output (shows per-table counts)
uv run MTSS clean --verbose
```

## Architecture Overview

```
EML Files → Parsing → Chunking → Embedding → Storage
                                                ↓
User Query → Embedding → Vector Search → Reranking → LLM → Answer
```

## Web Interface

MTSS includes a modern web interface built with Next.js and the AG-UI SDK for conversational document search.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js App (port 3000)                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  React Frontend                                         ││
│  │  - Custom AgentChat component with AG-UI SDK            ││
│  │  - Supabase Auth UI                                     ││
│  │  - Real-time progress via AG-UI state events            ││
│  │  - Langfuse browser SDK for user interaction tracking   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  /api/agent (API Route)                                 ││
│  │  - JWT validation                                       ││
│  │  - AG-UI HttpAgent proxy to Python backend              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (AG-UI Protocol / SSE)
┌─────────────────────────────────────────────────────────────┐
│              Python Agent (FastAPI, port 8000)              │
│  - LangGraph with AG-UI protocol via add_langgraph_endpoint │
│  - RAG tools with streaming progress updates                │
│  - copilotkit_emit_state() for real-time UI state sync      │
└─────────────────────────────────────────────────────────────┘
```

- **Web App:** Next.js with custom AG-UI SDK chat components and Langfuse browser tracking
- **Agent:** LangGraph agent with RAG capabilities and streaming progress updates via AG-UI protocol

### Quick Start (Web)

```bash
# 1. Install dependencies
uv sync --extra api        # Python agent
cd web && npm install      # Next.js app

# 2. Configure environment (see Environment Setup below)

# 3. Start services (2 terminals)

# Terminal 1: Python Agent (port 8000)
uv run python -m mtss.api.main

# Terminal 2: Next.js App (port 3000)
cd web && npm run dev

# 4. Open http://localhost:3000 in your browser
```

### Environment Setup

**Python Agent** (`.env`):
```bash
# Existing MTSS config plus:
CORS_ORIGINS=http://localhost:3000
API_HOST=0.0.0.0
API_PORT=8000
```

**Next.js App** (`web/.env.local`):
```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
AGENT_URL=http://localhost:8000/agent

# Optional: Langfuse browser SDK for user interaction tracking
NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY=pk-lf-xxx
NEXT_PUBLIC_LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### Features

- **Chat Interface:** Conversational AI for document Q&A powered by AG-UI SDK
- **Maritime Response Format:** Structured responses with vessel info, component details, resolution steps, and related incidents
- **Related Incidents:** Automatic grouping of search results by incident thread to surface similar past issues
- **Conversation History:** Browse, search, and continue previous conversations
- **Streaming Progress:** Real-time search progress updates (Searching → Reranking → Formatting)
- **Authentication:** Supabase Auth with JWT validation in API route
- **Interactive Citations:** Clickable citation badges with source viewer dialog and file downloads
- **Agent State Sync:** Bidirectional state between frontend and Python agent via AG-UI protocol
- **User Feedback:** Thumbs up/down feedback on assistant responses, tracked in Langfuse (backend + browser)
- **Langfuse Browser SDK:** Client-side user interaction tracking with consistent session IDs
- **Professional Design:** MTSS brand colors and responsive layout

### Response Format

The assistant provides structured responses in a maritime technical support format:

```
Vessel: MARAN CANOPUS
Vessel Class: Canopus Class
Date Resolved: December 30, 2025

---

Based on your query about "engine temperature sensor issues", I found 8 relevant incidents.

Most Relevant Solution:

Component: Engine Temperature Sensor
Issue: Sensor providing erratic temperature readings

Resolution Steps:
1. Check sensor wiring connections for corrosion
2. Verify sensor calibration using reference thermometer
3. If readings still inconsistent, replace sensor with OEM part

Critical Notes:
- This occurred on a similar vessel class with the same engine model
- Resolution time was approximately 4 hours

---

Related Incidents:

1. **Fuel Injector Sensor** - Similar calibration issue (MT Nordic, Jan 2025)
2. **Engine Coolant Sensor** - Replaced due to corrosion damage (Nov 2024)
```

### Conversation History

The web interface includes persistent conversation storage:

- **Conversation List:** Main page (`/conversations`) shows all conversations with search and date grouping
- **Auto-naming:** Conversations are automatically titled based on the first message
- **Vessel Filter:** UI placeholder for filtering by vessel (logic to be implemented)
- **Message Persistence:** Messages are automatically saved and restored via LangGraph's checkpointer
- **Private Conversations:** Each user only sees their own conversations (Row-Level Security)

**Database Setup:**

The conversations table is included in the main schema migration (`migrations/001_initial_schema.sql`).
LangGraph automatically creates its own checkpoint tables (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`) for message persistence.

See [docs/authentication.md](docs/authentication.md) for detailed auth flow documentation.

## API Reference

When running the API server, OpenAPI documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Conversations Endpoint

The conversations API manages conversation metadata (messages are handled by LangGraph):

```
GET    /conversations              # List conversations (with ?q=, ?archived=, ?limit=, ?offset=)
POST   /conversations              # Create new conversation
GET    /conversations/{thread_id}  # Get conversation by thread_id
PATCH  /conversations/{thread_id}  # Update title, vessel_filter, or archive status
DELETE /conversations/{thread_id}  # Delete conversation
POST   /conversations/{thread_id}/touch          # Update last_message_at timestamp
POST   /conversations/{thread_id}/generate-title # Auto-generate title from message
```

**Authentication:** Requires Supabase JWT token

### Archive Endpoint

The archive endpoint proxies files from Supabase Storage, serving browsable markdown previews and original file downloads:

```
GET /archive/{path}
```

**Authentication:** Requires Supabase JWT token (same as UI auth)

**Security:**
- Files stored in private Supabase Storage bucket (not publicly accessible)
- Path traversal prevention (rejects `..`, absolute paths)
- JWT validation for all requests
- Rate limited to 100 requests/minute

**Example Usage:**
```bash
# Get markdown preview of an email
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/archive/abc123def456/email.eml.md

# Download original attachment
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/archive/abc123def456/attachments/report.pdf
```

### Citations Endpoint

The citations endpoint returns source details for inline citation references:

```
GET /citations/{chunk_id}
```

**Authentication:** Requires Supabase JWT token

**Response:**
```json
{
  "chunk_id": "8f3a2b1c",
  "source_title": "Technical Manual",
  "page": 5,
  "lines": [10, 25],
  "archive_browse_uri": "abc123/email.eml.md",
  "archive_download_uri": "abc123/email.eml",
  "content": "Full markdown content of the source..."
}
```

### Feedback Endpoint

Submit user feedback (thumbs up/down) for assistant responses. Feedback is stored in Langfuse linked to the conversation session for quality analysis.

```
POST /feedback
```

**Request Body:**
```json
{
  "thread_id": "uuid-of-conversation",
  "message_id": "message-id",
  "value": 1
}
```

- `value`: `1` for positive (thumbs up), `0` for negative (thumbs down)

**Authentication:** Requires Supabase JWT token

**Langfuse Integration:**
- Each conversation (`thread_id`) maps to a Langfuse session
- Feedback scores are attached to traces for quality tracking
- View aggregated feedback in Langfuse dashboard under Sessions

## Technology Stack

- **CLI:** Typer + Rich
- **Web API:** FastAPI + AG-UI Protocol + LangGraph
- **Frontend:** Next.js + React + TypeScript + TailwindCSS + Radix UI + AG-UI SDK
- **Agent Framework:** LangGraph with AG-UI protocol integration
- **Observability:** Langfuse (backend + browser SDK)
- **Document Processing:** LlamaParse (PDFs, Office, legacy formats)
- **Image Processing:** OpenAI Vision API (classification + description)
- **Text Chunking:** LangChain text splitters with tiktoken
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o via LangChain
- **Reranking:** Cohere via LiteLLM
- **Database:** Supabase (PostgreSQL + pgvector)
- **Authentication:** Supabase Auth

## Docker

Run the entire application (backend + frontend) in a single container.

### Quick Start

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

### Configuration

1. Copy `.env.template` to `.env` and configure:
   - `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_DB_URL`
   - `OPENAI_API_KEY`
   - `SUPABASE_JWT_SECRET`
   - See `.env.template` for all options

2. Build with Git SHA (for version tracking):
```bash
export GIT_SHA=$(git rev-parse HEAD)
docker-compose build
```

### Endpoints

When running in Docker, everything is served on port 8000:
- **Frontend:** http://localhost:8000/
- **Health Check:** http://localhost:8000/health
- **API Docs:** http://localhost:8000/docs

### CI/CD

The GitHub Action (`.github/workflows/docker-build.yml`) automatically builds and pushes images to GitHub Container Registry:
- Push to `main` → `ghcr.io/<owner>/MTSS:latest`
- Push tag `v1.0.0` → `ghcr.io/<owner>/MTSS:1.0.0`

## Testing

Tests are designed to run without external dependencies (database, APIs). All external calls are mocked.

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage (target: 80%+)
uv run pytest tests/ --cov=src/mtss --cov-report=term-missing --cov-report=html

# Run specific test file
uv run pytest tests/test_eml_parser.py -v

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only

# Run ingest-specific tests
uv run pytest tests/test_ingest*.py tests/test_version_manager.py tests/test_archive_generator.py tests/test_vessel_matcher.py -v

# Run in parallel (requires pytest-xdist)
uv run pytest tests/ -v -n auto
```

### Test Documentation

See [docs/ingest-flow-tests.md](docs/ingest-flow-tests.md) for detailed documentation of:
- Each ingest processing step
- What each step does and why
- Which tests validate each step
- Current test coverage status

### Test Structure

| File | Description |
|------|-------------|
| `tests/conftest.py` | Shared fixtures for all tests |
| `tests/test_eml_parser.py` | Email parsing tests |
| `tests/test_attachment_processor.py` | Attachment extraction tests |
| `tests/test_ingest_processing.py` | Chunker, embeddings, hierarchy, attachment processor tests |
| `tests/test_ingest_storage.py` | Database client, helpers tests |
| `tests/test_ingest_flow.py` | Integration tests (full flow with mocks) |
| `tests/test_ingest_update_flow.py` | Update flow tests (fix missing data) |
| `tests/test_version_manager.py` | Version management and deduplication tests |
| `tests/test_archive_generator.py` | Archive generation tests |
| `tests/test_vessel_matcher.py` | Vessel name matching tests |
| `tests/test_ingest_consistency.py` | Ingest/update consistency validation |

Test fixtures are located in `tests/fixtures/` including a sample email with PDF, ZIP, and PNG attachments.

### Local Storage Output (Development)

For debugging ingest behavior without Supabase, tests can use local storage mocks:

```python
# In your test
def test_ingest_output(local_ingest_output):
    # Use local_ingest_output.db instead of SupabaseClient
    # Use local_ingest_output.bucket instead of ArchiveStorage

    # After running ingest, inspect output:
    docs = local_ingest_output.read_documents_jsonl()
    chunks = local_ingest_output.read_chunks_jsonl()
    events = local_ingest_output.read_events_jsonl()
```

Output files:
- `tmp/database/documents.jsonl` - Document records
- `tmp/database/chunks.jsonl` - Chunk records
- `tmp/bucket/{doc_id}/` - Archive files

## License

MIT
