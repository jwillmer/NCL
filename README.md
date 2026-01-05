# NCL - Email RAG Pipeline

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
uv run ncl --help
```

Alternatively, activate the virtual environment to use `ncl` directly:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Now you can use ncl directly
ncl --help
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
#    databases must be recreated (ncl clean) or migrated manually.

# 4. Import vessel register (optional - enables vessel filtering)
uv run ncl vessels import data/vessel-list.csv

# 5. Ingest emails
uv run ncl ingest --source ./data/emails

# 6. Query the system
uv run ncl query "What did John say about the project deadline?"

# 7. Search without generating answer
uv run ncl search "budget allocation" --top-k 10
```

## Documentation

See the [docs/](docs/) folder for detailed documentation:

- [Processing Flow](docs/processing-flow.md) - Visual flowcharts of the data pipeline
- [Architecture](docs/architecture.md) - Technical architecture and components
- [Features](docs/features.md) - Comprehensive feature guide with CLI examples

## CLI Commands

| Command | Description |
|---------|-------------|
| `uv run ncl ingest` | Process EML files into the RAG system |
| `uv run ncl query` | Ask questions with AI-generated answers |
| `uv run ncl search` | Search without generating an answer |
| `uv run ncl stats` | View processing statistics |
| `uv run ncl reset-stale` | Reset files stuck in processing |
| `uv run ncl reprocess` | Re-ingest documents with older ingest version |
| `uv run ncl vessels import` | Import vessel register from CSV |
| `uv run ncl vessels list` | List all vessels in registry |
| `uv run ncl clean` | Delete all data (database + processed files) |

### Ingest Options

```bash
# Verbose mode - show detailed processing info
uv run ncl ingest --source ./data/emails --verbose

# Retry failed files
uv run ncl ingest --retry-failed

# Process without resuming from previous state
uv run ncl ingest --no-resume

# Reprocess files ingested with an older version
uv run ncl ingest --reprocess-outdated

# Process 10 emails concurrently (default: 5)
MAX_CONCURRENT_FILES=10 uv run ncl ingest
```

### Reprocess Command

Re-ingest documents that were processed with an older ingest version. Useful after upgrading the processing logic:

```bash
# Show documents needing reprocessing (dry run)
uv run ncl reprocess --dry-run

# Reprocess documents below current version
uv run ncl reprocess

# Reprocess documents below a specific version
uv run ncl reprocess --target-version 2

# Limit number of documents processed
uv run ncl reprocess --limit 50
```

### Vessel Commands

Import and manage the vessel register for document filtering:

```bash
# Import vessels from CSV (semicolon-delimited)
uv run ncl vessels import data/vessel-list.csv

# Clear existing vessels before import
uv run ncl vessels import --clear

# List all vessels in registry
uv run ncl vessels list
```

CSV format (semicolon-delimited):
```csv
IMO_Number;Vessel_Name;Vessel_type;DWT
9527295;MARAN THALEIA;VLCC;321225
```

### Clean Command

Delete all data from the database and local processed files. Useful for testing:

```bash
# Clean all data (prompts for confirmation)
uv run ncl clean

# Skip confirmation prompt
uv run ncl clean --yes

# Clean with verbose output (shows per-table counts)
uv run ncl clean --verbose
```

## Architecture Overview

```
EML Files → Parsing → Chunking → Embedding → Storage
                                                ↓
User Query → Embedding → Vector Search → Reranking → LLM → Answer
```

## Web Interface

NCL includes a modern web interface built with Next.js and CopilotKit for conversational document search.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js App (port 3000)                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  React Frontend                                         ││
│  │  - CopilotKit chat interface                            ││
│  │  - Supabase Auth UI                                     ││
│  │  - Real-time progress rendering via useCoAgentStateRender││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  /api/copilotkit (API Route)                            ││
│  │  - JWT validation                                       ││
│  │  - CopilotRuntime with LangGraphAgent                   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Agent (FastAPI, port 8000)              │
│  - LangGraph with CopilotKit integration                    │
│  - RAG tools with streaming progress updates                │
│  - copilotkit_emit_state() for real-time UI updates         │
└─────────────────────────────────────────────────────────────┘
```

- **Web App:** Next.js with embedded API route for CopilotKit runtime
- **Agent:** LangGraph agent with RAG capabilities and streaming progress updates

### Quick Start (Web)

```bash
# 1. Install dependencies
uv sync --extra api        # Python agent
cd web && npm install      # Next.js app

# 2. Configure environment (see Environment Setup below)

# 3. Start services (2 terminals)

# Terminal 1: Python Agent (port 8000)
uv run python -m ncl.api.main

# Terminal 2: Next.js App (port 3000)
cd web && npm run dev

# 4. Open http://localhost:3000 in your browser
```

### Environment Setup

**Python Agent** (`.env`):
```bash
# Existing NCL config plus:
CORS_ORIGINS=http://localhost:3000
API_HOST=0.0.0.0
API_PORT=8000
```

**Next.js App** (`web/.env.local`):
```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
AGENT_URL=http://localhost:8000/copilotkit
```

### Features

- **Chat Interface:** Conversational AI for document Q&A powered by CopilotKit
- **Conversation History:** Browse, search, and continue previous conversations
- **Streaming Progress:** Real-time search progress updates (Searching → Reranking → Formatting)
- **Authentication:** Supabase Auth with JWT validation in API route
- **Interactive Citations:** Clickable citation badges with source viewer dialog and file downloads
- **Agent State Sync:** Bidirectional state between frontend and Python agent via LangGraph
- **Professional Design:** NCL brand colors and responsive layout

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

## Technology Stack

- **CLI:** Typer + Rich
- **Web API:** FastAPI + CopilotKit + LangGraph
- **Frontend:** Next.js + React + TypeScript + TailwindCSS + Radix UI
- **Agent Framework:** LangGraph with CopilotKit integration
- **Document Processing:** LlamaParse (PDFs, Office, legacy formats)
- **Image Processing:** OpenAI Vision API (classification + description)
- **Text Chunking:** LangChain text splitters with tiktoken
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o via LangChain
- **Reranking:** Cohere via LiteLLM
- **Database:** Supabase (PostgreSQL + pgvector)
- **Authentication:** Supabase Auth

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=ncl --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_eml_parser.py
```

Test fixtures are located in `tests/fixtures/` including a sample email with PDF, ZIP, and PNG attachments.

## License

MIT
