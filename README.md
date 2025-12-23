# NCL - Email RAG Pipeline

RAG pipeline for processing EML email files with attachments, preserving document hierarchy for context-aware question answering with source links.

## Features

- **Email Parsing:** Conversation-aware parsing with participant tracking
- **Multi-Format Support:** PDF, DOCX, PPTX, XLSX, images with OCR, ZIP archives
- **Image Understanding:** AI-powered image descriptions with SmolVLM
- **Semantic Chunking:** Structure-preserving chunks with heading paths
- **Vector Storage:** Supabase with pgvector for similarity search
- **Two-Stage Retrieval:** Vector search + cross-encoder reranking (20-35% accuracy improvement)
- **Source Attribution:** Every answer traces back to source documents

## Installation

```bash
# Using uv (recommended)
uv sync
```

> **Note:** First run will download PyTorch (~2GB) for OCR support via EasyOCR. Set `ENABLE_OCR=false` in `.env` to disable OCR and skip this download.

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

# 3. Run database migrations in Supabase
#    (see migrations/001_initial_schema.sql)

# 4. Ingest emails
uv run ncl ingest --source ./data/emails

# 5. Query the system
uv run ncl query "What did John say about the project deadline?"

# 6. Search without generating answer
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
| `uv run ncl clean` | Delete all data (database + processed files) |

### Ingest Options

```bash
# Verbose mode - show detailed processing info
uv run ncl ingest --source ./data/emails --verbose

# Retry failed files
uv run ncl ingest --retry-failed

# Process without resuming from previous state
uv run ncl ingest --no-resume
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

NCL includes a modern web interface built with React and CopilotKit for conversational document search.

### Quick Start (Web)

```bash
# 1. Install API dependencies
uv sync --extra api

# 2. Start the API server
uv run python -m ncl.api.main

# 3. In another terminal, start the frontend
cd frontend
npm install
npm run dev

# 4. Open http://localhost:5173 in your browser
```

### Environment Setup

Backend (`.env`):
```bash
# Add to existing .env file
SUPABASE_JWT_SECRET=your-jwt-secret-from-supabase-dashboard
CORS_ORIGINS=http://localhost:5173
API_HOST=0.0.0.0
API_PORT=8000
```

Frontend (`frontend/.env.local`):
```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
VITE_API_URL=http://localhost:8000
```

### Features

- **Chat Interface:** Conversational AI for document Q&A
- **Authentication:** Supabase Auth (Email, Magic Link, OAuth)
- **Source Attribution:** View sources with confidence scores
- **Professional Design:** NCL brand colors and responsive layout

See [frontend/README.md](frontend/README.md) for detailed frontend documentation.

## API Reference

When running the API server, OpenAPI documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technology Stack

- **CLI:** Typer + Rich
- **Web API:** FastAPI + CopilotKit
- **Frontend:** React + TypeScript + TailwindCSS + Radix UI
- **Document Processing:** Docling
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o-mini via LiteLLM
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
