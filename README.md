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

# Or using pip
pip install -e .
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
ncl ingest --source ./data/emails

# 5. Query the system
ncl query "What did John say about the project deadline?"

# 6. Search without generating answer
ncl search "budget allocation" --top-k 10
```

## Documentation

See the [docs/](docs/) folder for detailed documentation:

- [Processing Flow](docs/processing-flow.md) - Visual flowcharts of the data pipeline
- [Architecture](docs/architecture.md) - Technical architecture and components
- [Features](docs/features.md) - Comprehensive feature guide with CLI examples

## CLI Commands

| Command | Description |
|---------|-------------|
| `ncl ingest` | Process EML files into the RAG system |
| `ncl query` | Ask questions with AI-generated answers |
| `ncl search` | Search without generating an answer |
| `ncl stats` | View processing statistics |
| `ncl reset-stale` | Reset files stuck in processing |

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
python -m ncl.api.main

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
pytest tests/

# Run with coverage
pytest tests/ --cov=ncl --cov-report=term-missing

# Run specific test file
pytest tests/test_eml_parser.py
```

Test fixtures are located in `tests/fixtures/` including a sample email with PDF, ZIP, and PNG attachments.

## License

MIT
