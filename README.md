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

## Technology Stack

- **CLI:** Typer + Rich
- **Document Processing:** Docling
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o-mini via LiteLLM
- **Reranking:** Cohere via LiteLLM
- **Database:** Supabase (PostgreSQL + pgvector)

## License

MIT
