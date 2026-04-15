# MTSS Architecture

This document describes the technical architecture of the MTSS email RAG pipeline.

## System Overview

MTSS is a Retrieval-Augmented Generation (RAG) system designed specifically for email archives. It processes EML files with attachments, preserving document hierarchy for context-aware question answering with source attribution.

```
+------------------+     +------------------+     +------------------+
|   CLI Layer      |     |  Processing      |     |   Storage        |
|                  |     |  Layer           |     |   Layer          |
|  - ingest        | --> |  - Parsing       | --> |  - Supabase      |
|  - query         |     |  - Embedding     |     |  - pgvector      |
|  - search        |     |  - Reranking     |     |                  |
|  - stats         |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```

## Project Structure

```
src/mtss/
├── __init__.py, config.py, version.py, utils.py
├── cli/                         # Thin CLI command modules
│   ├── __init__.py              # App setup, sub-app registration
│   ├── _common.py               # Console, verbose helpers, shared formatting
│   ├── ingest_cmd.py            # ingest + estimate commands
│   ├── maintenance_cmd.py       # ingest-update, reprocess, reindex
│   ├── query_cmd.py             # query + search commands
│   ├── admin_cmd.py             # stats, failures, reset-stale, clean
│   └── entities_cmd.py          # vessels + topics commands
├── models/
│   ├── document.py              # Document, EmailMetadata, ParsedEmail
│   ├── chunk.py                 # Chunk, RetrievalResult, EnhancedRAGResponse
│   ├── topic.py, vessel.py
├── parsers/
│   ├── eml_parser.py            # EML parsing with conversation support
│   ├── attachment_processor.py  # LlamaParse + ZIP extraction
│   ├── chunker.py               # Document chunking + context generation
│   └── ...
├── ingest/                      # All ingest business logic
│   ├── components.py            # Component factory
│   ├── pipeline.py              # Single-email processing
│   ├── attachment_handler.py    # Attachment + ZIP processing
│   ├── repair.py                # Ingest-update/fix logic
│   ├── archive_generator.py     # Browsable archive generation
│   ├── hierarchy_manager.py     # Document tree management
│   ├── version_manager.py       # Ingest versioning/dedup
│   ├── lane_classifier.py       # Fast/slow lane classification
│   ├── estimator.py, helpers.py
├── processing/                  # Shared processing infrastructure
│   ├── embeddings.py            # Embeddings via LiteLLM (OpenRouter)
│   ├── topics.py                # Topic extraction + matching
│   ├── vessel_matcher.py        # Vessel name matching
│   └── image_processor.py       # Image processing
├── rag/                         # Retrieval pipeline
│   ├── retriever.py             # Embed + search + rerank
│   ├── query_engine.py          # LLM generation + citation validation
│   ├── citation_processor.py    # Citation processing
│   ├── reranker.py              # Cross-encoder reranking
│   └── topic_filter.py          # Topic pre-filtering
├── storage/                     # Infrastructure
│   ├── supabase_client.py       # Facade over repositories
│   ├── repositories/            # Focused DB access
│   │   ├── base.py, documents.py, search.py, domain.py
│   ├── archive_storage.py
│   ├── progress_tracker.py, failure_report.py
│   └── unsupported_file_logger.py
├── api/                         # Web layer
│   ├── agent.py                 # LangGraph agent
│   └── ...
└── observability/
```

## Core Components

### 1. Configuration (`config.py`)

Uses Pydantic Settings for type-safe configuration from environment variables.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | openrouter/openai/text-embedding-3-small | Embedding model via OpenRouter |
| `EMBEDDING_DIMENSIONS` | 1536 | Vector dimensions |
| `LLM_MODEL` | openrouter/openai/gpt-5-nano | Default LLM model |
| `THREAD_DIGEST_MODEL` | (fallback to LLM_MODEL) | Thread digest summarization model |
| `CHUNK_SIZE_TOKENS` | 1024 | Max tokens per chunk |
| `ENABLE_OCR` | true | Enable OCR for images/PDFs |
| `ENABLE_PICTURE_DESCRIPTION` | true | Enable AI image descriptions |
| `RERANK_ENABLED` | true | Enable two-stage retrieval |
| `RERANK_MODEL` | cohere/rerank-v3.5 | Cross-encoder model via OpenRouter |

### 2. EML Parser (`parsers/eml_parser.py`)

Parses email files using Python's `email` library with `policy.default`.

**Features:**
- Conversation parsing (splits threaded emails into messages)
- Participant extraction (from, to, cc, conversation history)
- Initiator detection (first sender in thread)
- HTML to plain text conversion
- Attachment extraction with deduplication

**Output:** `ParsedEmail` containing:
- `metadata`: EmailMetadata with conversation info
- `messages`: List of individual EmailMessage objects
- `full_text`: Combined conversation text
- `attachments`: List of ParsedAttachment with saved paths

### 3. Attachment Processor (`parsers/attachment_processor.py`)

Processes attachments using LlamaParse for document understanding.

**Supported Formats:**
| Format | MIME Type | Features |
|--------|-----------|----------|
| PDF | application/pdf | OCR, table extraction, picture description |
| Images | image/png, jpeg, tiff, bmp | OCR, AI description |
| Word | application/vnd.openxmlformats-officedocument.wordprocessingml.document | Full text extraction |
| PowerPoint | application/vnd.openxmlformats-officedocument.presentationml.presentation | Slide text + notes |
| Excel | application/vnd.openxmlformats-officedocument.spreadsheetml.sheet | Cell data extraction |
| HTML | text/html | Content extraction |
| ZIP | application/zip | Recursive extraction |

**ZIP Extraction Security:**
- Path traversal prevention (`../` detection)
- Absolute path blocking
- Hidden file filtering (`.files`, `__MACOSX/`)
- Nested ZIP support with depth limits

**Chunking:**
- Uses `HybridChunker` from docling-core
- tiktoken tokenizer for accurate token counting
- Preserves heading hierarchy
- Merges undersized peer chunks

### 4. Hierarchy Manager (`ingest/hierarchy_manager.py`)

Manages parent-child relationships between documents.

**Document Fields:**
- `id`: UUID primary key
- `parent_id`: Parent document UUID (null for emails)
- `root_id`: Root email UUID (for ancestry queries)
- `depth`: 0 for emails, 1+ for attachments
- `path`: Array of ancestor IDs for fast queries

**Hierarchy Example:**
```
Email (depth=0, root_id=self)
├── PDF Attachment (depth=1, root_id=email)
│   └── [chunks with heading_path]
├── ZIP Attachment (depth=1)
│   ├── Image 1 (depth=2, root_id=email)
│   └── Image 2 (depth=2, root_id=email)
└── DOCX Attachment (depth=1)
```

### 5. Embedding Generator (`processing/embeddings.py`)

Generates vector embeddings via LiteLLM (OpenRouter).

**Features:**
- Model: `text-embedding-3-small` via OpenRouter (configurable dimensions)
- Batch processing (100 texts per API call)
- Async operation for performance
- Cost estimation (~$0.001 per email)

### 6. Reranker (`rag/reranker.py`)

Implements two-stage retrieval for improved accuracy.

**How it works:**
1. **Stage 1 (Bi-encoder):** Fast vector search retrieves 20 candidates
2. **Stage 2 (Cross-encoder):** Reranker scores query+document pairs

**Improvement:** 20-35% accuracy gain over vector search alone

**Provider:** OpenRouter rerank API (direct HTTP)
- Default model: `cohere/rerank-v3.5`
- Browse models: https://openrouter.ai/models?type=rerank

### 7. Supabase Client (`storage/supabase_client.py`)

Handles all database operations with pgvector for similarity search.

**Tables:**
- `documents`: Document hierarchy with metadata
- `chunks`: Text chunks with embeddings
- `processing_log`: Progress tracking

**Key Operations:**
- `insert_document()`: Create document with hierarchy
- `insert_chunks()`: Bulk insert with embeddings
- `search_similar_chunks()`: pgvector cosine similarity
- `get_document_ancestry()`: Full path to root email

### 8. Query Engine (`rag/query_engine.py`)

Orchestrates the full RAG pipeline.

**Query Flow:**
1. Generate query embedding
2. Vector search (top_k candidates)
3. Rerank results (if enabled)
4. Build context with source headers
5. Generate answer with LLM
6. Format response with sources

**LLM Prompt Strategy:**
- System prompt enforces grounded answers
- Context includes source attribution headers
- Temperature: 0.3 for factual responses
- Max tokens: 1000

## Database Schema

### documents table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    parent_id UUID REFERENCES documents(id),
    root_id UUID REFERENCES documents(id),
    depth INTEGER NOT NULL DEFAULT 0,
    path TEXT[] NOT NULL DEFAULT '{}',

    document_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,

    -- Email metadata (when document_type = 'email')
    email_subject TEXT,
    email_participants TEXT[],
    email_initiator TEXT,
    email_date_start TIMESTAMPTZ,
    email_date_end TIMESTAMPTZ,

    -- Attachment metadata
    attachment_content_type TEXT,
    attachment_size_bytes BIGINT,

    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### chunks table
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,

    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading_path TEXT[] DEFAULT '{}',
    section_title TEXT,
    page_number INTEGER,

    embedding vector(1536),
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity search index
CREATE INDEX chunks_embedding_idx ON chunks
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### match_chunks function
```sql
CREATE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
) RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    similarity float,
    -- ... joined document fields
) AS $$
    SELECT
        c.id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity,
        d.file_path,
        d.document_type,
        -- ... more fields
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$ LANGUAGE sql;
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| CLI | Typer + Rich | Command-line interface |
| Config | Pydantic Settings | Environment configuration |
| Email Parsing | Python email | EML file processing |
| Document Processing | Docling | PDF, Office, Image extraction |
| Chunking | docling-core HybridChunker | Semantic chunking |
| OCR | EasyOCR | Text extraction from images |
| Image Description | SmolVLM | AI image understanding |
| Embeddings | text-embedding-3-small via OpenRouter | Vector generation |
| LLM | OpenRouter via LiteLLM | Answer generation |
| Reranking | OpenRouter rerank API | Result refinement |
| Database | Supabase (PostgreSQL) | Data persistence |
| Vector Search | pgvector | Similarity search |
| Async | asyncio + asyncpg | Non-blocking I/O |

## Security Considerations

1. **ZIP Extraction:** Path traversal prevention, hidden file filtering
2. **API Keys:** Environment variables, never in code
3. **Database:** Service role key for backend, RLS for frontend
4. **File Paths:** Sanitization of filenames and paths
5. **Input Validation:** Pydantic models for all data structures
