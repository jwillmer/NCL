# NCL Architecture

This document describes the technical architecture of the NCL email RAG pipeline.

## System Overview

NCL is a Retrieval-Augmented Generation (RAG) system designed specifically for email archives. It processes EML files with attachments, preserving document hierarchy for context-aware question answering with source attribution.

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
src/ncl/
├── __init__.py
├── cli.py                    # Typer CLI commands
├── config.py                 # Pydantic settings
├── models/
│   ├── document.py           # Document, EmailMetadata, ParsedEmail
│   └── chunk.py              # Chunk, SourceReference, RAGResponse
├── parsers/
│   ├── eml_parser.py         # EML parsing with conversation support
│   └── attachment_processor.py # Docling + ZIP extraction
├── processing/
│   ├── hierarchy_manager.py  # Document tree management
│   ├── embeddings.py         # OpenAI embeddings via LiteLLM
│   └── reranker.py           # Two-stage retrieval reranking
├── storage/
│   ├── supabase_client.py    # Database operations + vector search
│   └── progress_tracker.py   # Resumable processing state
└── rag/
    └── query_engine.py       # RAG query + answer generation
```

## Core Components

### 1. Configuration (`config.py`)

Uses Pydantic Settings for type-safe configuration from environment variables.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | 1536 | Vector dimensions |
| `LLM_MODEL` | gpt-4o-mini | Answer generation model |
| `CHUNK_SIZE_TOKENS` | 512 | Max tokens per chunk |
| `ENABLE_OCR` | true | Enable OCR for images/PDFs |
| `ENABLE_PICTURE_DESCRIPTION` | true | Enable AI image descriptions |
| `RERANK_ENABLED` | true | Enable two-stage retrieval |
| `RERANK_MODEL` | cohere/rerank-english-v3.0 | Cross-encoder model |

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

Processes attachments using Docling for document understanding.

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
- OpenAI tokenizer for accurate token counting
- Preserves heading hierarchy
- Merges undersized peer chunks

### 4. Hierarchy Manager (`processing/hierarchy_manager.py`)

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

Generates vector embeddings using OpenAI's API via LiteLLM.

**Features:**
- Model: `text-embedding-3-small` (1536 dimensions)
- Batch processing (100 texts per API call)
- Async operation for performance
- Cost estimation (~$0.001 per email)

### 6. Reranker (`processing/reranker.py`)

Implements two-stage retrieval for improved accuracy.

**How it works:**
1. **Stage 1 (Bi-encoder):** Fast vector search retrieves 20 candidates
2. **Stage 2 (Cross-encoder):** Reranker scores query+document pairs

**Improvement:** 20-35% accuracy gain over vector search alone

**Supported Providers (via LiteLLM):**
- Cohere: `cohere/rerank-english-v3.0`
- Azure AI: `azure_ai/cohere-rerank-v3.5`
- AWS Bedrock: `bedrock/rerank`
- Infinity (self-hosted): `infinity/<model>`

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
| Embeddings | OpenAI text-embedding-3-small | Vector generation |
| LLM | GPT-4o-mini via LiteLLM | Answer generation |
| Reranking | Cohere via LiteLLM | Result refinement |
| Database | Supabase (PostgreSQL) | Data persistence |
| Vector Search | pgvector | Similarity search |
| Async | asyncio + asyncpg | Non-blocking I/O |

## Security Considerations

1. **ZIP Extraction:** Path traversal prevention, hidden file filtering
2. **API Keys:** Environment variables, never in code
3. **Database:** Service role key for backend, RLS for frontend
4. **File Paths:** Sanitization of filenames and paths
5. **Input Validation:** Pydantic models for all data structures
