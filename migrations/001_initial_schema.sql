-- NCL RAG Pipeline - Complete Database Schema
-- Run this in Supabase SQL Editor
-- This script drops all existing objects and recreates them (development mode)

-- ============================================
-- CLEANUP EXISTING SCHEMA
-- ============================================

-- Drop tables first (CASCADE handles triggers and dependent objects)
DROP TABLE IF EXISTS unsupported_files CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS processing_log CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS match_chunks(extensions.vector(1536), FLOAT, INT);
DROP FUNCTION IF EXISTS get_document_ancestry(UUID);
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop custom types
DROP TYPE IF EXISTS document_type CASCADE;
DROP TYPE IF EXISTS processing_status CASCADE;

-- ============================================
-- EXTENSIONS
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ============================================
-- CUSTOM TYPES
-- ============================================

-- Document types enum
CREATE TYPE document_type AS ENUM (
    'email',
    'attachment_pdf',
    'attachment_image',
    'attachment_docx',
    'attachment_pptx',
    'attachment_xlsx',
    -- Legacy formats (via LlamaParse)
    'attachment_doc',
    'attachment_xls',
    'attachment_ppt',
    'attachment_csv',
    'attachment_rtf',
    'attachment_other'
);

-- Processing status enum
CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'skipped'
);

-- ============================================
-- DOCUMENTS TABLE (Hierarchy)
-- ============================================
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Hierarchy relationships
    parent_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    root_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    depth INTEGER NOT NULL DEFAULT 0,
    path TEXT[] NOT NULL DEFAULT '{}',  -- Array of ancestor IDs for fast ancestry queries

    -- Document identification
    document_type document_type NOT NULL,
    file_path TEXT NOT NULL,  -- Local file path for source linking
    file_name TEXT NOT NULL,
    file_hash TEXT,  -- SHA-256 hash for deduplication

    -- Email conversation metadata (nullable for attachments)
    email_subject TEXT,
    email_participants TEXT[],  -- All unique email addresses in conversation
    email_initiator TEXT,       -- First sender in thread
    email_date_start TIMESTAMPTZ,  -- Earliest message date
    email_date_end TIMESTAMPTZ,    -- Latest message date
    email_message_count INTEGER DEFAULT 1,

    -- Attachment-specific metadata
    attachment_content_type TEXT,
    attachment_size_bytes BIGINT,

    -- Processing metadata
    status processing_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    processed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for documents
CREATE INDEX idx_documents_parent_id ON documents(parent_id);
CREATE INDEX idx_documents_root_id ON documents(root_id);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_document_type ON documents(document_type);
CREATE INDEX idx_documents_path ON documents USING GIN(path);

-- ============================================
-- CHUNKS TABLE (with embeddings)
-- ============================================
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Relationship to document
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,  -- Order within document

    -- Hierarchy context (denormalized for query performance)
    heading_path TEXT[],  -- e.g., ['Chapter 1', 'Section 1.1']
    section_title TEXT,

    -- Source location in original document
    page_number INTEGER,
    start_char INTEGER,
    end_char INTEGER,

    -- Embedding vector (OpenAI text-embedding-3-small = 1536 dimensions)
    embedding extensions.vector(1536),

    -- Metadata for filtering
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for chunks
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks(document_id, chunk_index);

-- HNSW index for vector similarity search (cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- PROCESSING LOG TABLE (for resumable processing)
-- ============================================
CREATE TABLE processing_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,

    status processing_status NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_processing_log_status ON processing_log(status);
CREATE INDEX idx_processing_log_file_hash ON processing_log(file_hash);

-- ============================================
-- UNSUPPORTED FILES TABLE
-- Track files we cannot process for visibility
-- ============================================
CREATE TABLE unsupported_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- File identification
    file_path TEXT NOT NULL,              -- Path where file was found
    file_name TEXT NOT NULL,              -- Original filename
    file_hash TEXT,                       -- SHA-256 hash (may be null for very large files)
    file_size_bytes BIGINT,

    -- Source context
    source_eml_path TEXT,                 -- Which EML this came from
    source_zip_path TEXT,                 -- If extracted from ZIP
    parent_document_id UUID REFERENCES documents(id),

    -- File info
    mime_type TEXT,
    file_extension TEXT,
    reason TEXT NOT NULL,                 -- Why it's unsupported ('unsupported_format', 'too_large', 'corrupted', etc.)

    -- Timestamps
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique on path to avoid duplicates
    UNIQUE(file_path)
);

-- Indexes for unsupported files
CREATE INDEX idx_unsupported_files_source_eml ON unsupported_files(source_eml_path);
CREATE INDEX idx_unsupported_files_mime_type ON unsupported_files(mime_type);
CREATE INDEX idx_unsupported_files_reason ON unsupported_files(reason);

-- ============================================
-- VECTOR SEARCH FUNCTION
-- ============================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    heading_path TEXT[],
    similarity FLOAT,
    file_path TEXT,
    document_type document_type,
    email_subject TEXT,
    email_initiator TEXT,
    email_participants TEXT[],
    email_date TIMESTAMPTZ,
    root_file_path TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        c.heading_path,
        1 - (c.embedding <=> query_embedding) AS similarity,
        d.file_path,
        d.document_type,
        COALESCE(d.email_subject, root_doc.email_subject) AS email_subject,
        COALESCE(d.email_initiator, root_doc.email_initiator) AS email_initiator,
        COALESCE(d.email_participants, root_doc.email_participants) AS email_participants,
        COALESCE(d.email_date_start, root_doc.email_date_start) AS email_date,
        root_doc.file_path AS root_file_path
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    LEFT JOIN documents root_doc ON d.root_id = root_doc.id
    WHERE c.embedding IS NOT NULL
      AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- ============================================
-- GET DOCUMENT ANCESTRY FUNCTION
-- ============================================
CREATE OR REPLACE FUNCTION get_document_ancestry(doc_id UUID)
RETURNS TABLE (
    id UUID,
    document_type document_type,
    file_path TEXT,
    file_name TEXT,
    depth INTEGER,
    email_subject TEXT
)
LANGUAGE SQL
AS $$
    WITH RECURSIVE ancestry AS (
        SELECT d.id, d.document_type, d.file_path, d.file_name, d.depth,
               d.email_subject, d.parent_id
        FROM documents d WHERE d.id = doc_id

        UNION ALL

        SELECT d.id, d.document_type, d.file_path, d.file_name, d.depth,
               d.email_subject, d.parent_id
        FROM documents d
        JOIN ancestry a ON d.id = a.parent_id
    )
    SELECT a.id, a.document_type, a.file_path, a.file_name, a.depth, a.email_subject
    FROM ancestry a
    ORDER BY a.depth ASC;
$$;

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE unsupported_files ENABLE ROW LEVEL SECURITY;

-- Default policy: allow all for service role (adjust for multi-tenant if needed)
CREATE POLICY "Service role full access to documents" ON documents
    FOR ALL USING (true);
CREATE POLICY "Service role full access to chunks" ON chunks
    FOR ALL USING (true);
CREATE POLICY "Service role full access to processing_log" ON processing_log
    FOR ALL USING (true);
CREATE POLICY "Service role full access to unsupported_files" ON unsupported_files
    FOR ALL USING (true);

-- ============================================
-- TRIGGER FOR updated_at
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_processing_log_updated_at
    BEFORE UPDATE ON processing_log
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
