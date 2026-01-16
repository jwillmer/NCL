-- MTSS RAG Pipeline - Complete Database Schema
-- Run this in Supabase SQL Editor
-- This script drops all existing objects and recreates them (development mode)

-- ============================================
-- CLEANUP EXISTING SCHEMA
-- ============================================

-- Drop tables first (CASCADE handles triggers and dependent objects)
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS vessels CASCADE;
DROP TABLE IF EXISTS ingest_versions CASCADE;
DROP TABLE IF EXISTS unsupported_files CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS processing_log CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Drop functions (use CASCADE to handle dependent objects)
DROP FUNCTION IF EXISTS match_chunks CASCADE;
DROP FUNCTION IF EXISTS get_document_ancestry CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;

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

    -- STABLE IDENTIFICATION (for citations and versioning)
    source_id TEXT NOT NULL,              -- Normalized path: relative path lowercased
    doc_id TEXT NOT NULL UNIQUE,          -- Content-addressable: hash(source_id + file_hash)

    -- VERSIONING (for re-ingestion support)
    content_version INTEGER NOT NULL DEFAULT 1,   -- Increments when content changes
    ingest_version INTEGER NOT NULL DEFAULT 1,    -- Schema/logic version used during ingest

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

    -- SOURCE METADATA FOR CITATIONS
    source_title TEXT,                    -- Human-readable title (email subject, filename, etc.)

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

    -- ARCHIVE LINKS (for browsable content)
    archive_path TEXT,                    -- Relative path to archive folder
    archive_browse_uri TEXT,              -- URI to browsable .md file
    archive_download_uri TEXT,            -- URI to original file for download

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
CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_ingest_version ON documents(ingest_version);

-- ============================================
-- CHUNKS TABLE (with embeddings)
-- ============================================
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- STABLE IDENTIFICATION (for citations)
    chunk_id TEXT NOT NULL,               -- Deterministic: hash(doc_id + char_start + char_end)

    -- Relationship to document
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,  -- Order within document

    -- CONTEXTUAL CHUNKING (for improved retrieval)
    context_summary TEXT,                 -- Document-level context (LLM-generated)
    embedding_text TEXT,                  -- Full text used for embedding: context + content

    -- Hierarchy context (denormalized for query performance)
    section_path TEXT[],                  -- e.g., ['Chapter 1', 'Section 1.1'] (renamed from heading_path)
    section_title TEXT,

    -- CITATION METADATA (denormalized from document for fast retrieval)
    source_title TEXT,
    source_id TEXT,

    -- Source location in original document
    page_number INTEGER,
    line_from INTEGER,                    -- Line number start
    line_to INTEGER,                      -- Line number end
    char_start INTEGER,                   -- Character offset start
    char_end INTEGER,                     -- Character offset end

    -- ARCHIVE LINKS (denormalized for fast retrieval)
    archive_browse_uri TEXT,
    archive_download_uri TEXT,

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
CREATE UNIQUE INDEX idx_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX idx_chunks_source_id ON chunks(source_id);

-- HNSW index for vector similarity search (cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for fast JSONB containment queries on chunk metadata (vessel filtering)
CREATE INDEX idx_chunks_metadata ON chunks USING GIN (metadata);

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
-- INGEST VERSIONS TABLE (for re-processing)
-- ============================================
CREATE TABLE ingest_versions (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    breaking_changes BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed with initial version
INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (1, 'Initial schema with contextual chunking, citations, and browsable archive', FALSE);

-- ============================================
-- VESSELS TABLE (Vessel Register)
-- ============================================
CREATE TABLE vessels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    imo TEXT UNIQUE,                             -- 7-digit IMO number
    vessel_type TEXT,                            -- VLCC, Suezmax, Aframax, etc.
    dwt INTEGER,                                 -- Deadweight tonnage
    aliases TEXT[] NOT NULL DEFAULT '{}',        -- Alternative names for matching
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for vessels
CREATE INDEX idx_vessels_name ON vessels(name);
CREATE INDEX idx_vessels_imo ON vessels(imo);

-- ============================================
-- CONVERSATIONS TABLE
-- Stores conversation metadata only - messages are persisted
-- by LangGraph's AsyncPostgresSaver checkpointer
-- ============================================
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID UNIQUE NOT NULL,              -- Links to LangGraph checkpoints
    user_id UUID NOT NULL,                       -- Supabase auth user (no FK for flexibility)
    title TEXT,                                  -- Auto-generated or user-edited
    vessel_id UUID REFERENCES vessels(id),       -- Vessel filter (NULL = all vessels)
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ                  -- Updated on each message for sorting
);

-- Fast lookup by user
CREATE INDEX idx_conversations_user_id ON conversations(user_id);

-- Sort by recent activity (most common query)
CREATE INDEX idx_conversations_user_recent ON conversations(user_id, last_message_at DESC NULLS LAST);

-- Filter by archived status
CREATE INDEX idx_conversations_user_archived ON conversations(user_id, is_archived);

-- Filter by vessel
CREATE INDEX idx_conversations_vessel ON conversations(vessel_id);

-- Full-text search on title
CREATE INDEX idx_conversations_search ON conversations
    USING gin(to_tsvector('english', COALESCE(title, '')));

-- ============================================
-- VECTOR SEARCH FUNCTION
-- ============================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT NULL
)
RETURNS TABLE (
    -- Chunk identification
    chunk_id TEXT,
    document_uuid UUID,
    content TEXT,
    section_path TEXT[],
    similarity FLOAT,

    -- Document info
    doc_id TEXT,
    source_id TEXT,
    file_path TEXT,
    document_type document_type,

    -- Citation metadata
    source_title TEXT,
    page_number INTEGER,
    line_from INTEGER,
    line_to INTEGER,

    -- Archive links
    archive_browse_uri TEXT,
    archive_download_uri TEXT,

    -- Email metadata (from root document)
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
        c.chunk_id,
        c.document_id AS document_uuid,
        c.content,
        c.section_path,
        1 - (c.embedding <=> query_embedding) AS similarity,

        d.doc_id,
        c.source_id,
        d.file_path,
        d.document_type,

        c.source_title,
        c.page_number,
        c.line_from,
        c.line_to,

        c.archive_browse_uri,
        c.archive_download_uri,

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
      AND (metadata_filter IS NULL OR c.metadata @> metadata_filter)
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- ============================================
-- GET DOCUMENT ANCESTRY FUNCTION
-- ============================================
CREATE OR REPLACE FUNCTION get_document_ancestry(document_uuid UUID)
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
        FROM documents d WHERE d.id = document_uuid

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
ALTER TABLE ingest_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE vessels ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Service role policies for backend operations (restricted to service_role)
CREATE POLICY "Service role full access to documents" ON documents
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to chunks" ON chunks
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to processing_log" ON processing_log
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to unsupported_files" ON unsupported_files
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to ingest_versions" ON ingest_versions
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to vessels" ON vessels
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to conversations" ON conversations
    FOR ALL TO service_role USING (true);

-- Authenticated users can read chunks, documents, and vessels (for search/retrieval)
CREATE POLICY "Authenticated users can read documents" ON documents
    FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read chunks" ON chunks
    FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read vessels" ON vessels
    FOR SELECT TO authenticated USING (true);

-- Conversations: users can only access their own
CREATE POLICY "Users can view own conversations" ON conversations
    FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own conversations" ON conversations
    FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own conversations" ON conversations
    FOR UPDATE TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own conversations" ON conversations
    FOR DELETE TO authenticated USING (auth.uid() = user_id);

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

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vessels_updated_at
    BEFORE UPDATE ON vessels
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- NOTES
-- ============================================
-- LangGraph's AsyncPostgresSaver creates its own tables automatically:
--   - checkpoints
--   - checkpoint_writes
--   - checkpoint_blobs
-- These store the actual message history per thread_id.
