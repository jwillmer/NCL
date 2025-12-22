-- NCL RAG Pipeline - File Registry and Unsupported Files Tracking
-- Run this in Supabase SQL Editor after 001_initial_schema.sql

-- ============================================
-- FILE REGISTRY TABLE
-- Quick lookup for processed files to avoid reprocessing
-- Supports subdirectories and growing data folders
-- ============================================
CREATE TABLE file_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- File identification
    file_path TEXT NOT NULL,              -- Relative path from source root
    file_hash TEXT NOT NULL,              -- SHA-256 hash for content deduplication
    file_name TEXT NOT NULL,              -- Original filename
    file_size_bytes BIGINT NOT NULL,      -- File size for quick comparison

    -- Source tracking
    source_type TEXT NOT NULL,            -- 'eml', 'attachment', 'zip_extracted'
    parent_file_id UUID REFERENCES file_registry(id),  -- For attachments/extracted files
    root_eml_id UUID REFERENCES file_registry(id),     -- Always points to source EML

    -- Content type
    mime_type TEXT,
    is_supported BOOLEAN NOT NULL DEFAULT true,

    -- Processing status
    status processing_status NOT NULL DEFAULT 'pending',
    document_id UUID REFERENCES documents(id),  -- Link to processed document
    error_message TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint on path + hash (same file, same content = same record)
    UNIQUE(file_path, file_hash)
);

-- Indexes for file registry
CREATE INDEX idx_file_registry_hash ON file_registry(file_hash);
CREATE INDEX idx_file_registry_status ON file_registry(status);
CREATE INDEX idx_file_registry_source_type ON file_registry(source_type);
CREATE INDEX idx_file_registry_parent ON file_registry(parent_file_id);
CREATE INDEX idx_file_registry_root_eml ON file_registry(root_eml_id);
CREATE INDEX idx_file_registry_is_supported ON file_registry(is_supported);
CREATE INDEX idx_file_registry_path ON file_registry(file_path);

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
-- HELPER FUNCTION: Check if file needs processing
-- ============================================
CREATE OR REPLACE FUNCTION file_needs_processing(
    p_file_path TEXT,
    p_file_hash TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    existing_status processing_status;
BEGIN
    SELECT status INTO existing_status
    FROM file_registry
    WHERE file_path = p_file_path AND file_hash = p_file_hash
    LIMIT 1;

    -- File needs processing if:
    -- 1. Not in registry at all
    -- 2. Status is 'pending' or 'failed' (retry)
    IF existing_status IS NULL THEN
        RETURN true;
    ELSIF existing_status IN ('pending', 'failed') THEN
        RETURN true;
    ELSE
        RETURN false;
    END IF;
END;
$$;

-- ============================================
-- HELPER FUNCTION: Get processing statistics
-- ============================================
CREATE OR REPLACE FUNCTION get_file_registry_stats()
RETURNS TABLE (
    total_files BIGINT,
    pending_files BIGINT,
    processing_files BIGINT,
    completed_files BIGINT,
    failed_files BIGINT,
    unsupported_files BIGINT,
    eml_files BIGINT,
    attachment_files BIGINT,
    zip_extracted_files BIGINT
)
LANGUAGE SQL
AS $$
    SELECT
        COUNT(*) AS total_files,
        COUNT(*) FILTER (WHERE status = 'pending') AS pending_files,
        COUNT(*) FILTER (WHERE status = 'processing') AS processing_files,
        COUNT(*) FILTER (WHERE status = 'completed') AS completed_files,
        COUNT(*) FILTER (WHERE status = 'failed') AS failed_files,
        COUNT(*) FILTER (WHERE is_supported = false) AS unsupported_files,
        COUNT(*) FILTER (WHERE source_type = 'eml') AS eml_files,
        COUNT(*) FILTER (WHERE source_type = 'attachment') AS attachment_files,
        COUNT(*) FILTER (WHERE source_type = 'zip_extracted') AS zip_extracted_files
    FROM file_registry;
$$;

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
ALTER TABLE file_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE unsupported_files ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access to file_registry" ON file_registry
    FOR ALL USING (true);
CREATE POLICY "Service role full access to unsupported_files" ON unsupported_files
    FOR ALL USING (true);

-- ============================================
-- TRIGGER FOR updated_at
-- ============================================
CREATE TRIGGER update_file_registry_updated_at
    BEFORE UPDATE ON file_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
