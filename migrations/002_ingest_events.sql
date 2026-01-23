-- MTSS RAG Pipeline - Migration 002: Rename unsupported_files to ingest_events
-- This migration extends the unsupported_files table to track all ingest events
-- Run this in Supabase SQL Editor AFTER 001_initial_schema.sql

-- ============================================
-- RENAME AND EXTEND TABLE
-- ============================================

-- Rename the table
ALTER TABLE IF EXISTS unsupported_files RENAME TO ingest_events;

-- Add new columns for event classification
ALTER TABLE ingest_events
  ADD COLUMN IF NOT EXISTS severity TEXT DEFAULT 'warning',
  ADD COLUMN IF NOT EXISTS event_type TEXT DEFAULT 'unsupported_file';

-- ============================================
-- ADD CONSTRAINTS
-- ============================================

-- Add check constraint for severity values
ALTER TABLE ingest_events
  ADD CONSTRAINT chk_ingest_events_severity
  CHECK (severity IN ('error', 'warning', 'info'));

-- Add check constraint for event_type values
ALTER TABLE ingest_events
  ADD CONSTRAINT chk_ingest_events_type
  CHECK (event_type IN (
    'unsupported_file',
    'encoding_fallback',
    'parse_failure',
    'archive_failure',
    'context_failure',
    'empty_content',
    'zip_extraction_failure'
  ));

-- ============================================
-- INDEXES
-- ============================================

-- Index for querying events by document
CREATE INDEX IF NOT EXISTS idx_ingest_events_document
  ON ingest_events(parent_document_id);

-- Index for querying events by type
CREATE INDEX IF NOT EXISTS idx_ingest_events_type
  ON ingest_events(event_type);

-- Index for querying events by severity
CREATE INDEX IF NOT EXISTS idx_ingest_events_severity
  ON ingest_events(severity);

-- Index for sorting events by discovery time (common query pattern)
CREATE INDEX IF NOT EXISTS idx_ingest_events_discovered_at
  ON ingest_events(discovered_at DESC);

-- ============================================
-- UPDATE RLS POLICIES
-- ============================================

-- Drop old policy (if exists) and create new one
DROP POLICY IF EXISTS "Service role full access to unsupported_files" ON ingest_events;
CREATE POLICY "Service role full access to ingest_events" ON ingest_events
    FOR ALL TO service_role USING (true);

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE ingest_events IS 'Tracks processing events during ingest for visibility and debugging';
COMMENT ON COLUMN ingest_events.severity IS 'Event severity: error, warning, or info';
COMMENT ON COLUMN ingest_events.event_type IS 'Type of event: unsupported_file, encoding_fallback, parse_failure, etc.';

-- ============================================
-- UPDATE ingest_versions
-- ============================================

INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (2, 'Added ingest_events table (renamed from unsupported_files) with severity and event_type columns', FALSE)
ON CONFLICT (version) DO NOTHING;
