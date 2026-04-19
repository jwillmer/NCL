-- Migration 001: Add embedding_mode column to documents and chunks.
--
-- Introduces per-document / per-chunk mode tracking for the embedding
-- pipeline. Values: 'full', 'summary', 'metadata_only'.
--
-- Supabase is not live at the time of this change; this file ships so it can
-- be applied manually when Supabase is reintroduced. It is additive and safe
-- to run against an existing schema.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS embedding_mode TEXT;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS embedding_mode TEXT;

-- Optional: constrain valid values once data is backfilled.
-- ALTER TABLE documents
--     ADD CONSTRAINT documents_embedding_mode_chk
--     CHECK (embedding_mode IN ('full', 'summary', 'metadata_only'));
-- ALTER TABLE chunks
--     ADD CONSTRAINT chunks_embedding_mode_chk
--     CHECK (embedding_mode IN ('full', 'summary', 'metadata_only'));
