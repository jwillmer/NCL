-- Migration 003: Add vessel_class, remove unused columns, add type/class filters to conversations
-- This migration extends vessel tagging with type and class for RAG filtering.

-- ============================================
-- VESSELS TABLE - Add vessel_class, remove unused columns
-- ============================================

-- Add vessel_class column
ALTER TABLE vessels ADD COLUMN IF NOT EXISTS vessel_class TEXT;

-- Drop unused columns (imo, dwt, aliases)
ALTER TABLE vessels DROP COLUMN IF EXISTS imo;
ALTER TABLE vessels DROP COLUMN IF EXISTS dwt;
ALTER TABLE vessels DROP COLUMN IF EXISTS aliases;

-- Drop unused index for imo
DROP INDEX IF EXISTS idx_vessels_imo;

-- Add new indexes for type and class filtering
CREATE INDEX IF NOT EXISTS idx_vessels_vessel_type ON vessels(vessel_type);
CREATE INDEX IF NOT EXISTS idx_vessels_vessel_class ON vessels(vessel_class);

-- Clear vessels table and add unique constraint on name (required for upsert)
TRUNCATE TABLE vessels CASCADE;
ALTER TABLE vessels DROP CONSTRAINT IF EXISTS vessels_name_unique;
ALTER TABLE vessels ADD CONSTRAINT vessels_name_unique UNIQUE (name);

-- ============================================
-- CONVERSATIONS TABLE - Add type/class filter columns
-- ============================================

-- Add filter columns (mutually exclusive with vessel_id)
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS vessel_type TEXT;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS vessel_class TEXT;

-- Drop existing conversations (no backward compatibility needed)
TRUNCATE TABLE conversations CASCADE;

-- Also clear LangGraph checkpoint tables (they reference conversations by thread_id)
-- Use DO block since TRUNCATE doesn't support IF EXISTS
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoints') THEN
        TRUNCATE TABLE checkpoints CASCADE;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoint_writes') THEN
        TRUNCATE TABLE checkpoint_writes CASCADE;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoint_blobs') THEN
        TRUNCATE TABLE checkpoint_blobs CASCADE;
    END IF;
END $$;

-- Add comments documenting mutual exclusivity
COMMENT ON COLUMN conversations.vessel_id IS 'Vessel UUID filter (mutually exclusive with vessel_type/vessel_class)';
COMMENT ON COLUMN conversations.vessel_type IS 'Vessel type filter e.g. VLCC (mutually exclusive with vessel_id/vessel_class)';
COMMENT ON COLUMN conversations.vessel_class IS 'Vessel class filter e.g. Canopus Class (mutually exclusive with vessel_id/vessel_type)';

-- Add indexes for type/class filtering
CREATE INDEX IF NOT EXISTS idx_conversations_vessel_type ON conversations(vessel_type);
CREATE INDEX IF NOT EXISTS idx_conversations_vessel_class ON conversations(vessel_class);

-- ============================================
-- INGEST VERSIONS - Record this migration
-- ============================================

INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (3, 'Added vessel_class, removed imo/dwt/aliases, added type/class filters to conversations', FALSE)
ON CONFLICT (version) DO NOTHING;
