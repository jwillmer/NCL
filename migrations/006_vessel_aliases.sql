-- Migration 006: Re-add vessel aliases for improved matching
-- The aliases column was in the original schema (001) but dropped in migration 003.
-- Re-adding it to support alternative vessel name matching (e.g., "M/V MARAN CANOPUS").

ALTER TABLE vessels ADD COLUMN IF NOT EXISTS aliases TEXT[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_vessels_aliases ON vessels USING GIN(aliases);
