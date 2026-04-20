-- Migration 001: topic rewrite RPC
--
-- Rationale: Supabase stores chunk↔topic links inside chunks.metadata.topic_ids
-- (JSONB array of UUIDs) rather than in a junction table. When a local topic
-- consolidation runs (mtss topics consolidate --apply), absorbed topic rows
-- are deleted from the local `topics` table but any already-imported remote
-- chunks still carry the absorbed topic UUIDs inside their JSONB metadata —
-- orphan references that break topic-based filters.
--
-- This RPC takes a {old_uuid: new_uuid} mapping and rewrites every matching
-- topic_ids entry in a single server-side UPDATE. `mtss import` calls it
-- before pruning the stale remote topic rows so no chunk ends up pointing at
-- a deleted topic.

CREATE OR REPLACE FUNCTION rewrite_chunk_topic_ids(mapping JSONB)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    IF mapping IS NULL OR jsonb_typeof(mapping) <> 'object' THEN
        RAISE EXCEPTION 'mapping must be a JSON object of {old_uuid: new_uuid}';
    END IF;

    UPDATE chunks c
    SET metadata = jsonb_set(
        c.metadata,
        '{topic_ids}',
        (
            -- Deduplicate via DISTINCT — a keeper UUID the chunk already
            -- referenced would otherwise appear twice after rewrite.
            SELECT COALESCE(jsonb_agg(DISTINCT rewritten), '[]'::jsonb)
            FROM (
                SELECT COALESCE(mapping->>tid, tid) AS rewritten
                FROM jsonb_array_elements_text(c.metadata->'topic_ids') AS tid
            ) t
        )
    )
    WHERE c.metadata ? 'topic_ids'
      AND EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(c.metadata->'topic_ids') AS tid
          WHERE mapping ? tid
      );

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$;

COMMENT ON FUNCTION rewrite_chunk_topic_ids IS
    'Rewrite chunks.metadata.topic_ids using {old_uuid:new_uuid} map. Called by mtss import after local topic consolidation.';
