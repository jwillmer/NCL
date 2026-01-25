-- Migration 004: Add topics table for semantic categorization and pre-filtering
-- Topics are extracted from emails during ingest and matched from user queries.

-- ============================================
-- TOPICS TABLE - Main topic storage with embeddings
-- ============================================

CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,           -- Canonical lowercase
    display_name TEXT NOT NULL,          -- User-friendly
    description TEXT,
    embedding extensions.vector(1536),   -- For semantic similarity
    chunk_count INTEGER NOT NULL DEFAULT 0,
    document_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);
CREATE INDEX IF NOT EXISTS idx_topics_embedding ON topics
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE topics ENABLE ROW LEVEL SECURITY;

-- Service role gets full access
CREATE POLICY "Service role full access to topics" ON topics
    FOR ALL TO service_role USING (true);

-- Authenticated users can read topics
CREATE POLICY "Authenticated users can read topics" ON topics
    FOR SELECT TO authenticated USING (true);

-- ============================================
-- TRIGGERS
-- ============================================

-- Trigger for updated_at (uses existing function from 001_initial_schema.sql)
CREATE TRIGGER update_topics_updated_at
    BEFORE UPDATE ON topics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- SQL FUNCTIONS
-- ============================================

-- Function: find similar topics by embedding
CREATE OR REPLACE FUNCTION find_similar_topics(
    query_embedding extensions.vector(1536),
    similarity_threshold FLOAT DEFAULT 0.85,
    max_results INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    display_name TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        t.id,
        t.name,
        t.display_name,
        1 - (t.embedding <=> query_embedding) AS similarity
    FROM topics t
    WHERE t.embedding IS NOT NULL
      AND 1 - (t.embedding <=> query_embedding) > similarity_threshold
    ORDER BY t.embedding <=> query_embedding ASC
    LIMIT max_results;
$$;

-- Function: count chunks by single topic (with optional vessel filter)
CREATE OR REPLACE FUNCTION count_chunks_by_topic(
    topic_uuid UUID,
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    SELECT COUNT(*)::INTEGER
    FROM chunks c
    WHERE c.metadata->'topic_ids' ? topic_uuid::text
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

-- Function: count chunks by multiple topics (OR logic - match any)
CREATE OR REPLACE FUNCTION count_chunks_by_topics(
    topic_uuids UUID[],
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    SELECT COUNT(DISTINCT c.id)::INTEGER
    FROM chunks c
    WHERE c.metadata->'topic_ids' ?| ARRAY(SELECT t::text FROM unnest(topic_uuids) t)
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

-- Function: get all topics with counts
CREATE OR REPLACE FUNCTION get_topics_with_counts()
RETURNS TABLE (
    id UUID,
    name TEXT,
    display_name TEXT,
    description TEXT,
    chunk_count INTEGER,
    document_count INTEGER
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        t.id, t.name, t.display_name, t.description,
        t.chunk_count, t.document_count
    FROM topics t
    ORDER BY t.chunk_count DESC, t.name ASC;
$$;

-- ============================================
-- INGEST VERSIONS - Record this migration
-- ============================================

INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (4, 'Added topics table with embeddings for semantic categorization and pre-filtering', FALSE)
ON CONFLICT (version) DO NOTHING;
