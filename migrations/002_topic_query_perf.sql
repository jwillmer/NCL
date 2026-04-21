-- Migration 002: topic query performance
--
-- Two RPC fixes for the query-time topic filter path:
--
--   1. find_similar_topics bypassed the HNSW index on topics.embedding.
--      Per pgvector, the index only accelerates ORDER BY embedding <=> q
--      LIMIT k. The previous function put a distance predicate in the
--      WHERE clause, which forced a sequential scan over every topics row
--      (fetching 1536-dim vectors from heap for each). Rewritten so the
--      ORDER BY ... LIMIT stage hits the HNSW index first and the
--      threshold filter runs over the already-ranked top-k.
--
--   2. count_chunks_by_topic used `metadata->'topic_ids' ? topic_uuid::text`
--      — this is a function expression that cannot use the GIN index on
--      chunks.metadata, so it sequentially scanned the entire chunks
--      table. Rewritten as `metadata @> jsonb_build_object(...)` which the
--      GIN `jsonb_ops` index can serve directly. Same for
--      count_chunks_by_topics (OR'd containment on each topic).
--
-- At prod scale (~20M chunks, ~3k topics) the old forms took multiple
-- seconds per call; the new forms are index-backed and run in low ms.

-- ============================================
-- TOPICS: similarity search
-- ============================================

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
    SELECT x.id, x.name, x.display_name, x.similarity
    FROM (
        SELECT
            t.id,
            t.name,
            t.display_name,
            1 - (t.embedding <=> query_embedding) AS similarity
        FROM topics t
        WHERE t.embedding IS NOT NULL
        -- HNSW index on topics(embedding vector_cosine_ops) activates when
        -- the expression here matches the opclass and a LIMIT is present.
        ORDER BY t.embedding <=> query_embedding
        LIMIT GREATEST(max_results * 4, 20)
    ) x
    WHERE x.similarity > similarity_threshold
    ORDER BY x.similarity DESC
    LIMIT max_results;
$$;

COMMENT ON FUNCTION find_similar_topics IS
    'Top-k cosine-similar topics. HNSW-indexed. The inner LIMIT over-fetches (max_results * 4) so the threshold filter in the outer SELECT does not starve the result below max_results.';

-- ============================================
-- TOPICS: chunk counts
-- ============================================

CREATE OR REPLACE FUNCTION count_chunks_by_topic(
    topic_uuid UUID,
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    SELECT COUNT(*)::INTEGER
    FROM chunks c
    -- Containment (@>) can use the GIN index on chunks.metadata
    -- (jsonb_ops opclass). The previous `->'topic_ids' ? uuid` form was a
    -- function expression on a sub-path and fell back to a sequential
    -- scan at production scale.
    WHERE c.metadata @> jsonb_build_object(
              'topic_ids', jsonb_build_array(topic_uuid::text)
          )
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

CREATE OR REPLACE FUNCTION count_chunks_by_topics(
    topic_uuids UUID[],
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    -- Union of index-backed containment scans, one per topic_id. DISTINCT
    -- c.id dedupes chunks that carry multiple matching topics.
    SELECT COUNT(DISTINCT c.id)::INTEGER
    FROM chunks c,
         unnest(topic_uuids) AS tid
    WHERE c.metadata @> jsonb_build_object(
              'topic_ids', jsonb_build_array(tid::text)
          )
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

COMMENT ON FUNCTION count_chunks_by_topic IS
    'Count chunks tagged with a topic. Uses chunks.metadata @> jsonb_build_object(...) so the GIN index on metadata serves the query.';
COMMENT ON FUNCTION count_chunks_by_topics IS
    'OR-count of chunks for any of N topics. Same index-friendly containment form as count_chunks_by_topic.';
