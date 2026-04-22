-- ============================================================================
-- Production migration: RAG performance fixes
-- Date:      2026-04-22
-- Combines:  migrations 001, 002, 004
--            (003 is superseded by 004 — skipped)
-- ----------------------------------------------------------------------------
-- Effect on the test corpus (37-question eval harness, ef_search=100):
--     total agent latency          3198s  ->   466s   (-85%)
--     avg latency per question     86s    ->   12.6s  (6.8x faster)
--     overall_mean                 0.351  ->   0.532  (+52%)
--     citations_valid_pct          0.513  ->   0.784  (+53%)
--
-- The four underlying SQL changes are bundled here so prod can roll forward
-- in a single apply. All objects are CREATE OR REPLACE (or DROP+CREATE where
-- the function signature drifted), so running this migration a second time
-- is a no-op. Safe to re-run.
-- ----------------------------------------------------------------------------
-- Pre-flight checks to run BEFORE applying to prod:
--   1. Confirm you are connected to the prod Supabase project
--      (`SELECT current_database(), inet_server_addr();`).
--   2. Back up the four affected functions:
--         match_chunks, find_similar_topics, count_chunks_by_topic,
--         count_chunks_by_topics
--      Save their current definitions via `pg_get_functiondef(oid)` so you
--      can re-create them if you need to roll back.
--   3. Confirm extensions are present:
--         SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm');
--   4. Confirm the HNSW index exists on chunks.embedding:
--         SELECT indexname, indexdef FROM pg_indexes
--          WHERE tablename = 'chunks' AND indexdef ILIKE '%hnsw%';
--      (migration 004's win depends on this — without HNSW the rewrite is
--       neutral, not harmful.)
--   5. Confirm the GIN index on chunks.metadata exists with jsonb_ops:
--         SELECT indexname, indexdef FROM pg_indexes
--          WHERE tablename = 'chunks' AND indexdef ILIKE '%gin%metadata%';
--      (Migrations 002 + 004 rely on this to serve metadata @> containment
--       without a sequential scan.)
--   6. In the app, confirm `search_similar_chunks` sets
--      `SET LOCAL hnsw.ef_search = 100` inside its transaction; 40 (the
--      default) is too low for the 3x over-fetch candidate pool.
-- ----------------------------------------------------------------------------
-- Rollback plan:
--   `DROP FUNCTION IF EXISTS ...` then re-run 000_schema.sql's original
--   definitions of the four affected functions.
--   The function bodies and RETURNS shapes are captured in pre-flight step 2.
-- ============================================================================


-- ============================================================================
-- 001 · rewrite_chunk_topic_ids
-- ----------------------------------------------------------------------------
-- Server-side rewrite of chunks.metadata.topic_ids for local → remote topic
-- consolidation. Takes a {old_uuid: new_uuid} mapping and updates every
-- chunk that carries an absorbed UUID. Called by `mtss import` after the
-- local `mtss topics consolidate --apply` step.
-- ============================================================================

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
    'Rewrite chunks.metadata.topic_ids using {old_uuid:new_uuid} map. '
    'Called by mtss import after local topic consolidation.';


-- ============================================================================
-- 002.a · find_similar_topics
-- ----------------------------------------------------------------------------
-- Uses the HNSW index on topics.embedding. The old form put a distance
-- predicate in the WHERE and forced a sequential scan over every topic row;
-- now ORDER BY ... LIMIT is the first stage so the index activates, and the
-- similarity threshold runs on the already-ranked top-k. At prod scale this
-- drops from multiple seconds to low ms.
-- ============================================================================

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
        -- HNSW activates on ORDER BY embedding <=> query with LIMIT.
        ORDER BY t.embedding <=> query_embedding
        LIMIT GREATEST(max_results * 4, 20)
    ) x
    WHERE x.similarity > similarity_threshold
    ORDER BY x.similarity DESC
    LIMIT max_results;
$$;

COMMENT ON FUNCTION find_similar_topics IS
    'Top-k cosine-similar topics. HNSW-indexed. Inner LIMIT over-fetches '
    '(max_results * 4) so the threshold filter in the outer SELECT does not '
    'starve the result below max_results.';


-- ============================================================================
-- 002.b · count_chunks_by_topic + count_chunks_by_topics
-- ----------------------------------------------------------------------------
-- GIN-indexable containment (@>). The old form
--     chunks.metadata->'topic_ids' ? topic_uuid::text
-- is a function expression on a sub-path that the GIN index on metadata
-- cannot serve, so prod sequentially scanned the full chunks table (~20M
-- rows) on every topic filter call. The containment form
--     chunks.metadata @> jsonb_build_object('topic_ids', jsonb_build_array(...))
-- matches the jsonb_ops opclass and runs in low ms.
-- ============================================================================

CREATE OR REPLACE FUNCTION count_chunks_by_topic(
    topic_uuid UUID,
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    SELECT COUNT(*)::INTEGER
    FROM chunks c
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
    SELECT COUNT(DISTINCT c.id)::INTEGER
    FROM chunks c,
         unnest(topic_uuids) AS tid
    WHERE c.metadata @> jsonb_build_object(
              'topic_ids', jsonb_build_array(tid::text)
          )
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

COMMENT ON FUNCTION count_chunks_by_topic IS
    'Count chunks tagged with a topic via chunks.metadata @> jsonb_build_object(...). '
    'Serves from the GIN jsonb_ops index on metadata.';
COMMENT ON FUNCTION count_chunks_by_topics IS
    'OR-count of chunks for any of N topics. Same index-friendly containment form.';


-- ============================================================================
-- 004 · match_chunks (HNSW-direct + GIN-indexable topic filter)
-- ----------------------------------------------------------------------------
-- Rewrite to let pgvector's HNSW index fire inside the function body. The
-- prior form had
--     WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
--     ORDER BY sub.similarity DESC
-- which the planner could not map to the HNSW access method (derived
-- expression + derived ORDER BY column). It fell back to a sequential scan
-- over all chunks. Direct probe on 2026-04-22:
--     match_chunks hybrid (query_text set)      27-36s   (seq scan)
--     raw "ORDER BY embedding <=> v LIMIT 40"   50-800ms (HNSW used)
-- Post-fix: match_chunks hybrid drops to ~55ms. EXPLAIN Execution Time
-- collapses from 21806ms to 4.3ms.
--
-- Fix:
--   (a) Express the similarity threshold as a distance bound:
--       embedding <=> query_embedding < (1 - match_threshold)
--   (b) ORDER BY the raw operator inside a CTE (HNSW-eligible).
--   (c) Over-fetch match_count * 3 candidates (min 60) via HNSW, then
--       re-rank in the outer SELECT with the hybrid BM25 blend and LIMIT
--       to match_count. ts_rank runs on a bounded ~120-row set.
--
-- Topic filter block is the GIN-indexable unnest/@> form from migration 003.
-- All other behavior (hybrid BM25 weight 0.7/0.3, RETURNS shape, filter
-- handling) is identical to the pre-004 function.
--
-- WHY THE DROP: the live function signature has drifted from 000_schema.sql
-- over time (added columns), so Postgres refuses CREATE OR REPLACE when the
-- RETURNS TABLE shape differs. Dropping first lets this migration's shape
-- win. ``document_type`` is the live schema's custom enum — keep the enum
-- type reference unchanged to avoid "cannot change return type".
-- ============================================================================

DROP FUNCTION IF EXISTS match_chunks(vector, float, int, JSONB, TEXT);
DROP FUNCTION IF EXISTS match_chunks(extensions.vector, float, int, JSONB, TEXT);

CREATE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold float,
    match_count int,
    metadata_filter JSONB DEFAULT NULL,
    query_text TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id TEXT,
    document_uuid UUID,
    content TEXT,
    section_path TEXT[],
    similarity FLOAT,
    context_summary TEXT,
    doc_id TEXT,
    source_id TEXT,
    file_path TEXT,
    document_type document_type,
    source_title TEXT,
    page_number INTEGER,
    line_from INTEGER,
    line_to INTEGER,
    archive_browse_uri TEXT,
    archive_download_uri TEXT,
    email_subject TEXT,
    email_initiator TEXT,
    email_participants TEXT[],
    email_date TIMESTAMPTZ,
    root_file_path TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    topic_ids_filter TEXT[];
    other_filter JSONB;
    tsquery_val tsquery;
    distance_limit FLOAT;
    candidate_count INT;
BEGIN
    IF metadata_filter IS NOT NULL AND metadata_filter ? 'topic_ids' THEN
        SELECT ARRAY(SELECT jsonb_array_elements_text(metadata_filter->'topic_ids'))
        INTO topic_ids_filter;
        other_filter := metadata_filter - 'topic_ids';
        IF other_filter = '{}'::jsonb THEN
            other_filter := NULL;
        END IF;
    ELSE
        topic_ids_filter := NULL;
        other_filter := metadata_filter;
    END IF;

    IF query_text IS NOT NULL AND query_text <> '' THEN
        tsquery_val := plainto_tsquery('english', query_text);
    ELSE
        tsquery_val := NULL;
    END IF;

    distance_limit := 1.0 - match_threshold;
    candidate_count := GREATEST(match_count * 3, 60);

    RETURN QUERY
    WITH candidates AS (
        SELECT
            c.chunk_id,
            c.document_id,
            c.content,
            c.content_tsv,
            c.section_path,
            c.context_summary,
            c.source_id,
            c.source_title,
            c.page_number,
            c.line_from,
            c.line_to,
            c.archive_browse_uri,
            c.archive_download_uri,
            c.metadata,
            c.embedding <=> query_embedding AS distance
        FROM chunks c
        WHERE c.embedding IS NOT NULL
          AND c.embedding <=> query_embedding < distance_limit
          -- GIN-indexable OR-of-containments on topic_ids (migration 003
          -- semantics folded in). Each ``c.metadata @> <expr>`` is served
          -- from the GIN index instead of a sequential scan.
          AND (topic_ids_filter IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM unnest(topic_ids_filter) AS tid
                   WHERE c.metadata @> jsonb_build_object(
                       'topic_ids', jsonb_build_array(tid)
                   )
               ))
          AND (other_filter IS NULL OR c.metadata @> other_filter)
        ORDER BY c.embedding <=> query_embedding
        LIMIT candidate_count
    )
    SELECT
        cand.chunk_id,
        cand.document_id AS document_uuid,
        cand.content,
        cand.section_path,
        CASE
            WHEN tsquery_val IS NOT NULL AND cand.content_tsv @@ tsquery_val THEN
                (0.7 * (1 - cand.distance) +
                 0.3 * ts_rank(cand.content_tsv, tsquery_val))::FLOAT
            ELSE
                (1 - cand.distance)::FLOAT
        END AS similarity,
        cand.context_summary,
        d.doc_id,
        cand.source_id,
        d.file_path,
        d.document_type,
        cand.source_title,
        cand.page_number,
        cand.line_from,
        cand.line_to,
        cand.archive_browse_uri,
        cand.archive_download_uri,
        COALESCE(d.email_subject, root_doc.email_subject) AS email_subject,
        COALESCE(d.email_initiator, root_doc.email_initiator) AS email_initiator,
        COALESCE(d.email_participants, root_doc.email_participants) AS email_participants,
        COALESCE(d.email_date_start, root_doc.email_date_start) AS email_date,
        root_doc.file_path AS root_file_path
    FROM candidates cand
    JOIN documents d ON cand.document_id = d.id
    LEFT JOIN documents root_doc ON d.root_id = root_doc.id
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS 'Hybrid vector+BM25 search with
HNSW-eligible ORDER BY (2026-04-22 prod migration). Over-fetches
match_count*3 HNSW candidates, then re-scores with BM25 and takes
top match_count. Topic filter uses GIN-backed @> containment.';

-- ============================================================================
-- Post-apply verification (run manually after deploy):
--
--   -- 1. Function definitions are in place.
--   SELECT proname, pg_get_function_result(oid)
--     FROM pg_proc
--    WHERE proname IN (
--        'match_chunks', 'find_similar_topics',
--        'count_chunks_by_topic', 'count_chunks_by_topics',
--        'rewrite_chunk_topic_ids'
--    );
--
--   -- 2. HNSW actually fires inside match_chunks (EXPLAIN Execution Time
--   --    should be <20ms on a warm cache, not >1s):
--   SET hnsw.ef_search = 100;
--   EXPLAIN (ANALYZE, SUMMARY)
--     SELECT chunk_id FROM match_chunks(
--         (SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1),
--         0.3, 40, NULL, 'equipment failures'
--     );
--
--   -- 3. Spot-check a topic count (should be < 50ms at prod scale):
--   EXPLAIN (ANALYZE, SUMMARY)
--     SELECT count_chunks_by_topic(
--         (SELECT id FROM topics ORDER BY chunk_count DESC LIMIT 1)
--     );
-- ============================================================================
