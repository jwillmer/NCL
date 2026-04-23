-- Migration 004: make HNSW actually fire inside match_chunks.
--
-- Direct probe on 2026-04-22 attributed the 45+ second search latency
-- entirely to db_search_ms (the pgvector call), not the reranker:
--
--     Raw "ORDER BY embedding <=> v LIMIT 40"       ->  50-800ms   (HNSW used)
--     match_chunks vector-only (query_text=NULL)    ->  24-32s     (seq scan)
--     match_chunks hybrid (query_text set)          ->  27-36s     (+ts_rank)
--
-- EXPLAIN on match_chunks shows "Function Scan on match_chunks
-- actual time=21773ms", confirming the planner never opened the
-- HNSW index. Why: migration 003 (and 000 before it) wrote the
-- WHERE clause as
--     AND 1 - (c.embedding <=> query_embedding) > match_threshold
-- and ORDER BY on a derived ``sub.similarity`` column. pgvector
-- HNSW only engages for ``ORDER BY embedding <op> constant`` with
-- an optional distance-operator filter. A filter on 1-distance is
-- a derived expression the planner cannot match to the index, so
-- it falls back to a seq scan over all 43k chunks for every call.
--
-- Fix: express the similarity threshold as a direct distance bound
-- (1 - match_threshold) and ORDER BY the raw operator inside the
-- subquery. We over-fetch match_count * 3 candidates via HNSW,
-- then re-rank them with the hybrid BM25 expression on a ~120-row
-- outer set so ts_rank cost stays bounded. Final LIMIT match_count
-- returns the same row count as before.
--
-- Semantics preserved:
--   - ``similarity > match_threshold``  equivalent to
--     ``embedding <=> query_embedding < 1 - match_threshold``
--     for any embedding_distance in [0, 2].
--   - Topic/other metadata filter blocks unchanged from migration 003.
--   - Hybrid BM25 blend (0.7 vector + 0.3 ts_rank when tsquery hits)
--     unchanged, just evaluated after HNSW cuts the candidate set
--     instead of across every row.
--   - Return shape unchanged (RETURNS TABLE matches migration 003).
--
-- Over-fetch factor: 3x. HNSW with ef_search=100 (session default
-- bumped for this workload) produces ~ef_search sorted candidates;
-- taking top 3*match_count keeps recall high enough for the
-- re-rank step to restore the exact hybrid top-K.

DROP FUNCTION IF EXISTS match_chunks(vector, float, int, JSONB, TEXT);

CREATE FUNCTION match_chunks(
    query_embedding vector(1536),
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
SET search_path = public, extensions, pg_temp
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

    -- HNSW fires on ORDER BY embedding <=> const and on
    -- embedding <=> const < const; translate similarity threshold
    -- into the equivalent distance bound.
    distance_limit := 1.0 - match_threshold;
    -- Over-fetch so the hybrid re-rank has enough candidates.
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

COMMENT ON FUNCTION match_chunks IS 'Hybrid vector+BM25 search with HNSW-eligible
ORDER BY (migration 004). Over-fetches match_count*3 HNSW candidates, then re-scores
with BM25 and takes top match_count. Topic filter uses GIN-backed @> (migration 003).';
