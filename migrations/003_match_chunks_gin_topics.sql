-- Migration 003: match_chunks topic_ids filter uses the GIN index.
--
-- baseline-02 (2026-04-21) instrumentation showed search_rerank_ms p50 =
-- 42s on 36/37 questions, with broad (fleet-wide) queries hitting 47s
-- median and 91s p95. Confirmed via smoke-06 (local cross-encoder, 18 ms
-- warm) that Cohere rerank is NOT the cost — the pgvector search itself
-- is slow on broad queries. Smoke-08 confirmed disabling hybrid BM25
-- made things WORSE, so ts_rank isn't the culprit either.
--
-- The real culprit is the topic_ids filter in match_chunks:
--
--     AND (topic_ids_filter IS NULL
--          OR EXISTS (
--              SELECT 1
--              FROM jsonb_array_elements_text(c.metadata->'topic_ids') AS stored_tid
--              WHERE stored_tid = ANY(topic_ids_filter)
--          ))
--
-- ``jsonb_array_elements_text(c.metadata->'topic_ids')`` is a function
-- expression on a sub-path — it cannot use the GIN index on
-- chunks.metadata. After HNSW returns the top-K by cosine distance, each
-- candidate's metadata is unnested inline, which at scale forces a
-- sequential-scan shape even though the GIN index is right there.
--
-- Migration 002 already proved this pattern on ``count_chunks_by_topic``:
-- replacing ``metadata->'topic_ids' ? uuid`` with
-- ``metadata @> jsonb_build_object('topic_ids', jsonb_build_array(uuid))``
-- took that function from "multiple seconds" to "low ms" at prod scale.
-- Apply the same fix inside match_chunks.
--
-- The EXISTS-over-unnest form preserves ANY-of-N semantics (match a chunk
-- if it carries AT LEAST ONE of the filter topics) while letting the
-- planner use the GIN ``jsonb_ops`` index to serve each ``@>`` check.
--
-- No data changes; this is a CREATE OR REPLACE on a function. The body
-- below is identical to 000_schema.sql:414 except for the topic filter
-- block (lines 512-517 in the original).

-- Live DB's match_chunks return-row signature has drifted from
-- 000_schema.sql (added fields over time); Postgres refuses CREATE OR
-- REPLACE when the RETURNS row shape changes. Drop first so this
-- migration's RETURNS TABLE wins.
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
    -- document_type is a custom enum in the live schema (not plain TEXT).
    -- 000_schema.sql:~50 defines it. Match the live shape or CREATE FUNCTION
    -- fails with "cannot change return type".
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
BEGIN
    -- Extract topic_ids for special OR handling
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

    RETURN QUERY
    SELECT * FROM (
        SELECT
            c.chunk_id,
            c.document_id AS document_uuid,
            c.content,
            c.section_path,
            CASE
                WHEN tsquery_val IS NOT NULL AND c.content_tsv @@ tsquery_val THEN
                    (0.7 * (1 - (c.embedding <=> query_embedding)) +
                     0.3 * ts_rank(c.content_tsv, tsquery_val))::FLOAT
                ELSE
                    (1 - (c.embedding <=> query_embedding))::FLOAT
            END AS similarity,
            c.context_summary,
            d.doc_id,
            c.source_id,
            d.file_path,
            d.document_type,
            c.source_title,
            c.page_number,
            c.line_from,
            c.line_to,
            c.archive_browse_uri,
            c.archive_download_uri,
            COALESCE(d.email_subject, root_doc.email_subject) AS email_subject,
            COALESCE(d.email_initiator, root_doc.email_initiator) AS email_initiator,
            COALESCE(d.email_participants, root_doc.email_participants) AS email_participants,
            COALESCE(d.email_date_start, root_doc.email_date_start) AS email_date,
            root_doc.file_path AS root_file_path
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        LEFT JOIN documents root_doc ON d.root_id = root_doc.id
        WHERE c.embedding IS NOT NULL
          AND 1 - (c.embedding <=> query_embedding) > match_threshold
          -- GIN-indexable OR-of-containments (one @> check per topic_id in
          -- the filter). The planner serves each c.metadata @> <expr>
          -- from the GIN index instead of sequentially scanning.
          AND (topic_ids_filter IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM unnest(topic_ids_filter) AS tid
                   WHERE c.metadata @> jsonb_build_object(
                       'topic_ids', jsonb_build_array(tid)
                   )
               ))
          AND (other_filter IS NULL OR c.metadata @> other_filter)
    ) sub
    ORDER BY sub.similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS 'Hybrid vector+BM25 search with metadata filtering.
topic_ids filter uses GIN-backed @> containment (migration 003); other metadata
filters use @> directly. OR-of-containments preserves ANY-of-N semantics.';
