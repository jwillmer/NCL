-- Migration 007: Hybrid search (BM25 + vector) and context summary in results
-- Adds:
-- 1. Generated tsvector column + GIN index for full-text search
-- 2. Updated match_chunks() with optional query_text for hybrid scoring
-- 3. context_summary in match_chunks() output for LLM context assembly

-- ============================================
-- 1. ADD TSVECTOR COLUMN (auto-populated)
-- ============================================

-- PostgreSQL GENERATED ALWAYS AS ... STORED auto-populates on INSERT/UPDATE.
-- Existing rows will be populated when this migration runs.
ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
  ON chunks USING GIN (content_tsv);

-- ============================================
-- 2. UPDATE match_chunks() FOR HYBRID SEARCH
-- ============================================

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT NULL,
    query_text TEXT DEFAULT NULL
)
RETURNS TABLE (
    -- Chunk identification
    chunk_id TEXT,
    document_uuid UUID,
    content TEXT,
    section_path TEXT[],
    similarity FLOAT,

    -- Context summary (LLM-generated document summary)
    context_summary TEXT,

    -- Document info
    doc_id TEXT,
    source_id TEXT,
    file_path TEXT,
    document_type document_type,

    -- Citation metadata
    source_title TEXT,
    page_number INTEGER,
    line_from INTEGER,
    line_to INTEGER,

    -- Archive links
    archive_browse_uri TEXT,
    archive_download_uri TEXT,

    -- Email metadata (from root document)
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

    -- Pre-compute tsquery if query_text provided
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
            -- Hybrid score: blend vector similarity with BM25 when query_text matches
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
          AND (topic_ids_filter IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM jsonb_array_elements_text(c.metadata->'topic_ids') AS stored_tid
                   WHERE stored_tid = ANY(topic_ids_filter)
               ))
          AND (other_filter IS NULL OR c.metadata @> other_filter)
    ) sub
    ORDER BY sub.similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS 'Hybrid vector+BM25 search with metadata filtering.
When query_text is provided, score = 0.7*vector + 0.3*bm25 for text-matching chunks.
Chunks that don''t match the text query keep their pure vector score.
topic_ids uses OR logic (match ANY topic), other filters use AND logic (match ALL).
Returns context_summary for LLM context assembly.';
