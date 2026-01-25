-- Migration 004b: Fix match_chunks to support topic_ids with OR logic
-- The @> operator requires ALL elements to match, but we need ANY (OR logic)

-- Update match_chunks to handle topic_ids specially
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT NULL
)
RETURNS TABLE (
    -- Chunk identification
    chunk_id TEXT,
    document_uuid UUID,
    content TEXT,
    section_path TEXT[],
    similarity FLOAT,

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
BEGIN
    -- Extract topic_ids for special OR handling
    IF metadata_filter IS NOT NULL AND metadata_filter ? 'topic_ids' THEN
        -- Convert JSONB array to TEXT array for ?| operator
        SELECT ARRAY(SELECT jsonb_array_elements_text(metadata_filter->'topic_ids'))
        INTO topic_ids_filter;
        -- Remove topic_ids from filter, keep rest for @> matching
        other_filter := metadata_filter - 'topic_ids';
        IF other_filter = '{}'::jsonb THEN
            other_filter := NULL;
        END IF;
    ELSE
        topic_ids_filter := NULL;
        other_filter := metadata_filter;
    END IF;

    RETURN QUERY
    SELECT
        c.chunk_id,
        c.document_id AS document_uuid,
        c.content,
        c.section_path,
        1 - (c.embedding <=> query_embedding) AS similarity,

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
      -- topic_ids uses OR logic (?| = any of these exist)
      AND (topic_ids_filter IS NULL OR c.metadata->'topic_ids' ?| topic_ids_filter)
      -- Other filters use AND logic (@> = contains all)
      AND (other_filter IS NULL OR c.metadata @> other_filter)
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- Add comment explaining the behavior
COMMENT ON FUNCTION match_chunks IS 'Vector similarity search with metadata filtering.
topic_ids uses OR logic (match ANY topic), other filters use AND logic (match ALL).';
