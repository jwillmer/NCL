-- MTSS RAG Pipeline - Complete Database Schema (merged from migrations 001–008)
-- Run this in Supabase SQL Editor to create a clean database from scratch.
-- WARNING: This drops ALL existing tables and recreates them.

-- ============================================
-- CLEANUP EXISTING SCHEMA
-- ============================================

DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS vessels CASCADE;
DROP TABLE IF EXISTS topics CASCADE;
DROP TABLE IF EXISTS ingest_versions CASCADE;
DROP TABLE IF EXISTS ingest_events CASCADE;
DROP TABLE IF EXISTS unsupported_files CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS processing_log CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

DROP FUNCTION IF EXISTS match_chunks CASCADE;
DROP FUNCTION IF EXISTS get_document_ancestry CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
DROP FUNCTION IF EXISTS find_similar_topics CASCADE;
DROP FUNCTION IF EXISTS count_chunks_by_topic CASCADE;
DROP FUNCTION IF EXISTS count_chunks_by_topics CASCADE;
DROP FUNCTION IF EXISTS get_topics_with_counts CASCADE;

DROP TYPE IF EXISTS document_type CASCADE;
DROP TYPE IF EXISTS processing_status CASCADE;

-- Also clear LangGraph checkpoint tables if present
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

-- ============================================
-- EXTENSIONS
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ============================================
-- CUSTOM TYPES
-- ============================================

CREATE TYPE document_type AS ENUM (
    'email',
    'attachment_pdf',
    'attachment_image',
    'attachment_docx',
    'attachment_pptx',
    'attachment_xlsx',
    'attachment_doc',
    'attachment_xls',
    'attachment_ppt',
    'attachment_csv',
    'attachment_rtf',
    'attachment_other'
);

CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'skipped'
);

-- ============================================
-- TRIGGER FUNCTION
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = public, pg_temp;

-- ============================================
-- DOCUMENTS TABLE
-- ============================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Stable identification
    source_id TEXT NOT NULL,
    doc_id TEXT NOT NULL UNIQUE,

    -- Versioning
    content_version INTEGER NOT NULL DEFAULT 1,
    ingest_version INTEGER NOT NULL DEFAULT 1,

    -- Hierarchy
    parent_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    root_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    depth INTEGER NOT NULL DEFAULT 0,
    path TEXT[] NOT NULL DEFAULT '{}',

    -- Document identification
    document_type document_type NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,

    -- Citation metadata
    source_title TEXT,

    -- Email metadata (nullable for attachments)
    email_subject TEXT,
    email_participants TEXT[],
    email_initiator TEXT,
    email_date_start TIMESTAMPTZ,
    email_date_end TIMESTAMPTZ,
    email_message_count INTEGER DEFAULT 1,

    -- Attachment metadata
    attachment_content_type TEXT,
    attachment_size_bytes BIGINT,

    -- Archive links
    archive_path TEXT,
    archive_browse_uri TEXT,
    archive_download_uri TEXT,

    -- Processing
    status processing_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    processed_at TIMESTAMPTZ,

    -- Embedding mode: 'full', 'summary', 'metadata_only' (see embedding_decider.py)
    embedding_mode TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_parent_id ON documents(parent_id);
CREATE INDEX idx_documents_root_id ON documents(root_id);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_document_type ON documents(document_type);
CREATE INDEX idx_documents_path ON documents USING GIN(path);
CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_ingest_version ON documents(ingest_version);

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- CHUNKS TABLE
-- ============================================

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Stable identification
    chunk_id TEXT NOT NULL,

    -- Relationship
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,

    -- Contextual chunking
    context_summary TEXT,
    embedding_text TEXT,

    -- Hierarchy context
    section_path TEXT[],
    section_title TEXT,

    -- Citation metadata (denormalized)
    source_title TEXT,
    source_id TEXT,

    -- Source location
    page_number INTEGER,
    line_from INTEGER,
    line_to INTEGER,
    char_start INTEGER,
    char_end INTEGER,

    -- Archive links (denormalized)
    archive_browse_uri TEXT,
    archive_download_uri TEXT,

    -- Vector embedding (1536 dimensions)
    embedding extensions.vector(1536),

    -- Embedding mode inherited from parent document
    embedding_mode TEXT,

    -- Metadata for filtering
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Full-text search (auto-populated)
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks(document_id, chunk_index);
CREATE UNIQUE INDEX idx_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX idx_chunks_source_id ON chunks(source_id);
CREATE INDEX idx_chunks_metadata ON chunks USING GIN (metadata);
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);

CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- PROCESSING LOG TABLE
-- ============================================

CREATE TABLE processing_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,

    status processing_status NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_processing_log_status ON processing_log(status);
CREATE INDEX idx_processing_log_file_hash ON processing_log(file_hash);

CREATE TRIGGER update_processing_log_updated_at
    BEFORE UPDATE ON processing_log
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- INGEST EVENTS TABLE
-- ============================================

CREATE TABLE ingest_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,
    file_size_bytes BIGINT,

    source_eml_path TEXT,
    source_zip_path TEXT,
    parent_document_id UUID REFERENCES documents(id),

    mime_type TEXT,
    file_extension TEXT,
    reason TEXT NOT NULL,

    severity TEXT DEFAULT 'warning',
    event_type TEXT DEFAULT 'unsupported_file',

    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(file_path),

    CONSTRAINT chk_ingest_events_severity
        CHECK (severity IN ('error', 'warning', 'info')),
    CONSTRAINT chk_ingest_events_type
        CHECK (event_type IN (
            'unsupported_file',
            'encoding_fallback',
            'parse_failure',
            'archive_failure',
            'context_failure',
            'empty_content',
            'zip_extraction_failure'
        ))
);

CREATE INDEX idx_ingest_events_source_eml ON ingest_events(source_eml_path);
CREATE INDEX idx_ingest_events_mime_type ON ingest_events(mime_type);
CREATE INDEX idx_ingest_events_reason ON ingest_events(reason);
CREATE INDEX idx_ingest_events_document ON ingest_events(parent_document_id);
CREATE INDEX idx_ingest_events_type ON ingest_events(event_type);
CREATE INDEX idx_ingest_events_severity ON ingest_events(severity);
CREATE INDEX idx_ingest_events_discovered_at ON ingest_events(discovered_at DESC);

COMMENT ON TABLE ingest_events IS 'Tracks processing events during ingest for visibility and debugging';

-- ============================================
-- VESSELS TABLE
-- ============================================

CREATE TABLE vessels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    vessel_type TEXT,
    vessel_class TEXT,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT vessels_name_unique UNIQUE (name)
);

CREATE INDEX idx_vessels_name ON vessels(name);
CREATE INDEX idx_vessels_vessel_type ON vessels(vessel_type);
CREATE INDEX idx_vessels_vessel_class ON vessels(vessel_class);
CREATE INDEX idx_vessels_aliases ON vessels USING GIN(aliases);

CREATE TRIGGER update_vessels_updated_at
    BEFORE UPDATE ON vessels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- TOPICS TABLE
-- ============================================

CREATE TABLE topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT,
    embedding extensions.vector(1536),
    chunk_count INTEGER NOT NULL DEFAULT 0,
    document_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_topics_name ON topics(name);
CREATE INDEX idx_topics_embedding ON topics
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TRIGGER update_topics_updated_at
    BEFORE UPDATE ON topics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- CONVERSATIONS TABLE
-- ============================================

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID UNIQUE NOT NULL,
    user_id UUID NOT NULL,
    title TEXT,
    vessel_id UUID REFERENCES vessels(id),
    vessel_type TEXT,
    vessel_class TEXT,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ
);

COMMENT ON COLUMN conversations.vessel_id IS 'Vessel UUID filter (mutually exclusive with vessel_type/vessel_class)';
COMMENT ON COLUMN conversations.vessel_type IS 'Vessel type filter e.g. VLCC (mutually exclusive with vessel_id/vessel_class)';
COMMENT ON COLUMN conversations.vessel_class IS 'Vessel class filter e.g. Canopus Class (mutually exclusive with vessel_id/vessel_type)';

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
-- Composite tuned for GET /api/conversations: filter by (user_id, is_archived),
-- sort by (last_message_at DESC NULLS LAST, created_at DESC). Satisfies the
-- list query directly from the index — earlier three narrower indexes each
-- covered only part of the shape and forced an in-memory sort.
CREATE INDEX idx_conversations_list
    ON conversations(user_id, is_archived, last_message_at DESC NULLS LAST, created_at DESC);
CREATE INDEX idx_conversations_vessel ON conversations(vessel_id);
CREATE INDEX idx_conversations_vessel_type ON conversations(vessel_type);
CREATE INDEX idx_conversations_vessel_class ON conversations(vessel_class);
CREATE INDEX idx_conversations_search ON conversations
    USING gin(to_tsvector('english', COALESCE(title, '')));

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- INGEST VERSIONS TABLE
-- ============================================

CREATE TABLE ingest_versions (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    breaking_changes BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO ingest_versions (version, description, breaking_changes) VALUES
    (1, 'Initial schema with contextual chunking, citations, and browsable archive', FALSE),
    (2, 'Added ingest_events table with severity and event_type columns', FALSE),
    (3, 'Added vessel_class, removed imo/dwt, added type/class filters to conversations', FALSE),
    (4, 'Added topics table with embeddings for semantic categorization and pre-filtering', FALSE);

-- ============================================
-- FUNCTIONS: VECTOR / HYBRID SEARCH
-- ============================================

-- Hybrid vector+BM25 search. HNSW-eligible ORDER BY + GIN-indexable topic
-- filter. Over-fetches match_count*3 HNSW candidates then re-scores with
-- BM25 on the bounded set. See git log for the performance rationale
-- (derived expression on 1-distance defeated HNSW; direct distance bound
-- restores it).
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding extensions.vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
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
          -- GIN-indexable OR-of-containments on topic_ids. Each
          -- c.metadata @> <expr> is served from the GIN index on
          -- chunks.metadata (jsonb_ops).
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

COMMENT ON FUNCTION match_chunks IS 'Hybrid vector+BM25 search with metadata filtering.
Over-fetches match_count*3 HNSW candidates, then re-scores with BM25 on a bounded set.
When query_text is provided, score = 0.7*vector + 0.3*bm25 for text-matching chunks.
topic_ids uses OR logic (match ANY topic), other filters use AND logic (match ALL).';

-- ============================================
-- FUNCTIONS: DOCUMENT ANCESTRY
-- ============================================

CREATE OR REPLACE FUNCTION get_document_ancestry(document_uuid UUID)
RETURNS TABLE (
    id UUID,
    document_type document_type,
    file_path TEXT,
    file_name TEXT,
    depth INTEGER,
    email_subject TEXT
)
LANGUAGE SQL
SET search_path = public, pg_temp
AS $$
    WITH RECURSIVE ancestry AS (
        SELECT d.id, d.document_type, d.file_path, d.file_name, d.depth,
               d.email_subject, d.parent_id
        FROM documents d WHERE d.id = document_uuid

        UNION ALL

        SELECT d.id, d.document_type, d.file_path, d.file_name, d.depth,
               d.email_subject, d.parent_id
        FROM documents d
        JOIN ancestry a ON d.id = a.parent_id
    )
    SELECT a.id, a.document_type, a.file_path, a.file_name, a.depth, a.email_subject
    FROM ancestry a
    ORDER BY a.depth ASC;
$$;

-- ============================================
-- FUNCTIONS: TOPICS
-- ============================================

-- HNSW index on topics(embedding vector_cosine_ops) activates when the
-- ORDER BY matches the opclass and a LIMIT is present. Inner over-fetch
-- (max_results * 4) keeps the threshold filter from starving the result.
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
SET search_path = public, extensions, pg_temp
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
        ORDER BY t.embedding <=> query_embedding
        LIMIT GREATEST(max_results * 4, 20)
    ) x
    WHERE x.similarity > similarity_threshold
    ORDER BY x.similarity DESC
    LIMIT max_results;
$$;

-- Containment (@>) uses the GIN index on chunks.metadata (jsonb_ops).
-- The prior `->'topic_ids' ? uuid` form was a function expression on a
-- sub-path and fell back to a sequential scan at production scale.
CREATE OR REPLACE FUNCTION count_chunks_by_topic(
    topic_uuid UUID,
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
SET search_path = public, pg_temp
AS $$
    SELECT COUNT(*)::INTEGER
    FROM chunks c
    WHERE c.metadata @> jsonb_build_object(
              'topic_ids', jsonb_build_array(topic_uuid::text)
          )
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

-- Union of index-backed containment scans, one per topic_id. DISTINCT
-- c.id dedupes chunks that carry multiple matching topics.
CREATE OR REPLACE FUNCTION count_chunks_by_topics(
    topic_uuids UUID[],
    vessel_filter JSONB DEFAULT NULL
)
RETURNS INTEGER
LANGUAGE SQL STABLE
SET search_path = public, pg_temp
AS $$
    SELECT COUNT(DISTINCT c.id)::INTEGER
    FROM chunks c,
         unnest(topic_uuids) AS tid
    WHERE c.metadata @> jsonb_build_object(
              'topic_ids', jsonb_build_array(tid::text)
          )
      AND (vessel_filter IS NULL OR c.metadata @> vessel_filter);
$$;

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
SET search_path = public, pg_temp
AS $$
    SELECT
        t.id, t.name, t.display_name, t.description,
        t.chunk_count, t.document_count
    FROM topics t
    ORDER BY t.chunk_count DESC, t.name ASC;
$$;

-- Server-side rewrite of chunks.metadata.topic_ids for local → remote topic
-- consolidation. Takes a {old_uuid: new_uuid} mapping and updates every
-- chunk that carries an absorbed UUID. Called by `mtss import` after the
-- local `mtss topics consolidate --apply` step so no chunk ends up
-- pointing at a deleted topic.
CREATE OR REPLACE FUNCTION rewrite_chunk_topic_ids(mapping JSONB)
RETURNS INTEGER
LANGUAGE plpgsql
SET search_path = public, pg_temp
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
            -- DISTINCT dedupes — a keeper UUID the chunk already
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
    'Rewrite chunks.metadata.topic_ids using {old_uuid:new_uuid} map. '
    'Called by mtss import after local topic consolidation.';

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingest_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingest_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE vessels ENABLE ROW LEVEL SECURITY;
ALTER TABLE topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Service role: full access
CREATE POLICY "Service role full access to documents" ON documents
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to chunks" ON chunks
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to processing_log" ON processing_log
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to ingest_events" ON ingest_events
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to ingest_versions" ON ingest_versions
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to vessels" ON vessels
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to topics" ON topics
    FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access to conversations" ON conversations
    FOR ALL TO service_role USING (true);

-- Authenticated users: read access to data tables
CREATE POLICY "Authenticated users can read documents" ON documents
    FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read chunks" ON chunks
    FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read vessels" ON vessels
    FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read topics" ON topics
    FOR SELECT TO authenticated USING (true);

-- Conversations: users own their data
CREATE POLICY "Users can view own conversations" ON conversations
    FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own conversations" ON conversations
    FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own conversations" ON conversations
    FOR UPDATE TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own conversations" ON conversations
    FOR DELETE TO authenticated USING (auth.uid() = user_id);

-- ============================================
-- NOTES
-- ============================================
-- LangGraph's AsyncPostgresSaver creates its own tables automatically:
--   - checkpoints
--   - checkpoint_writes
--   - checkpoint_blobs
-- These store the actual message history per thread_id.
