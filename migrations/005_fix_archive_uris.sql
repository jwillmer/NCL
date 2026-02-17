-- Migration 005: Fix archive URI issues
--
-- Fixes 4 related bugs with archive URIs:
-- 1. Double-encoded URIs (%2520 instead of %20)
-- 2. Missing archive_download_uri on attachment documents
-- 3. Chunks missing archive URIs that their parent documents have
--
-- Safe to re-run (all statements are idempotent).

-- Fix double-encoded URIs (e.g., %2520 → %20)
UPDATE documents
SET archive_browse_uri = replace(archive_browse_uri, '%25', '%')
WHERE archive_browse_uri LIKE '%!%25%' ESCAPE '!';

UPDATE documents
SET archive_download_uri = replace(archive_download_uri, '%25', '%')
WHERE archive_download_uri LIKE '%!%25%' ESCAPE '!';

UPDATE chunks
SET archive_browse_uri = replace(archive_browse_uri, '%25', '%')
WHERE archive_browse_uri LIKE '%!%25%' ESCAPE '!';

UPDATE chunks
SET archive_download_uri = replace(archive_download_uri, '%25', '%')
WHERE archive_download_uri LIKE '%!%25%' ESCAPE '!';

-- Set missing download URIs (derive from browse URI by stripping .md)
UPDATE documents
SET archive_download_uri = regexp_replace(archive_browse_uri, '\.md$', '')
WHERE document_type != 'email'
  AND archive_browse_uri IS NOT NULL
  AND archive_download_uri IS NULL;

-- Propagate all URIs from documents to chunks
UPDATE chunks c
SET archive_browse_uri = d.archive_browse_uri,
    archive_download_uri = d.archive_download_uri
FROM documents d
WHERE c.document_id = d.id
  AND (c.archive_browse_uri IS DISTINCT FROM d.archive_browse_uri
       OR c.archive_download_uri IS DISTINCT FROM d.archive_download_uri);
