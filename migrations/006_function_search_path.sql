-- Migration 006: pin search_path on all public functions.
--
-- Supabase advisor (lint 0011, function_search_path_mutable) flagged ten
-- public functions with a role-mutable search_path. If any of these ever
-- run under SECURITY DEFINER (or are invoked by a privileged role that
-- has CREATE on another schema earlier in its path), an attacker can
-- shadow operators/types/functions the body relies on. Pinning
-- search_path to `public, pg_temp` removes that vector.
--
-- Applied via ALTER FUNCTION so we don't need to know each signature
-- ahead of time. A DO block iterates over every function in the `public`
-- schema that matches one of the advisor's names and pins its
-- search_path. Idempotent — safe to re-run; ALTER FUNCTION ... SET is a
-- no-op when the setting already matches.
--
-- Two of the flagged functions (`get_file_registry_stats`,
-- `file_needs_processing`) don't exist in the repo migrations; they're
-- prod-only residue from an earlier schema. The dynamic ALTER catches
-- them without us having to guess their argument lists.

DO $$
DECLARE
    f record;
    target_names TEXT[] := ARRAY[
        'update_updated_at_column',
        'match_chunks',
        'find_similar_topics',
        'count_chunks_by_topic',
        'count_chunks_by_topics',
        'get_topics_with_counts',
        'get_document_ancestry',
        'rewrite_chunk_topic_ids',
        'get_file_registry_stats',
        'file_needs_processing'
    ];
BEGIN
    FOR f IN
        SELECT n.nspname AS schema_name,
               p.proname  AS func_name,
               pg_get_function_identity_arguments(p.oid) AS args
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND p.proname = ANY(target_names)
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%I(%s) SET search_path = public, pg_temp',
            f.schema_name, f.func_name, f.args
        );
    END LOOP;
END
$$;
