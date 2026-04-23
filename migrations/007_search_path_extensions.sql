-- Migration 007: restore extensions schema on the two vector-using functions.
--
-- Migration 006 pinned search_path = `public, pg_temp` on all ten public
-- functions flagged by Supabase advisor lint 0011. That is correct for
-- functions whose bodies only touch pg_catalog / public objects — but
-- `match_chunks` and `find_similar_topics` both use pgvector's `<=>`
-- operator, which lives in the `extensions` schema (000_schema.sql:47
-- `CREATE EXTENSION vector WITH SCHEMA extensions`). With `extensions`
-- dropped from the path, operator lookup fails at call time:
--
--     asyncpg.exceptions.UndefinedFunctionError: operator does not exist:
--     extensions.vector <=> extensions.vector
--
-- End-user impact: every /api/agent RAG search returned an error for the
-- window between 006 and this migration. Topic filtering also failed
-- (`Embedding-only topic matching failed`). Other endpoints (ingest,
-- conversations, stats, health) unaffected because they don't invoke
-- either function.
--
-- Fix: pin those two functions to `public, extensions, pg_temp`. The
-- other eight functions stay at `public, pg_temp` because their bodies
-- use only pg_catalog and public objects — narrower path, smaller
-- attack surface, same intent as 006.
--
-- Why adding `extensions` is still safe under lint 0011:
--   0011 warns about privilege escalation when a role can CREATE into a
--   schema that appears earlier in the path than the one a function
--   resolves objects from — an attacker plants a shadow operator/type
--   and the function picks it up. The `extensions` schema in Supabase
--   is owned by `supabase_admin`; application roles have no CREATE
--   privilege there, so it is not a viable shadow target. Adding it to
--   the pin is the Supabase-documented pattern for functions that
--   depend on extension operators.
--
-- Idempotent. Safe to re-run — ALTER FUNCTION ... SET is a no-op when
-- the setting already matches.

DO $$
DECLARE
    f record;
    target_names TEXT[] := ARRAY[
        'match_chunks',
        'find_similar_topics'
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
            'ALTER FUNCTION %I.%I(%s) SET search_path = public, extensions, pg_temp',
            f.schema_name, f.func_name, f.args
        );
    END LOOP;
END
$$;
