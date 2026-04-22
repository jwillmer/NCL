-- 005_conversations_list_perf.sql
--
-- Tune the indexes backing GET /api/conversations.
--
-- The list query filters by (user_id, is_archived) and orders by
-- (last_message_at DESC NULLS LAST, created_at DESC). The original schema
-- had three separate indexes that each covered part of that shape:
--   - idx_conversations_user_recent   (user_id, last_message_at DESC)
--   - idx_conversations_user_archived (user_id, is_archived)
--   - idx_conversations_user_id       (user_id)   -- kept, covers tsearch joins
-- The planner had to pick one of them and then sort/filter in memory. A
-- single composite that matches the full filter + sort key lets Postgres
-- satisfy the query directly from the index.
--
-- NOTE: CREATE INDEX CONCURRENTLY must run OUTSIDE a transaction block. The
-- helper `scripts/apply_migration.py` uses asyncpg's simple-query mode,
-- which is the correct way to run this migration.

-- New composite index tuned for the list query.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_list
    ON conversations(user_id, is_archived, last_message_at DESC NULLS LAST, created_at DESC);

-- The two narrower indexes are now redundant — the composite above is a
-- strict superset for every query shape they cover.
DROP INDEX IF EXISTS idx_conversations_user_recent;
DROP INDEX IF EXISTS idx_conversations_user_archived;
