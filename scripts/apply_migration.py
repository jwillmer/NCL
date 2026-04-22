"""Apply a SQL migration file to a Postgres DB.

Unlike the bootstrap migration files in ``migrations/000_schema.sql`` which
can be loaded with standard tooling inside a transaction, some statements
(notably ``CREATE INDEX CONCURRENTLY``) MUST run outside a transaction
block. This helper opens an asyncpg connection and executes each top-level
SQL statement in the file one at a time, outside any explicit transaction.

Usage:
    uv run python scripts/apply_migration.py \\
        migrations/005_conversations_list_perf.sql --env-file .env.test

    uv run python scripts/apply_migration.py \\
        migrations/005_conversations_list_perf.sql --env-file .env    # PROD

It is intentionally verbose — prints the exact DB host, the file path, and
each statement about to be executed. Production rollout requires an
explicit ``.env`` pointer, so there is no default env-file: you have to
name it on every invocation.

Safety
------
* Read-only preflight: prints ``server_version``, the current DB, and the
  statement plan without executing anything when ``--dry-run`` is set.
* ``--env-file`` is required; there is no default, to prevent an accidental
  prod apply.
* Each top-level statement is executed without an explicit transaction
  wrap. A failure on statement N does NOT roll back statements 1..N-1 —
  this is the cost of running outside a transaction. The migration file
  is deliberately idempotent (``IF NOT EXISTS`` / ``IF EXISTS``).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
from dotenv import dotenv_values


# Very small SQL splitter: splits on top-level semicolons that end a line,
# ignoring semicolons inside strings / dollar-quoted bodies. Good enough for
# the migration files in this repo, which are hand-written and simple.
def _split_sql_statements(sql: str) -> list[str]:
    # Strip block comments first so they don't interfere.
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Strip -- line comments but preserve newlines.
    sql = "\n".join(re.sub(r"--[^\n]*$", "", line) for line in sql.splitlines())

    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    in_dollar: str | None = None

    i = 0
    while i < len(sql):
        ch = sql[i]
        # Dollar-quoted bodies: opening tag = $tag$
        if in_dollar is None and ch == "$":
            m = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if m and not in_single and not in_double:
                in_dollar = m.group(0)
                buf.append(in_dollar)
                i += len(in_dollar)
                continue
        elif in_dollar is not None and sql[i:].startswith(in_dollar):
            buf.append(in_dollar)
            i += len(in_dollar)
            in_dollar = None
            continue

        if in_dollar is None:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == ";" and not in_single and not in_double:
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _redact_host(db_url: str) -> str:
    parsed = urlparse(db_url)
    return f"{parsed.hostname}:{parsed.port or 5432} / db={parsed.path.lstrip('/')}"


async def _apply(db_url: str, migration_path: Path, dry_run: bool) -> int:
    sql = migration_path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)

    print(f"Target DB: {_redact_host(db_url)}")
    print(f"Migration: {migration_path}")
    print(f"Statements: {len(statements)} top-level SQL statement(s)")

    if dry_run:
        print("\n--dry-run: not connecting. Plan:")
        for i, stmt in enumerate(statements, 1):
            first_line = stmt.splitlines()[0][:120]
            print(f"  [{i}] {first_line}{'...' if len(stmt) > 120 else ''}")
        return 0

    conn = await asyncpg.connect(db_url, timeout=60)
    try:
        version = await conn.fetchval("SELECT version()")
        print(f"Connected: {version}")

        for i, stmt in enumerate(statements, 1):
            first_line = stmt.splitlines()[0][:120]
            print(f"\n[{i}/{len(statements)}] {first_line}{'...' if len(stmt) > 120 else ''}")
            # asyncpg's .execute() runs in simple-query protocol mode when
            # not inside an explicit transaction — which is what CREATE
            # INDEX CONCURRENTLY requires.
            await conn.execute(stmt)
            print(f"    -> OK")
    finally:
        await conn.close()

    print("\nDone.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("migration", type=Path, help="Path to the .sql migration file")
    ap.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="Path to the dotenv file to read SUPABASE_DB_URL from. Required — no default.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned statements without connecting to the DB.",
    )
    args = ap.parse_args()

    if not args.migration.exists():
        print(f"Migration file not found: {args.migration}", file=sys.stderr)
        return 2

    if not args.env_file.exists():
        print(f"env file not found: {args.env_file}", file=sys.stderr)
        return 2

    values = dotenv_values(args.env_file)
    db_url = values.get("SUPABASE_DB_URL")
    if not db_url:
        print(f"SUPABASE_DB_URL not set in {args.env_file}", file=sys.stderr)
        return 2

    return asyncio.run(_apply(db_url, args.migration, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
