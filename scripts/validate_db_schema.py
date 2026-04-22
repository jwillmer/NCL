"""Audit a Supabase Postgres DB against the repo's migrations.

Parses `migrations/*.sql` to extract the expected set of extensions, tables,
indexes, functions (with return-table column counts), and trigger functions.
Then introspects a live DB via read-only `pg_catalog` queries and reports
anything MISSING. Extra objects on the DB side are informational.

Read-only — only SELECTs against `pg_catalog` / `information_schema`.

Usage:
    uv run python scripts/validate_db_schema.py --env-file .env
    uv run python scripts/validate_db_schema.py --env-file .env.test
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import asyncpg
from dotenv import dotenv_values
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"

# LangGraph-managed auto-created tables we intentionally do NOT audit.
LANGGRAPH_TABLES = {"checkpoints", "checkpoint_writes", "checkpoint_blobs"}

# Tables that should have an `updated_at` trigger (per 000_schema.sql).
TABLES_WITH_UPDATED_AT_TRIGGER = {
    "documents",
    "processing_log",
    "vessels",
    "topics",
    "conversations",
}


@dataclass
class ExpectedSchema:
    """Expected objects parsed from migration SQL files."""

    extensions: set[str] = field(default_factory=set)
    tables: set[str] = field(default_factory=set)
    indexes: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    # Latest RETURNS TABLE column count, keyed by function name.
    function_return_cols: dict[str, int] = field(default_factory=dict)
    # Tables that have DROP TABLE ... CASCADE (informational; used to validate
    # we aren't missing re-creation)
    trigger_functions: set[str] = field(default_factory=set)


@dataclass
class ActualSchema:
    """Live DB state."""

    extensions: set[str] = field(default_factory=set)
    tables: set[str] = field(default_factory=set)
    indexes: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    function_return_defs: dict[str, str] = field(default_factory=dict)
    triggers_by_table: dict[str, set[str]] = field(default_factory=dict)
    ingest_version: int | None = None
    ingest_version_desc: str | None = None
    db_host: str = ""


# ---------------------------------------------------------------------------
# Parse migrations
# ---------------------------------------------------------------------------

# Regex helpers. Intentionally forgiving; migrations are hand-written SQL.
RE_CREATE_EXTENSION = re.compile(
    r"CREATE\s+EXTENSION\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
RE_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
RE_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
RE_DROP_INDEX = re.compile(
    r"DROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
RE_CREATE_FUNCTION = re.compile(
    r"CREATE(?:\s+OR\s+REPLACE)?\s+FUNCTION\s+([A-Za-z_][\w]*)\s*\(",
    re.IGNORECASE,
)
RE_CREATE_TRIGGER = re.compile(
    r"CREATE\s+TRIGGER\s+([A-Za-z_][\w]*)\s",
    re.IGNORECASE,
)
RE_RETURNS_TABLE = re.compile(
    r"RETURNS\s+TABLE\s*\(",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    # Remove /* ... */ and -- ... line comments to simplify parsing.
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    return sql


def _split_top_level_columns(text: str) -> int:
    """Given the body of a RETURNS TABLE (...) paren-group, return the top-level
    comma count + 1 (i.e. number of columns). Handles nested parens in type
    expressions like vector(1536) or NUMERIC(10,2)."""
    depth = 0
    cols = 1
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cols += 1
    return cols


def _extract_returns_table_cols(sql_after_returns: str) -> int | None:
    """From the substring that starts at RETURNS TABLE (, extract the column
    count. Returns None if the block doesn't start with a paren."""
    m = re.match(r"\s*RETURNS\s+TABLE\s*\(", sql_after_returns, re.IGNORECASE)
    if not m:
        return None
    i = m.end()  # position just after '('
    depth = 1
    start = i
    while i < len(sql_after_returns):
        ch = sql_after_returns[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body = sql_after_returns[start:i]
                return _split_top_level_columns(body)
        i += 1
    return None


def parse_migrations(files: list[Path]) -> ExpectedSchema:
    """Parse the ordered list of migration files. Later files override earlier
    ones for per-function return shape — this mirrors last-writer-wins as
    executed on the DB."""
    expected = ExpectedSchema()
    # Trigger function name emitted by 000_schema.sql.
    # We treat `update_updated_at_column` as a required trigger function.
    for path in files:
        raw = path.read_text(encoding="utf-8")
        sql = _strip_sql_comments(raw)

        for m in RE_CREATE_EXTENSION.finditer(sql):
            expected.extensions.add(m.group(1).lower())

        for m in RE_CREATE_TABLE.finditer(sql):
            expected.tables.add(m.group(1).lower())

        for m in RE_CREATE_INDEX.finditer(sql):
            expected.indexes.add(m.group(1).lower())

        # Later migrations can retire indexes they've superseded. Honour
        # DROP INDEX so the expected set reflects the final post-migration
        # state, not the union of every CREATE ever written.
        for m in RE_DROP_INDEX.finditer(sql):
            expected.indexes.discard(m.group(1).lower())

        for m in RE_CREATE_FUNCTION.finditer(sql):
            fname = m.group(1).lower()
            expected.functions.add(fname)
            # Header ends at "AS $$" (or "AS $body$" etc.) - only consider
            # the RETURNS clause that appears before the body, otherwise we'd
            # pick up RETURNS TABLE from a later function in the same file.
            tail = sql[m.end() :]
            body_m = re.search(r"\bAS\s+\$[^$]*\$", tail, re.IGNORECASE)
            header = tail[: body_m.start()] if body_m else tail[:4000]
            rmatch = re.search(r"RETURNS\s+TABLE\s*\(", header, re.IGNORECASE)
            if rmatch:
                cols = _extract_returns_table_cols(header[rmatch.start() :])
                if cols is not None:
                    # Last writer wins — later migrations can change shape.
                    expected.function_return_cols[fname] = cols

        for m in RE_CREATE_TRIGGER.finditer(sql):
            # Trigger names are informational; we don't assert them directly.
            pass

    # `update_updated_at_column` is emitted as a CREATE OR REPLACE FUNCTION
    # and will be in expected.functions already. Mirror it into
    # trigger_functions for clarity.
    if "update_updated_at_column" in expected.functions:
        expected.trigger_functions.add("update_updated_at_column")

    # LangGraph tables are carved out from expected even if mentioned.
    expected.tables -= LANGGRAPH_TABLES

    return expected


# ---------------------------------------------------------------------------
# Introspect live DB
# ---------------------------------------------------------------------------


async def introspect_db(db_url: str) -> ActualSchema:
    """Connect read-only and gather live schema state."""
    actual = ActualSchema()

    parsed = urlparse(db_url)
    actual.db_host = f"{parsed.hostname}:{parsed.port or 5432}"

    conn = await asyncpg.connect(db_url, timeout=30)
    try:
        rows = await conn.fetch("SELECT extname FROM pg_extension")
        actual.extensions = {r["extname"].lower() for r in rows}

        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        actual.tables = {r["tablename"].lower() for r in rows}

        rows = await conn.fetch(
            "SELECT tablename, indexname FROM pg_indexes WHERE schemaname = 'public'"
        )
        actual.indexes = {r["indexname"].lower() for r in rows}

        rows = await conn.fetch(
            """
            SELECT p.proname, pg_get_function_result(p.oid) AS returns_def
              FROM pg_proc p
              JOIN pg_namespace n ON p.pronamespace = n.oid
             WHERE n.nspname = 'public'
            """
        )
        for r in rows:
            name = r["proname"].lower()
            actual.functions.add(name)
            # Keep the last one seen; usually there's only one per name.
            actual.function_return_defs[name] = r["returns_def"] or ""

        rows = await conn.fetch(
            """
            SELECT t.tgname AS trigger_name,
                   c.relname AS table_name
              FROM pg_trigger t
              JOIN pg_class c ON t.tgrelid = c.oid
              JOIN pg_namespace n ON c.relnamespace = n.oid
             WHERE NOT t.tgisinternal
               AND n.nspname = 'public'
            """
        )
        for r in rows:
            actual.triggers_by_table.setdefault(
                r["table_name"].lower(), set()
            ).add(r["trigger_name"].lower())

        # Latest ingest_versions row (if table exists).
        if "ingest_versions" in actual.tables:
            row = await conn.fetchrow(
                "SELECT version, description FROM ingest_versions ORDER BY version DESC LIMIT 1"
            )
            if row:
                actual.ingest_version = row["version"]
                actual.ingest_version_desc = row["description"]
    finally:
        await conn.close()

    return actual


def count_cols_in_returns_def(returns_def: str) -> int | None:
    """pg_get_function_result returns one of:
       - 'TABLE(col1 type, col2 type, ...)'
       - 'SETOF some_type'
       - 'integer'
    Only the TABLE(...) form has a column count we can check."""
    m = re.match(r"TABLE\s*\(", returns_def, re.IGNORECASE)
    if not m:
        return None
    # Extract body inside the outer parens.
    tail = returns_def[m.end() :]
    depth = 1
    body_end = 0
    for i, ch in enumerate(tail):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    body = tail[:body_end]
    return _split_top_level_columns(body)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def scrub_host(db_url: str) -> str:
    try:
        parsed = urlparse(db_url)
        userpart = parsed.username or ""
        hostpart = f"{parsed.hostname}:{parsed.port or 5432}"
        return f"{userpart}@{hostpart}" if userpart else hostpart
    except Exception:
        return "<unparseable>"


def _section(
    console: Console,
    title: str,
    expected: set[str],
    actual: set[str],
    *,
    ignore_extras: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    missing = expected - actual
    extra = actual - expected - (ignore_extras or set())

    t = Table(title=title, show_header=True, header_style="bold")
    t.add_column("Status")
    t.add_column("Name")
    for n in sorted(missing):
        t.add_row("[red]MISSING[/red]", n)
    for n in sorted(extra):
        t.add_row("[dim]extra[/dim]", f"[dim]{n}[/dim]")
    if not missing and not extra:
        t.add_row("[green]OK[/green]", f"[dim]{len(expected)} objects match[/dim]")
    console.print(t)
    return missing, extra


def report(
    console: Console,
    env_file: Path,
    db_url: str,
    expected: ExpectedSchema,
    actual: ActualSchema,
) -> int:
    """Emit a report for one environment. Returns number of missing items."""
    console.rule(f"[bold cyan]DB schema audit: {env_file.name}[/bold cyan]")

    summary = Table(title="Environment", show_header=False, box=None)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("env_file", str(env_file))
    summary.add_row("db_host", scrub_host(db_url))
    summary.add_row(
        "ingest_version",
        (
            f"{actual.ingest_version} — {actual.ingest_version_desc}"
            if actual.ingest_version is not None
            else "[red]<no ingest_versions row>[/red]"
        ),
    )
    console.print(summary)

    total_missing = 0

    # Extensions
    missing, _extra = _section(
        console, "Extensions", expected.extensions, actual.extensions
    )
    total_missing += len(missing)

    # Tables — ignore LangGraph checkpoints as extras
    missing, _ = _section(
        console,
        "Tables (public schema)",
        expected.tables,
        actual.tables,
        ignore_extras=LANGGRAPH_TABLES,
    )
    total_missing += len(missing)

    # Indexes — many extras are fine (pkeys, uniques created implicitly).
    # Only highlight missing ones parsed from migrations.
    missing_idx = expected.indexes - actual.indexes
    t = Table(title="Indexes (migration-declared only)", show_header=True, header_style="bold")
    t.add_column("Status")
    t.add_column("Name")
    for n in sorted(missing_idx):
        t.add_row("[red]MISSING[/red]", n)
    if not missing_idx:
        t.add_row(
            "[green]OK[/green]",
            f"[dim]{len(expected.indexes)} declared indexes present[/dim]",
        )
    console.print(t)
    total_missing += len(missing_idx)

    # Functions — shape check on RETURNS TABLE
    func_t = Table(title="Functions", show_header=True, header_style="bold")
    func_t.add_column("Status")
    func_t.add_column("Name")
    func_t.add_column("Expected cols")
    func_t.add_column("Actual cols")

    missing_funcs = expected.functions - actual.functions
    for n in sorted(missing_funcs):
        func_t.add_row("[red]MISSING[/red]", n, "-", "-")
    total_missing += len(missing_funcs)

    # For each expected function present, do a return-shape check if we have
    # an expected column count.
    shape_mismatches = 0
    for fname in sorted(expected.functions):
        if fname not in actual.functions:
            continue
        exp_cols = expected.function_return_cols.get(fname)
        act_cols = count_cols_in_returns_def(actual.function_return_defs.get(fname, ""))
        if exp_cols is None:
            # No RETURNS TABLE declared; informational only when present.
            func_t.add_row("[green]present[/green]", fname, "-", "-")
        elif act_cols is None:
            func_t.add_row(
                "[yellow]WRONG-KIND[/yellow]",
                fname,
                f"TABLE({exp_cols})",
                "not a TABLE-returning function",
            )
            shape_mismatches += 1
        elif act_cols != exp_cols:
            func_t.add_row(
                "[red]SHAPE-MISMATCH[/red]",
                fname,
                str(exp_cols),
                str(act_cols),
            )
            shape_mismatches += 1
        else:
            func_t.add_row(
                "[green]OK[/green]",
                fname,
                str(exp_cols),
                str(act_cols),
            )
    total_missing += shape_mismatches
    console.print(func_t)

    # Spotlight: match_chunks must be 21 cols
    mc_expected = 21
    mc_actual = count_cols_in_returns_def(
        actual.function_return_defs.get("match_chunks", "")
    )
    mc_t = Table(
        title="Spotlight: match_chunks post-004/prod shape", show_header=False, box=None
    )
    mc_t.add_column(style="bold")
    mc_t.add_column()
    mc_t.add_row("Expected columns", str(mc_expected))
    mc_t.add_row(
        "Actual columns",
        str(mc_actual) if mc_actual is not None else "[red]missing or not TABLE-returning[/red]",
    )
    if mc_actual != mc_expected:
        mc_t.add_row("Verdict", "[red]NOT matching post-004/prod shape[/red]")
    else:
        mc_t.add_row("Verdict", "[green]OK — includes root_file_path[/green]")
    console.print(mc_t)

    # Trigger functions + updated_at triggers per table
    tf_missing = expected.trigger_functions - actual.functions
    trg_t = Table(title="Trigger functions + updated_at triggers", show_header=True, header_style="bold")
    trg_t.add_column("Status")
    trg_t.add_column("Name")
    for n in sorted(tf_missing):
        trg_t.add_row("[red]MISSING FUNCTION[/red]", n)
    total_missing += len(tf_missing)

    # Every table in TABLES_WITH_UPDATED_AT_TRIGGER should have at least one
    # trigger on it.
    for tbl in sorted(TABLES_WITH_UPDATED_AT_TRIGGER):
        if tbl not in actual.tables:
            # The table-missing case is already reported above; skip here.
            continue
        triggers = actual.triggers_by_table.get(tbl, set())
        if not triggers:
            trg_t.add_row("[red]NO TRIGGERS[/red]", f"{tbl} (expected updated_at trigger)")
            total_missing += 1
        else:
            trg_t.add_row(
                "[green]OK[/green]", f"{tbl} -> {', '.join(sorted(triggers))}"
            )
    console.print(trg_t)

    # Verdict
    if total_missing == 0:
        console.print(
            f"[bold green]PASS[/bold green] — {env_file.name} matches migration declarations."
        )
    else:
        console.print(
            f"[bold red]FAIL[/bold red] — {env_file.name} is missing {total_missing} object(s) / has shape mismatches."
        )

    return total_missing


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------


def load_db_url(env_file: Path) -> str:
    if not env_file.exists():
        raise SystemExit(f"env file not found: {env_file}")
    values = dotenv_values(env_file)
    # Fall back to process env if the file doesn't have it (unlikely here).
    db_url = values.get("SUPABASE_DB_URL") or os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit(f"SUPABASE_DB_URL not set in {env_file}")
    return db_url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_for_env(console: Console, env_file: Path, expected: ExpectedSchema) -> int:
    db_url = load_db_url(env_file)
    try:
        actual = await introspect_db(db_url)
    except Exception as exc:  # pragma: no cover - connection errors
        console.print(
            f"[red]ERROR[/red] connecting to DB for {env_file.name}: {exc!r}"
        )
        return 1
    return report(console, env_file, db_url, expected, actual)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to the dotenv file to load SUPABASE_DB_URL from (default: .env)",
    )
    ap.add_argument(
        "--both",
        action="store_true",
        help="Run against both .env and .env.test in one invocation",
    )
    args = ap.parse_args()

    console = Console()

    # Apply order matters — later files override earlier function shapes
    # (last-writer-wins) and DROP INDEX statements prune expected objects.
    # Standard numbered migrations first; dated prod rollups after.
    migration_files = sorted(
        [p for p in MIGRATIONS_DIR.glob("*.sql") if re.match(r"\d{3}_", p.name)]
    ) + sorted(
        [p for p in MIGRATIONS_DIR.glob("prod_*.sql")]
    )
    missing_files = [f for f in migration_files if not f.exists()]
    if missing_files:
        raise SystemExit(
            f"Missing migration files: {', '.join(str(f) for f in missing_files)}"
        )

    expected = parse_migrations(migration_files)

    if args.both:
        total = 0
        for envf in (REPO_ROOT / ".env", REPO_ROOT / ".env.test"):
            total += asyncio.run(run_for_env(console, envf, expected))
        return 0 if total == 0 else 1

    missing = asyncio.run(run_for_env(console, args.env_file, expected))
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
