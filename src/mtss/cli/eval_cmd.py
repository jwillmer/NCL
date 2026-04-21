"""External evaluation harness CLI — strictly targets the test Supabase.

Workflow:
    mtss eval seed    ...  # push local data/output/ingest.db → test Supabase
    mtss eval verify       # ping test instance, compare row counts vs local
    mtss eval run     ...  # Phase 1: execute goldens, log everything
    mtss eval score   ...  # Phase 2: run auto-graders (humans judge responses)
    mtss eval diff    ...  # Phase 3: side-by-side delta vs another run
    mtss eval list         # show prior runs
    mtss eval show    ...  # cat one run's summary

Safety: every command in this group resolves SUPABASE_URL after loading the
env file and REFUSES to run if the project ref matches a known prod instance,
unless ``--allow-prod`` is passed. Production seeding is reserved for the
top-level ``mtss import`` command and never reachable from here.

Default --env-file is `.env.test`. Override only with extreme care.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from ._common import console

DEFAULT_ENV_FILE = ".env.test"
DEFAULT_QUESTIONS = "tests/eval/goldens/questions.yaml"
DEFAULT_RUNS_ROOT = Path("tests/eval/runs")

# Known production Supabase project refs. Any URL whose host begins with
# `<ref>.` is treated as prod and rejected by `_resolve_target` unless the
# caller explicitly passes --allow-prod. Override via comma-separated env var
# MTSS_PROD_SUPABASE_REFS so future projects can extend without code changes.
_BUILTIN_PROD_REFS = ("oupmykuvurhzxoaelyjs",)


def _prod_refs() -> tuple[str, ...]:
    extra = os.environ.get("MTSS_PROD_SUPABASE_REFS", "")
    return _BUILTIN_PROD_REFS + tuple(r.strip() for r in extra.split(",") if r.strip())


def _project_ref_from_url(url: str) -> str:
    """Extract `bcbssdemwtcsylijmlox` from `https://bcbssdemwtcsylijmlox.supabase.co`."""
    if not url:
        return ""
    host = url.replace("https://", "").replace("http://", "").rstrip("/").split("/", 1)[0]
    return host.split(".", 1)[0]


def _resolve_target(env_file: str, allow_prod: bool, *, enable_langfuse: bool) -> tuple[str, str]:
    """Load env, identify the Supabase target, refuse if prod (unless allowed).

    Returns (project_ref, supabase_url) — project_ref is also printed in the
    banner so the user always knows which database is about to be touched.
    """
    from tests.eval.env import setup_eval_env

    setup_eval_env(env_file, enable_langfuse=enable_langfuse)

    url = os.environ.get("SUPABASE_URL", "")
    ref = _project_ref_from_url(url)

    if ref in _prod_refs() and not allow_prod:
        console.print(
            f"[red bold]REFUSED:[/red bold] env file {env_file!r} resolves to a "
            f"production Supabase project (ref={ref!r}, url={url!r}).\n"
            f"[dim]The 'mtss eval' group is for the test instance only. "
            f"Pass --allow-prod to override (and double-check first), or use "
            f"`mtss import` for production data writes.[/dim]"
        )
        raise typer.Exit(2)

    badge = "[red bold]PROD[/red bold]" if ref in _prod_refs() else "[green]TEST[/green]"
    console.print(f"[bold]EVAL TARGET:[/bold] {badge} · ref={ref} · {url}")
    return ref, url


def register(app: typer.Typer) -> None:
    eval_app = typer.Typer(
        help="External evaluation framework — TEST INSTANCE ONLY. Use `mtss import` for prod.",
        no_args_is_help=True,
    )
    app.add_typer(eval_app, name="eval")

    # ------------------------------------------------------------------
    # SEED — push local ingest.db to the test Supabase
    # ------------------------------------------------------------------

    @eval_app.command("seed")
    def seed_cmd(
        env_file: str = typer.Option(DEFAULT_ENV_FILE, "--env-file"),
        output_dir: str = typer.Option("data/output", "--output-dir",
                                        help="Local ingest output dir to push from"),
        apply: bool = typer.Option(False, "--apply",
                                    help="Actually push (default is dry-run)"),
        include_archives: bool = typer.Option(False, "--include-archives",
                                              help="Also upload archive markdown + attachments to "
                                                   "Supabase Storage (default skipped — LLM never reads them)"),
        limit: Optional[int] = typer.Option(None, "--limit",
                                            help="Wave mode: push first N pending only (resumable)"),
        allow_prod: bool = typer.Option(False, "--allow-prod", hidden=True),
    ) -> None:
        """Push local data/output/ingest.db → test Supabase. Dry-run by default."""
        _resolve_target(env_file, allow_prod, enable_langfuse=False)

        # Verify required creds
        missing = [k for k in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_DB_URL")
                   if not os.environ.get(k)]
        if missing:
            console.print(f"[red]Missing in {env_file}: {', '.join(missing)}[/red]")
            raise typer.Exit(2)

        out = Path(output_dir).resolve()
        if not (out / "ingest.db").exists():
            console.print(f"[red]No ingest.db at {out}[/red]")
            raise typer.Exit(3)

        console.print(f"[cyan]Source[/cyan]      {out}")
        console.print(f"[cyan]Apply[/cyan]       {apply}  (skip archives: {not include_archives})")
        if not apply:
            console.print("\n[yellow][DRY RUN][/yellow] Re-run with --apply to push.\n")

        from mtss.cli.import_cmd import _import_data
        asyncio.run(_import_data(
            output_dir=out,
            skip_archives=not include_archives,
            skip_vessels=False,
            dry_run=not apply,
            verbose=False,
            limit=limit,
            rewrite_topics_from=None,
        ))

    # ------------------------------------------------------------------
    # VERIFY — ping test instance + compare row counts to local
    # ------------------------------------------------------------------

    @eval_app.command("verify")
    def verify_cmd(
        env_file: str = typer.Option(DEFAULT_ENV_FILE, "--env-file"),
        output_dir: str = typer.Option("data/output", "--output-dir"),
        allow_prod: bool = typer.Option(False, "--allow-prod", hidden=True),
    ) -> None:
        """Ping test Supabase, show row counts side-by-side with local ingest.db."""
        _resolve_target(env_file, allow_prod, enable_langfuse=False)

        async def _run():
            from mtss.storage.supabase_client import SupabaseClient
            client = SupabaseClient()
            pool = await client.get_pool()
            async with pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                remote = {
                    "documents": await conn.fetchval("SELECT COUNT(*) FROM documents"),
                    "chunks":    await conn.fetchval("SELECT COUNT(*) FROM chunks"),
                    "chunks_with_embedding": await conn.fetchval(
                        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"),
                    "topics":    await conn.fetchval("SELECT COUNT(*) FROM topics"),
                    "vessels":   await conn.fetchval("SELECT COUNT(*) FROM vessels"),
                }
            await client.close()
            return version, remote

        version, remote = asyncio.run(_run())

        # Local counts
        import sqlite3
        db = Path(output_dir).resolve() / "ingest.db"
        local: dict[str, int | None] = {k: None for k in remote}
        if db.exists():
            con = sqlite3.connect(db)
            try:
                cur = con.cursor()
                local["documents"] = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                local["chunks"]    = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                local["chunks_with_embedding"] = cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
                local["topics"]    = cur.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
                # vessels in SQLite live in chunk_topics? actually no — try a best effort
                try:
                    local["vessels"] = cur.execute("SELECT COUNT(*) FROM vessels").fetchone()[0]
                except sqlite3.OperationalError:
                    local["vessels"] = None
            finally:
                con.close()

        console.print(f"[dim]{version[:60]}[/dim]")
        table = Table(title="Row counts")
        table.add_column("Table", style="cyan")
        table.add_column("Test", justify="right")
        table.add_column("Local", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("State")
        for k in ("documents", "chunks", "chunks_with_embedding", "topics", "vessels"):
            r = remote[k]
            l = local[k]
            delta = "-" if l is None else f"{r - l:+d}"
            state = "ok" if (l is not None and r == l) else ("missing" if r == 0 else "partial")
            color = {"ok": "green", "partial": "yellow", "missing": "red"}[state]
            table.add_row(k, str(r), "?" if l is None else str(l), delta, f"[{color}]{state}[/{color}]")
        console.print(table)

    # ------------------------------------------------------------------
    # RUN — Phase 1
    # ------------------------------------------------------------------

    @eval_app.command("run")
    def run_cmd(
        run_id: Optional[str] = typer.Option(None, "--run-id", help="Identifier; defaults to timestamp"),
        env_file: str = typer.Option(DEFAULT_ENV_FILE, "--env-file"),
        questions: str = typer.Option(DEFAULT_QUESTIONS, "--questions"),
        out_dir: Optional[str] = typer.Option(None, "--out"),
        concurrency: int = typer.Option(2, "--concurrency", "-c"),
        filter_ids: Optional[str] = typer.Option(None, "--filter", help="Comma-separated question IDs"),
        notes: Optional[str] = typer.Option(None, "--notes"),
        allow_prod: bool = typer.Option(False, "--allow-prod", hidden=True),
    ) -> None:
        """Phase 1: execute goldens through the agent, write results.jsonl."""
        _resolve_target(env_file, allow_prod, enable_langfuse=True)

        from tests.eval.runner import execute_run

        rid = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out = Path(out_dir) if out_dir else DEFAULT_RUNS_ROOT / rid
        ids = [s.strip() for s in filter_ids.split(",")] if filter_ids else None

        console.print(f"[cyan]Run[/cyan] {rid} → [dim]{out}[/dim]")
        manifest = asyncio.run(execute_run(
            run_id=rid,
            questions_path=Path(questions),
            out_dir=out,
            env_file=env_file,
            concurrency=concurrency,
            filter_ids=ids,
            notes=notes,
        ))
        console.print(f"[green]Done[/green] · {manifest.question_count} questions · {out}/results.jsonl")

    # ------------------------------------------------------------------
    # SCORE — Phase 2 (auto-graders only; humans judge responses manually)
    # ------------------------------------------------------------------

    @eval_app.command("score")
    def score_cmd(
        run_dir: str = typer.Argument(..., help="Path to a runs/<id>/ directory"),
        env_file: str = typer.Option(DEFAULT_ENV_FILE, "--env-file"),
        questions: str = typer.Option(DEFAULT_QUESTIONS, "--questions"),
        concurrency: int = typer.Option(4, "--concurrency", "-c"),
        allow_prod: bool = typer.Option(False, "--allow-prod", hidden=True),
    ) -> None:
        """Phase 2: apply auto-graders to a run. No LLM calls.

        Humans judge response quality directly against the goldens —
        this just computes citation coverage, format adherence, and
        (when goldens carry labeled chunk_ids) retrieval metrics.
        """
        _resolve_target(env_file, allow_prod, enable_langfuse=False)

        from tests.eval.scoring import execute_judge

        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            console.print(f"[red]Run dir not found: {run_dir_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Score[/cyan] {run_dir_path}")
        summary = asyncio.run(execute_judge(
            run_dir=run_dir_path,
            questions_path=Path(questions),
            concurrency=concurrency,
        ))
        console.print(
            f"[green]Done[/green] · overall_mean=[bold]{summary.overall_mean}[/bold] · "
            f"see {run_dir_path}/summary.md"
        )

    # ------------------------------------------------------------------
    # DIFF — Phase 3 (no env / target needed; pure file ops)
    # ------------------------------------------------------------------

    @eval_app.command("diff")
    def diff_cmd(
        base: str = typer.Argument(..., help="Baseline run directory"),
        candidate: str = typer.Argument(..., help="Candidate run directory"),
        out: Optional[str] = typer.Option(None, "--out"),
    ) -> None:
        """Phase 3: per-question + aggregate delta between two runs."""
        from tests.eval.diff import diff_runs

        base_dir = Path(base)
        cand_dir = Path(candidate)
        for d in (base_dir, cand_dir):
            if not d.exists():
                console.print(f"[red]Missing: {d}[/red]")
                raise typer.Exit(1)

        report = diff_runs(base_dir, cand_dir)
        out_path = Path(out) if out else cand_dir / f"diff_vs_{base_dir.name}.md"
        out_path.write_text(report, encoding="utf-8")
        console.print(f"[green]Diff written[/green] · {out_path}")

    # ------------------------------------------------------------------
    # LIST / SHOW (no target needed)
    # ------------------------------------------------------------------

    @eval_app.command("list")
    def list_cmd(
        runs_root: str = typer.Option(str(DEFAULT_RUNS_ROOT), "--runs-root"),
    ) -> None:
        """Show prior eval runs with their headline scores."""
        root = Path(runs_root)
        if not root.exists():
            console.print(f"[yellow]No runs at {root}[/yellow]")
            return
        table = Table(title=f"Eval runs in {root}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Started")
        table.add_column("Qs", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Cost $", justify="right")
        table.add_column("Notes")

        from tests.eval.types import RunManifest, ScoreSummary
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest_path = run_dir / "manifest.json"
            summary_path = run_dir / "summary.json"
            if not manifest_path.exists():
                continue
            try:
                m = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            mean = "-"
            cost = "-"
            if summary_path.exists():
                try:
                    s = ScoreSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
                    mean = f"{s.overall_mean:.3f}"
                    cost = f"{s.total_cost_usd:.4f}"
                except Exception:
                    pass
            table.add_row(
                m.run_id,
                m.started_at.strftime("%Y-%m-%d %H:%M") if m.started_at else "-",
                str(m.question_count),
                mean,
                cost,
                (m.notes or "")[:40],
            )
        console.print(table)

    @eval_app.command("show")
    def show_cmd(run_dir: str = typer.Argument(...)) -> None:
        """Print a run's summary.md to the console."""
        path = Path(run_dir) / "summary.md"
        if not path.exists():
            console.print(f"[red]No summary.md at {path}[/red]")
            raise typer.Exit(1)
        console.print(path.read_text(encoding="utf-8"))
