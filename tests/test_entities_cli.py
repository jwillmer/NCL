"""CLI-level tests for ``mtss topics list / audit / consolidate``.

Invokes the actual Typer app via ``typer.testing.CliRunner`` against a
fresh on-disk SQLite DB built per-test. This guards the full CLI surface
(option parsing, exit codes, JSON side-effects, Rich output) against
regressions when the underlying storage helpers change.

The commands are entirely local — no Supabase calls — so these tests
don't need to mock the supabase client.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from mtss.models.topic import Topic
from mtss.storage.sqlite_client import SqliteStorageClient


# ── helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def runner() -> CliRunner:
    # PYTHONIOENCODING so Rich's Unicode box chars don't choke under
    # Windows cp1252 when pytest captures subprocess-ish stdout.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # mix_stderr=False so we can assert on stderr separately when needed.
    return CliRunner(mix_stderr=False)


@pytest.fixture
def app():
    """Import the CLI app once — avoids module-load side effects per test."""
    from mtss.cli import app as _app
    return _app


def _seed_topics(output_dir: Path, topics: list[Topic]) -> None:
    """Seed the given Topic rows into a fresh ingest.db under output_dir."""
    client = SqliteStorageClient(output_dir=output_dir)
    try:
        for topic in topics:
            # insert_topic is async but the method body is synchronous SQL.
            import asyncio
            asyncio.run(client.insert_topic(topic))
    finally:
        client._conn.close()


def _populated_output_dir(tmp_path: Path) -> Path:
    """Build an output/ with a handful of topics covering sort/filter cases."""
    out = tmp_path / "output"
    out.mkdir()
    _seed_topics(
        out,
        [
            Topic(
                id=uuid4(), name="cargo damage",
                display_name="Cargo Damage",
                embedding=[1.0, 0.0, 0.0, 0.0],
                chunk_count=10, document_count=4,
            ),
            Topic(
                id=uuid4(), name="engine issues",
                display_name="Engine Issues",
                embedding=[0.0, 1.0, 0.0, 0.0],
                chunk_count=3, document_count=2,
            ),
            Topic(
                id=uuid4(), name="port call planning",
                display_name="Port Call Planning",
                embedding=[0.0, 0.0, 1.0, 0.0],
                chunk_count=1, document_count=1,
            ),
        ],
    )
    return out


# ── topics list ─────────────────────────────────────────────────────

class TestTopicsList:
    def test_default_sort_by_chunks(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(app, ["topics", "list", "--output-dir", str(out)])
        assert result.exit_code == 0, result.output
        # All three topics appear, most-chunks-first.
        cargo_idx = result.output.index("Cargo Damage")
        engine_idx = result.output.index("Engine Issues")
        port_idx = result.output.index("Port Call Planning")
        assert cargo_idx < engine_idx < port_idx

    def test_sort_by_name(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(out), "--sort", "name"]
        )
        assert result.exit_code == 0, result.output
        # Alphabetical: Cargo → Engine → Port.
        cargo_idx = result.output.index("Cargo Damage")
        engine_idx = result.output.index("Engine Issues")
        port_idx = result.output.index("Port Call Planning")
        assert cargo_idx < engine_idx < port_idx

    def test_filter_narrows_results(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(out), "--filter", "engine"]
        )
        assert result.exit_code == 0, result.output
        assert "Engine Issues" in result.output
        assert "Cargo Damage" not in result.output
        assert "Port Call Planning" not in result.output

    def test_min_chunks_filters_low_count_topics(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(out), "--min-chunks", "3"]
        )
        assert result.exit_code == 0, result.output
        assert "Cargo Damage" in result.output
        assert "Engine Issues" in result.output
        # chunk_count=1 < 3 → filtered out
        assert "Port Call Planning" not in result.output

    def test_limit_zero_shows_all(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(out), "--limit", "0"]
        )
        assert result.exit_code == 0, result.output
        # All three present; no "more row(s) hidden" trailer.
        assert "Cargo Damage" in result.output
        assert "Engine Issues" in result.output
        assert "Port Call Planning" in result.output
        assert "more row(s) hidden" not in result.output

    def test_json_out_writes_full_filtered_list(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = _populated_output_dir(tmp_path)
        json_path = tmp_path / "topics.json"
        result = runner.invoke(
            app,
            [
                "topics", "list",
                "--output-dir", str(out),
                "--json", str(json_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        names = {p["display_name"] for p in payload}
        assert names == {"Cargo Damage", "Engine Issues", "Port Call Planning"}
        # Each entry carries chunk/doc counts.
        for p in payload:
            assert "chunk_count" in p and "document_count" in p

    def test_missing_ingest_db_exits_nonzero(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        empty_dir = tmp_path / "empty_output"
        empty_dir.mkdir()
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(empty_dir)]
        )
        assert result.exit_code == 1
        assert "ingest.db" in result.output

    def test_bad_sort_value_rejects(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app, ["topics", "list", "--output-dir", str(out), "--sort", "bogus"]
        )
        assert result.exit_code == 2


# ── topics audit ────────────────────────────────────────────────────

class TestTopicsAudit:
    def test_no_band_hits_exits_clean(self, runner: CliRunner, app, tmp_path: Path):
        out = _populated_output_dir(tmp_path)  # all pairs orthogonal → 0.0 sim
        result = runner.invoke(
            app,
            [
                "topics", "audit",
                "--output-dir", str(out),
                "--lower", "0.75",
                "--upper", "0.85",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No near-duplicate pairs" in result.output

    def test_hits_render_table_with_similarity_and_counts(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = tmp_path / "output"
        out.mkdir()
        _seed_topics(
            out,
            [
                Topic(
                    id=uuid4(), name="mid_a", display_name="Mid A",
                    embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    chunk_count=10, document_count=3,
                ),
                # Cosine similarity ≈ 0.80
                Topic(
                    id=uuid4(), name="mid_b", display_name="Mid B",
                    embedding=[0.80, 0.60, 0.0, 0.0, 0.0, 0.0],
                    chunk_count=2, document_count=1,
                ),
            ],
        )
        result = runner.invoke(
            app,
            [
                "topics", "audit",
                "--output-dir", str(out),
                "--lower", "0.75",
                "--upper", "0.95",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Mid A" in result.output
        assert "Mid B" in result.output
        # Banner line reports the band + pair count.
        assert "1 pair" in result.output

    def test_invalid_band_returns_error(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "topics", "audit",
                "--output-dir", str(out),
                "--lower", "0.9",
                "--upper", "0.8",
            ],
        )
        assert result.exit_code != 0

    def test_missing_ingest_db_exits_nonzero(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        empty_dir = tmp_path / "empty_output"
        empty_dir.mkdir()
        result = runner.invoke(
            app, ["topics", "audit", "--output-dir", str(empty_dir)]
        )
        assert result.exit_code == 1


# ── topics consolidate ──────────────────────────────────────────────

class TestTopicsConsolidate:
    def test_name_strategy_zero_collisions(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        """Normalized-name buckets with distinct keys yield no merges."""
        out = _populated_output_dir(tmp_path)  # 3 distinct names
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "name",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No merges at this threshold" in result.output

    def test_name_strategy_dry_run_detects_collision(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = tmp_path / "output"
        out.mkdir()
        _seed_topics(
            out,
            [
                Topic(
                    id=uuid4(), name="pre inspection documentation",
                    display_name="Pre-Inspection Documentation",
                    embedding=[1.0, 0.0, 0.0, 0.0], chunk_count=9,
                ),
                Topic(
                    id=uuid4(), name="pre-inspection, documentation",
                    display_name="Pre-inspection, Documentation",
                    embedding=[0.1, 0.9, 0.0, 0.0], chunk_count=3,
                ),
            ],
        )
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "name",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "DRY-RUN" in result.output
        # Exactly one merge previewed; "APPLY" must not appear.
        assert "1 merge" in result.output
        assert "APPLY" not in result.output

        # Dry-run did NOT touch the DB.
        import sqlite3
        conn = sqlite3.connect(out / "ingest.db")
        try:
            count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        finally:
            conn.close()
        assert count == 2

    def test_name_strategy_apply_with_yes_mutates_db(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = tmp_path / "output"
        out.mkdir()
        _seed_topics(
            out,
            [
                Topic(
                    id=uuid4(), name="pre inspection documentation",
                    display_name="Pre-Inspection Documentation",
                    embedding=[1.0, 0.0, 0.0, 0.0], chunk_count=9,
                ),
                Topic(
                    id=uuid4(), name="pre-inspection, documentation",
                    display_name="Pre-inspection, Documentation",
                    embedding=[0.1, 0.9, 0.0, 0.0], chunk_count=3,
                ),
                # Distinct — must survive.
                Topic(
                    id=uuid4(), name="port call planning",
                    display_name="Port Call Planning",
                    embedding=[0.0, 0.0, 1.0, 0.0], chunk_count=2,
                ),
            ],
        )
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "name",
                "--apply", "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Merged 1 topic pair" in result.output

        import sqlite3
        conn = sqlite3.connect(out / "ingest.db")
        try:
            count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            names = {r[0] for r in conn.execute("SELECT display_name FROM topics")}
        finally:
            conn.close()
        assert count == 2  # 3 - 1 absorbed
        # Keeper kept (higher chunk count), absorbed gone.
        assert "Pre-Inspection Documentation" in names
        assert "Pre-inspection, Documentation" not in names

    def test_cluster_strategy_dry_run_plan(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        """Cluster merge collapses a transitive {a,b,c} chain at threshold 0.75."""
        out = tmp_path / "output"
        out.mkdir()
        _seed_topics(
            out,
            [
                Topic(
                    id=uuid4(), name="a", display_name="A",
                    embedding=[1.0, 0.0, 0.0, 0.0], chunk_count=10,
                ),
                # sim(a,b)=0.80, sim(b,c)≈0.91, sim(a,c)≈0.527
                Topic(
                    id=uuid4(), name="b", display_name="B",
                    embedding=[0.80, 0.60, 0.0, 0.0], chunk_count=3,
                ),
                Topic(
                    id=uuid4(), name="c", display_name="C",
                    embedding=[0.48, 0.877, 0.0, 0.0], chunk_count=2,
                ),
                Topic(
                    id=uuid4(), name="far", display_name="Far",
                    embedding=[0.0, 0.0, 0.0, 1.0], chunk_count=5,
                ),
            ],
        )
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "cluster",
                "--threshold", "0.75",
            ],
        )
        assert result.exit_code == 0, result.output
        # {a, b, c} cluster → 2 merges (keeper=a absorbs b and c).
        assert "2 merge" in result.output
        assert "DRY-RUN" in result.output

    def test_pairwise_strategy_high_threshold_no_candidates(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = _populated_output_dir(tmp_path)  # all orthogonal
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "pairwise",
                "--threshold", "0.95",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No merges at this threshold" in result.output

    def test_unknown_strategy_errors(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        out = _populated_output_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "topics", "consolidate",
                "--output-dir", str(out),
                "--strategy", "foo",
            ],
        )
        assert result.exit_code == 2
        # Output mentions the accepted values so operators know how to fix it.
        assert "pairwise" in result.output or "cluster" in result.output

    def test_missing_ingest_db_exits_nonzero(
        self, runner: CliRunner, app, tmp_path: Path
    ):
        empty_dir = tmp_path / "empty_output"
        empty_dir.mkdir()
        result = runner.invoke(
            app,
            ["topics", "consolidate", "--output-dir", str(empty_dir)],
        )
        assert result.exit_code == 1
