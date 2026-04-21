"""Phase 1 orchestrator: load goldens, execute, append-write results.jsonl.

Append-write means: a crashed/cancelled run still produces partial output,
and a re-run with the same `--run-id` skips already-completed questions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from .env import settings_snapshot
from .fixtures.agent_runner import run_questions
from .types import GoldenQuestion, RunManifest, RunResult

logger = logging.getLogger(__name__)


def load_goldens(path: Path) -> List[GoldenQuestion]:
    """Load and validate all golden questions from a YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "questions" not in raw:
        raise ValueError(f"{path}: expected top-level 'questions:' list")
    return [GoldenQuestion(**q) for q in raw["questions"]]


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def _completed_question_ids(results_path: Path) -> set[str]:
    """Return question_ids already present in results.jsonl (resume support)."""
    if not results_path.exists():
        return set()
    seen: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(json.loads(line)["question_id"])
        except Exception:
            continue
    return seen


async def execute_run(
    *,
    run_id: str,
    questions_path: Path,
    out_dir: Path,
    env_file: str,
    concurrency: int = 2,
    filter_ids: Optional[List[str]] = None,
    session_prefix: str = "eval",
    langfuse_tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> RunManifest:
    """Run all (or a filtered subset of) golden questions and write results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    manifest_path = out_dir / "manifest.json"

    all_questions = load_goldens(questions_path)
    if filter_ids:
        ids = set(filter_ids)
        all_questions = [q for q in all_questions if q.id in ids]
        if not all_questions:
            raise ValueError(f"No goldens matched --filter-ids={sorted(ids)}")

    already_done = _completed_question_ids(results_path)
    pending = [q for q in all_questions if q.id not in already_done]

    logger.info(
        "Run %s: %d total, %d already done, %d to execute",
        run_id, len(all_questions), len(already_done), len(pending),
    )

    started = datetime.utcnow()
    manifest = RunManifest(
        run_id=run_id,
        started_at=started,
        git_sha=_git_sha(),
        env_file=env_file,
        questions_file=str(questions_path),
        question_count=len(all_questions),
        settings_snapshot=settings_snapshot(),
        notes=notes,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    if not pending:
        logger.info("Nothing to run — all questions already in results.jsonl")
        return manifest

    # Append-write callback: each completed question hits disk immediately
    def _append(result: RunResult) -> None:
        with results_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    await run_questions(
        pending,
        run_id=run_id,
        concurrency=concurrency,
        session_prefix=session_prefix,
        langfuse_tags=langfuse_tags,
        on_complete=_append,
    )

    manifest.completed_at = datetime.utcnow()
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_results(results_path: Path) -> List[RunResult]:
    """Read results.jsonl back into RunResult objects."""
    out: List[RunResult] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(RunResult.model_validate_json(line))
    return out
